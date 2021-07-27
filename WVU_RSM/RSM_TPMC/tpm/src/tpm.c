/* Calculates the drag coefficient of a given object (from a .stl mesh file) using */
/* the test particle method */

/* INPUTS: */

/* GS_MODEL: Specifies gas-surface interaction model */
/* 0 = Maxwell's Model */
/* 1 = Diffuse Reflection with Incomplete Accommodation (DRIA) */
/* 2 = Cercignani-Lampis-Lord (CLL) */

/* Ux, Uy, Uz = satellite speed relative to atmosphere [m/s] */
/* Ts = satellite surface temperature [K] */
/* Ta = atmospheric translational temperature [K] */
/* sigmat = tangential momentum accommodation coefficient [unitless] (ONLY USED IN CLL) */
/* alphan = normal energy accommodation coefficient [unitless] (ONLY USED IN CLL) */
/* alpha = energy accommodation coefficient (ONLY USED IN DRIA) */
/* epsilon = fraction of particles specularly reflected (ONLY USED IN MAXWELL'S MODEL) */
/* X = species mole fractions list in order [O, O2, N, N2, He, H] */
/* f = directory path for the mesh file to be used e.g "./Mesh Files/sphere_res05_ascii.stl" */

/* OUTPUT: */
/* Cd = Drag Coefficient [unitless] */

#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include "hdf5.h"
#include "tpm.h"
#include "../config.h"

#if PARALLEL
#include "mpi.h"
#endif /* PARALLEL */

gsl_rng * rr;
gsl_rng *rrr;

long int seed;
int nProcs;

#if PARALLEL
int rank;
MPI_Comm io_comm;
#endif /* PARALLEL */

  

int main(int argc, char *argv[]) {

#if PARALLEL
    /***************************** MPI Initialization *******************************/
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    /*Create communicator for the I/O functions */
    MPI_Comm_dup(MPI_COMM_WORLD, &io_comm);
    /* Find my ID number */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* Find the total number of procs */
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    double CdT;
#endif /* PARALLEL */

    /************************** Random Number Generator Setup *******************/
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    rr = gsl_rng_alloc(T);
#if DYNAMIC_SEED
#if PARALLEL
    seed = time(NULL) * (rank + 1);
#else /* PARALLEL */
    seed = time (NULL);
#endif /*PARALLEL */
#else /* DYNAMIC SEED */
    seed = 0;
#endif /* DYNAMIC SEED */
    //printf("seed = %ld\n", seed);
    gsl_rng_set(rr, seed);

    double Cd;
    char outfilename[1024];
    char areaoutfilename[1024];
    char filename[1024];
    char objname[1024];
    char deflection[1024];
    char tempdeflection[1024];
    char areafilename[1024];

#if PARALLEL
    /************* Set up number of 0 to use before processor number ****************/
    char *zeros = "p";
    if (nProcs > 9) {
        if (rank < 10) zeros = "p0";
    }
    if (nProcs > 99) {
        if (rank < 10) {
            zeros = "p00";
        } else if (rank < 100) {
            zeros = "p0";
        }
    }
#endif /*PARALLEL*/


    char line[1024];
#if PARALLEL
    sprintf(outfilename, "tempfiles/Cdout_%s%d.dat", zeros, rank);
#else /* PARALLEL */
    sprintf(outfilename, "tempfiles/Cdout.dat");
#endif /* PARALLEL */

    FILE *fout = fopen(outfilename, "w");
    if (fout == NULL) {
        printf("Error opening %s\n", outfilename);
        exit(1);
    }
#if PARALLEL
    FILE *ftot = fopen("tempfiles/Cdout.dat", "w");
    FILE *ftota = fopen("Outputs/Area_Regression_Models/Aout.dat","w");
    if (ftot == NULL) {
        printf("Error opening %s\n", "tempfiles/Cdout.dat");
        exit(1);
    }
     if (ftota == NULL) {
        printf("Error opening %s\n", "Outputs/Area_Regression_Models/Aout.dat");
        exit(1);
    }
#endif /* PARALLEL */

    double X[NSPECIES] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int GSI_MODEL = 0;
    int ROTATION_FLAG = 0;
    int NUM_POINTS = 0;
    int NCOMP = 0;
    int NVAR = 0; 

#if PARALLEL
    int ppN; // Particles per processor
#endif /* PARALLEL */

    int pc, iTOT;

    /* READ IN ENSEMBLE PARAMETERS */
    read_input(objname, X, &GSI_MODEL, &ROTATION_FLAG);



/*     Read HDF5 file      */

    hid_t file_id, dataset_id, space_id; /* identifiers */
    herr_t status;
    hsize_t dims[2];

    int i, j, k, ndims;
    double **rdata;

    //printf("Opening HDF5 tpm.ensemble.h5... \n");

    /* Open an existing file. */
    file_id = H5Fopen(FILEhf, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, "/tpm", H5P_DEFAULT);

    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(space_id, dims, NULL);

    printf("Dims: %ld %ld\n", dims[0], dims[1]);

    /*Allocate array of pointers to rows*/
    rdata = (double **) malloc(dims[0] * sizeof(double *));


    /*Allocate space for integer data.*/
    rdata[0] = (double *) malloc(dims[0] * dims[1] * sizeof(double));

    /*Set the rest of the pointers to rows to the correct addresses*/
    for (i=1; i<dims[0]; i++)
        rdata[i] = rdata[0] + i * dims[1];

    /*Read the data using the default properties*/
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata[0]);

    //printf("%o \n", rdata[0]);
    //for (i=0; i<dims[1]; i++)
    //	printf("%20lf %10p\n", rdata[0][i], &(rdata[0][i]));
    //printf("\n");

    /* Close the space */
    status = H5Sclose(space_id);

    /* Close the dataset */
    status = H5Dclose(dataset_id);

    /* Close the file */
    status = H5Fclose(file_id);

    NUM_POINTS = dims[0]; //number of rows in tpm.ensemble.h5
    NVAR = dims[1]; //number of columns in tpm.ensemble.h5
    //Get number of Components being deflected
    if(GSI_MODEL == 0){NCOMP = NVAR - 6; }
    else if (GSI_MODEL == 1) {NCOMP = NVAR - 6;}
    else {NCOMP = NVAR - 7;}
	
    printf("GSI_MODEL=%d NCOMP=%d\n",GSI_MODEL, NCOMP);

    /* Define new data variables */
    double *LHS_Umag, *LHS_theta, *LHS_phi, *LHS_Ts, *LHS_Ta, *LHS_An, *LHS_St, *LHS_Al, *LHS_Ep;
    double **Comp_Deflec;
    double area;
    int ncols;
    LHS_Umag = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_theta = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_phi = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_Ts = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_Ta = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_An = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_St = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_Al = (double *) calloc(NUM_POINTS, sizeof(double));
    LHS_Ep = (double *) calloc(NUM_POINTS, sizeof(double));

    Comp_Deflec = (double **)malloc(NUM_POINTS*sizeof(double *));
    Comp_Deflec[0] = (double *) malloc(NUM_POINTS * NCOMP * sizeof(double));

    for (i=1; i<NUM_POINTS; i++)
        Comp_Deflec[i] = Comp_Deflec[0] + i * NCOMP;

    if (GSI_MODEL == 0) ncols=6;
    else if (GSI_MODEL == 1) ncols=6;
    else if (GSI_MODEL == 2) ncols=7;

    for (i = 0; i < NUM_POINTS; i++) {
        //printf("GSI_MODEL= %d\n", GSI_MODEL);
        if (GSI_MODEL == 0) {
           LHS_Umag[i] =  rdata[i][0] ;
           LHS_theta[i] = rdata[i][1] ;
           LHS_phi[i] =   rdata[i][2];
           LHS_Ts[i] =    rdata[i][3];
           LHS_Ta[i] =    rdata[i][4];
           LHS_Ep[i] =    rdata[i][5];
        } else if (GSI_MODEL == 1) {
           LHS_Umag[i] =  rdata[i][0];
           LHS_theta[i] = rdata[i][1];
           LHS_phi[i] =   rdata[i][2];
           LHS_Ts[i] =    rdata[i][3];
           LHS_Ta[i] =    rdata[i][4];
           LHS_Al[i] =    rdata[i][5];
        } else {
           LHS_Umag[i] =  rdata[i][0];
           LHS_theta[i] = rdata[i][1];
           LHS_phi[i] =   rdata[i][2];
           LHS_Ts[i] =    rdata[i][3];
           LHS_Ta[i] =    rdata[i][4];
           LHS_An[i] =    rdata[i][5];
           LHS_St[i] =    rdata[i][6];
        }
	printf("[%5d] Umag= %12lf theta= %12lf phi= %12lf ", i, LHS_Umag[i], LHS_theta[i], LHS_phi[i]);

	if (ncols+NCOMP>dims[1])
	{
	   printf("ERROR: Not enough columns in data" );
           return -1;	
	}

       for(j = 0;j<NCOMP;j++) {
	        Comp_Deflec[i][j] = rdata[i][j+ncols];
		printf("C%d= %12lf ", j,Comp_Deflec[i][j]);        
	}
        printf("\n");

   }

#if PARALLEL
  if (NUM_POINTS%nProcs==0) {
     ppN = NUM_POINTS/nProcs;
     if(rank==0) {
       printf("NUM_POINTS=%d nProcs=%d ppN=%d\n", NUM_POINTS, nProcs, ppN);
     }
  }
  else
  {
    printf("ERROR: The number of points is not integer divisible by the number MPI processes");
    exit(1);
  }
#endif /* PARALLEL */


#if PARALLEL

  /* Output filename for error diagnostics */
  char *speciesname, istr[3]; // BAD IDEA, limited to 999
  if(X[0] == 1.0) { /* Atomic Oxygen */
    speciesname = "O";
  } else if (X[1] == 1.0) { /* Diatomic Oxygen */
    speciesname = "O2";
  } else if (X[2] == 1.0) { /* Atomic Nitrogen */
    speciesname = "N";
  } else if (X[3] == 1.0) { /* Diatomic Nitrogen */
    speciesname = "N2";
  } else if (X[4] == 1.0) { /* Atomic Helium */
    speciesname = "He";
  } else { /* Atomic Hydrogen */
    speciesname = "H";
  }
  sprintf(areafilename, "Outputs/Area_Regression_Models/area_%s_%s%d.dat", objname,zeros,rank);
  FILE *farea = fopen(areafilename, "w");
  if(farea == NULL)
  {
	printf("Error opening %s\n",areafilename);  
  	exit(1);             
  }
  //fprintf(farea,"Projected Area of Satellite \n");




  for(i=rank*ppN; i<(rank+1)*ppN; i++) {


    // If rotation is being performed, then the appropriate filename must be read 
    if(ROTATION_FLAG == 2){
    sprintf(filename,"%s",objname);
    sprintf(istr,"%d",i);
    strcat(filename,istr);
    strcat(filename,".stl");}
    else{
    strcpy(filename,objname);
    }  // Otherwise use the objectname as the filename 
    printf("I am rank %d, computing case %d : %s\n",rank,i, filename);



    iTOT=0;
    if(rank==0) {
      printf("Sim for Processor = %d \n",i);
      // WARNING: ppN is now Integer the resulting division will be integer too 
      //if((int)(100*i/ppN) % 1 == 0 && (int)(100*i/ppN)/1 > 0) {
	//printf("%d percent complete\r", (int)(100*i/ppN));
      //}
    }

    if(rank>-1) {    //Serial Loop

      Cd = testparticle(filename, LHS_Umag[i], LHS_theta[i], LHS_phi[i], LHS_Ts[i], LHS_Ta[i], LHS_Ep[i], \
		      LHS_Al[i], LHS_An[i], LHS_St[i], X, GSI_MODEL, iTOT, &area);
      //printf("Cd = %e\n", Cd);

      if(GSI_MODEL==1){ //DRIA
         fprintf(fout, "%e %e %e %e %e %e %e \n",
      LHS_Umag[i], LHS_Ts[i], LHS_Ta[i], LHS_Al[i],LHS_theta[i], LHS_phi[i], Cd);
      }


      else{fprintf(fout, "%e %e %e %e %e %e %e %e \n", 
	    LHS_Umag[i], LHS_Ts[i], LHS_Ta[i], LHS_An[i], LHS_St[i], LHS_theta[i], LHS_phi[i], Cd);} //CLL
      if(ROTATION_FLAG == 2){
             fprintf(farea, "%s : Area =  %e Yaw = %e Pitch = %e ", filename,area,LHS_phi[i],LHS_theta[i]);
             for(j = 0;j<NCOMP;j++) {
                   Comp_Deflec[i][j] = rdata[i][j+ncols];
                   fprintf(farea,"C%d= %12lf ", j,Comp_Deflec[i][j]);
             }
             fprintf(farea,"\n");
                  }
       else{fprintf(farea, "%s : Area =  %e Yaw = %e Pitch = %e \n", filename,area,LHS_phi[i],LHS_theta[i]);}
      } //Serial loop end 
    
  }
#else /* PARALLEL */

  //printf("Loop on NUM_POINTS\n");
  sprintf(filename,"%s",objname);
  for(i=0; i<NUM_POINTS; i++) {    // Parallel Loop
    iTOT=0;
     
    Cd = testparticle(filename, LHS_Umag[i], LHS_theta[i], LHS_phi[i], LHS_Ts[i], LHS_Ta[i], LHS_Ep[i], \
		      LHS_Al[i], LHS_An[i], LHS_St[i], X, GSI_MODEL, iTOT, &area);
    pc = 100*i/NUM_POINTS; //percentage complete
    //printf("Cd = %e\n", Cd);

         
    if(GSI_MODEL==1){ //DRIA
         fprintf(fout, "%e %e %e %e %e %e %e\n", 
              LHS_Umag[i], LHS_Ts[i], LHS_Ta[i], LHS_Al[i],LHS_theta[i], LHS_phi[i],  Cd);

    }
    else{ //CLL
    fprintf(fout, "%e %e %e %e %e %e %e %e\n", 
	    LHS_Umag[i], LHS_Ts[i], LHS_Ta[i], LHS_An[i], LHS_St[i], LHS_theta[i], LHS_phi[i],  Cd);
     } 
    
    if(ROTATION_FLAG == 2){ 
        fprintf(farea, "%s : Area =  %e Yaw = %e Pitch = %e ", filename,area,LHS_phi[i],LHS_theta[i]);
        for(j = 0;j<NCOMP;j++){
                 Comp_Deflec[i][j] = rdata[i][j+ncols];
                 fprintf(farea,"C%d= %12lf ", j,Comp_Deflec[i][j]);
         }
        fprintf(farea,"\n");
     }
    else{fprintf(farea, "%s : Area =  %e Yaw = %e Pitch = %e \n", filename,area,LHS_phi[i],LHS_theta[i]);}
  }


#endif /* PARALLEL */

  fclose(fout);
  fclose(farea);
#if PARALLEL
  //printf("[rank=%d] MPI Barrier\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Merge Files */
  if(rank==0) {
    for(i=0; i<nProcs; i++) {
      defzeros(i, &zeros);
      //printf("Creating tempfiles/Cdout_%s%d.dat\n", zeros,i);
      sprintf(outfilename, "tempfiles/Cdout_%s%d.dat", zeros, i);
      fout = fopen(outfilename, "r");
      sprintf(areaoutfilename,"Outputs/Area_Regression_Models/area_%s_%s%d.dat", objname,zeros,i);
      farea = fopen(areaoutfilename,"r");    
      if(fout == NULL)
      {printf("Error opening %s\n",outfilename);  
  	   exit(1);             
      }
     if(farea == NULL)
      {printf("Error opening %s\n",areaoutfilename);  
      exit(1);             
      }
    
     while(!feof(fout)) {
	     if(fgets(line, 1024, fout)) {
	          fprintf(ftot, line);
         }
     }   
    while(!feof(farea)) {
         if(fgets(line,1024,farea)){
             fprintf(ftota,line);
         }
      }

      fclose(fout);
      fclose(farea);
    }
    fclose(ftot);
    fclose(ftota);
  }
  
#endif /* PARALLEL */
   

  free(LHS_Umag);
  free(LHS_theta);
  free(LHS_phi);
  free(LHS_Ts);
  free(LHS_Ta);
  free(LHS_An);
  free(LHS_St);
  free(LHS_Al);
  free(LHS_Ep);

 


  gsl_rng_free(rr);

#if PARALLEL
  MPI_Finalize();
#endif /* PARALLEL */

#if PARALLEL
  if(rank==0) {
    printf("Simulation complete\n");
  }
#endif /* PARALLEL */

  return(0);

}

  
 /******************************** READ INPUT ***************************/
void read_input(char objname[1024], double X[], int *GSI_MODEL, int *ROTATION_FLAG)
{

  /* Read in ensemble input file */
  /* Input: Ensemble input file */
  /* Output: Ensemble parameters */

  FILE *f = fopen("tempfiles/temp_variables.txt", "r");

  if(f == NULL)
  {
  printf("Error opening %s\n","temp_variables.txt");
    exit(1);             
  }

  char line[1024];
  char *temp;
  char *data;
  int i;


  /* MESH OBJECT NAME */
  fgets(line, 1024, f);
  temp = strtok(line, "#");
  data = strtok(NULL, "#");
  sscanf(data, "%s\n", objname);

  /* GAS SURFACE INTERACTION MODEL */
  fgets(line, 1024, f);
  temp = strtok(line, "#");
  data = strtok(NULL, "#");
  sscanf(data, "%d\n", GSI_MODEL);

  /* SPECIES MOLE FRACTIONS X = [O, O2, N, N2, He, H] */
  fgets(line, 1024, f);
  temp = strtok(line, "#");
  data = strtok(NULL, "#");
  sscanf(data, "%lf %lf %lf %lf %lf %lf\n", &X[0], &X[1], &X[2], &X[3], &X[4], &X[5]);

  fgets(line, 1024, f);
  temp = strtok(line, "#");
  data = strtok(NULL, "#");
  sscanf(data, "%d\n", ROTATION_FLAG);

  fclose(f);

}

/***************************** DEFINE ZEROS FORMATTING ***************************/ 
 void defzeros(int i, char **zeros)
 {

   if (nProcs>9) {
     if (i<10) {
       *zeros="p0";
     } else {
       *zeros = "p";
     }
   }
   if (nProcs>99) {
     if (i<10) {
       *zeros="p00";
     }
     else if (i<100) {
       *zeros="p0";
     }
   }
   
 }


/***************************** TEST PARTICLE MONTE CARLO ***************************/ 
double testparticle(char filename[1024], double Umag, double theta, double phi, double Ts, double Ta, double epsilon, double alpha, double alphan, double sigmat, double X[], int GSI_MODEL, int iTOT,double *area)

{

  struct particle_struct *pparticle=NULL;
  struct facet_struct *pfacet=NULL;
  struct cell_struct *pcell = NULL;
  struct samp_struct *psample=NULL;

#if SAMPLE
  struct samp_struct *sample;
#endif /* SAMPLE */

  struct particle_struct *particle;
  struct cell_struct *cell;

  int i, j, k, ipart, species; 
  int particle_surf = 0;
  int nfacets = 0;
  int min_fc = -1;
  int scount = 0;
  int temp;
  int pcounter[NSURF][NSPECIES];
  int spcounter[NSPECIES];
  int surfcounter[NSURF];
  double sflux[NSURF];
  double tsflux[NSURF];
  double spflux[NSPECIES];

  double Weight;
  double proj_area;
  double Cd;
  double pveln[3], dF[3];

  double r1, r2;

/* INITIALIZE MIN AND MAX POSITIONS FOR EACH DIMENSION */
  double XMIN = 0.0;
  double YMIN = 0.0;
  double ZMIN = 0.0;

  double XMAX = 0.0;
  double YMAX = 0.0;
  double ZMAX = 0.0;

/* COMPUTE SURFACE AREA OF EACH DOMAIN BOUNDARY */
  double surf_area[NSURF];

  double surf_flux[NSURF][NSPECIES];
  double s[NSURF][NSPECIES];
  double cumul_dist[NSURF*NSPECIES];

  double Ux = Umag*cos(theta)*cos(phi);
  double Uy = Umag*sin(theta)*cos(phi);
  double Uz = Umag*sin(phi);
  double Usurf[3] = {Ux, Uy, Uz};
  double Ubulk[NSURF] = {Ux, -Ux, Uy, -Uy, Uz, -Uz};
  int direction[NSURF] = {1, -1, 1, -1, 1, -1};
  double sp_flux[NSURF];
  double tot_flux = 0.0;

/* INITIALIZE TOTAL FORCE ON OBJECT */
  double Fx = 0.0;
  double Fy = 0.0;
  double Fz = 0.0;

/* COMPUTE AVERAGE MASS */
/******************** {O,            O2,           N,            N2,           He,           H        } */
  double m[NSPECIES] = {2.65676e-26, 5.31352e-26,  2.32586e-26,  4.65173e-26,  6.65327e-27,  1.67372e-27};

  double m_avg = 0.0;
  
  for(i=0; i<NSPECIES; i++) {
    m_avg = m_avg + X[i]*m[i];
  }

  /* INITIALIZE SPECIES PARTICLE COUNTERS */
  int nspec[NSPECIES] = {0, 0, 0, 0, 0, 0};

/* DEFINE ATMOSPHERIC/SATELLITE PROPERTIES */
  double kB = 1.38065e-23;                           /* BOLTZMANN CONSTANT [J/(Km)] */
  double Vw;                                         /* WALL SPEED [m/s] */
  double Cmp[NSPECIES];                              /* MOST PROBABLE SPEED [m/s] */

  double surf_temp;

  int cell_number;
  int nc[3] = {0, 0, 0};
  int particle_cell_track[MAXCELLS];

  int *total_facet_list;
  int fcount = 0;
  double update_step;

/* DETERMINE NUMBER OF FACETS FROM MESH FILE */
  nfacets = read_num_lines(filename);
  pfacet = (struct facet_struct *) calloc(nfacets, sizeof(struct facet_struct));
  pparticle = (struct particle_struct *) calloc(NPART, sizeof(struct particle_struct));
#if SAMPLE
  psample = (struct samp_struct *) calloc(NPART, sizeof(struct samp_struct));
#endif /* SAMPLE */

  total_facet_list = (int *) calloc(nfacets, sizeof(int));
  for(i=0; i<nfacets; i++) {
    total_facet_list[i] = -1;
  }

  /* READ IN FACET PROPERTIES FROM MESH FILE */
  facet_properties(filename, Ux, Uy, Uz, nfacets, pfacet, Umag);

  /* DYNAMIC COMPUTATION OF DOMAIN BOUNDARIES */
  domain_boundary(&XMIN, &YMIN, &ZMIN, &XMAX, &YMAX, &ZMAX, nfacets, pfacet);

  surf_area[0] = surf_area[1] = (YMAX-YMIN)*(ZMAX-ZMIN);
  surf_area[2] = surf_area[3] = (XMAX-XMIN)*(ZMAX-ZMIN);
  surf_area[4] = surf_area[5] = (XMAX-XMIN)*(YMAX-YMIN);

  /* DETERMINE MAXIMUM SIZE OF FACETS IN EACH DIMENSION */
  max_facet_dimension(XMIN, YMIN, ZMIN, XMAX, YMAX, ZMAX, pfacet, nfacets, nc);
  pcell = (struct cell_struct *) calloc(nc[0]*nc[1]*nc[2], sizeof(struct cell_struct));

  /* CREATE CELL GRID */
  create_grid(XMIN, YMIN, ZMIN, XMAX, YMAX, ZMAX, pcell, pfacet, nfacets, nc);

  sort_facets(pfacet, pcell, nfacets, nc);

  /* COMPUTE PROJECTED AREA OF MESH */
  proj_area = projected_area(XMIN, YMIN, ZMIN, XMAX, YMAX, ZMAX, Ux, Uy, Uz, nfacets, pfacet);
  *area = proj_area; //return projected area for  data files 
  /* COMPUTE SPECIES PARTICLE COUNTERS */
  nspec[0] = (int)(NPART*X[0]);
  for(j=1; j<NSPECIES; j++) {
    nspec[j] = nspec[j-1] + (int)(NPART*X[j]);
  }

  /* INTIALIZE PROPERTIES RELATED TO PARTICLE FLUX */
  for(i=0; i<NSURF; i++) {
    tsflux[i] = 0.0;
    for(j=0; j<NSPECIES; j++) {
      surf_flux[i][j] = 0.0;
      s[i][j] = 0.0;
      pcounter[i][j] = 0;
    }
  }
  
  for(k=0; k<NSPECIES*NSURF; k++) {
    cumul_dist[k] = 0.0;
  }

  for(j=0; j<NSPECIES; j++) {
    sp_flux[j] = 0.0;
  }
  
  /* COMPUTE SPEED RATIO, SURFACE FLUX, AND TOTAL FLUX */
  for(i=0; i<NSURF; i++) {
    for(j=0; j<NSPECIES; j++) {
      Cmp[j] = sqrt(2.0*kB*Ta/m[j]);             
      s[i][j] = Ubulk[i]/Cmp[j];
      surf_flux[i][j] = AT_NUM_DENSITY*X[j]*Cmp[j]*surf_area[i]*
	(exp(-s[i][j]*s[i][j]) + sqrt(TPM_PI)*s[i][j]*(1.0 + erf(s[i][j])))/(2.0*sqrt(TPM_PI));
      tsflux[i] += surf_flux[i][j];
      spflux[j] += surf_flux[i][j];
    }
  }

  /* COMPUTE TOTAL FLUX BY SUMMING OF SPECIES FLUXES */
  for(i=0; i<NSURF; i++) {
    tot_flux += tsflux[i];
  } 

  /* COMPUTE PARTICLE WEIGHT */
  Weight = tot_flux/NPART;

  /* COMPUTE CUMULATIVE PROBABILITY DISTRIBUTION */
  cumul_dist[0] = surf_flux[0][0]/tot_flux;
  for(k=1; k<NSURF*NSPECIES; k++) {
    i = k / 6;
    j = k % 6;
    cumul_dist[k] = cumul_dist[k-1] + surf_flux[i][j]/tot_flux;
  }

  update_step = (double)NPART/100.0;

  for(ipart=0; ipart<NPART; ipart++) {

    //#if !PARALLEL
    /* if(ipart % (int)update_step == 0) { */
    /*  printf("%3.0lf percent complete\n", 100.0*((double)ipart/(double)NPART)); */
    /* } */
    //#endif /* !PARALLEL */

    if (ipart%100000==0) printf(">>>%s<<<  Particle No %d\n", filename, ipart);

    particle = pparticle + ipart;
    
#if SAMPLE
    sample = psample + ipart;
    sample->t = 1.0e6;
#endif /* SAMPLE */
    
    /* DETERMINE SURFACE TO CREATE PARTICLE ON */
    r2 = ranf0(rr);
    for(k=0; k<NSURF*NSPECIES; k++) {
      if(r2 < cumul_dist[k]) {
	i = k / 6;
	j = k % 6;
	particle_surf = i;
	species = j;
	break;
      }
    }

    pcounter[particle_surf][species]++;

    /* COMPUTE PARTICLE POSITION */
    pposition(XMIN, YMIN, ZMIN, XMAX, YMAX, ZMAX, pparticle, ipart, particle_surf);
      
    /* COMPUTE PARTICLE VELOCITY */
    pvelocity(s, Usurf, direction, Cmp, particle_surf, pparticle, ipart, species);
    
#if SAMPLE
    for(i=0; i<3; i++) {
      sample->ivel[i] = particle->vel[i];
      sample->ipos[i] = particle->pos[i];
    }
#endif /* SAMPLE */

    /* TRACK NUMBER OF SURFACES HIT IN CASE A PARTICLE IS TRAPPED */
      scount = 0;

      do {

	//if ((scount+1)%10 == 0)
	//{
      //printf("scount=%d\n",scount);	
	//} 	
	

	/* INITIALIZE PARTICLE CELL TRACK */
	for(i=0; i<MAXCELLS; i++) {
	  particle_cell_track[i] = -1;
	}

	/* COMPUTE CELLS THAT THE PARTICLE PASSED THROUGH */
	cell_track(pparticle, ipart, pcell, nc, particle_cell_track);

	/* COMPILE THE LIST OF FACETS THAT A PARTICLE MAY HAVE INTERACTED WITH */
	compile_facet_list(pfacet, nfacets, pcell, particle_cell_track, total_facet_list, &fcount, ipart);
      
	/* COMPUTE INTERSECTION OF PARTICLE PATH AND FACET PLANE */
	pf_intercept(&min_fc, pfacet, pparticle, psample, ipart, particle_surf, nfacets, pcell, 
		     particle_cell_track, total_facet_list, fcount);
      
	/* PARTICLE INCIDENT ON FINITE FACET PLANE */
	if(min_fc >=0) {

	  scount++;

	  /* COMPUTE THE GAS-SURFACE INTERACTION */
	  Vw = sqrt(2.0*kB*Ts/m[species]);
	  gsi(pparticle, ipart, pfacet, min_fc, Vw, pveln, GSI_MODEL, epsilon, alpha, alphan, sigmat);
	
	  /* COMPUTE FORCE ON OBJECT BASED ON CHANGE IN VELOCITY VECTOR */
	  for(i=0; i<3; i++) {
 
	    dF[i] = Weight*m[species]*(particle->vel[i] - pveln[i]);

	    /* UPDATE VELOCITY */
	    particle->vel[i] = pveln[i];
#if SAMPLE
	    sample->rvel[i] = particle->vel[i];
#endif /* SAMPLE */
	  }
	
	  Fx = Fx + dF[0];
	  Fy = Fy + dF[1];
	  Fz = Fz + dF[2];
	
	}

	if(scount>MAXCOUNT) {
	  printf("Problem! Particle #%d has hit the satellite surface %d times!\n", ipart, scount);
	}

      } while (min_fc >= 0);
      
    } /* END LOOP OVER PARTICLES */

#if SAMPLE
  /* Write out sampled particle data */
  sample_write(iTOT, psample);
#endif /* SAMPLE */

  for(j=0; j<NSPECIES; j++) {
    spcounter[j] = 0;
  }

  for(i=0; i<NSURF; i++) {
    surfcounter[i] = 0;
    sflux[i] = 0.0;
  }

  for(i=0; i<NSURF; i++) {
    for(j=0; j<NSPECIES; j++) {
      sflux[i] += surf_flux[i][j]/tot_flux;
    }
  }

  Cd = compute_Cd(Umag, theta, phi, Fx, Fy, Fz, proj_area, AT_NUM_DENSITY, m_avg);

  free(total_facet_list);
  free(pfacet);
  free(pparticle);
  free(pcell);

#if SAMPLE
  free(psample);
#endif /* SAMPLE */

  return(Cd);

}

  
 /******************************** MESH READ ***************************/
int read_num_lines(char filename[1024])
{

  /* Read in the STL mesh file and determine number of lines*/
  /* Input: STL mesh filename */
  /* Output: Number of Facets */

  int nfacets = 0;
  int num_lines = 0;
  int ch;

  printf("read_num_lines: %s\n",filename);

  void chdir();
  chdir("tempfiles/Mesh_Files");  // Get the STL file
  FILE *f = fopen(filename, "r");
  if(f == NULL)
  {
	printf("Error opening %s\n",filename);  
  	exit(1);             
  }

  /*Check that file exists*/
  if(!f) {
    printf("Mesh File does not exist\n");
    exit(1);
  }
  
  while (EOF != (ch=fgetc(f))) 
    if (ch=='\n')
      num_lines++;

  fclose(f);

  /* DETERMINE NUMBER OF FACETS */
  nfacets = (num_lines-2)/7;
  
  chdir("../..");

  return(nfacets);

}


/******************************** FACET PROPERTIES ***************************/
void facet_properties(char filename[1024], double Ux, double Uy, double Uz, int nfacets, struct facet_struct *pfacet, 
		      double Umag)
{

  struct facet_struct *facet;

  /* Input: STL mesh filename */
  /* Output: Facet Properties Stucture Containing: */
  /*         Facet Normal [x, y, z]  */
  /*         Vertex1 [x, y, z] */
  /*         Vertex2 [x, y, z] */
  /*         Vertex3 [z, y, z] */
  void chdir();
  chdir("tempfiles/Mesh_Files"); // Get the STL file
  FILE *f = fopen(filename, "r");
  if(f == NULL)
  {
  printf("Error opening %s\n",filename);  
    exit(1);             
  }


  /* READ IN STL FILE HEADER */
  char header[1024];
  char line1[1024], line2[1024], line3[1024], line4[1024], line5[1024], line6[1024], line7[1024];
  char *vert1x, *vert1y, *vert1z;
  char *vert2x, *vert2y, *vert2z;
  char *vert3x, *vert3y, *vert3z;
  char *normx, *normy, *normz;
  char *temp;
  int HeaderSize = 1;
  int i, ifacet;

  double v[3];
  double dist1[3], dist2[3];
  double tc[3];

  for(i=0; i<3; i++) {
    dist1[i] = 0.0;
    dist2[i] = 0.0;
    tc[i] = 0.0;
  }

  for(i=0; i<HeaderSize; i++) {
    fgets(header, 1024, f);
  }

  for(ifacet=0; ifacet<nfacets; ifacet++) {
    /* Read the first line and assign normal components */
    fgets(line1, 1024, f);
    temp = strtok(line1, " ");
    temp = strtok(NULL, " ");
    normx = strtok(NULL, " ");
    normy = strtok(NULL, " ");
    normz = strtok(NULL, " ");
    /* Throw away second line */
    fgets(line2, 1024, f);
    /* Read the third line and assign vertex #1 position */
    fgets(line3, 1024, f);
    temp = strtok(line3, " ");
    vert1x = strtok(NULL, " ");
    vert1y = strtok(NULL, " ");
    vert1z = strtok(NULL, " ");
    /* Read the fourth line and assign vertex #2 position */
    fgets(line4, 1024, f);
    temp = strtok(line4, " ");
    vert2x = strtok(NULL, " ");
    vert2y = strtok(NULL, " ");
    vert2z = strtok(NULL, " ");
    /* Read the fifth line and assign vertex #3 position */
    fgets(line5, 1024, f);
    temp = strtok(line5, " ");
    vert3x = strtok(NULL, " ");
    vert3y = strtok(NULL, " ");
    vert3z = strtok(NULL, " ");
    /* Throw away the sixth and seventh lines */
    fgets(line6, 1024, f);
    fgets(line7, 1024, f);

    facet = pfacet + ifacet;

    /* Assign facet normal vector to "facet" structure */
    facet->normal[0] = atof(normx);
    facet->normal[1] = atof(normy);
    facet->normal[2] = atof(normz);

    /* Assign vertex #1 position to "facet" structure */
    facet->vertex1[0] = atof(vert1x);
    facet->vertex1[1] = atof(vert1y);
    facet->vertex1[2] = atof(vert1z);

    /* Assign vertex #2 position to "facet" structure */
    facet->vertex2[0] = atof(vert2x);
    facet->vertex2[1] = atof(vert2y);
    facet->vertex2[2] = atof(vert2z);

    /* Assign vertex #3 position to "facet" structure */
    facet->vertex3[0] = atof(vert3x);
    facet->vertex3[1] = atof(vert3y);
    facet->vertex3[2] = atof(vert3z);

    /* COMPUTE UNIT VELOCITY VECTOR */
    v[0] = Ux/Umag;
    v[1] = Uy/Umag;
    v[2] = Uz/Umag;

    /* Find length of 1st side of the triangle */
    dist1[0] = facet->vertex2[0] - facet->vertex1[0];
    dist1[1] = facet->vertex2[1] - facet->vertex1[1];
    dist1[2] = facet->vertex2[2] - facet->vertex1[2];
    
    /* Find length of 2st side of the triangle */
    dist2[0] = facet->vertex1[0] - facet->vertex3[0];
    dist2[1] = facet->vertex1[1] - facet->vertex3[1];
    dist2[2] = facet->vertex1[2] - facet->vertex3[2];

    /* Compute the cross product between the two sides of the triangle */
    cross(dist1, dist2, tc);

    facet->area = 0.5*sqrt(dot(tc, tc));
     
  }

  chdir("../..");
  fclose(f);
}


/***************************** PROJECTED AREA **********************************/
double projected_area(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, double Ux, double Uy, double Uz, int nfacets, struct facet_struct *pfacet)
{

  /* Calculate the projected area of the mesh */
  /* Translated from F. Lumpkin's Fortran Area routine for DAC */
  /* Inputs: facet_normal = Facet Normal [x, y, z] */
  /*         vertex1 = x, y, z positions of 1st vertex */
  /*         vertex2 = x, y, z positions of 2nd vertex */
  /*         vertex3 = x, y, z positions of 3rd vertex */
           
  /* Outputs: facet_area = Facet Area [m^2] */
  /*          proj_area = Projected Area of the Mesh [m^2] */ 

  struct facet_struct *facet;
  double proj_area = 0.0;
  double v[3];
  double Umag = sqrt(Ux*Ux+Uy*Uy+Uz*Uz);
  double theta, phi;
  double **fv1, **fv2, **fv3;
  double transform[3][3];
  double sp, cp, st, ct;

  int ifacet, i, j, k, n;
  
  int idir0 = 0;
  int msort = 64;
  int mgrid = 4096;
  int melms = 2000000;
  int melm2 = 2*melms;
  int msort2 = msort*msort;
  int icount;
  int j1, j2, j3, k1, k2, k3;
  int maxj, minj, maxk, mink;
  int jj;
  int iarea;
  int ix[3], iy[3];
  int *isort, *j1sort, *j2sort;

  double a = 2.0e-7;
  double eps;
  double xmax[3] = {0.0, 0.0, 0.0};
  double xmin[3] = {0.0, 0.0, 0.0};
  double fact;
  double dx[3], dy[3];
  double dxi[3], dyi[3];
  double **xp, **yp;
  double *x1, *x2, *x3, *y1, *y2, *y3;
  double *dx21, *dx32, *dx13, *dy21, *dy32, *dy13;
  double cx1, cx2, cx3;
  double area, diffarea;
  double ddiv, ddx, ddy, ddxy;

  /* ADD SMALL VALUE TO BOUNDARIES */
  /* NOTE - SHOULD REMOVE 1.1. IN FAVOR OF A CONSTANT VARIABLE DEFINED IN MAIN */
  /* CURRENTLY 1.1. COMES FROM THE VALUE SET IN DOMAIN_BOUNDARY */
  double fxmin = 1.0e6;
  double fxmax = -1.0e6;
  double fymin = 1.0e6;
  double fymax = -1.0e6;
  double fzmin = 1.0e6;
  double fzmax = -1.0e6;

  /* ALLOCATE MEMORY */
  xp = (double **) calloc(3, sizeof(double));
  yp = (double **) calloc(3, sizeof(double));

  for(i=0; i<3; i++) {
    xp[i] = (double *) calloc(mgrid, sizeof(double));
    yp[i] = (double *) calloc(mgrid, sizeof(double));
  }

  isort = (int *) calloc(msort2, sizeof(int));
  j1sort = (int *) calloc(melm2, sizeof(int));
  j2sort = (int *) calloc(melm2, sizeof(int));

  x1 = (double *) calloc(nfacets, sizeof(double));
  x2 = (double *) calloc(nfacets, sizeof(double));
  x3 = (double *) calloc(nfacets, sizeof(double));
  y1 = (double *) calloc(nfacets, sizeof(double));
  y2 = (double *) calloc(nfacets, sizeof(double));
  y3 = (double *) calloc(nfacets, sizeof(double));

  dx21 = (double *) calloc(nfacets, sizeof(double));
  dx32 = (double *) calloc(nfacets, sizeof(double));
  dx13 = (double *) calloc(nfacets, sizeof(double));
  dy21 = (double *) calloc(nfacets, sizeof(double));
  dy32 = (double *) calloc(nfacets, sizeof(double));
  dy13 = (double *) calloc(nfacets, sizeof(double));

  fv1 = (double **) calloc(nfacets, sizeof(double));
  fv2 = (double **) calloc(nfacets, sizeof(double));
  fv3 = (double **) calloc(nfacets, sizeof(double));
  for(ifacet=0; ifacet<nfacets; ifacet++) {
    fv1[ifacet] = (double *) calloc(3, sizeof(double));
    fv2[ifacet] = (double *) calloc(3, sizeof(double));
    fv3[ifacet] = (double *) calloc(3, sizeof(double));
  }

  /* Define orientation angles in radians*/
  if(Ux==0.0 && Uy==0.0) {
    theta = 0.0;
  } else {
    theta = atan2(Uy, Ux);
  }
  phi = asin(Uz/sqrt(Ux*Ux+Uy*Uy+Uz*Uz));

  /* Transform vertices based on theta and phi */
  if(theta != 0.0 || phi != 0.0) {
    sp = sin(phi);
    cp = cos(phi);
    st = sin(theta);
    ct = cos(theta);
    transform[0][0] = ct*cp;
    transform[0][1] = st*cp;
    transform[0][2] = -sp;
    transform[1][0] = -st;
    transform[1][1] = ct;
    transform[1][2] = 0.0;
    transform[2][0] = ct*sp;
    transform[2][1] = st*sp;
    transform[2][2] = cp;
    for(ifacet=0; ifacet<nfacets; ifacet++) {
      facet = pfacet + ifacet;
      for(i=0; i<3; i++) {
  	fv1[ifacet][i] = 0.0;
  	fv2[ifacet][i] = 0.0;
  	fv3[ifacet][i] = 0.0;
  	for(j=0; j<3; j++) {
  	  fv1[ifacet][i] += transform[j][i]*facet->vertex1[j];
  	  fv2[ifacet][i] += transform[j][i]*facet->vertex2[j];
  	  fv3[ifacet][i] += transform[j][i]*facet->vertex3[j];
  	}
      }
    }
  } else {
    for(ifacet=0; ifacet<nfacets; ifacet++) {
      facet = pfacet + ifacet;
      for(i=0; i<3; i++) {
  	fv1[ifacet][i] = facet->vertex1[i];
  	fv2[ifacet][i] = facet->vertex2[i];
  	fv3[ifacet][i] = facet->vertex3[i];
      }
    }
  }

  for(ifacet=0; ifacet<nfacets; ifacet++) {

    /* Determine X mininimum and maximum for this facet */
    if(fv1[ifacet][0] < fxmin) fxmin = fv1[ifacet][0];
    if(fv2[ifacet][0] < fxmin) fxmin = fv2[ifacet][0];
    if(fv3[ifacet][0] < fxmin) fxmin = fv3[ifacet][0];

    if(fv1[ifacet][0] > fxmax) fxmax = fv1[ifacet][0];
    if(fv2[ifacet][0] > fxmax) fxmax = fv2[ifacet][0];
    if(fv3[ifacet][0] > fxmax) fxmax = fv3[ifacet][0];

    /* Determine Y mininimum and maximum for this facet */
    if(fv1[ifacet][1] < fymin) fymin = fv1[ifacet][1];
    if(fv2[ifacet][1] < fymin) fymin = fv2[ifacet][1];
    if(fv3[ifacet][1] < fymin) fymin = fv3[ifacet][1];

    if(fv1[ifacet][1] > fymax) fymax = fv1[ifacet][1];
    if(fv2[ifacet][1] > fymax) fymax = fv2[ifacet][1];
    if(fv3[ifacet][1] > fymax) fymax = fv3[ifacet][1];

    /* Determine Z mininimum and maximum for this facet */
    if(fv1[ifacet][2] < fzmin) fzmin = fv1[ifacet][2];
    if(fv2[ifacet][2] < fzmin) fzmin = fv2[ifacet][2];
    if(fv3[ifacet][2] < fzmin) fzmin = fv3[ifacet][2];

    if(fv1[ifacet][2] > fzmax) fzmax = fv1[ifacet][2];
    if(fv2[ifacet][2] > fzmax) fzmax = fv2[ifacet][2];
    if(fv3[ifacet][2] > fzmax) fzmax = fv3[ifacet][2];

  }

  eps = a*((xmax[0] + xmax[1] + xmax[2]) - (xmin[0] + xmin[1] + xmin[2]));

  xmin[0] = fxmin - eps;
  xmax[0] = fxmax + eps;
  xmin[1] = fymin - eps;
  xmax[1] = fymax + eps;
  xmin[2] = fzmin - eps;
  xmax[2] = fzmax + eps;

  /* CALCULATE DOMAIN SIZE AND INVERSE DOMAIN GRID SIZE */
  for(i=0; i<3; i++) {
    ix[i] = (int)((i+1) % 3);
    iy[i] = (int)(((i+2) % 3));
    dx[i] = xmax[ix[i]] - xmin[ix[i]];
    dy[i] = xmax[iy[i]] - xmin[iy[i]];
    dxi[i] = (double)msort/dx[i];
    dyi[i] = (double)msort/dy[i];
  }

  /* CALCULATE GRID POINT CENTERS */
  fact = 1.0/(double)mgrid;
  for(i=0; i<3; i++) {
    for(n=0; n<mgrid; n++) {
      xp[i][n] = fact*dx[i]*(double)(n+0.5) + xmin[ix[i]];
      yp[i][n] = fact*dy[i]*(double)(n+0.5) + xmin[iy[i]];
    }
  }

  /* INITIALIZE SORTING ARRAYS */
  icount = 0;
  for(i=0; i<msort2; i++) {
    isort[i] = 0;
  }
  for(i=0; i<melm2; i++) {
    j1sort[i] = 0;
    j2sort[i] = 0;
  }
  
  for(ifacet=0; ifacet<nfacets; ifacet++) { 

    facet = pfacet + ifacet;

    x1[ifacet] = fv1[ifacet][ix[idir0]];
    x2[ifacet] = fv2[ifacet][ix[idir0]];
    x3[ifacet] = fv3[ifacet][ix[idir0]];
    y1[ifacet] = fv1[ifacet][iy[idir0]];
    y2[ifacet] = fv2[ifacet][iy[idir0]];
    y3[ifacet] = fv3[ifacet][iy[idir0]];

    dx21[ifacet] = x2[ifacet] - x1[ifacet];
    dx32[ifacet] = x3[ifacet] - x2[ifacet];
    dx13[ifacet] = x1[ifacet] - x3[ifacet];
    dy21[ifacet] = y2[ifacet] - y1[ifacet];
    dy32[ifacet] = y3[ifacet] - y2[ifacet];
    dy13[ifacet] = y1[ifacet] - y3[ifacet];
    
    j1 = (int)((x1[ifacet] - xmin[ix[idir0]])*dxi[idir0]);
    j2 = (int)((x2[ifacet] - xmin[ix[idir0]])*dxi[idir0]);
    j3 = (int)((x3[ifacet] - xmin[ix[idir0]])*dxi[idir0]);

    k1 = (int)((y1[ifacet] - xmin[iy[idir0]])*dyi[idir0]);
    k2 = (int)((y2[ifacet] - xmin[iy[idir0]])*dyi[idir0]);
    k3 = (int)((y3[ifacet] - xmin[iy[idir0]])*dyi[idir0]);

    
  
    maxj = (int)fmax((double)j1, (double)j2);
    maxj = (int)fmax((double)j3, (double)maxj);

    minj = (int)fmin((double)j1, (double)j2);
    minj = (int)fmin((double)j3, (double)minj);

    maxk = (int)fmax((double)k1, (double)k2);
    maxk = (int)fmax((double)k3, (double)maxk);

    mink = (int)fmin((double)k1, (double)k2);
    mink = (int)fmin((double)k3, (double)mink);

    //TEMPORARY FIX
    if(maxj > 63) {
      maxj = 63;
    }
    if(maxk > 63) {
      maxk = 63;
    }
    if(minj < 0) {
      minj = 0;
    }
    if(mink < 0) {
      mink = 0;
    }
   
    for(j=minj; j<=maxj; j++) {
      for(k=mink; k<=maxk; k++) {
	icount = icount + 1;
	jj = msort*(k) + j;
	j2sort[icount] = ifacet;
	j1sort[icount] = isort[jj];
	isort[jj] = icount;
      }
    }

  }
   
  area = 0.0;
  for(i=0; i<mgrid; i++) {
    iarea = 0;
    for(j=0; j<mgrid; j++) {
      jj = msort*(int)((yp[idir0][j]-xmin[iy[idir0]])*dyi[idir0]) + (int)((xp[idir0][i]-xmin[ix[idir0]])*dxi[idir0]);
      k = isort[jj];
      while(k!=0) {
	ifacet = j2sort[k];
	cx1 = dx21[ifacet]*(yp[idir0][j]-y1[ifacet]) - dy21[ifacet]*(xp[idir0][i]-x1[ifacet]);
	cx2 = dx32[ifacet]*(yp[idir0][j]-y2[ifacet]) - dy32[ifacet]*(xp[idir0][i]-x2[ifacet]);
	if(cx1*cx2 > 0.0) {
	  cx3 = dx13[ifacet]*(yp[idir0][j]-y3[ifacet]) - dy13[ifacet]*(xp[idir0][i]-x3[ifacet]);
	  if(cx1*cx3 > 0.0) {
	    iarea++;
	    break;
	  }
	}
	k = j1sort[k];
      }
    }
    diffarea = (double)iarea;
    area += diffarea;
  }
  ddiv = (double)mgrid;
  ddx = (double)(dx[idir0])/ddiv;
  ddy = (double)(dy[idir0])/ddiv;
  ddxy = ddx*ddy;
  proj_area = ddxy*area;

  for(i=0; i<3; i++) {
    free(xp[i]);
    free(yp[i]);
  }

  free(xp);
  free(yp);
  
  free(isort);
  free(j1sort);
  free(j2sort);

  free(x1);
  free(x2);
  free(x3);
  free(y1);
  free(y2);
  free(y3);

  free(dx21);
  free(dx32);
  free(dx13);
  free(dy21);
  free(dy32);
  free(dy13);

  for(ifacet=0; ifacet<nfacets; ifacet++) {
    free(fv1[ifacet]);
    free(fv2[ifacet]);
    free(fv3[ifacet]);
  }

  free(fv1);
  free(fv2);
  free(fv3);

  return(proj_area);

}


/***************************** DOMAIN BOUNDARY **********************************/
void domain_boundary(double *XMIN, double *YMIN, double *ZMIN, double *XMAX, double *YMAX, double *ZMAX, int nfacets, struct facet_struct *pfacet)
{

  /* Calculate the domain boundaries based on mesh */
  /* Inputs: nfacets = Number of Facets */
  /*         pfacet = Facet structure pointer          */
            
  /* Outputs: = Xmin = X-direction domain minimum boundary [m] */
  /*            Xmax = X-direction domain maximum boundary [m] */
  /*            Ymin = Y-direction domain minimum boundary [m] */
  /*            Ymax = Y-direction domain maximum boundary [m] */
  /*            Zmin = Z-direction domain minimum boundary [m] */
  /*            Zmax = Z-direction domain maximum boundary [m] */

  struct facet_struct *facet;
  int ifacet;
  double fxmin = 1.0e6;
  double fxmax = -1.0e6;
  double fymin = 1.0e6;
  double fymax = -1.0e6;
  double fzmin = 1.0e6;
  double fzmax = -1.0e6;

  double A = 0.1; /* Safety factor on domain bounds */

  for(ifacet=0; ifacet<nfacets; ifacet++) {
    
    facet = pfacet + ifacet;

    /* Determine X mininimum and maximum for this facet */
    if(facet->vertex1[0] < fxmin) fxmin = facet->vertex1[0];
    if(facet->vertex2[0] < fxmin) fxmin = facet->vertex2[0];
    if(facet->vertex3[0] < fxmin) fxmin = facet->vertex3[0];

    if(facet->vertex1[0] > fxmax) fxmax = facet->vertex1[0];
    if(facet->vertex2[0] > fxmax) fxmax = facet->vertex2[0];
    if(facet->vertex3[0] > fxmax) fxmax = facet->vertex3[0];

    /* Determine Y mininimum and maximum for this facet */
    if(facet->vertex1[1] < fymin) fymin = facet->vertex1[1];
    if(facet->vertex2[1] < fymin) fymin = facet->vertex2[1];
    if(facet->vertex3[1] < fymin) fymin = facet->vertex3[1];

    if(facet->vertex1[1] > fymax) fymax = facet->vertex1[1];
    if(facet->vertex2[1] > fymax) fymax = facet->vertex2[1];
    if(facet->vertex3[1] > fymax) fymax = facet->vertex3[1];

    /* Determine Z mininimum and maximum for this facet */
    if(facet->vertex1[2] < fzmin) fzmin = facet->vertex1[2];
    if(facet->vertex2[2] < fzmin) fzmin = facet->vertex2[2];
    if(facet->vertex3[2] < fzmin) fzmin = facet->vertex3[2];

    if(facet->vertex1[2] > fzmax) fzmax = facet->vertex1[2];
    if(facet->vertex2[2] > fzmax) fzmax = facet->vertex2[2];
    if(facet->vertex3[2] > fzmax) fzmax = facet->vertex3[2];

  }

  *XMIN = fxmin - A*(fxmax - fxmin);
  *XMAX = fxmax + A*(fxmax - fxmin);
  *YMIN = fymin - A*(fymax - fymin);
  *YMAX = fymax + A*(fymax - fymin);
  *ZMIN = fzmin - A*(fzmax - fzmin);
  *ZMAX = fzmax + A*(fzmax - fzmin);

}


/***************************** MAX FACET DIMENSION ******************************/
void max_facet_dimension(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct facet_struct *pfacet, int nfacets, int nc[3]) {

  int i, j, k, m;
  int ifacet;
  double fmax[3];
  struct facet_struct *facet;

  for(m=0; m<3; m++) {
    fmax[m] = -1.0e6;
  }

  /* Search through facets to find largest dimension in X, Y, and Z */
  for(ifacet=0; ifacet<nfacets; ifacet++) {
    facet = pfacet + ifacet;

    /* Compare Vertex #1 and #2 */
    for(m=0; m<3; m++) {
      if(fabs(facet->vertex1[m] - facet->vertex2[m]) > fmax[m]) {
	fmax[m] = fabs(facet->vertex1[m] - facet->vertex2[m]);
      }
    }

    /* Compare Vertex #1 and #3 */
    for(m=0; m<3; m++) {
      if(fabs(facet->vertex1[m] - facet->vertex3[m]) > fmax[m]) {
	fmax[m] = fabs(facet->vertex1[m] - facet->vertex3[m]);
      }
    }

    /* Compare Vertex #2 and #3 */
    for(m=0; m<3; m++) {
      if(fabs(facet->vertex2[m] - facet->vertex3[m]) > fmax[m]) {
	fmax[m] = fabs(facet->vertex2[m] - facet->vertex3[m]);
      }
    }

  }
    
  /* Compute cell size based on largest facet dimension in X, Y, and Z */
  nc[0] = (int)((XMAX-XMIN)/fmax[0]);
  nc[1] = (int)((YMAX-YMIN)/fmax[1]);
  nc[2] = (int)((ZMAX-ZMIN)/fmax[2]);

  for(m=0; m<3; m++) {
    if(nc[m] < 2) {
      nc[m] = 2;
    } else if(nc[m] > NCELLS) {
      nc[m] = NCELLS; 
    }
  }

}

/***************************** INITIALIZE CELL GRID *****************************/
void create_grid(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct cell_struct *pcell, struct facet_struct *pfacet, int nfacets, int nc[3]) {

  /* Determine the positional boundaries for the cell grid */
  
  /* Inputs: pcell = Starting address of cell structure */
  /*         XMIN = X-minimum domain boundary */
  /*         YMIN = Y-minimum domain boundary */
  /*         ZMIN = Z-minimum domain boundary */
  /*         XMAX = X-maximum domain boundary */
  /*         YMAX = Y-maximum domain boundary */
  /*         ZMAX = Z-maximum domain boundary */

  /* Outputs: Defined boundaries of cell structure */

  int i, j, k, m;
  int cell_number;
  struct cell_struct *cell;

  for(i=0; i<nc[0]; i++) { /* X-position */
    for(j=0; j<nc[1]; j++) { /* Y-position */
      for(k=0; k<nc[2]; k++) { /* Z-position */
	cell_number = i*nc[1]*nc[2] + nc[2]*j + k;
	cell = pcell + cell_number;
	
	cell->top = ZMIN + (ZMAX - ZMIN)*((double)(k+1)/(double)nc[2]);
	cell->bottom = ZMIN + (ZMAX - ZMIN)*((double)k/(double)nc[2]);

	cell->right = YMIN + (YMAX - YMIN)*((double)(j+1)/(double)nc[1]);
	cell->left = YMIN + (YMAX - YMIN)*((double)j/(double)nc[1]);

	cell->front = XMIN + (XMAX - XMIN)*((double)(i+1)/(double)nc[0]);
	cell->back = XMIN + (XMAX - XMIN)*((double)i/(double)nc[0]);

	/* Initialize the list of facets in the cell; -1 is equivalent to empty */
	for(m=0; m<MAXFACETS; m++) {
	  cell->facet_list[m] = -1;
	}

      }
    }
  }

}


/**************************** SORT FACETS IN CELLS ******************************/
void sort_facets(struct facet_struct *pfacet, struct cell_struct *pcell, int nfacets, int nc[3]) {

  /* Determines which facets are in which cells */

  /* Input: pfacet = Facet structure     */
  /*        pcell  = Cell structure      */

  /* Outputs: Facets indexed to cells */
  
  int i, j, k;
  int ifacet;
  int counter;
  int cell_number;
  int total_counter = 0;
  int sum = 0;
  int *facet_af = NULL;
  int *facet_overlaps = NULL;
  double ***facet_ijk = NULL;
  double **facet_ijkmin = NULL;
  double **facet_ijkmax = NULL;
  double ***cellcounter = NULL;
  int ijkdiff[3][3];
  int vindx;
  int overlap_counter = 0;
  int ijkcounter;
  double edgevec[3];
  double edgepos[3];
  int ief, ifo, oflag;
  int iter, dupflag;
  int ixyz;
  facet_af = (int *) calloc(nfacets, sizeof(int));
  facet_overlaps = (int *) calloc(nfacets, sizeof(int));
  facet_ijk = (double ***) calloc(nfacets, sizeof(double));
  for(i=0; i<nfacets; i++) {
    facet_ijk[i] = (double **) calloc(3, sizeof(double));
    for(j=0; j<3; j++) {
      facet_ijk[i][j] = (double *) calloc(3, sizeof(double));
    }
  }
  facet_ijkmin = (double **) calloc(nfacets, sizeof(double));
  facet_ijkmax = (double **) calloc(nfacets, sizeof(double));
  for(i=0; i<nfacets; i++) {
    facet_ijkmin[i] = (double *) calloc(3, sizeof(double));
    facet_ijkmax[i] = (double *) calloc(3, sizeof(double));
  }    

  cellcounter = (double ***) calloc(nc[0], sizeof(double));
  for(i=0; i<nc[0]; i++) {
    cellcounter[i] = (double **) calloc(nc[1], sizeof(double));
    for(j=0; j<nc[1]; j++) {
      cellcounter[i][j] = (double *) calloc(nc[2], sizeof(double));
    }
  }

  for(i=0; i<nfacets; i++) { /* Loop over facets */
    facet_af[i] = 0;
    facet_overlaps[i] = -1;
    for(j=0; j<3; j++) {
      facet_ijkmin[i][j] = 1.0e6;
      facet_ijkmax[i][j] = -1.0e6;
    }
  }

  for(i=0; i<nfacets; i++) { /* Loop over facets */
    for(j=0; j<3; j++) { /* Loop over vertices */
      for(k=0; k<3; k++) { /* Loop over x, y, z */
	facet_ijk[i][j][k] = -1;
      }
    }
  }

  struct cell_struct *cell;
  struct facet_struct *facet;

  for(i=0; i<nc[0]; i++) { /* X-position */
    for(j=0; j<nc[1]; j++) { /* Y-position */
      for(k=0; k<nc[2]; k++) { /* Z-position */
	cell_number = i*nc[1]*nc[2] + nc[2]*j + k;
	cell = pcell + cell_number;

	counter = 0;
	cellcounter[i][j][k] = 0;

	for(ifacet=0; ifacet<nfacets; ifacet++) {
	  facet = pfacet + ifacet;

	  /* Check Vertex #1 */
	  if((facet->vertex1[0] > cell->back) && (facet->vertex1[0] <= cell->front)) {
	    if((facet->vertex1[1] > cell->left) && (facet->vertex1[1] <= cell->right)) {
	      if((facet->vertex1[2] > cell->bottom) && (facet->vertex1[2] <= cell->top)) {
		cell->facet_list[(int)cellcounter[i][j][k]] = ifacet;
		facet_ijk[ifacet][0][0] = i;
		facet_ijk[ifacet][0][1] = j;
		facet_ijk[ifacet][0][2] = k;
		cellcounter[i][j][k]++;
		total_counter++;
		facet_af[ifacet]++;
		if(cellcounter[i][j][k]>=MAXFACETS) {
		  printf("MAXFACETS is too low! Increase its value!\n");
		  exit(1);
		}
	      }
	    }
	  }

	  /* Check Vertex #2 */
	  if((facet->vertex2[0] > cell->back) && (facet->vertex2[0] <= cell->front)) {
	    if((facet->vertex2[1] > cell->left) && (facet->vertex2[1] <= cell->right)) {
	      if((facet->vertex2[2] > cell->bottom) && (facet->vertex2[2] <= cell->top)) {
		cell->facet_list[(int)cellcounter[i][j][k]] = ifacet;
		facet_ijk[ifacet][1][0] = i;
		facet_ijk[ifacet][1][1] = j;
		facet_ijk[ifacet][1][2] = k;
		cellcounter[i][j][k]++;
		total_counter++;
		facet_af[ifacet]++;
		if(cellcounter[i][j][k]>=MAXFACETS) {
		  printf("MAXFACETS is too low! Increase its value!\n");
		  exit(1);
		}
	      }
	    }
	  }

	  /* Check Vertex #3 */
	  if((facet->vertex3[0] > cell->back) && (facet->vertex3[0] <= cell->front)) {
	    if((facet->vertex3[1] > cell->left) && (facet->vertex3[1] <= cell->right)) {
	      if((facet->vertex3[2] > cell->bottom) && (facet->vertex3[2] <= cell->top)) {
		cell->facet_list[(int)cellcounter[i][j][k]] = ifacet;
		facet_ijk[ifacet][2][0] = i;
		facet_ijk[ifacet][2][1] = j;
		facet_ijk[ifacet][2][2] = k;
		cellcounter[i][j][k]++;
		total_counter++;
		facet_af[ifacet]++;
		if(cellcounter[i][j][k]>=MAXFACETS) {
		  printf("MAXFACETS is too low! Increase its value!\n");
		  exit(1);
		}
	      }
	    }
	  }

	} /* End loop over facets */

      } /* End k loop */
    } /* End j loop */
  } /* End i loop */

  /* Loop over facets to find possible facet overlap not counted */
  for(ifacet=0; ifacet<nfacets; ifacet++) {
    facet = pfacet + ifacet;
    ijkcounter=0;
    /*Compare 1st and 2nd vertex i, j, k indicies */
    for(i=0; i<3; i++) { /* Vertex combination */
      if(ijkcounter==2) {
	break;
      }
      ijkcounter=0;
      for(j=0; j<3; j++) { /* ijk index */
	if(i<2) {
	  vindx = i+1;
	} else {
	  vindx = 0;
	}
	ijkdiff[i][j] = abs(facet_ijk[ifacet][i][j] - facet_ijk[ifacet][vindx][j]);
	if(ijkdiff[i][j] > 0) {
	  ijkcounter++;
	}
	if(ijkcounter == 2) {
	  facet_overlaps[overlap_counter]=ifacet;
	  overlap_counter++;
	  break;
	}	  
      }
    }
  }
	
  /* Loop over possibly overlapping facets  */
  for(ifo=0; ifo<overlap_counter; ifo++) {
    oflag = 0;
    
    ifacet = facet_overlaps[ifo];
    
    /* Compute facet ijk min and max */
    for(i=0; i<3; i++) { /* Loop over vertices */
      for(j=0; j<3; j++) { /* Loop over ijk index */
	if(facet_ijk[ifacet][i][j] < facet_ijkmin[ifacet][j]) {
	  facet_ijkmin[ifacet][j] = facet_ijk[ifacet][i][j];
	}
	if (facet_ijk[ifacet][i][j] > facet_ijkmax[ifacet][j]) {
	  facet_ijkmax[ifacet][j] = facet_ijk[ifacet][i][j];
	}
      }
    }
    
    facet = pfacet + ifacet;
    
    for(ixyz=0; ixyz<3; ixyz++) { /* Loop over facet edges */
      
      /* Calculate edge vector for each side*/
      for(i=0; i<3; i++) {
	if(ixyz==0) {
	  edgevec[i] = facet->vertex2[i] - facet->vertex1[i];
	} else if (ixyz==1) {
	  edgevec[i] = facet->vertex3[i] - facet->vertex2[i];
	} else {
	  edgevec[i] = facet->vertex1[i] - facet->vertex3[i];
	}
      }
      
      for(ief=1; ief<NITER; ief++) {
	
	/* Compute iterated position along facet edge */	
	for(i=0; i<3; i++) {
	  if(ixyz==0) {
	    edgepos[i] = facet->vertex1[i] + ((double)ief/(double)NITER)*edgevec[i];
	  } else if(ixyz==1) {
	    edgepos[i] = facet->vertex2[i] + ((double)ief/(double)NITER)*edgevec[i];
	  } else {
	    edgepos[i] = facet->vertex3[i] + ((double)ief/(double)NITER)*edgevec[i];
	  }
	}
	
	/* Loop over cells and check if iterated position lies within the cell */
	for(i=facet_ijkmin[ifacet][0]; i<=facet_ijkmax[ifacet][0]; i++) { /* X-position */ 
	  for(j=facet_ijkmin[ifacet][1]; j<=facet_ijkmax[ifacet][1]; j++) { /* Y-position */
	    for(k=facet_ijkmin[ifacet][2]; k<=facet_ijkmax[ifacet][2]; k++) { /* Z-position */
	      cell_number = i*nc[1]*nc[2] + nc[2]*j + k;
	      cell = pcell + cell_number;
	      dupflag = 0;
	      
	      /* Check iterated position */
	      if((edgepos[0] > cell->back) && (edgepos[0] <= cell->front)) {
		if((edgepos[1] > cell->left) && (edgepos[1] <= cell->right)) {
		  if((edgepos[2] > cell->bottom) && (edgepos[2] <= cell->top)) {
		    for(iter=0; iter<cellcounter[i][j][k]; iter++) {
		      if(ifacet==cell->facet_list[iter]) {
			dupflag = 1;
		      }
		    }
		    if(dupflag==0) {		    
		      cell->facet_list[(int)cellcounter[i][j][k]] = ifacet;
		      cellcounter[i][j][k]++;
		      total_counter++;
		      facet_af[ifacet]++;
		      oflag = 1;
		      if(cellcounter[i][j][k]>=MAXFACETS) {
			printf("MAXFACETS is too low! Increase its value!\n");
			exit(1);
		      }
		      break;
		    }
		  }
		}
	      }	
	    }
	  }  
	}
      } 
    }
  }
  

  free(facet_af);
  free(facet_overlaps);
  for(i=0; i<nfacets; i++) {
    for(j=0; j<3; j++) {
      free(facet_ijk[i][j]);
    }
  }
  for(i=0; i<nfacets; i++) {
    free(facet_ijk[i]);
    free(facet_ijkmin[i]);
    free(facet_ijkmax[i]);
  }
  free(facet_ijk);
  for(i=0; i<nc[0]; i++) {
    for(j=0; j<nc[1]; j++) {
      free(cellcounter[i][j]);
    }
  }
  for(i=0; i<nc[0]; i++) {
    free(cellcounter[i]);
  }
  free(cellcounter);
  free(facet_ijkmin);
  free(facet_ijkmax);

}	  

/******************************* PARTICLE_POSITION ******************************/
void pposition(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct particle_struct *pparticle, int ipart, int particle_surf)
{

  /* Determine the surface to create the particle on and then */
  /* calculate the X, Y, and Z position of the particle. */

  /* Inputs: cumul_dist = Cumulative distribution of surface fluxes */
  /*         XMIN = X-minimum domain boundary */
  /*         YMIN = Y-minimum domain boundary */
  /*         ZMIN = Z-minimum domain boundary */
            
  /* Outputs: ppos = Particle position [x, y, z] [m] */
  /*          particle_surf = Surface where particle was generated */
  /*          ihatX = normal to that surface */
  /*          ihatY = tangential component #1 to that surface */
  /*          ihatZ = tangential component #2 to that surface */

  struct particle_struct *particle;
  int i;

  particle = pparticle + ipart;

  /* COMPUTE RANDOM POSITION ON SURFACE */
  for(i=0; i<3; i++) {
    particle->pos[i] = 0.0;
  }

  particle->pos[0] = XMIN + ranf0(rr)*(XMAX - XMIN);
  particle->pos[1] = YMIN + ranf0(rr)*(YMAX - YMIN);
  particle->pos[2] = ZMIN + ranf0(rr)*(ZMAX - ZMIN);

  if(particle_surf==0)
    particle->pos[0] = XMIN;
  if(particle_surf==1) 
    particle->pos[0] = XMAX; 
  if(particle_surf==2) 
    particle->pos[1] = YMIN;
  if(particle_surf==3) 
    particle->pos[1] = YMAX;
  if(particle_surf==4) 
    particle->pos[2] = ZMIN;
  if(particle_surf==5) 
    particle->pos[2] = ZMAX;

}
  

/******************************* PARTICLE_VELOCITY **********************************/    
void pvelocity(double s[NSURF][NSPECIES], double Usurf[], int direction[], double Cmp[], int particle_surf, struct particle_struct *pparticle, int ipart, int species)
{

        
  /* Compute the proper velocity components for each particle based */
  /* on the surface and the bulk velocity vector. */

  /* Inputs: s = Speed ratio array */
  /*         Usurf = Bulk velocity for each surface */
  /*         direction = provides correct normal for molecule velocities */
  /*         Cmp = Most probable speed [m/s] */
  /*         ihatX = normal to that surface */
  /*         ihatY = tangential component #1 to that surface */
  /*         ihatZ = tangential component #2 to that surface */
            
  /* Outputs: pvel = Particle velocity vector [m/s]  */
  struct particle_struct *particle;
  int i;
  double phi;
  double un, a, fs1, fs2;
  double qa, utemp;
  double ss;
  int ihat[3], ips, sp;
 
  ips = particle_surf;
  sp = species;

  ss = s[ips][sp];
    
  if(ips==0 || ips==1) {
    ihat[0] = 0;
    ihat[1] = 1;
    ihat[2] = 2;
  }
      
  if(ips==2 || ips==3) {
    ihat[0] = 1;
    ihat[1] = 0;
    ihat[2] = 2;
  }
  
  if(ips==4 || ips==5) {
    ihat[0] = 2;
    ihat[1] = 1;
    ihat[2] = 0;
  }

  particle = pparticle + ipart;

  for(i=0; i<3; i++) {
    particle->vel[i] = 0.0;
  }

  phi = 2.0*TPM_PI*ranf0(rr);

  /* COMPUTE VELOCITY VECTOR FROM MAXWELLIAN DISTRIBUTION */
 
  un = -1.0;
  a = 1.0;
  fs1 = ss + sqrt(ss*ss + 2.0);
  fs2 = 0.5*(1.0 + ss*(2.0*ss-fs1));
  if(fabs(Usurf[ihat[0]]) > 1.0e-6) {
    qa = 3.0;
    if(ss < -3.0) qa = fabs(ss)+1.0;
    while(un < 0.0 || a < ranf0(rr)) {
      utemp = -qa + 2.0*qa*ranf0(rr);
      un = utemp+ss;
      a = (2.0*un/fs1)*exp(fs2-utemp*utemp);
    }
    particle->vel[ihat[0]] = direction[ips]*un*Cmp[sp]; 
  }     
  else {
    particle->vel[ihat[0]] = direction[ips]*sqrt(-log(ranf0(rr)))*Cmp[sp];
  }

  a = sqrt(-log(ranf0(rr)));
  particle->vel[ihat[1]] = Usurf[ihat[1]] + a*Cmp[sp]*cos(phi);
  particle->vel[ihat[2]] = Usurf[ihat[2]] + a*Cmp[sp]*sin(phi);  

}



/**************************** TRACK PARTICLE THROUGH CELLS ***************************/
void cell_track(struct particle_struct *pparticle, int ipart, struct cell_struct *pcell, int nc[3], int particle_cell_track[MAXCELLS]) {

  /* Compute the cells that the particle pass through */

  /* Inputs: pparticle = particle structure */
  /*         ipart = current particle       */
  /* 	      pcell = cell structure         */

  /* Outputs: Cell numbers that the particle passes through */ 

  int i, j, k, m;
  int iside;
  struct particle_struct *particle;
  struct cell_struct *cell0, *cell;
  double a;
  double ix1, ix2;
  int cell_number;
  int counter = 0;
  int xflag;
  double cpos[3]; /* Current particle position */
  double eps1 = 2.0e-15;
  double eps2 = 1.0e-15;
  int diff;

  double xside[6], boundmin[3], boundmax[3];
  int ipar[6], iperp1[6], iperp2[6];

  for(i=0; i<nc[0]; i++) { /* X-position */
    for(j=0; j<nc[1]; j++) { /* Y-position */
      for(k=0; k<nc[2]; k++) { /* Z-position */
	cell_number = i*nc[2]*nc[1] + j*nc[2] + k;
	cell = pcell + cell_number;
      }
    }
  }

  particle = pparticle + ipart;
  cell0 = pcell;

  for(m=0; m<3; m++) {
    cpos[m] = particle->pos[m];
  }

  /* Determine the initial cell that the particle is located within */
  i = (cpos[0] - cell0->back)/(cell0->front - cell0->back);
  j = (cpos[1] - cell0->left)/(cell0->right - cell0->left);
  k = (cpos[2] - cell0->bottom)/(cell0->top - cell0->bottom);
  
  if(i==nc[0] && cpos[0] > ((cell0->back)+nc[0]*(cell0->front-cell0->back) - eps1) && 
                 cpos[0] < ((cell0->back)+nc[0]*(cell0->front-cell0->back) + eps1))
    i--; /* Particle is on X-inclusive boundary */
  if(j==nc[1] && cpos[1] > ((cell0->left)+nc[1]*(cell0->right-cell0->left) - eps1) &&
                 cpos[1] < ((cell0->left)+nc[1]*(cell0->right-cell0->left) + eps1))
    j--; /* Particle is on Y-inclusive boundary */
  if(k==nc[2] && cpos[2] > ((cell0->bottom)+nc[2]*(cell0->top-cell0->bottom) - eps1) && 
                 cpos[2] < ((cell0->bottom)+nc[2]*(cell0->top-cell0->bottom) + eps1))
    k--; /* Particle is on Z-inclusive boundary */

  while((i>=0) && (i<nc[0]) && (j>=0) && (j<nc[1]) && (k>=0) && (k<nc[2])) {

    xflag = 0;

    cell_number = i*nc[2]*nc[1] + j*nc[2] + k;
    cell = pcell + cell_number;

    particle_cell_track[counter] = cell_number;

    counter++;

    if(counter==MAXCELLS) {
	break;
    }

    /* Determine which neighboring cell the particles moves to */
    /* For now, simply do a static order: front, top, bottom, right, left, back */
 
    xside[0] = cell->front;
    xside[1] = cell->top; 
    xside[2] = cell->bottom;
    xside[3] = cell->right;
    xside[4] = cell->left;
    xside[5] = cell->back;

    /* Front */
    ipar[0] = 0;
    iperp1[0] = 1;
    iperp2[0] = 2;

    /* Back */
    ipar[5] = 0;
    iperp1[5] = 1;
    iperp2[5] = 2;
  
    /* Top & Bottom */
    for(m=1; m<3; m++) {
      ipar[m] = 2;
      iperp1[m] = 0;
      iperp2[m] = 1;
    }

    /* Left & Right */      
    for(m=3; m<5; m++) {
      ipar[m] = 1;
      iperp1[m] = 0;
      iperp2[m] = 2;
    }

    boundmin[0] = cell->back;
    boundmin[1] = cell->left;
    boundmin[2] = cell->bottom;
    
    boundmax[0] = cell->front;
    boundmax[1] = cell->right;
    boundmax[2] = cell->top;
    
    for(iside=0; iside<6; iside++) {

      a = (xside[iside] - cpos[ipar[iside]])/particle->vel[ipar[iside]];
      ix1 = cpos[iperp1[iside]] + a*particle->vel[iperp1[iside]];
      ix2 = cpos[iperp2[iside]] + a*particle->vel[iperp2[iside]];

      if(a>0.0) {
	if((ix1 > boundmin[iperp1[iside]]) && (ix1 <= boundmax[iperp1[iside]])) {
	  if((ix2 > boundmin[iperp2[iside]]) && (ix2 <= boundmax[iperp2[iside]])) {
	    
	    /* Move the particle to the new position and cell */
	    /* Use a small offset to ensure the particle isn't on the cell boundary */
	    for(m=0; m<3; m++) {
	      cpos[m] += (a+eps2)*particle->vel[m];
	    }
	    
	    xflag = 1;
	    if(iside==0) {
	      i++;
	    } else if(iside==1) {
	      k++;
	    } else if(iside==2) {
	      k--;
	    } else if(iside==3) {
	      j++;
	    } else if(iside==4) {
	      j--;
	    } else if(iside==5) {
	      i--;
	    } 
	    break;
	    
	  }
	}
      }

    } /* End for loop over sides */

  } /* End while loop over domain */

}
  
/******************************* COMPILE FACET LIST ***************************/
void compile_facet_list(struct facet_struct *pfacet, int nfacets, struct cell_struct *pcell, int particle_cell_track[MAXCELLS], int *total_facet_list, int *fcount, int ipart) {

  /* Compiles the total facet list for a particle */
  /* Key point is removing duplicates from the list */

  /* Inputs: pfacet = Facet structure   */
  /*         nfacets = Number of facets */
  /*         pcell = Cell structure     */
  /*         Particle Cell Track        */
  
  /* Outputs: Total Facet List          */

    /* DETERMINE THE FACETS TO LOOP THROUGH */
  int icell = 0;
  int ipf, cell_number;
  int i;
  int ifl, ifl2;

  struct cell_struct *cell;

  *fcount = 0;

  while(particle_cell_track[icell] >=0) { /* Break the loop when the cell track ends */

    ipf=0;
    cell_number = particle_cell_track[icell];
    cell = pcell + cell_number;

    // CHICAGO CODE HAVE THIS ADDED
    if(icell==MAXCELLS) {
       // printf("Trapped particle deleted\n");

      // printf("icell=%d\n",icell);
      // printf("cell_number=%d\n",cell_number);
      // printf("nfacets=%d\n", nfacets);
      // printf("fcount=%d\n", fcount);
      // printf("ipart=%d\n", ipart);
      // printf("MAXCELLS=%d\n",MAXCELLS);
       //for(i=icell-10;i<MAXCELLS;i++){
         // printf("%d : %d\n", i, particle_cell_track[i]);
      // }
       //if (icell>MAXCELLS){
       // printf("ERROR: icell was beyond MAXCELLS, this will crash...\n");
      // }
 

       break;
     }

    // DEBUGGER MENTION ERROR with while(cell->facet_list[ipf] >=0) {
    //
    //
    if (cell->facet_list)
    {
    //printf("cell->facet_list[0]=%d\n", cell->facet_list[0]);
    // Proceed
    }
    else
    {
    printf("ERROR:Null pointer for cell->facet_list");
    exit(1);
    // Handle null-pointer error
   }
    //printf("cell_number=%d\n",cell_number);
    while(cell->facet_list[ipf] >=0) {
      for(i=0; i<*fcount; i++) {
	if(total_facet_list[i]==cell->facet_list[ipf]) {
	  ipf++;
	  break;
	}
      }
      if(i==*fcount) {
	total_facet_list[*fcount] = cell->facet_list[ipf];
	*fcount = *fcount + 1;
	ipf++;
      }
    }

    icell++;
  } /* End while loop */

}

/****************************** PATH_FACET_INTERSECTION ******************************/    
void pf_intercept(int *min_fc, struct facet_struct *pfacet, struct particle_struct *pparticle, struct samp_struct *psample, int ipart, int particle_surf, int nfacets, struct cell_struct *pcell, int particle_cell_track[MAXCELLS], int *total_facet_list, int fcount) 
{

  /* Determines which facet planes the particle path would intersect */
  /* (in an infinite sense) and the corresponding point that the intersection */
  /* would occur. Next, it determines whether that point lines in the */
  /* finite triangular facet plane by converting to Barycentric coordinates. */

  /* Inputs: vertex1 = x, y, z positions of 1st vertex */
  /*         vertex2 = x, y, z positions of 2nd vertex */
  /*         vertex3 = x, y, z positions of 3rd vertex */
  /*         pvel = Particle velocity vector [u, v, w] [m/s] */
  /*         ppos = Particle position [x, y, z] [m] */
            
  /* Outputs: pvel = Particle velocity vector [m/s] */
  struct facet_struct *facet;
  struct particle_struct *particle;
  struct cell_struct *cell;
#if SAMPLE
  struct samp_struct *sample;
#endif /* SAMPLE */

  int inside[nfacets];
  int singular[nfacets];
  double intersection[nfacets];
  double xvec[3];
  double a, b;
  double r, w[3];
  double u[3], v[3];
  double uu, uv, vv;
  double wu, wv;
  double s, t;
  double D;
  double pixn[3];
  int i, fc;
  int icell, ipf, ifl;
  int cell_number;
  double min_int;
  double rndoff = 1.0e-15;
  int ifl2;

  for(i=0; i<nfacets; i++) {
    inside[i] = 0;
    intersection[i] = 1.0e6;
    singular[i] = 0;
  }

  for(i=0; i<3; i++) {
    xvec[i] = 0.0;
    u[i] = 0.0;
    v[i] = 0.0;
  }
 
  particle = pparticle + ipart;

#if SAMPLE
  sample = psample + ipart;
#endif /* SAMPLE */
                           
  /* DETERMINE WHETHER PATH OF THE PARTICLE INTERSECTS ANY OF THE FACETS */  
  /* LOOP ONLY OVER FACETS TO WHICH THE PARTICLE PASSED NEARBY */

  for(ifl=0; ifl<fcount; ifl++) {
    
    fc = total_facet_list[ifl];

    facet = pfacet + fc;

    for(i=0; i<3; i++) {
      u[i] = facet->vertex2[i] - facet->vertex1[i];
      v[i] = facet->vertex3[i] - facet->vertex1[i];
    }
  
    for(i=0; i<3; i++) {
      xvec[i] = particle->pos[i]-facet->vertex1[i];
    }

    a = -dot(facet->normal, xvec);
    b = dot(facet->normal, particle->vel);

    if(a == 0 || b == 0) {
      inside[fc] = 0;
      singular[fc] = 1;
    }
    else {
	
      singular[fc] = 0;
      r = a/b;
      intersection[fc] = r;
      if (r < 0.0) {
	inside[fc] = 0;
      }
      else {
	  
	for(i=0; i<3; i++) {
	  pixn[i] = particle->pos[i] + r*particle->vel[i];
	}
	  
	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	  
	for(i=0; i<3; i++) {
	  w[i] = pixn[i] - facet->vertex1[i];
	}
	
	wu = dot(w, u);
	wv = dot(w, v);
	  
	D = uv*uv - uu*vv;
	  
	s = (uv*wv - vv*wu)/D;
	if(s < 0.0 || s > 1.0) {
	  inside[fc] = 0;
	}
	else {
	  t = (uv * wu - uu * wv) / D;
	  if(t < 0.0 || (s+t) > 1.0) {
	    inside[fc] = 0;
	  }
	  else {
	    inside[fc] = 1;
	  }
	}
      }
    }
  } /* END LOOP OVER FACETS */

  min_int = 1.0e6;
  *min_fc = -1;

  for(ifl=0; ifl<fcount; ifl++) {
    fc = total_facet_list[ifl];
    if(singular[fc]==0) {
      if(inside[fc]==1) {
  	if(intersection[fc] > 0.0 && intersection[fc] > rndoff) {
  	  if(fabs(intersection[fc]) < min_int) {
  	    min_int = fabs(intersection[fc]);
  	    *min_fc = fc;
  	  }
  	}
      }
    }
  }

  /* UPDATE PARTICLE POSITION IF IT HIT A BOUNDARY */
  if(*min_fc >=0) {
    for(i=0; i<3; i++) {
      particle->pos[i] = particle->pos[i] + intersection[*min_fc]*particle->vel[i];
    }
  } 

#if SAMPLE
  if(*min_fc >=0 ) {
    sample->t = intersection[*min_fc];
  }
#endif /* SAMPLE */

}
  
/******************************** GAS_SURFACE_INTERACTION ******************************/    
void gsi(struct particle_struct *pparticle, int ipart, struct facet_struct *pfacet, int min_fc, double Vw, double pveln[], int GSI_MODEL, double epsilon, double alpha, double alphan, double sigmat)
{

  /* Computes the gas-surface interaction between the test particle */
  /* and the satellite surface. Can use one of three models: */
  /* Maxwell's Model, Diffuse Reflection with Incomplete Accommodation */
  /* or the Cercignani-Lampis-Lord (CLL) model. In essence, the GSI changes */
  /* the magnitude and direction of the particle velocity vector according */
  /* to some algorithm. */

  /* Inputs: pvel = Particle velocity vector [u, v, w] [m/s] */
  /*         facet_normal = Facet Normal [x, y, z] */
  /*         vertex1 = x, y, z positions of 1st vertex */
  /*         vertex2 = x, y, z positions of 2nd vertex */
  /*         vertex3 = x, y, z positions of 3rd vertex */
            
  /* Outputs: pveln = Reflected particle velocity vector */
  /*          in domain coordinate system [m/s] */

  struct facet_struct *facet;
  struct particle_struct *particle;
  double pvelf[3];
  double nvec[3], t1vec[3], t2vec[3];
  double norm;
  double pvelr[3];
  int i;

  for(i=0; i<3; i++) {
    pvelf[i] = 0.0;
    pveln[i] = 0.0;
    pvelr[i] = 0.0;
    nvec[i] = 0.0;
    t1vec[i] = 0.0;
    t2vec[i] = 0.0;
  }

  facet = pfacet + min_fc;
  particle = pparticle + ipart;

  norm = 0.0;
  for(i=0; i<3; i++) {
    norm += (facet->vertex2[i]-facet->vertex1[i])*(facet->vertex2[i]-facet->vertex1[i]);
  }

  norm = sqrt(norm);
   
  /* COMPUTE TANGENTIAL VECTOR */
  for(i=0; i<3; i++) {
    nvec[i] = facet->normal[i];
    t1vec[i] = (facet->vertex2[i] - facet->vertex1[i])/norm;
  }

  cross(nvec, t1vec, t2vec);

  /* FIRST, CONVERT VELOCITY VECTORS TO FACET COORDINATE SYSTEM */
  pvelf[0] = particle->vel[0]*nvec[0]  + particle->vel[1]*nvec[1]  + particle->vel[2]*nvec[2];
  pvelf[1] = particle->vel[0]*t1vec[0] + particle->vel[1]*t1vec[1] + particle->vel[2]*t1vec[2];
  pvelf[2] = particle->vel[0]*t2vec[0] + particle->vel[1]*t2vec[1] + particle->vel[2]*t2vec[2];
  
  if(GSI_MODEL == 0) { /* MAXWELL'S MODEL */
    maxwell(pvelf, pvelr, Vw, epsilon);
  }
  else if(GSI_MODEL == 1) { /* DIFFUSE REFLECTION WITH INCOMPLETE ACCOMMODATION */
    dria(pvelf, pvelr, Vw, GSI_MODEL, alpha, alphan, sigmat);
  }
  else if(GSI_MODEL == 2) { /* CLL MODEL */
    cll(pvelf, pvelr, Vw, GSI_MODEL, alphan, sigmat, alpha);
  }
 
  /* CONVERT VELOCITIES BACK TO THEIR ORIGINAL COORDINATE FRAME */
  pveln[0] = pvelr[0]*nvec[0] + pvelr[1]*t1vec[0] + pvelr[2]*t2vec[0];
  pveln[1] = pvelr[0]*nvec[1] + pvelr[1]*t1vec[1] + pvelr[2]*t2vec[1];
  pveln[2] = pvelr[0]*nvec[2] + pvelr[1]*t1vec[2] + pvelr[2]*t2vec[2];

}


/********************************* MAXWELL'S MODEL *********************************/      
void maxwell(double pvelf[], double pvelr[], double Vw, double epsilon)
{

  /* Computes the velocity transformation according to Maxwell's Model */

  /* Inputs: pvelf = Particle velocity vector in facet plane frame [u, v, w] [m/s] */
  /*         epsilon = Fraction of particles specularly reflected */
  /*         Vw = Wall velocity [m/s] */
            
  /* Outputs: pvelr = Reflected particle velocity vector in facet plane frame [m/s] */
  
  double phi;
  double a;
  int i;
   
  if(ranf0(rr) > epsilon) { /* DIFFUSE */
      
    pvelr[0] = sqrt(-log(ranf0(rr)))*Vw;
    phi = 2.0*TPM_PI*ranf0(rr);
    a = sqrt(-log(ranf0(rr)));
    pvelr[1] = a*Vw*cos(phi);
    pvelr[2] = a*Vw*sin(phi);
  }
  else { /* SPECULAR */
    pvelr[0] = -pvelf[0];
    pvelr[1] = pvelf[1];
    pvelr[2] = pvelf[2];
  }

}

/************************* DIFFUSE REFLECTION WITH INCOMPLETE ACCOMMODATION **********************/      
void dria(double pvelf[], double pvelr[], double Vw, int GSI_MODEL, double alpha, double alphan, double sigmat)
{

  /* Computes the velocity transformation according to the Diffuse */
  /* Reflection with Incomplete Accommodation Model (DRIA) */

  /* Inputs: pvelf = Particle velocity vector in facet plane frame [u, v, w] [m/s] */
  /*         alpha = Energy Accommodation Coefficient */
  /*         Vw = Wall velocity [m/s] */
            
  /* Outputs: pvelr = Reflected particle velocity vector in facet plane frame [m/s] */

  double pvu[3];
  double Vtot;
  double phi, theta;
  int i;

  for(i=0; i<3; i++) {
    pvelr[i] = 0.0;
    pvu[i] = 0.0;
  }

  cll(pvelf, pvelr, Vw, GSI_MODEL, alphan, sigmat, alpha);
    
  Vtot = sqrt(dot(pvelr, pvelr));
    
  /* RESELECT VELOCITY COMPONENTS FROM COSINE DISTRIBUTION */
  phi = 2.0*TPM_PI*ranf0(rr);
  theta = asin(sqrt(ranf0(rr)));
  pvelr[0] = Vtot*cos(theta);
  pvelr[1] = Vtot*sin(theta)*cos(phi);
  pvelr[2] = Vtot*sin(theta)*sin(phi);

}


/*********************** CERCIGNANI-LAMPIS-LORD MODEL *****************************/       
void cll(double pvelf[], double pvelr[], double Vw, int GSI_MODEL, double alphan, double sigmat, double alpha) 
{

  /* Computes the velocity transformation according to the */
  /* Cercignani-Lampis-Lord Model (CLL) Model */

  /* Inputs: pvelf = Particle velocity vector in facet plane frame [u, v, w] [m/s] */
  /*         alphan = Normal Energy Accommodation Coefficient */
  /*         sigmat = Tangential Momentum Accommodation Coefficient */
  /*         Vw = Wall velocity [m/s] */
            
  /* Outputs: pvelr = Reflected particle velocity vector in facet plane frame [m/s] */

  double pvelt[3];
  double an, at;
  double delta, Vtang, r, theta, W;
  double vCLL, wCLL;
  int i;

  for(i=0; i<3; i++) {
    pvelr[i] = 0.0;
    pvelt[i] = 0.0;
  }

  for(i=0; i<3; i++) {
    pvelt[i] = pvelf[i]/Vw;
  }
  
  if(GSI_MODEL==1) 
    an = alpha;
  else 
    an = alphan;
  
  delta = atan2(pvelt[2],pvelt[1]);
  Vtang = sqrt(pvelt[1]*pvelt[1]+pvelt[2]*pvelt[2]);
  r = sqrt(-an*log(ranf0(rr)));
  theta = 2.0*TPM_PI*ranf0(rr);
  W = pvelt[0]*sqrt(1.0-an);
  pvelr[0] = sqrt(r*r+W*W+2.0*r*W*cos(theta));

  if(GSI_MODEL==1) 
    at = alpha;   
  else 
    at = sigmat*(2.0-sigmat);
      
  r = sqrt(-at*log(ranf0(rr)));
  theta = 2.0*TPM_PI*ranf0(rr);
  W = Vtang*sqrt(1.0-at);
  vCLL = W + r*cos(theta);
  wCLL = r*sin(theta);

  pvelr[1] = vCLL*cos(delta)-wCLL*sin(delta);
  pvelr[2] = vCLL*sin(delta)+wCLL*cos(delta);

  for(i=0; i<3; i++) {
    pvelr[i] *= Vw;
  }

}


/********************************* COMPUTE_CD *************************************/    
double compute_Cd(double Umag, double theta, double phi, double Fx, double Fy, double Fz, double proj_area, double n, double m_avg)
{

  /* Computes the drag coefficient of the object */

  /* Inputs: Umag  = Magnitude Bulk Velocity  */
  /*         theta = Pitch Angle              */
  /*         phi   = Yaw Angle                */
  /*         Fx = X-direction force [N]       */
  /*         Fy = Y-direction force [N]       */
  /*         Fz = Z-direction force [N]       */
  /*         Umag = Magnitude of the bulk velocity vector [m/s] */
  /*         proj_area = Projected Area of the Object [m^2]     */
  /*         n = Atmospheric number density [m^-3]              */
  /*         m_avg = Average mass of real particles             */
            
  /* Outputs: Cd = Drag coefficient of the object */

  double Cnorm, Caxial;
  double Cd;

  Cnorm = Fx/(0.5*Umag*Umag*proj_area*n*m_avg);
  Caxial = (fabs(Fy*sin(theta)*cos(phi)) + fabs(Fz*sin(phi)))/(0.5*Umag*Umag*proj_area*n*m_avg);

  Cd = Cnorm*cos(theta)*cos(phi) + Caxial;

  return(Cd);

}

/*********************************** DOT PRODUCT *************************************/
double dot(double V[], double W[]) {
   
  /* Computes dot product between vectors V and W */
  /* Inputs: V = Vector #1 [Vx, Vy, Vz] */
  /*         W = Vector #2 [Wx, Wy, Wz] */
  /* Output: a = Dot Product of V and W */

  double a = 0.0;

  a = V[0]*W[0] + V[1]*W[1] + V[2]*W[2];
  return(a);

}

/*********************************** DOT PRODUCT *************************************/
void cross(double V[], double W[], double VEC[]) {
  /* Computes cross product between vectors V and W */

  /* Inputs: V = Vector #1 [Vx, Vy, Vz] */
  /*         W = Vector #2 [Wx, Wy, Wz] */

  /* Output: b = Cross Product of V and W */

  int i;

  VEC[0] = V[1]*W[2] - V[2]*W[1];
  VEC[1] = V[2]*W[0] - V[0]*W[2];
  VEC[2] = V[0]*W[1] - V[1]*W[0];

}


/********************************* RANDOM NUMBER GENERATOR **************************/
double ranf0(gsl_rng *rr)
{
  return gsl_rng_uniform_pos (rr);
}
 
