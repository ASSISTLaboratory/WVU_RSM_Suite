#define PARALLEL       1          /* 1 = PARALLEL COMPUTATION; 0 = SERIAL */
#define NPART          1.0e6      /* NUMBER OF TEST PARTICLES */
#define NSURF          6          /* NUMBER OF DOMAIN SURFACES */
#define NSPECIES       6          /* NUMBER OF ATMOSPHERIC SPECIES */
#define NCELLS         100        /* MAXIMUM NUMBER OF CELLS IN 1-DIMENSION  */
#define MAXCELLS       10*NCELLS  /* MAXIMUM NUMBER OF ENTRIES IN PARTICLE-CELL TRACKING ARRAY */
#define MAXFACETS      30000      /* MAXIMUM NUMBER OF FACETS THAT CAN BE IN A SINGLE CELL */
#define NITER          100        /* NUMBER OF SEGMENTS TO BREAK UP FACET EDGES INTO FOR ITERATION */
#define MAXCOUNT       10000
#define FILEhf         "tempfiles/tpm.ensemble.h5"
#define TPM_PI         3.141593   /* PI */
#define DYNAMIC_SEED   1          /* USES A DIFFERENT SEED FOR EACH RUN BASED ON CLOCK TIME */
#define SAMPLE         0          /* SAMPLE PARTICLE POSITION/VELOCITY FOR VISUALIZATION OUTPUT */
#define NSPECIES       6
#define AT_NUM_DENSITY 7.5e14     /* ATMOSPHERIC NUMBER DENSITY [#/m^3] */

struct particle_struct {
  double pos[3];
  double vel[3];
};

struct facet_struct {
  double normal[3];
  double vertex1[3];
  double vertex2[3];
  double vertex3[3];
  double area;
};

struct cell_struct {
  double top;                   /* Maximum Z-position */
  double bottom;                /* Minimum Z-position */
  double right;                 /* Maximum Y-position */
  double left;                  /* Minimum Y-position */
  double front;                 /* Maximum X-position */
  double back;                  /* Minimum X-position */
  int facet_list[MAXFACETS];    /* List of the facets in cell */
};

struct samp_struct {
  double ivel[3];
  double rvel[3];
  double ipos[3];
  double t;
};

void read_input(char objname[1024], double X[], int *GSI_MODEL, int *ROTATION_FLAG);

void defzeros(int i, char **zeros);

double testparticle(char filename[1024], double Umag, double theta, double phi, double Ts, double Ta, double epsilon, double alpha, double alphan, double sigmat, double X[], int GSI_MODEL, int iTOT,double *area);

int read_num_lines(char filename[1024]);

void facet_properties(char filename[1024], double Ux, double Uy, double Uz, int nfacets, struct facet_struct *pfacet, double Umag);

void domain_boundary(double *XMIN, double *YMIN, double *ZMIN, double *XMAX, double *YMAX, double *ZMAX, int nfacets, struct facet_struct *pfacet);

void max_facet_dimension(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct facet_struct *pfacet, int nfacets, int nc[3]);

void create_grid(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct cell_struct *pcell, struct facet_struct *pfacet, int nfacets, int nc[3]);

void sort_facets(struct facet_struct *pfacet, struct cell_struct *pcell, int nfacets, int nc[3]);

double projected_area(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, double Ux, double Uy, double Uz, int nfacets, struct facet_struct *pfacet);

void pposition(double XMIN, double YMIN, double ZMIN, double XMAX, double YMAX, double ZMAX, struct particle_struct *pparticle, int ipart, int particle_surf);

void pvelocity(double s[NSURF][NSPECIES], double Usurf[], int direction[], double Cmp[], int particle_surf, struct particle_struct *pparticle, int ipart, int species);

void cell_track(struct particle_struct *pparticle, int ipart, struct cell_struct *pcell, int nc[3], int particle_cell_track[MAXCELLS]);

void compile_facet_list(struct facet_struct *pfacet, int nfacets, struct cell_struct *pcell, int particle_cell_track[MAXCELLS], int *total_facet_list, int *fcount, int ipart);

void pf_intercept(int *min_fc, struct facet_struct *pfacet, struct particle_struct *pparticle, struct samp_struct *psample, int ipart, int particle_surf, int nfacets, struct cell_struct *pcell, int particle_cell_track[MAXCELLS], int *total_facet_list, int fcount);

void gsi(struct particle_struct *pparticle, int ipart, struct facet_struct *pfacet, int min_fc, double Vw, double pveln[], int GSI_MODEL, double epsilon, double alpha, double alphan, double sigmat);

void maxwell(double pvelf[], double pvelr[], double Vw, double epsilon);

void dria(double pvelf[], double pvelr[], double Vw, int GSI_MODEL, double alpha, double alphan, double sigmat);

void cll(double pvelf[], double pvelr[], double Vw, int GSI_MODEL, double alphan, double sigmat, double alpha);

double compute_Cd(double Umag, double theta, double phi, double Fx, double Fy, double Fz, double proj_area, double n, double m_avg);

double dot(double V[], double W[]);

void cross(double V[], double W[], double VEC[]);

double ranf0(gsl_rng *rr);
