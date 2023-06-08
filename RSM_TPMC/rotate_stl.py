#!/usr/bin/env python3

import numpy as np
import os
import math
import json
import logging 

def CWT(meshdir,filename):       #Function created by Logan Sheridan at WVU

####################### Analyze STL #######################
    #meshpath = os.path.join(meshdir,filename)
    meshpath = meshdir
    stl = open(meshpath,'r')    #Open stl file
    data=stl.readlines()        #Read all lines of stl file
    lst=[]                      #Create empty list so self-updating is possible
    nfacet=0
    for i in range(len(data)):
        x='vertex' in data[i]   #See if the word 'vertex' exists within each line
        if x==1:
            lst.append(data[i]) #If so, add to the list
            vertices=np.zeros([len(lst),3])    #Create matrix for vertices

    for i in range(len(lst)):
        st=lst[i]
        sta=st.strip()          #Remove all trailing spaces in line
        sta=sta.replace('vertex','').strip()    #Remove the word 'vertex'
        sta=np.array(sta.split())         #Split remaining string into 1x3 vector containing vertices
        vertices[i,:]=sta                 #Add vertices to matrix

    for i in range(len(data)):
        j='endfacet' in data[i]
        if j==1:
            nfacet=nfacet+1         #Find the number of facets

################## Check Water Tightness ##################

    stl.close()
    nvert = len(vertices) #number of vertices
    f = np.arange(nvert).reshape((nfacet, 3)) #acts as a placeholder for faces (faster than a for loop)
    c, ia, ic= np.unique(vertices,return_index=True,return_inverse=True, axis=0)  #find the index of the unique vertices
    e=ic[f] #These are essentially the edges of the facet. Each should occur 3 times
    edges = np.concatenate([e[:,[0,1]], e[:,[1,2]], e[:,[2,0]] ]) #create list of edges

    #Sort edges and find unique edges and how many
    sortedges= np.sort(edges) #sort edges
    unqedges= np.unique(sortedges, axis=0) #number of unique edges

    if len(edges) == len(unqedges)*2: #because every edge is used twice (each edge touches two triangles)
        print(filename+' is Watertight \n')

    else:
        print(filename +' is NOT Watertight! \n Terminating program...')
        exit()


def rotate_stl(base_folder):

    logging.basicConfig(level=logging.DEBUG)

    #### Read Inputs ####
    print("[rotate_stl] Base folder: "+base_folder)

    ## Checking for needed folders
    inputpath = os.path.join(base_folder,"Inputs/STL_Rotation_Inputs")
    meshdir = os.path.join(base_folder, "Inputs/STL_Files")

    for i in [inputpath, meshdir]:
        if not os.path.isdir(i):
            raise ValueError("Could not find folder: " % i)

    ## Checking for needed files
    parentfile=inputpath+os.sep+'parent.txt'
    hinge1=inputpath+os.sep+'hinge_points1.txt'
    hinge2=inputpath+os.sep+'hinge_points2.txt'
    deflectionfile=base_folder+os.sep+'Outputs/deflections.txt'

    for i in [parentfile, hinge1, hinge2, deflectionfile ]:
        if not os.path.isfile(i):
            raise ValueError("Could find mandatory file: " % i)

    if not os.path.isdir(base_folder+os.sep+'tempfiles'):
        os.path.mkdir(base_folder+os.sep+'tempfiles')

    ## Determine number of components to be rotated
    parents = open(parentfile,'r')
    parentnames = parents.read().split()
    num_of_components = len(parentnames)
    parents.close()

    hp1_coordinates = open(hinge1,'r')
    hp1 = np.loadtxt(hp1_coordinates)

    hp2_coordinates = open(hinge2,'r')
    hp2 = np.loadtxt(hp2_coordinates)

    ## Assign values of all inputs
    deflections = open(deflectionfile)

    deflec = np.loadtxt(deflections)
    deflecholder = np.zeros(np.shape(deflec))

    ## Check to make sure there are the correct number of values in each input
    if len(deflec[0]) != np.size(parentnames):  #Columns = number of parent files
        print("Number of components and deflections do not match")
        exit()

    if len(hp1) != np.size(parentnames): #Rows = number of parent files
        print("Number of components and number of first hinge points do not match")
        exit()

    if len(hp2) != np.size(parentnames): #Rows = number of parent files
        print("Number of components and number of second hinge points do not match")
        exit()

    #### End Inputs ####

    world = np.array([0,0,0]) #define world orgin
    h = np.subtract(hp2,hp1)

    for defl in range(len(deflec)):  ## top most for loop
        logging.debug("Deflection No: %d", defl)   
        ### Deflection for each Parent Name  ###
        for plcv in range(len(parentnames)):
            logging.debug("Parent Name No: %d", plcv)

            hi = h[plcv,:]  # Define hinge point 1 for stl
            p2 = hp2[plcv,:]  # Define hinge point 2 for stl
            filename = parentnames[plcv] #STL being opened

                ### Read in values/attributes of mesh file ###
            meshpath = os.path.join(meshdir,filename)
            mesh_stl = open(meshpath,'r')    #Open stl file
            data=mesh_stl.readlines()        #Read all lines of stl file
            lst=[]                      #Create empty list so self-updating is possible
            nst=[]
            nfacet=0
            for i in range(len(data)):
                xx='vertex' in data[i] #See if the word 'vertex' exists within each line
                yy= 'normal' in data[i] #See if the word 'normal' exists within each line
                if xx==1:
                    lst.append(data[i]) #If so, add to the list
                    vertices=np.zeros([len(lst),3])    #Create matrix for vertices
                elif yy==1:
                     nst.append(data[i])
                     normalStore = np.zeros([len(nst),3])


            for i in range(len(lst)):
                st=lst[i]
                sta=st.strip()          #Remove all trailing spaces in line
                sta=sta.replace('vertex','').strip()    #Remove the word 'vertex'
                sta=np.array(sta.split())         #Split remaining string into 1x3 vector containing vertices
                vertices[i,:]=sta                 #Add vertices to matrix

            for i in range(len(nst)):
                nt=nst[i]
                nta=nt.strip()          #Remove all trailing spaces in line
                nta=nta.replace('facet normal','').strip()    #Remove the word 'facet normal'
                nta=np.array(nta.split())         #Split remaining string into 1x3 vector containing vertices
                normalStore[i,:]=nta                 #Add vertices to matrix


            for i in range(len(data)):
                j='endfacet' in data[i]
                if j==1:
                    nfacet=nfacet+1         #Find the number of facets

            v1 = vertices[0::3] #Store first vertex for normal calculations


            magh = math.sqrt(hi[0]**2 + hi[1]**2 + hi[2]**2)

            ## Direction cosines to world body directions
            alph = (180/np.pi)*math.acos(hi[0]/magh)
            bet = (180/np.pi)*math.acos(hi[1]/magh)
            gam = (180/np.pi)*math.acos(hi[2]/magh)

            ## Account for any singularities in DCM's
            if alph == 90:
                alph = 90-.0000001
            else:
                pass

            if gam == 180:
                gam = 180 - 0.000001
            elif gam == 0:
                gam = 0 + 0.000001
            else:
                pass

            delta = [p2 - world] #Translation vector from world orgin to a point on hinge line
            delta =np.array(delta)
            delta = np.transpose(delta)
            #First rotation angle
            theta1 = (180/np.pi)*math.acos(((math.cos(alph*np.pi/180)**2) + (math.sin(gam*np.pi/180)**2) - (math.cos(bet*np.pi/180)**2))/(2*math.cos(alph*np.pi/180)*math.sin(gam*np.pi/180)))
            #Second rotation angle
            theta2 = 90 - gam

            #===== Determine octant to which the hinge line vector points =====#

            if hi[0] >= 0 and hi[1] >= 0 and hi[2]>=0:
                pass #Do Nothing
            elif hi[0] < 0 and hi[1] >= 0 and hi[2] >= 0:
                pass #Do Nothing
                #OCTANT 3
            elif hi[0] < 0 and hi[1] < 0 and hi[2] >= 0:
                theta1 = -theta1;
                #OCTANT 4
            elif hi[0] >= 0 and hi[1] < 0 and hi[2] >= 0:
                theta1 = -theta1;
                #OCTANT 5
            elif hi[0] >= 0 and hi[1] >= 0 and hi[2] < 0:
                theta2 = -theta2;
                #OCTANT 6
            elif hi[0] < 0 and hi[1] >= 0 and hi[2] < 0:
                theta2 = -theta2;
                #OCTANT 7
            elif hi[0] < 0 and hi[1] < 0 and hi[2] < 0:
                theta1 = -theta1;
                theta2 = -theta2;
                #OCTANT 8
            elif hi[0] >= 0 and hi[1] < 0 and hi[2] < 0:
                theta1 = -theta1;
                theta2 = -theta2;
            else:
                pass

            #====== Calculate a point at the end of surface normals =====#
            Px = np.zeros(len(v1))
            Py = np.zeros(len(v1))
            Pz = np.zeros(len(v1))
            for normlcv in range(len(v1)):
                rv1i =v1[normlcv,:] - world


                #Equation of line in direction of normal vector, staring at
                #vertex 1; Equivalently the position vector from the origin
                #to point P on a line passing through the normal vector and
                #vertex 1
                rpi = rv1i + normalStore[normlcv,:]

                Px[normlcv] = world[0] + rpi[0]
                Py[normlcv] = world[1] + rpi[1]
                Pz[normlcv] = world[2] + rpi[2]


            #======================= Perform a rotation ===========================#

            thetaH = deflec[defl,plcv]

            #Rotation Matrix 1
            L1 = [[math.cos(theta1*np.pi/180),math.sin(theta1*np.pi/180),0],
                   [-1*math.sin(theta1*np.pi/180),math.cos(theta1*np.pi/180),0],
                   [0,0,1]]
            L1 = np.array(L1)
            #Rotation Matrix 2
            L2 = [[math.cos(theta2*np.pi/180),0,math.sin(theta2*np.pi/180)],
                   [0,1,0],
                   [-1*math.sin(theta2*np.pi/180),0,math.cos(theta2*np.pi/180)]]
            L2 = np.array(L2)
            #Rotation Matrix 3
            L3 = [[1,0,0],
                  [0,math.cos(thetaH*np.pi/180),math.sin(thetaH*np.pi/180)],
                  [0,-1*math.sin(thetaH*np.pi/180),math.cos(thetaH*np.pi/180)]]
            L3 = np.array(L3)


            invL1 = np.linalg.inv(L1)
            invL2 = np.linalg.inv(L2)

            #Execute Rotation of vertex points

            newVertices = np.zeros((len(vertices),len(vertices[0])))

            for rowlcv in range(len(vertices)):
                for collcv in range(len(vertices[0])):
                    if collcv==0:
                        x=vertices[rowlcv,collcv]
                    elif collcv == 1:
                        y=vertices[rowlcv,collcv]
                    elif collcv == 2:
                        z=vertices[rowlcv,collcv]

                Xb = [[x],[y],[z]]
                Xb = np.array(Xb)
                ## Rotation
                newVertices[rowlcv,:] = np.transpose(invL1@invL2@L3@L2@L1@(Xb-delta) + delta)

            normal = np.zeros((len(v1),len(v1[0])))
            for rotnlcv in range(len(v1)):
                XbP = [[Px[rotnlcv]],[Py[rotnlcv]],[Pz[rotnlcv]]]
                XbP = np.array(XbP)
                Xbv1 = [[v1[rotnlcv,0]],[v1[rotnlcv,1]],[v1[rotnlcv,2]]]
                Xbv1 = np.array(Xbv1)
                ## Rotation
                newP = invL1@invL2@L3@L2@L1@(XbP-delta) + delta
                newv1= invL1@invL2@L3@L2@L1@(Xbv1-delta)+ delta

                newnorm = newP-newv1

                unitnormal = newnorm/np.linalg.norm(newnorm)
                normal[rotnlcv,:] = np.transpose(unitnormal)


       #=================== Create  STL files =======================#

            # Get name for whole body or part
            basedir = os.getcwd()
            inputdir = os.path.join(base_folder, "Inputs") #code shares base directory of rsm_run_script.py

            inputpath = os.path.join(inputdir, "Simulation.json")
            rf=open(inputpath)
            md=json.load(rf)
            RSMNAME = md['Object Name']

            #=================== Create new Component STL file =======================#
    #Can uncomment this section to get individual component rotation#
            ## Create file path
            # filename = RSMNAME
            # newfilename = filename+str(thetaH)+'.stl'
            # outdir = os.path.join(localpath,'Simulation/Mesh_Component_Rotation/Rotated_Files(Output)')
            # filedir = os.path.join(outdir, newfilename)

            # ## Create and write to file
            # outfile = open(filedir,'w')

            # title = ['solid',filename,'\n']
            # title = " ".join(title)
            # outfile.write(title)
            # normal = normal.tolist()
            # newVertices = newVertices.tolist()

            # vertcount = 0


            # for i in range(nfacet):
            #     normalstring = ' '.join(map(str,normal[i]))
            #     outfile.write('     facet normal %s\n'% normalstring)
            #     outfile.write('          outer loop\n')
            #     for clcv in range(3):
            #         newVertstring= ' '.join(map(str,newVertices[vertcount]))
            #         outfile.write('               vertex %s\n' % newVertstring)
            #         if clcv == 2 and i == len(normalStore):
            #             pass
            #         else:
            #             vertcount = vertcount + 1 #to write vertices in groups of 3
            #     outfile.write('          endloop \n')
            #     outfile.write('     endfacet \n')


            # outfile.write('endsolid')
            # outfile.close()
            # CWT(filedir,newfilename)

            #Save vertices for stl combination
            if plcv == 0:
                wholebodyvert = newVertices
                wholebodynormal = normal
                wholebodynfacet = nfacet
                wholenormalstore = normalStore
            else:
                wholebodyvert = np.vstack((wholebodyvert,newVertices))
                wholebodynormal = np.vstack((wholebodynormal,normal))
                wholebodynfacet = wholebodynfacet + nfacet
                wholenormalstore =  np.vstack((wholenormalstore,normalStore))


        #===================== Create new Whole Body STL file ========================#


        #Get new name of mesh
    #    for line in lines:   #search though each line
    #
    #        if "Mesh Filename" in line:
    #            #RSMNAME -- NAME OF OUTPUT RSM FILE
    #            filename=line[line.find("#")+1:].split()[0]
    #            RSMNAME = os.path.splitext(filename)[0]

        ## Create file path
        obj = RSMNAME
        for plcv in range(len(parentnames)):
            value = deflec[defl,plcv]
            RSMNAME = RSMNAME+'_'+str(value)
        RSMNAMEpath = os.path.join(base_folder,'tempfiles/Mesh_Files')
        if not os.path.isdir(RSMNAMEpath):
            os.mkdir(RSMNAMEpath)

        RSMNAMEpath = os.path.join(RSMNAMEpath,RSMNAME)

        Usablefile =  obj + str(defl)  +'.stl'
        #os.symlink(RSMNAMEpath, Usablefile)

        solidname = RSMNAME
        RSMNAME = RSMNAME+'.stl' #add .stl extension
        RSMoutdir = os.path.join(base_folder,'tempfiles/Mesh_Files')
        RSMfiledir = os.path.join(RSMoutdir, Usablefile)

        ## Create and write to file
        print("Writing rotated STL: %s" % RSMfiledir)
        RSMoutfile = open(RSMfiledir,'w')

        title = ['solid',solidname,'\n']
        title = " ".join(title)

        RSMoutfile.write(title)
        wholebodynormal = wholebodynormal.tolist()
        wholebodyvert = wholebodyvert.tolist()
        vertcount = 0

        for i in range(wholebodynfacet):
            normalstring = ' '.join(map(str,wholebodynormal[i]))
            RSMoutfile.write('     facet normal %s\n'% normalstring)
            RSMoutfile.write('          outer loop\n')
            for clcv in range(3):
                newVertstring= ' '.join(map(str,wholebodyvert[vertcount]))
                RSMoutfile.write('               vertex %s\n' % newVertstring)
                if clcv == 2 and i == len(wholenormalstore):
                    pass
                else:
                    vertcount = vertcount +1 #to write vertices in groups of 3
            RSMoutfile.write('          endloop \n')
            RSMoutfile.write('     endfacet \n')


        RSMoutfile.write('endsolid')

        RSMoutfile.close()
    # Need to CWT each body
        CWT(RSMfiledir,RSMNAME)

        del RSMNAME
        del wholebodynormal
        del wholebodyvert
        del wholebodynfacet
        del wholenormalstore

if __name__ == "__main__":

    rotate_stl('.')

