#!/usr/bin/env python3

"""
Code to calculate Projected area given n-dimensions
"""
import numpy as np
import os 
from scipy.interpolate import  NearestNDInterpolator



def area(base_folder):
    """
    Parameters
    ----------
    base_folder : TYPE
        Top level folder of the RSM Suite

    Returns
    -------
    Projected Area

    """
    #Read in information from area files 
    values=[]
    areafile = (base_folder+os.sep+"Outputs/Projected_Area/Area_Total.dat")
    fcount = open(areafile,'r')
    line_count =0
    for i in fcount:
        if i != "\n":
            line_count += 1
    fcount.close()
    
    f = open(areafile,'r')
    j=0
    for line in f: 
        
        data = line.split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
        if j == 0:
            columns = len(floats)
            values = np.zeros([line_count,columns])
        values[j,:] = np.array(floats)
        j+= 1

    f.close()
     

    points = values[:,1:]
    areavalues = np.array(values[:,0])
   
    
    # Read Inputs from csv
    csvfile = (base_folder+os.sep+"Inputs/Model_Evaluation_Inputs/Model_Input_Data.csv")
    inp = np.loadtxt(csvfile,delimiter=',',skiprows=1)
    

    yawreq = np.reshape(inp[:,3],(len(inp),1))
    pitchreq = np.reshape(inp[:,4],(len(inp),1))
    rotreq = inp[:,12:]
    request = np.hstack((yawreq,pitchreq,rotreq))
    
    #Create model and evaluate 
    interp=NearestNDInterpolator(points, areavalues)
    
    project_area = interp(request)
    print(project_area)
                       




    
    
 
    
if __name__ == "__main__":
    
    area('.')
    
    