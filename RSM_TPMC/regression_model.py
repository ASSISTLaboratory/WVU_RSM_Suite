#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression model for Satellite Drag Coefficents 

"""
import numpy as np
import os 
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, ExpSineSquared as Exp, DotProduct as Lin
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.gaussian_process.kernels import Matern 
import json 


def regression_model(base_folder):
    #os.chdir('..')
    inputpath = (base_folder+os.sep+"Inputs/Simulation.json")
    rf=open(inputpath)
    md=json.load(rf)
    RSMNAME = md['Object Name']
    GSI_MODEL = md['Gas Surface Interaction Model (1=DRIA,2=CLL)']


    # if GSI_MODEL ==1:
    #     Model_ones = np.ones(6)
    # elif GSI_MODEL ==2:
    #     Model_ones = np.ones(7)

    # Loading training data
    file_train_H  =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_H_training_set.dat" )
    file_train_He =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_He_training_set.dat" )
    file_train_N  =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_N_training_set.dat" )
    file_train_N2 =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_N2_training_set.dat" )
    file_train_O  =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_O_training_set.dat" )
    file_train_O2 =     np.loadtxt("./Outputs/RSM_Regression_Models/data/Training Set/" + RSMNAME + "_O2_training_set.dat" )

    # Loading test data
    file_test_H   = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_H_test_set.dat")
    file_test_He  = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_He_test_set.dat")
    file_test_N   = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_N_test_set.dat")
    file_test_N2  = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_N2_test_set.dat")
    file_test_O   = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_O_test_set.dat")
    file_test_O2  = np.loadtxt("./Outputs/RSM_Regression_Models/data/Test Set/" + RSMNAME +"_O2_test_set.dat")

    species = ['H','He','N','N2','O','O2']

    # checking if there is a column of all zeros 
    zerostd_H_col = np.where(~file_train_H.any(axis=0))[0]
    
    file_train_H  = np.delete(file_train_H, zerostd_H_col,axis=1)
    file_train_He = np.delete(file_train_He,zerostd_H_col,axis=1)
    file_train_N  = np.delete(file_train_N, zerostd_H_col,axis=1)
    file_train_N2 = np.delete(file_train_N2,zerostd_H_col,axis=1)
    file_train_O  = np.delete(file_train_O, zerostd_H_col,axis=1)
    file_train_O2 = np.delete(file_train_O2,zerostd_H_col,axis=1)
    
    
    file_test_H  = np.delete(file_test_H, zerostd_H_col,axis=1)
    file_test_He = np.delete(file_test_He,zerostd_H_col,axis=1)
    file_test_N  = np.delete(file_test_N, zerostd_H_col,axis=1)
    file_test_N2 = np.delete(file_test_N2,zerostd_H_col,axis=1)
    file_test_O  = np.delete(file_test_O, zerostd_H_col,axis=1)
    file_test_O2 = np.delete(file_test_O2,zerostd_H_col,axis=1)
    
    
    
    
    Model_ones = np.ones(len(file_train_H[0])-1)
    
 



    # Training x vector
    x_train_H = file_train_H[:,:-1]
    x_train_He = file_train_He[:,:-1]
    x_train_N = file_train_N[:,:-1]
    x_train_N2 = file_train_N2[:,:-1]
    x_train_O = file_train_O[:,:-1]
    x_train_O2 = file_train_O2[:,:-1]  #everything before Cd 

    # Mean and STD for x Training Data
    x_train_H_mean = np.mean(x_train_H, axis=0)
    x_train_H_std = np.std(x_train_H, axis=0)
    x_train_H = (x_train_H-x_train_H_mean)/x_train_H_std #normalize data

        
    x_train_He_mean = np.mean(x_train_He, axis=0)
    x_train_He_std = np.std(x_train_He, axis=0)
    x_train_He = (x_train_He-x_train_He_mean)/x_train_He_std #normalize data


    x_train_N_mean = np.mean(x_train_N, axis=0)
    x_train_N_std = np.std(x_train_N, axis=0)
    x_train_N = (x_train_N-x_train_N_mean)/x_train_N_std #normalize data


    x_train_N2_mean = np.mean(x_train_N2, axis=0)
    x_train_N2_std = np.std(x_train_N2, axis=0)
    x_train_N2 = (x_train_N2-x_train_N2_mean)/x_train_N2_std #normalize data


    x_train_O_mean = np.mean(x_train_O, axis=0)
    x_train_O_std = np.std(x_train_O, axis=0)
    x_train_O = (x_train_O-x_train_O_mean)/x_train_O_std #normalize data


    x_train_O2_mean = np.mean(x_train_O2, axis=0)
    x_train_O2_std = np.std(x_train_O2, axis=0)
    x_train_O2 = (x_train_O2-x_train_O2_mean)/x_train_O2_std #normalize data
     
    ## Save std and mean models 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_H_mean.pkl', 'wb') as f_Hm:
        pickle.dump(x_train_H_mean, f_Hm)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_H_std.pkl', 'wb') as f_Hs:
        pickle.dump(x_train_H_std, f_Hs)  
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_He_mean.pkl', 'wb') as f_Hem:
        pickle.dump(x_train_He_mean, f_Hem)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_He_std.pkl', 'wb') as f_Hes:
        pickle.dump(x_train_He_std, f_Hes) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_N_mean.pkl', 'wb') as f_Nm:
        pickle.dump(x_train_N_mean, f_Nm)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_N_std.pkl', 'wb') as f_Ns:
        pickle.dump(x_train_N_std, f_Ns) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_N2_mean.pkl', 'wb') as f_N2m:
        pickle.dump(x_train_N2_mean, f_N2m)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_N2_std.pkl', 'wb') as f_N2s:
        pickle.dump(x_train_N2_std, f_N2s)     
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_O_mean.pkl', 'wb') as f_Om:
        pickle.dump(x_train_O_mean, f_Om)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_O_std.pkl', 'wb') as f_Os:
        pickle.dump(x_train_O_std, f_Os)     
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_O2_mean.pkl', 'wb') as f_O2m:
        pickle.dump(x_train_O2_mean, f_O2m)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_x_train_O2_std.pkl', 'wb') as f_O2s:
        pickle.dump(x_train_O2_std, f_O2s)
        
    # Training y vector
    y_train_H = file_train_H[:,-1:]
    y_train_He = file_train_He[:,-1:]
    y_train_N = file_train_N[:,-1:]
    y_train_N2 = file_train_N2[:,-1:]
    y_train_O = file_train_O[:,-1:]
    y_train_O2 = file_train_O2[:,-1:]

    ## Mean and STD for y Train Data

    #Hydrogen
    y_train_H_mean = np.mean(y_train_H, axis=0)
    y_train_H_std = np.std(y_train_H, axis=0)
    y_train_H = (y_train_H-y_train_H_mean)/y_train_H_std
     

    y_train_He_mean = np.mean(y_train_He, axis=0)
    y_train_He_std = np.std(y_train_He, axis=0)
    y_train_He = (y_train_He-y_train_He_mean)/y_train_He_std


    y_train_N_mean = np.mean(y_train_N, axis=0)
    y_train_N_std = np.std(y_train_N, axis=0)
    y_train_N = (y_train_N-y_train_N_mean)/y_train_N_std


    y_train_N2_mean = np.mean(y_train_N2, axis=0)
    y_train_N2_std = np.std(y_train_N2, axis=0)
    y_train_N2 = (y_train_N2-y_train_N2_mean)/y_train_N2_std


    y_train_O_mean = np.mean(y_train_O, axis=0)
    y_train_O_std = np.std(y_train_O, axis=0)
    y_train_O = (y_train_O-y_train_O_mean)/y_train_O_std


    y_train_O2_mean = np.mean(y_train_O2, axis=0)
    y_train_O2_std = np.std(y_train_O2, axis=0)
    y_train_O2 = (y_train_O2-y_train_O2_mean)/y_train_O2_std



    ## Save std and mean models 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_H_mean.pkl', 'wb') as f_Hm1:
        pickle.dump(y_train_H_mean, f_Hm1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_H_std.pkl', 'wb') as f_Hs1:
        pickle.dump(y_train_H_std, f_Hs1)  
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_He_mean.pkl', 'wb') as f_Hem1:
        pickle.dump(y_train_He_mean, f_Hem1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_He_std.pkl', 'wb') as f_Hes1:
        pickle.dump(y_train_He_std, f_Hes1) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_N_mean.pkl', 'wb') as f_Nm1:
        pickle.dump(y_train_N_mean, f_Nm1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_N_std.pkl', 'wb') as f_Ns1:
        pickle.dump(y_train_N_std, f_Ns1) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_N2_mean.pkl', 'wb') as f_N2m1:
        pickle.dump(y_train_N2_mean, f_N2m1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_N2_std.pkl', 'wb') as f_N2s1:
        pickle.dump(y_train_N2_std, f_N2s1) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_O_mean.pkl', 'wb') as f_Om1:
        pickle.dump(y_train_O_mean, f_Om1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_O_std.pkl', 'wb') as f_Os1:
        pickle.dump(y_train_O_std, f_Os1) 
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_O2_mean.pkl', 'wb') as f_O2m1:
        pickle.dump(y_train_O2_mean, f_O2m1)
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME + '_y_train_O2_std.pkl', 'wb') as f_O2s1:
        pickle.dump(y_train_O2_std, f_O2s1) 





    # Test x vector
    x_test_H = file_test_H[:,:-1]
    x_test_He = file_test_He[:,:-1]
    x_test_N = file_test_N[:,:-1]
    x_test_N2 = file_test_N2[:,:-1]
    x_test_O = file_test_O[:,:-1]
    x_test_O2 = file_test_O2[:,:-1]

    x_test_H = (x_test_H-x_train_H_mean)/x_train_H_std #using traing data 
    x_test_He= (x_test_He-x_train_He_mean)/x_train_He_std
    x_test_N= (x_test_N-x_train_N_mean)/x_train_N_std
    x_test_N2= (x_test_N2-x_train_N2_mean)/x_train_N2_std
    x_test_O= (x_test_O-x_train_O_mean)/x_train_O_std
    x_test_O2= (x_test_O2-x_train_O2_mean)/x_train_O2_std


    # Test y vector
    y_test_H = file_test_H[:,-1:]
    y_test_He = file_test_He[:,-1:]
    y_test_N = file_test_N[:,-1:]
    y_test_N2 = file_test_N2[:,-1:]
    y_test_O = file_test_O[:,-1:]
    y_test_O2 = file_test_O2[:,-1:]

    y_test_H = (y_test_H-y_train_H_mean)/y_train_H_std    
    y_test_He= (y_test_He-y_train_He_mean)/y_train_He_std
    y_test_N= (y_test_N-y_train_N_mean)/y_train_N_std
    y_test_N2= (y_test_N2-y_train_N2_mean)/y_train_N2_std
    y_test_O= (y_test_O-y_train_O_mean)/y_train_O_std
    y_test_O2= (y_test_O2-y_train_O2_mean)/y_train_O2_std
    

    print('Beginning Species Model Creation')
    print('Creating Plots of Predictions') 
    ########################################################################    
    ############################   Hydrogen   ##############################
    ########################################################################

    np.random.seed(1011)
    # Instantiate a Gaussian Process kernel (Matern Kernel 2)
    lscale=Model_ones
    kernel_H_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_H_matern2 = GaussianProcessRegressor(kernel=kernel_H_matern2, n_restarts_optimizer=9)


    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp_H_matern2.fit(x_train_H, y_train_H)

    # Make the prediction on the test data
    y_pred_H_matern2, sigma_H_matern2 = gp_H_matern2.predict(x_test_H, return_std=True)
    y_pred_H_matern2 = y_pred_H_matern2*y_train_H_std + y_train_H_mean #converting back the mean 
    sigma_H_matern2 = sigma_H_matern2*y_train_H_std #converting back the std
    y_test_H = y_test_H*y_train_H_std + y_train_H_mean # converts test data to non normalized version 

    # obtain the rmse for the test data
    #rmse_H_matern2 = mean_squared_error(y_pred_H_matern2, y_test_H, squared=False) #rms between pred and test 
    rmspe_sum = 0
    for j in range(0,len(y_pred_H_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_H_matern2[j]-y_test_H[j])/y_test_H[j])*100)**2
    rmspe_H_matern2 = np.sqrt(rmspe_sum/len(y_pred_H_matern2)) #give rms percentage error

    # Plot the observation, prediction, and the 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_H)), int(len(x_test_H)))).T
    plt.figure()
    plt.plot(x, y_test_H[0:int(len(x_test_H))], 'r.', label='Observations')
    plt.plot(x, y_pred_H_matern2[0:int(len(x_test_H))], 'g.', label='Predictions')
    plt.plot(x, y_pred_H_matern2[0:int(len(x_test_H))]+3*sigma_H_matern2[0:int(len(x_test_H))].reshape(int(len(x_test_H)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_H_matern2[0:int(len(x_test_H))]-3*sigma_H_matern2[0:int(len(x_test_H))].reshape(int(len(x_test_H)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for H using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_datasets_H_Matern2.png", dpi=1200)
         
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_H_matern2.pkl', 'wb') as file_GP_reg_GRACEH_matern2:
        pickle.dump(gp_H_matern2, file_GP_reg_GRACEH_matern2)    
        
    plt.figure()
    plt.plot(y_test_H,y_pred_H_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data H')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, H using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_H_Matern2.png", dpi=120)
     
    plt.figure()
    residual_H = y_test_H - y_pred_H_matern2
    plt.hist(residual_H)
    plt.title('Residual of H GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_H_Matern2.png", dpi=120)
        
        
    ########################################################################    
    #############################   Helium   ###############################
    ########################################################################   
        
    np.random.seed(1011)   
        
    lscale=Model_ones
    kernel_He_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_He_matern2 = GaussianProcessRegressor(kernel=kernel_He_matern2, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp_He_matern2.fit(x_train_He, y_train_He)
    # Make the prediction on the test data
    y_pred_He_matern2, sigma_He_matern2 = gp_He_matern2.predict(x_test_He, return_std=True)
    y_pred_He_matern2 = y_pred_He_matern2*y_train_He_std + y_train_He_mean 
    sigma_He_matern2 = sigma_He_matern2*y_train_He_std 
    y_test_He = y_test_He*y_train_He_std + y_train_He_mean 
    # obtain the rmse for the test data
    #rmse_He_matern2 = mean_squared_error(y_pred_He_matern2, y_test_He, squared=False)
    rmspe_sum = 0
    for j in range(0,len(y_pred_He_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_He_matern2[j]-y_test_He[j])/y_test_He[j])*100)**2
    rmspe_He_matern2 = np.sqrt(rmspe_sum/len(y_pred_He_matern2))
    # Plot the observation, prediction, and the 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_He)), int(len(x_test_He)))).T
    plt.figure()
    plt.plot(x, y_test_He[0:int(len(x_test_He))], 'r.', label='Observations')
    plt.plot(x, y_pred_He_matern2[0:int(len(x_test_He))], 'g.', label='Predictions')
    plt.plot(x, y_pred_He_matern2[0:int(len(x_test_He))]+3*sigma_He_matern2[0:int(len(x_test_He))].reshape(int(len(x_test_He)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_He_matern2[0:int(len(x_test_He))]-3*sigma_He_matern2[0:int(len(x_test_He))].reshape(int(len(x_test_He)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for He using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+"Outputs/RSM_Regression_Models/Plots_Output/"+RSMNAME+"_datasets_He_Matern2.png", dpi=120)

    plt.figure()
    plt.plot(y_test_He,y_pred_He_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data He')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, He using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_He_Matern2.png", dpi=120)
     
    plt.figure()
    residual_He = y_test_He - y_pred_He_matern2
    plt.hist(residual_He)
    plt.title('Residual of He GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_He_Matern2.png", dpi=120)
    
          
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_He_matern2.pkl', 'wb') as file_GP_reg_GRACEHe_matern2:
        pickle.dump(gp_He_matern2, file_GP_reg_GRACEHe_matern2)    

    ########################################################################    
    ############################   Nitrogen   ##############################
    ########################################################################  

     
    np.random.seed(1011)   
        
    lscale=Model_ones
    kernel_N_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_N_matern2 = GaussianProcessRegressor(kernel=kernel_N_matern2, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of tN parameters
    gp_N_matern2.fit(x_train_N, y_train_N)
    # Make tN prediction on tN test data
    y_pred_N_matern2, sigma_N_matern2 = gp_N_matern2.predict(x_test_N, return_std=True)
    y_pred_N_matern2 = y_pred_N_matern2*y_train_N_std + y_train_N_mean 
    sigma_N_matern2 = sigma_N_matern2*y_train_N_std 
    y_test_N = y_test_N*y_train_N_std + y_train_N_mean 
    # obtain tN rmse for tN test data
    #rmse_N_matern2 = mean_squared_error(y_pred_N_matern2, y_test_N, squared=False)
    rmspe_sum = 0
    for j in range(0,len(y_pred_N_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_N_matern2[j]-y_test_N[j])/y_test_N[j])*100)**2
    rmspe_N_matern2 = np.sqrt(rmspe_sum/len(y_pred_N_matern2))
     # Plot tN observation, prediction, and tN 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_N)), int(len(x_test_N)))).T
    plt.figure()
    plt.plot(x, y_test_N[0:int(len(x_test_N))], 'r.', label='Observations')
    plt.plot(x, y_pred_N_matern2[0:int(len(x_test_N))], 'g.', label='Predictions')
    plt.plot(x, y_pred_N_matern2[0:int(len(x_test_N))]+3*sigma_N_matern2[0:int(len(x_test_N))].reshape(int(len(x_test_N)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_N_matern2[0:int(len(x_test_N))]-3*sigma_N_matern2[0:int(len(x_test_N))].reshape(int(len(x_test_N)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for N using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+"Outputs/RSM_Regression_Models/Plots_Output/"+RSMNAME+"_datasets_N_Matern2.png", dpi=120)

    plt.figure()
    plt.plot(y_test_N,y_pred_N_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data N')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, N using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_N_Matern2.png", dpi=120)
     
    plt.figure()
    residual_N = y_test_N - y_pred_N_matern2
    plt.hist(residual_N)
    plt.title('Residual of N GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_N_Matern2.png", dpi=120)
     
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_N_matern2.pkl', 'wb') as file_GP_reg_GRACEN_matern2:
        pickle.dump(gp_N_matern2, file_GP_reg_GRACEN_matern2)       
          
    ########################################################################    
    ##########################   Nitrogen  - 2  ############################
    ########################################################################     
        
    np.random.seed(1011)   
        
    lscale=Model_ones
    kernel_N2_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_N2_matern2 = GaussianProcessRegressor(kernel=kernel_N2_matern2, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of tN2 parameters
    gp_N2_matern2.fit(x_train_N2, y_train_N2)
    # Make tN2 prediction on tN2 test data
    y_pred_N2_matern2, sigma_N2_matern2 = gp_N2_matern2.predict(x_test_N2, return_std=True)
    y_pred_N2_matern2 = y_pred_N2_matern2*y_train_N2_std + y_train_N2_mean 
    sigma_N2_matern2 = sigma_N2_matern2*y_train_N2_std 
    y_test_N2 = y_test_N2*y_train_N2_std + y_train_N2_mean 
    # obtain tN2 rmse for tN2 test data
    #rmse_N2_matern2 = mean_squared_error(y_pred_N2_matern2, y_test_N2, squared=False)
    rmspe_sum = 0
    for j in range(0,len(y_pred_N2_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_N2_matern2[j]-y_test_N2[j])/y_test_N2[j])*100)**2
    rmspe_N2_matern2 = np.sqrt(rmspe_sum/len(y_pred_N2_matern2))
     # Plot tN2 observation, prediction, and tN2 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_N2)), int(len(x_test_N2)))).T
    plt.figure()
    plt.plot(x, y_test_N2[0:int(len(x_test_N2))], 'r.', label='Observations')
    plt.plot(x, y_pred_N2_matern2[0:int(len(x_test_N2))], 'g.', label='Predictions')
    plt.plot(x, y_pred_N2_matern2[0:int(len(x_test_N2))]+3*sigma_N2_matern2[0:int(len(x_test_N2))].reshape(int(len(x_test_N2)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_N2_matern2[0:int(len(x_test_N2))]-3*sigma_N2_matern2[0:int(len(x_test_N2))].reshape(int(len(x_test_N2)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for N2 using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+"Outputs/RSM_Regression_Models/Plots_Output/"+RSMNAME+"_datasets_N2_Matern2.png", dpi=120)

    plt.figure()
    plt.plot(y_test_N2,y_pred_N2_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data N')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, N2 using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_N2_Matern2.png", dpi=120)
     
    plt.figure()
    residual_N2 = y_test_N2 - y_pred_N2_matern2
    plt.hist(residual_N2)
    plt.title('Residual of N2 GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_N2_Matern2.png", dpi=120)
    

    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_N2_matern2.pkl', 'wb') as file_GP_reg_GRACEN2_matern2:
        pickle.dump(gp_N2_matern2, file_GP_reg_GRACEN2_matern2) 
        
    ########################################################################    
    #############################   Oxygen   ###############################
    ########################################################################      
        
    np.random.seed(1011)   
        
    lscale=Model_ones
    kernel_O_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_O_matern2 = GaussianProcessRegressor(kernel=kernel_O_matern2, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of tO parameters
    gp_O_matern2.fit(x_train_O, y_train_O)
    # Make tO prediction on tO test data
    y_pred_O_matern2, sigma_O_matern2 = gp_O_matern2.predict(x_test_O, return_std=True)
    y_pred_O_matern2 = y_pred_O_matern2*y_train_O_std + y_train_O_mean 
    sigma_O_matern2 = sigma_O_matern2*y_train_O_std 
    y_test_O = y_test_O*y_train_O_std + y_train_O_mean 
    # obtain tO rmse for tO test data
    #rmse_O_matern2 = mean_squared_error(y_pred_O_matern2, y_test_O, squared=False)
    rmspe_sum = 0
    for j in range(0,len(y_pred_O_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_O_matern2[j]-y_test_O[j])/y_test_O[j])*100)**2
    rmspe_O_matern2 = np.sqrt(rmspe_sum/len(y_pred_O_matern2))
    # Plot tO observation, prediction, and tO 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_O)), int(len(x_test_O)))).T
    plt.figure()
    plt.plot(x, y_test_O[0:int(len(x_test_O))], 'r.', label='Observations')
    plt.plot(x, y_pred_O_matern2[0:int(len(x_test_O))], 'g.', label='Predictions')
    plt.plot(x, y_pred_O_matern2[0:int(len(x_test_O))]+3*sigma_O_matern2[0:int(len(x_test_O))].reshape(int(len(x_test_O)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_O_matern2[0:int(len(x_test_O))]-3*sigma_O_matern2[0:int(len(x_test_O))].reshape(int(len(x_test_O)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for O using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+"Outputs/RSM_Regression_Models/Plots_Output/"+RSMNAME+"_datasets_O_Matern2.png", dpi=120)

    plt.figure()
    plt.plot(y_test_O,y_pred_O_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data O')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, O using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_O_Matern2.png", dpi=120)
     
    plt.figure()
    residual_O = y_test_O - y_pred_O_matern2
    plt.hist(residual_O)
    plt.title('Residual of O GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_O_Matern2.png", dpi=120)
    
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_O_matern2.pkl', 'wb') as file_GP_reg_GRACEO_matern2:
        pickle.dump(gp_O_matern2, file_GP_reg_GRACEO_matern2)        
        
        
    ########################################################################    
    ###########################   Oxygen  - 2  #############################
    ######################################################################## 
        
    np.random.seed(1011)   
        
    lscale=Model_ones
    kernel_O2_matern2= Matern(length_scale=lscale, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gp_O2_matern2 = GaussianProcessRegressor(kernel=kernel_O2_matern2, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of tO2 parameters
    gp_O2_matern2.fit(x_train_O2, y_train_O2)
    # Make tO2 prediction on tO2 test data
    y_pred_O2_matern2, sigma_O2_matern2 = gp_O2_matern2.predict(x_test_O2, return_std=True)
    y_pred_O2_matern2 = y_pred_O2_matern2*y_train_O2_std + y_train_O2_mean 
    sigma_O2_matern2 = sigma_O2_matern2*y_train_O2_std 
    y_test_O2 = y_test_O2*y_train_O2_std + y_train_O2_mean 
    # obtain tO2 rmse for tO2 test data
    #rmse_O2_matern2 = mean_squared_error(y_pred_O2_matern2, y_test_O2, squared=False)
    rmspe_sum = 0
    for j in range(0,len(y_pred_O2_matern2)):
        rmspe_sum = rmspe_sum + (((y_pred_O2_matern2[j]-y_test_O2[j])/y_test_O2[j])*100)**2
    rmspe_O2_matern2 = np.sqrt(rmspe_sum/len(y_pred_O2_matern2))
     # Plot tO2 observation, prediction, and tO2 3-sigma error bounds
    x = np.atleast_2d(np.linspace(1, int(len(x_test_O2)), int(len(x_test_O2)))).T
    plt.figure()
    plt.plot(x, y_test_O2[0:int(len(x_test_O2))], 'r.', label='Observations')
    plt.plot(x, y_pred_O2_matern2[0:int(len(x_test_O2))], 'g.', label='Predictions')
    plt.plot(x, y_pred_O2_matern2[0:int(len(x_test_O2))]+3*sigma_O2_matern2[0:int(len(x_test_O2))].reshape(int(len(x_test_O2)),1), 'b-', label='3 Sigma Bound')
    plt.plot(x, y_pred_O2_matern2[0:int(len(x_test_O2))]-3*sigma_O2_matern2[0:int(len(x_test_O2))].reshape(int(len(x_test_O2)),1), 'b-')
    plt.legend(loc='upper left')
    plt.title(' $C_d$ for '+ RSMNAME +'model for O2 using Matern kernel (nu=2.5)')
    plt.xlabel('Test Data Number')
    plt.ylabel('$C_d$')
    plt.grid()
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_datasets_O2_Matern2.png", dpi=120)

    plt.figure()
    plt.plot(y_test_O2,y_pred_O2_matern2,'bo',markerfacecolor='none')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.grid()
    plt.xlabel('Test Data O2')
    plt.ylabel('Predicted Data')
    plt.title('GPR Results for '+ RSMNAME +' model, O2 using Matern kernel (nu=2.5)')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_GPRresults_O2_Matern2.png", dpi=120)
     
    plt.figure()
    residual_O2 = y_test_O2 - y_pred_O2_matern2
    plt.hist(residual_O2)
    plt.title('Residual of O2 GPR Test Set Comparison' )
    plt.xlabel('Test Set Residuals')
    plt.savefig(base_folder+os.sep+'Outputs/RSM_Regression_Models/Plots_Output/'+RSMNAME+"_residuals_O2_Matern2.png", dpi=120)
    
    with open(base_folder+os.sep+'Outputs/RSM_Regression_Models/'+RSMNAME+'_GP_reg_O2_matern2.pkl', 'wb') as file_GP_reg_GRACEO2_matern2:
        pickle.dump(gp_O2_matern2, file_GP_reg_GRACEO2_matern2)  


if __name__ == "__main__":

    regression_model('.')

