#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Logan Sheridan, Smriti Paul 
"""


import numpy as np
import json 
import os 
import pandas as pd
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, ExpSineSquared as Exp, DotProduct as Lin
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern 
from projected_area import area

def read_input_info(basedir):
    #os.chdir('..')
    topdir = os.getcwd()
    inputdir = os.path.join(topdir,'Inputs')
    print('Reading Model Input Information \n')
    jsonpath = os.path.join(inputdir, "Model_Evaluation.json")
    rf=open(jsonpath)
    md=json.load(rf)
    
    MODEL_NAME = md['Model Name']
    GSI_MODEL = md['Gas Surface Interaction Model (1=DRIA,2=CLL)']  
    Rotation = md['Component Rotation (1=No,2=Yes)']  
    Sat_Mass = md['Satellite Mass (kg)']
    surf_part_mass = md['Surface Particle Mass (kg)']
    ads_model = md['Adsorption Model (1=Freundlich,2=Langmuir)']
    
    #os.chdir(basedir)
    
    return MODEL_NAME,GSI_MODEL,ads_model, Rotation, Sat_Mass, surf_part_mass



def read_input_data(basedir):
    #os.chdir('..')
    topdir = os.getcwd()
    datadir = os.path.join(topdir,'Inputs/Model_Evaluation_Inputs') 
    print('Reading Input Data \n')
    datapath = os.path.join(datadir, 'Model_Input_Data.csv')
    file_input = pd.read_csv(datapath,header = 0)
    file_input = file_input.to_numpy()
    
    datalength = np.size(file_input,0)
    
    
    
    mole_frac = np.zeros([datalength,6]) 
    
    
    
    INPUTS = np.stack((file_input[:,0],file_input[:,1],file_input[:,2],file_input[:,3],file_input[:,4],file_input[:,5]),axis=1)   
                                                                            #alpha             #sigmat                                                                         
    # INPUT = np.stack((file_input[:,1],file_input[:,2],file_input[:,3],0*file_input[:,6]+1,0*file_input[:,6]+1,file_input[:,7],file_input[:,8]),axis=1)        
    mole_frac[:,0] =  file_input[:,6]    
    mole_frac[:,1] =  file_input[:,7]
    mole_frac[:,2] =  file_input[:,8]
    mole_frac[:,3] =  file_input[:,9]
    mole_frac[:,4] =  file_input[:,10]
    mole_frac[:,5] =  file_input[:,11]
    
    #os.chdir(basedir)
   
    return INPUTS, mole_frac, datalength

def CdSetup(MODEL_NAME,GSI_MODEL,ads_model,surf_part_mass,Inputs,mole_frac,datalength):
    NSPECIES = 6
    sigmat = 1.0
    alpha_ads = 1.0
    alphan_ads = 1.0
    Po = np.zeros([datalength,1])
    n= np.zeros([datalength,1])
    kB = 1.3806488e-23; # Boltzmann's constant [J/K]
    kL_CLL = 2.89e6 #CLL - Define Langmuir Adsorpate Constant [unitless]
    kL_DRIA = 1.44e6
    
    aF_CLL = 0.089 #CLL - Define Freundlich Alpha Constant [unitless]
    kF_CLL = 2.35       #CLL - Define Freundlich K constant [unitless] 

    
    mass = np.zeros([NSPECIES,1])
    mass[0] = 1.674e-27;  # hydrogen 
    mass[1] = 6.646e-27;  # helium 
    mass[2] = 2.326e-26;  # atomic nitrogen 
    mass[3] = 4.652e-26;  # diatomic nitrogen 
    mass[4] = 2.657e-26;  # atomic oxygen 
    mass[5] = 5.314e-26;  # diatomic oxygen 

    n[:,0] = Inputs[:,5] # number density 
    
    m_avg = np.zeros([datalength,1])

    m_avg[:,0] = mass[0] * mole_frac[:,0]+mass[1] * mole_frac[:,1]+ mass[2] * mole_frac[:,2]+mass[3] * mole_frac[:,3]+mass[4] * mole_frac[:,4]+mass[5] * mole_frac[:,5]
    
    
    mu = m_avg/surf_part_mass
    alpha_sub = 3.0*mu/((1+mu)**2)
    
    alphan_sub = 2.0*alpha_sub-1.0
    
    #alphan_sub cannont be less than 0 
    alphan_sub[alphan_sub<0] = 0

    
    
    if GSI_MODEL == 1: #DRIA
        ads_inputDRIA =np.zeros([datalength,6])
        srf_inputDRIA =np.zeros([datalength,6])
        if np.all(Inputs[0] < 100):
            print('Make sure Velocity is in m/s')
            
        ads_inputDRIA[:,0] = srf_inputDRIA[:,0] = Inputs[:,0] #Velocity (m/s)
        ads_inputDRIA[:,1] = srf_inputDRIA[:,1] = Inputs[:,1] #Surface Temp (K)
        ads_inputDRIA[:,2] = srf_inputDRIA[:,2] = Inputs[:,2] #Atm Temp (K)
        ads_inputDRIA[:,4] = srf_inputDRIA[:,4] = Inputs[:,3] #Yaw (rad)
        ads_inputDRIA[:,5] = srf_inputDRIA[:,5] = Inputs[:,4] #Pitch (rad)

        #Specific inputs for adsorbate and clean surfaces for accommodation coefficients 
        ads_inputDRIA[:,3] = alpha_ads #always one
        srf_inputDRIA[:,3] = alpha_sub[:,0]
        
        #Emulator 
        print('Beginning Emulator')
        Cd_ads,uncertainty_ads = emu(ads_inputDRIA,MODEL_NAME,mole_frac,datalength,m_avg)
        Cd_srf,uncertainty_srf = emu(srf_inputDRIA,MODEL_NAME,mole_frac,datalength,m_avg)
    
    elif GSI_MODEL == 2:
        ads_inputCLL = np.zeros([datalength,7])
        srf_inputCLL = np.zeros([datalength,7])
        if np.all(Inputs[0] < 100):
            print('Make sure Velocity is in m/s')
        
        ads_inputCLL[:,0] = srf_inputCLL[:,0] = Inputs[:,0] #Velocity (m/s)
        ads_inputCLL[:,1] = srf_inputCLL[:,1] = Inputs[:,1] #Surface Temp (K)
        ads_inputCLL[:,2] = srf_inputCLL[:,2] = Inputs[:,2] #Atm Temp (K)
        ads_inputCLL[:,4] = srf_inputCLL[:,4] = sigmat
        ads_inputCLL[:,5] = srf_inputCLL[:,5] = Inputs[:,3] #Yaw (rad)
        ads_inputCLL[:,6] = srf_inputCLL[:,6] = Inputs[:,4] #Pitch (rad)
        
        ads_inputCLL[:,3] = alphan_ads #always one
        srf_inputCLL[:,3] = alphan_sub[:,0]
        
        #Emulator
        print('Beginning Emulator')
        Cd_ads,uncertainty_ads = emu(ads_inputCLL,mole_frac,datalength,m_avg)
        Cd_srf,uncertainty_srf = emu(srf_inputCLL,mole_frac,datalength,m_avg)
        
    else:
        print('Input valid GSI Flag in Model_Evaluation.json')
    
    print('Calculating Cd with ADS model')    
    #Compute partial pressure of oxygen 
    Po[:,0] = n[:,0]*mole_frac[:,4]*kB*Inputs[:,2]
    if ads_model == 1: #Freundlich
        fsc = kF_CLL*(Po**aF_CLL) 
        if fsc>1.0:
            fsc = 1.0
            
    elif ads_model == 2: #Langmuir
        if GSI_MODEL == 1:
            fsc = (kL_DRIA*Po)/(1+kL_DRIA*Po)
        elif GSI_MODEL == 2:
            fsc = (kL_CLL*Po)/(1+kL_CLL*Po)
    else:
        print('Enter valid ADS Model Flag')
    
    Cd_TOTAL = fsc[:,0]*Cd_ads[:,0] + (1.0-fsc[:,0])*Cd_srf[:,0]
    CD_var = np.multiply(fsc**2,uncertainty_ads**2) + np.multiply((1-fsc)**2,uncertainty_srf**2)     # assuming covariance between CD_ads and CD_surf to be zero
    Cd_STD = np.sqrt(CD_var)
    
    
    
    
    return Cd_TOTAL, Cd_STD



def emu(INPUT,MODEL_NAME,mole_frac,datalength,m_avg):
    basedir = os.getcwd()
    #os.chdir('..')
    topdir = os.getcwd() 
    regdir = os.path.join(topdir,'Inputs/Model_Evaluation_Inputs/Regression_Models')
    os.chdir(regdir)
    
    Cd = np.zeros([datalength,1])
    
    print('Evaluating H model')
    #### Hydrogen H
    # Loading up the trained GP model (Matern kernel) and the x,y normalization constants
    with open(MODEL_NAME +'_GP_reg_H_matern2.pkl', 'rb') as file:
        gp_H_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_H_mean.pkl', 'rb') as f_Hm:
        x_train_H_mean = pickle.load(f_Hm)
    with open(MODEL_NAME + '_x_train_H_std.pkl', 'rb') as f_Hs:
        x_train_H_std = pickle.load(f_Hs)    
    with open(MODEL_NAME + '_y_train_H_mean.pkl', 'rb') as f_Hm1:
        y_train_H_mean = pickle.load(f_Hm1)
    with open(MODEL_NAME + '_y_train_H_std.pkl', 'rb') as f_Hs1:
        y_train_H_std = pickle.load(f_Hs1)     
          
     
    
    # Test dataset of actual satellite for one day
    x_test_H = INPUT
    # Normalize
    x_test_H = (x_test_H-x_train_H_mean)/x_train_H_std #normalize input 
    
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_H_Matern, sigma_H_Matern = gp_H_Matern.predict(x_test_H, return_std=True) #directly predicting Cd 
    y_pred_H_Matern = y_pred_H_Matern*y_train_H_std + y_train_H_mean #make non normalized form 
    sigma_H_Matern = sigma_H_Matern*y_train_H_std 
    print('Evaluating He model')
    #### Helium He
    # Loading up the trained GP model (Matern kernel) and the x,y normalization constants
    with open(MODEL_NAME +'_GP_reg_He_matern2.pkl', 'rb') as file:
        gp_He_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_He_mean.pkl', 'rb') as f_Hem:
        x_train_He_mean = pickle.load(f_Hem)
    with open(MODEL_NAME + '_x_train_He_std.pkl', 'rb') as f_Hes:
        x_train_He_std = pickle.load(f_Hes)    
    with open(MODEL_NAME + '_y_train_He_mean.pkl', 'rb') as f_Hem1:
        y_train_He_mean = pickle.load(f_Hem1)
    with open(MODEL_NAME + '_y_train_He_std.pkl', 'rb') as f_Hes1:
        y_train_He_std = pickle.load(f_Hes1) 
        
    # Test dataset of actual satellite for one day
    x_test_He = INPUT
    # Normalize
    x_test_He = (x_test_He-x_train_He_mean)/x_train_He_std
            
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_He_Matern, sigma_He_Matern = gp_He_Matern.predict(x_test_He, return_std=True)
    y_pred_He_Matern = y_pred_He_Matern*y_train_He_std + y_train_He_mean 
    sigma_He_Matern = sigma_He_Matern*y_train_He_std 
    print('Evaluating N model')
    #### Atomic Nitrogen N
    with open(MODEL_NAME + '_GP_reg_N_matern2.pkl', 'rb') as file:
        gp_N_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_N_mean.pkl', 'rb') as f_Nm:
        x_train_N_mean = pickle.load(f_Nm)
    with open(MODEL_NAME + '_x_train_N_std.pkl', 'rb') as f_Ns:
        x_train_N_std = pickle.load(f_Ns)    
    with open(MODEL_NAME + '_y_train_N_mean.pkl', 'rb') as f_Nm1:
        y_train_N_mean = pickle.load(f_Nm1)
    with open(MODEL_NAME + '_y_train_N_std.pkl', 'rb') as f_Ns1:
        y_train_N_std = pickle.load(f_Ns1) 
    
    # Test dataset of actual satellite for one day
    x_test_N = INPUT
    # Normalize
    x_test_N = (x_test_N-x_train_N_mean)/x_train_N_std
            
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_N_Matern, sigma_N_Matern = gp_N_Matern.predict(x_test_N, return_std=True)
    y_pred_N_Matern = y_pred_N_Matern*y_train_N_std + y_train_N_mean 
    sigma_N_Matern = sigma_N_Matern*y_train_N_std 
    print('Evaluating N2 model')
    #### Molecular Nitrogen N2
    # Loading up the trained GP model (Matern kernel) and the x,y normalization constants
    with open(MODEL_NAME + '_GP_reg_N2_matern2.pkl', 'rb') as file:
        gp_N2_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_N2_mean.pkl', 'rb') as f_N2m:
        x_train_N2_mean = pickle.load(f_N2m)
    with open(MODEL_NAME + '_x_train_N2_std.pkl', 'rb') as f_N2s:
        x_train_N2_std = pickle.load(f_N2s)    
    with open(MODEL_NAME + '_y_train_N2_mean.pkl', 'rb') as f_N2m1:
        y_train_N2_mean = pickle.load(f_N2m1)
    with open(MODEL_NAME + '_y_train_N2_std.pkl', 'rb') as f_N2s1:
        y_train_N2_std = pickle.load(f_N2s1) 
        
    # Test dataset of actual satellite for one day
    x_test_N2 = INPUT
    # Normalize
    x_test_N2 = (x_test_N2-x_train_N2_mean)/x_train_N2_std
            
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_N2_Matern, sigma_N2_Matern = gp_N2_Matern.predict(x_test_N2, return_std=True)
    y_pred_N2_Matern = y_pred_N2_Matern*y_train_N2_std + y_train_N2_mean 
    sigma_N2_Matern = sigma_N2_Matern*y_train_N2_std 
    print('Evaluating O model')
    #### Atomic Oxygen O
    # Loading up the trained GP model (Matern kernel) and the x,y normalization constants
    with open(MODEL_NAME + '_GP_reg_O_matern2.pkl', 'rb') as file:
        gp_O_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_O_mean.pkl', 'rb') as f_Om:
        x_train_O_mean = pickle.load(f_Om)
    with open(MODEL_NAME + '_x_train_O_std.pkl', 'rb') as f_Os:
        x_train_O_std = pickle.load(f_Os)    
    with open(MODEL_NAME + '_y_train_O_mean.pkl', 'rb') as f_Om1:
        y_train_O_mean = pickle.load(f_Om1)
    with open(MODEL_NAME + '_y_train_O_std.pkl', 'rb') as f_Os1:
        y_train_O_std = pickle.load(f_Os1) 
        
    # Test dataset of actual satellite for one day
    x_test_O = INPUT
    # Normalize
    x_test_O = (x_test_O-x_train_O_mean)/x_train_O_std
            
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_O_Matern, sigma_O_Matern = gp_O_Matern.predict(x_test_O, return_std=True)
    y_pred_O_Matern = y_pred_O_Matern*y_train_O_std + y_train_O_mean 
    sigma_O_Matern = sigma_O_Matern*y_train_O_std 
    print('Evaluating O2 model \n')
    #### Molecular Oxygen O2
    # Loading up the trained GP model (Matern kernel) and the x,y normalization constants
    with open(MODEL_NAME + '_GP_reg_O2_matern2.pkl', 'rb') as file:
        gp_O2_Matern = pickle.load(file)
    with open(MODEL_NAME + '_x_train_O2_mean.pkl', 'rb') as f_O2m:
        x_train_O2_mean = pickle.load(f_O2m)
    with open(MODEL_NAME + '_x_train_O2_std.pkl', 'rb') as f_O2s:
        x_train_O2_std = pickle.load(f_O2s)    
    with open(MODEL_NAME + '_y_train_O2_mean.pkl', 'rb') as f_O2m1:
        y_train_O2_mean = pickle.load(f_O2m1)
    with open(MODEL_NAME + '_y_train_O2_std.pkl', 'rb') as f_O2s1:
        y_train_O2_std = pickle.load(f_O2s1) 
        
    # Test dataset of actual satellite for one day
    x_test_O2 = INPUT
    # Normalize
    x_test_O2 = (x_test_O2-x_train_O2_mean)/x_train_O2_std
            
    # Predict the mean and covariance using the GP model and remove the normalization
    y_pred_O2_Matern, sigma_O2_Matern = gp_O2_Matern.predict(x_test_O2, return_std=True)
    y_pred_O2_Matern = y_pred_O2_Matern*y_train_O2_std + y_train_O2_mean 
    sigma_O2_Matern = sigma_O2_Matern*y_train_O2_std 
    
    # Combined Cd    

    print('Calculating Cd from model')    
    # Sampling and obtaining mean/std for the net CD using the sampling
    CD_net_mean_sampling = np.zeros([datalength,1])
    CD_net_std_sampling = np.zeros([datalength,1])
    denominator = np.zeros([datalength,1])
    denominator[:,0] = m_avg[:,0]
    nsamples = 500
    for j in range(0, datalength): #loop uses sampling to get mean and std 
        np.random.seed(j*6)
        s_H = np.random.normal(y_pred_H_Matern[j], sigma_H_Matern[j], nsamples)
        np.random.seed(j*6+1)
        s_He = np.random.normal(y_pred_He_Matern[j], sigma_He_Matern[j], nsamples)     
        np.random.seed(j*6+2)
        s_N = np.random.normal(y_pred_N_Matern[j], sigma_N_Matern[j], nsamples)         
        np.random.seed(j*6+3)
        s_N2 = np.random.normal(y_pred_N2_Matern[j], sigma_N2_Matern[j], nsamples)         
        np.random.seed(j*6+4)
        s_O = np.random.normal(y_pred_O_Matern[j], sigma_O_Matern[j], nsamples)         
        np.random.seed(j*6+5)
        s_O2 = np.random.normal(y_pred_O2_Matern[j], sigma_O2_Matern[j], nsamples)    
        CD_samples = np.zeros(nsamples)
        for k in range(0, nsamples):
            CD_samples[k] = (s_H[k]*mole_frac[j,0]*1.674e-27 + 
            s_He[k]*mole_frac[j,1]*6.646e-27 +
            s_N[k]*mole_frac[j,2]*2.326e-26 +
            s_N2[k]*mole_frac[j,3]*2.326e-26*2 +
            s_O[k]*mole_frac[j,4]*2.657e-26 + 
            s_O2[k]*mole_frac[j,5]*2.657e-26*2
            )/denominator[j]
        CD_net_mean_sampling[j,0] = np.mean(CD_samples) 
        CD_net_std_sampling[j,0] = np.std(CD_samples)
    
        
    ycombined = np.array([y_pred_H_Matern,y_pred_He_Matern,y_pred_N_Matern,y_pred_N2_Matern,y_pred_O_Matern,y_pred_O2_Matern]).T    
    sigmacombined = np.array([sigma_H_Matern,sigma_He_Matern,sigma_N_Matern,sigma_N2_Matern,sigma_O_Matern,sigma_O2_Matern]).T    

    
    Cd = CD_net_mean_sampling
    uncertainty=CD_net_std_sampling
    os.chdir(basedir)


    return Cd, uncertainty





if __name__ == '__main__':

    basedir = os.getcwd()
   # os.chdir('..')
    topdir = os.getcwd()
    os.chdir(basedir)    
    # inputdir = os.path.join(basedir, "Inputs")
    
    #Get the basic information from .json file 
    MODEL_NAME,GSI_MODEL,ads_model, Rotation, Sat_Mass, surf_part_mass = read_input_info(basedir)
    
    #Get the actual input data i.e. Velocity, Temperature, etc. 
    Inputs, mole_fractions, datalength = read_input_data(basedir)
    
    #Calculate the Cd 
    print('Preparing Cd Calculation...')
    RSMCd, UNCERTAINTY = CdSetup(MODEL_NAME,GSI_MODEL,ads_model,surf_part_mass,Inputs,mole_fractions,datalength)
    RSMA = area(basedir)
    #Save Results
    np.savetxt('Outputs/Cd_Results/'+MODEL_NAME+'_Cd_results.txt', RSMCd)   
    np.savetxt('Outputs/Cd_Results/'+MODEL_NAME+'_Uncertainty_Bounds.txt', UNCERTAINTY)  
    #Save Area Results
    np.savetext('Outputs/Projected_Area/'+MODEL_NAME+'_projectedarea.txt', RSMA)   
    

    
    
    
    
    




