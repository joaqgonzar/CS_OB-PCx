#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:02:59 2023

@author: pcanalisis2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.io 
import pandas as pd
import os
import h5py
import mat73
from scipy import signal
from scipy import stats
from sklearn.decomposition import FastICA, PCA
import matplotlib.gridspec as gridspec
from sklearn.cross_decomposition import CCA 
from sklearn.metrics import silhouette_score
import umap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def eegfilt(data,srate,flow,fhigh):
    
    # fir LS
    trans = 0.15
    nyq = srate*0.5
    f=[0, (1-trans)*flow/nyq, flow/nyq, fhigh/nyq, (1+trans)*fhigh/nyq, 1]
    m=[0,0,1,1,0,0]
    filt_order = 3*np.fix(srate/flow)
    if filt_order % 2 == 0:
        filt_order = filt_order + 1
      
    filtwts = signal.firls(filt_order,f,np.double(m))
    data_filt = signal.filtfilt(filtwts,1, data) 
    
    return(data_filt)

def get_spike_matrix(spike_times):
    
    srate_spikes = 30000
    srate_resample = 2000
    
    recording_time = np.max(np.concatenate(spike_times))

    neuron_number = len(spike_times)
    multi_units_1 = np.zeros([neuron_number,int(recording_time*srate_spikes)],dtype = np.bool_)
    
    indice = 1
    for x in range(neuron_number):
        spike_range = np.where(spike_times[x]<recording_time)[0]
        spikes = spike_times[x][spike_range]
        s_unit = np.rint(spikes*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1
        indice = indice+1
    
    
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)
    
    return(sum_units_1)

def phase_amp_hist(amp,fase_lenta,numbin):
    
    position=np.zeros(numbin) # this variable will get the beginning (not the center) of each phase bin (in rads)
    winsize = 2*np.pi/numbin # bin de fase
    
    position = []
    for j in np.arange(1,numbin+1):
        position.append(-np.pi+(j-1)*winsize)
        

    nbin=numbin 
    mean_amp = []
    for j in np.arange(0,nbin):  
        boolean_array = np.logical_and(fase_lenta >=  position[j], fase_lenta < position[j]+winsize)
        I = np.where(boolean_array)[0]
        mean_amp.append(np.mean(amp[I]))
        
    mean_amp = [x for x in mean_amp if str(x) != 'nan']
 
    return(mean_amp)

def pac_histograms(spikes_conv,resp,lenta_BandWidth,srate,numbin):
        
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(resp,srate,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
    
    hist_freqs = phase_amp_hist(spikes_conv,faselenta,numbin)
    
    return(hist_freqs) 



def plot_3d_confidence_ellipse(x, y, z, num_std_dev=1):
    # Calculate mean and covariance matrix
    data = np.column_stack((x, y, z))
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)

    # Set degrees of freedom and critical value
    #dof = 3
    #critical_value = chi2.ppf(confidence_level, df=dof)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Calculate semi-axes lengths
    semi_axes_lengths = np.sqrt(eigenvalues) * num_std_dev

    # Create rotation matrix
    rotation_matrix = eigenvectors

    # Create points on the ellipse and rotate them
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_ellipse = semi_axes_lengths[0] * np.outer(np.cos(u), np.sin(v))
    y_ellipse = semi_axes_lengths[1] * np.outer(np.sin(u), np.sin(v))
    z_ellipse = semi_axes_lengths[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate the ellipse
    for i in range(len(x_ellipse)):
        for j in range(len(x_ellipse)):
            [x_ellipse[i, j], y_ellipse[i, j], z_ellipse[i, j]] = np.dot([x_ellipse[i, j], y_ellipse[i, j], z_ellipse[i, j]], rotation_matrix) + mean

    
    return(x_ellipse, y_ellipse, z_ellipse)

# Example usage:
# Replace x_data, y_data, and z_data with your data arrays.



#%% general information 

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

os.chdir(directory+'/Simul')
exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)
region = np.array(exp_data[4])
ob_bank_info = list(exp_data[0][region == 'B'])
pc_bank_info = list(exp_data[0][region == 'P'])

ob_bank = []
for x in ob_bank_info:
    ob_bank.append(x[-1])
    
pc_bank = []
for x in pc_bank_info:
    pc_bank.append(x[-1])
    
loading = exp_data[3][1:14]
    

#%% loop through animals 

ob_loadings_animals = []
pc_loadings_animals = []

smells_trig_ob_odor_animals = []
smells_trig_pc_odor_animals = []

smells_trig_ob_concentration_animals = []
smells_trig_pc_concentration_animals = []

ob_loadings_animals_odorless = []
pc_loadings_animals_odorless = []

ob_loadings_animals_odor = []
pc_loadings_animals_odor = []


smells_trig_ob_odorless_animals = []
smells_trig_pc_odorless_animals = []


odorants_animals = []
concentration_animals = []

proj_trig_ob_concentration_animals = []
proj_trig_pc_concentration_animals = []

spontaneous_proj_trig_ob_concentration_animals = []
spontaneous_proj_trig_pc_concentration_animals = []

proj_trig_ob_odor_animals = []
proj_trig_pc_odor_animals = []

spontaneous_proj_trig_ob_odor_animals = []
spontaneous_proj_trig_pc_odor_animals = []

proj_trig_ob_odorless_animals = []
proj_trig_pc_odorless_animals = []

spontaneous_proj_trig_ob_odorless_animals = []
spontaneous_proj_trig_pc_odorless_animals = []


smells_trig_ob_concentration_1_animals = []
smells_trig_pc_concentration_1_animals = []

smells_trig_ob_concentration_2_animals = []
smells_trig_pc_concentration_2_animals = []

concentration_animals_1 = []
concentration_animals_2 = []



for index, name in enumerate(names):
        
    print(name)
    
    # get recordings
    
    os.chdir(directory+'/Simul/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index])+'.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index])+'.mat')['SpikeTimes']['tsec']
    
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    srate_resp = 2000
        
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')
    lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
    os.chdir(directory+'/Simul/processed/'+name)
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate_resp
    inh_start = np.squeeze(inh_start)
    inh_start = inh_start[inh_start<lfp.shape[1]]


    # get ob neurons 

    ob_spikes = []
    for x in range(len(spike_times_ob)):
        ob_spikes.append(spike_times_ob[int(x)][0])

    units_ob = get_spike_matrix(ob_spikes)


    # get pc neurons 

    pc_spikes = []
    for x in range(len(spike_times_pc)):
        pc_spikes.append(spike_times_pc[int(x)][0])

    units_pc = get_spike_matrix(pc_spikes)

    recording_time = np.min([resp.shape[0],units_ob.shape[1],units_pc.shape[1]])
    
    srate = 2000
    
    units_ob = units_ob[:,0:recording_time]
    units_pc = units_pc[:,0:recording_time]
    resp = resp[0:recording_time]
    
    # get odor inhalations
    os.chdir(directory+'/Simul/processed/'+name)
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    
    if loading[index+1] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
    elif loading[index+1] == 'C':
        odor_series = list(np.array([11,7,8,6,12,10])-1)
    
    
    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate   
    odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate,axis = 1))
    
    # get inh starts
    resp_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    
    # exclude odor deliveries and non-awake times
    inh_no_odor = np.setxor1d(odor_times_srate,resp_start)
    inh_no_odor = inh_no_odor[inh_no_odor<resp.shape[0]]
  
    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[1]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    odorless_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[1]))
    odor_mask = np.intersect1d(odor_times,np.arange(0,lfp.shape[1]))
    
    
    resp = resp[odorless_mask]
    # faselenta = faselenta[odorless_mask]
    
    # convolve spike trains

    srate_resample = 2000
    kernel = signal.gaussian(int(0.1*srate_resample),20) 

    conv_neurons_ob = []
    for x in range(units_ob.shape[0]):
        conv = signal.convolve(np.squeeze(units_ob[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_ob.append(conv)
        
    conv_neurons_pc = []
    for x in range(units_pc.shape[0]):
        conv = signal.convolve(np.squeeze(units_pc[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_pc.append(conv)

    
    
    conv_neurons_ob = np.array(conv_neurons_ob)
    conv_neurons_pc = np.array(conv_neurons_pc)

    
    smells_trig_ob = []
    smells_trig_pc = []
    
    # CCA odor cycles
    
    X = np.array(conv_neurons_ob)[1:,odor_mask].T
    Y = np.array(conv_neurons_pc)[1:,odor_mask].T
    
    #del conv_neurons_ob, conv_neurons_pc
    
    firing_neurons_ob = np.sum(X,axis = 0)>0
    firing_neurons_pc = np.sum(Y,axis = 0)>0
    
    X = stats.zscore(X[:,firing_neurons_ob],axis = 0)
    Y = stats.zscore(Y[:,firing_neurons_pc],axis = 0)

    #
    lenght = X.shape[0]-4000
    
    x = 50 
    
    start_x = 0+2000
    start_y = x+2000
    end_x = start_x+lenght
    end_y = start_y+lenght
    
    n_components = 15
    
    cca = CCA(n_components=n_components,max_iter=100000,tol = 1e-12)
    cca.fit(X[start_x:end_x,:], Y[start_y:end_y,:])
    
    ob_loadings_odor = cca.x_weights_
    pc_loadings_odor = cca.y_weights_
    
    
    
    # CCA odorless cycles
    
    X = np.array(conv_neurons_ob)[1:,odorless_mask].T
    Y = np.array(conv_neurons_pc)[1:,odorless_mask].T
    
    #del conv_neurons_ob, conv_neurons_pc
    
    firing_neurons_ob = np.sum(X,axis = 0)>0
    firing_neurons_pc = np.sum(Y,axis = 0)>0
    
    X = stats.zscore(X[:,firing_neurons_ob],axis = 0)
    Y = stats.zscore(Y[:,firing_neurons_pc],axis = 0)

    #
    lenght = X.shape[0]-4000
    
    x = 50 
    
    start_x = 0+2000
    start_y = x+2000
    end_x = start_x+lenght
    end_y = start_y+lenght
    
    n_components = 15
    
    cca = CCA(n_components=n_components,max_iter=100000,tol = 1e-12)
    cca.fit(X[start_x:end_x,:], Y[start_y:end_y,:])
    
    ob_loadings_odorless = cca.x_weights_
    pc_loadings_odorless = cca.y_weights_
    

    X_all = np.array(conv_neurons_ob)[1:,:].T
    Y_all = np.array(conv_neurons_pc)[1:,:].T
    
    ob_proj_odor = stats.zscore(np.sum(X_all*ob_loadings_odor[:,0].T,axis = 1),axis = 0)
    pc_proj_odor = stats.zscore(np.sum(Y_all*pc_loadings_odor[:,0].T,axis = 1),axis = 0)
    
    ob_proj_odorless = stats.zscore(np.sum(X_all*ob_loadings_odorless[:,0].T,axis = 1),axis = 0)
    pc_proj_odorless = stats.zscore(np.sum(Y_all*pc_loadings_odorless[:,0].T,axis = 1),axis = 0)
    
    
    # check odors
    
    os.chdir(directory+'/Simul/processed/'+name)
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]

    if loading[index+1] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
    elif loading[index+1] == 'C':
        odor_series = list(np.array([11,7,8,6,12,10])-1)


    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate   

    odor_times_awake = []
    odorants = []


    for x in range(odor_times_srate.shape[0]):
        
        odor_mask = odor_times_srate[x]<lfp.shape[1]

        odor_times_awake.append(odor_times_srate[x][odor_mask])
        odorants.append(np.repeat(x,repeats = np.sum(odor_mask)))
        
    odor_times_awake = np.concatenate(odor_times_awake)
    odorants = np.concatenate(odorants)


    smells_trig_ob = []
    smells_trig_pc = []
    
    proj_trig_ob_odor = []
    proj_trig_pc_odor = []
    
    spontaneous_proj_trig_ob_odor = []
    spontaneous_proj_trig_pc_odor = []
    
    
    time = 2000

    listss = []
    
    for index_odor, x in enumerate(odor_times_awake):
        
        
        smells_trig_ob.append(conv_neurons_ob[1:,int(x):int(int(x)+time)])
        smells_trig_pc.append(conv_neurons_pc[1:,int(x):int(int(x)+time)])
        
        if conv_neurons_ob[:,int(int(x)-time):int(int(x)+time)].shape[1] == 4000:
            
            
            proj_trig_ob_odor.append(ob_proj_odor[int(int(x)-time):int(int(x)+time)])
            proj_trig_pc_odor.append(pc_proj_odor[int(int(x)-time):int(int(x)+time)])
            
            spontaneous_proj_trig_ob_odor.append(ob_proj_odorless[int(int(x)-time):int(int(x)+time)])
            spontaneous_proj_trig_pc_odor.append(pc_proj_odorless[int(int(x)-time):int(int(x)+time)])
            
            
            


    
    smells_trig_ob_odor = np.array(smells_trig_ob)
    smells_trig_pc_odor = np.array(smells_trig_pc)
    
    proj_trig_ob_odor_animals.append(proj_trig_ob_odor)
    proj_trig_pc_odor_animals.append(proj_trig_pc_odor)
    
    spontaneous_proj_trig_ob_odor_animals.append(spontaneous_proj_trig_ob_odor)
    spontaneous_proj_trig_pc_odor_animals.append(spontaneous_proj_trig_pc_odor)
    
    
    # check odorless inh starts
    
    # get odor inhalations
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    inh_odor = []
    for x in odor_onset_srate:
        index_smells = np.logical_and((inh_start>x),(inh_start<x+2000))
        inh_odor.append(inh_start[index_smells])
            
    inh_odor = np.concatenate(inh_odor,axis = 0)
    inh_no_odor = np.setdiff1d(inh_start,inh_odor)
    
    
    smells_trig_ob = []
    smells_trig_pc = []
    
    proj_trig_ob_odorless = []
    proj_trig_pc_odorless = []
    
    spontaneous_proj_trig_ob_odorless = []
    spontaneous_proj_trig_pc_odorless = []
    
    
    
    for index_odor, x in enumerate(inh_no_odor):
        
        if conv_neurons_ob[:,int(int(x)-time):int(int(x)+time)].shape[1] == 4000:
            
            
            proj_trig_ob_odorless.append(ob_proj_odor[int(int(x)-time):int(int(x)+time)])
            proj_trig_pc_odorless.append(pc_proj_odor[int(int(x)-time):int(int(x)+time)])
            
            spontaneous_proj_trig_ob_odorless.append(ob_proj_odorless[int(int(x)-time):int(int(x)+time)])
            spontaneous_proj_trig_pc_odorless.append(pc_proj_odorless[int(int(x)-time):int(int(x)+time)])
            
    
    smells_trig_ob_odorless = np.array(smells_trig_ob)
    smells_trig_pc_odorless = np.array(smells_trig_pc)
    
    proj_trig_ob_odorless_animals.append(proj_trig_ob_odorless)
    proj_trig_pc_odorless_animals.append(proj_trig_pc_odorless)
    
    spontaneous_proj_trig_ob_odorless_animals.append(spontaneous_proj_trig_ob_odorless)
    spontaneous_proj_trig_pc_odorless_animals.append(spontaneous_proj_trig_pc_odorless)
    
    
    
    
    # check odor concentration 
    
    if loading[index+1] == 'A':
        conc_series1 = list(np.array([2,3,4,5])-1)
        conc_series2 = list(np.array([10,11,12,13])-1)
    
        odor_data_conc = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
        odor_times_1 = odor_data_conc[conc_series1]
        odor_times_srate_1 = odor_times_1*srate   
        odor_times_2 = odor_data_conc[conc_series2]
        odor_times_srate_2 = odor_times_2*srate  
    
        odor_times_awake_concentration = []
        concentration = []
    
        for x in range(odor_times_srate_1.shape[0]):
            
            onsets_odor_1 = odor_times_srate_1[x][0]
            onsets_odor_2 = odor_times_srate_2[x][0]
            onsets_odor = np.concatenate([onsets_odor_1,onsets_odor_2])
            
            odor_mask = onsets_odor<lfp.shape[1]
            
            odor_times_awake_concentration.append(onsets_odor[odor_mask])
            concentration.append(np.repeat(x,repeats = np.sum(odor_mask)))
            
        odor_times_awake_concentration = np.concatenate(odor_times_awake_concentration)
        concentration = np.concatenate(concentration)
    
        
        smells_trig_ob = []
        smells_trig_pc = []
        
        proj_trig_ob_concentration = []
        proj_trig_pc_concentration = []
        
        spontaneous_proj_trig_ob_concentration = []
        spontaneous_proj_trig_pc_concentration = []
        
    
        time = 2000
    
        for index_odor, x in enumerate(odor_times_awake_concentration):
            
            if conv_neurons_ob[:,int(int(x)-time):int(int(x)+time)].shape[1] == 4000:
                
                smells_trig_ob.append(conv_neurons_ob[1:,int(int(x)-time):int(int(x)+time)])
                smells_trig_pc.append(conv_neurons_pc[1:,int(int(x)-time):int(int(x)+time)])
                
                proj_trig_ob_concentration.append(ob_proj_odor[int(int(x)-time):int(int(x)+time)])
                proj_trig_pc_concentration.append(pc_proj_odor[int(int(x)-time):int(int(x)+time)])
                
                spontaneous_proj_trig_ob_concentration.append(ob_proj_odorless[int(int(x)-time):int(int(x)+time)])
                spontaneous_proj_trig_pc_concentration.append(pc_proj_odorless[int(int(x)-time):int(int(x)+time)])
                
                

        
        smells_trig_ob_concentration = np.array(smells_trig_ob)
        smells_trig_pc_concentration = np.array(smells_trig_pc)
        
    
        smells_trig_ob_concentration_animals.append(smells_trig_ob_concentration)
        smells_trig_pc_concentration_animals.append(smells_trig_pc_concentration)
        concentration_animals.append(concentration)
        
        proj_trig_ob_concentration_animals.append(proj_trig_ob_concentration)
        proj_trig_pc_concentration_animals.append(proj_trig_pc_concentration)
        
        spontaneous_proj_trig_ob_concentration_animals.append(spontaneous_proj_trig_ob_concentration)
        spontaneous_proj_trig_pc_concentration_animals.append(spontaneous_proj_trig_pc_concentration)
        
        
        # separate concentration into two odors
        
        # odor 1
        
        if loading[index+1] == 'A':
            conc_series1 = list(np.array([2,3,4,5])-1)
            conc_series2 = list(np.array([10,11,12,13])-1)
        
            odor_data_conc = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
            odor_times_1 = odor_data_conc[conc_series1]
            odor_times_srate_1 = odor_times_1*srate   
            odor_times_2 = odor_data_conc[conc_series2]
            odor_times_srate_2 = odor_times_2*srate  
        
            odor_times_awake_concentration = []
            concentration = []
        
            for x in range(odor_times_srate_1.shape[0]):
                
                onsets_odor_1 = odor_times_srate_1[x][0]
                onsets_odor_2 = odor_times_srate_2[x][0]
                onsets_odor = np.concatenate([onsets_odor_1])
                
                odor_mask = onsets_odor<lfp.shape[1]
                
                odor_times_awake_concentration.append(onsets_odor[odor_mask])
                concentration.append(np.repeat(x,repeats = np.sum(odor_mask)))
                
            odor_times_awake_concentration = np.concatenate(odor_times_awake_concentration)
            concentration_1 = np.concatenate(concentration)
        
            
            smells_trig_ob_1 = []
            smells_trig_pc_1 = []
            
            proj_trig_ob_concentration = []
            proj_trig_pc_concentration = []
            
            spontaneous_proj_trig_ob_concentration = []
            spontaneous_proj_trig_pc_concentration = []
            
        
            time = 2000
        
            for index_odor, x in enumerate(odor_times_awake_concentration):
                
                if conv_neurons_ob[:,int(int(x)-time):int(int(x)+time)].shape[1] == 4000:
                    
                    smells_trig_ob_1.append(conv_neurons_ob[1:,int(int(x)-time):int(int(x)+time)])
                    smells_trig_pc_1.append(conv_neurons_pc[1:,int(int(x)-time):int(int(x)+time)])
                    

     
            smells_trig_ob_concentration_1 = np.array(smells_trig_ob_1)
            smells_trig_pc_concentration_1 = np.array(smells_trig_pc_1)
            
        
            smells_trig_ob_concentration_1_animals.append(smells_trig_ob_concentration_1)
            smells_trig_pc_concentration_1_animals.append(smells_trig_pc_concentration_1)
            
            concentration_animals_1.append(concentration_1)
            
            
            # odor 2
            
            if loading[index+1] == 'A':
                conc_series1 = list(np.array([2,3,4,5])-1)
                conc_series2 = list(np.array([10,11,12,13])-1)
            
                odor_data_conc = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
                odor_times_1 = odor_data_conc[conc_series1]
                odor_times_srate_1 = odor_times_1*srate   
                odor_times_2 = odor_data_conc[conc_series2]
                odor_times_srate_2 = odor_times_2*srate  
            
                odor_times_awake_concentration = []
                concentration = []
            
                for x in range(odor_times_srate_1.shape[0]):
                    
                    onsets_odor_1 = odor_times_srate_1[x][0]
                    onsets_odor_2 = odor_times_srate_2[x][0]
                    onsets_odor = np.concatenate([onsets_odor_2])
                    
                    odor_mask = onsets_odor<lfp.shape[1]
                    
                    odor_times_awake_concentration.append(onsets_odor[odor_mask])
                    concentration.append(np.repeat(x,repeats = np.sum(odor_mask)))
                    
                odor_times_awake_concentration = np.concatenate(odor_times_awake_concentration)
                concentration_2 = np.concatenate(concentration)
            
                
                smells_trig_ob_2 = []
                smells_trig_pc_2 = []
                
                proj_trig_ob_concentration = []
                proj_trig_pc_concentration = []
                
                spontaneous_proj_trig_ob_concentration = []
                spontaneous_proj_trig_pc_concentration = []
                
            
                time = 2000
            
                for index_odor, x in enumerate(odor_times_awake_concentration):
                    
                    if conv_neurons_ob[:,int(int(x)-time):int(int(x)+time)].shape[1] == 4000:
                        
                        smells_trig_ob_2.append(conv_neurons_ob[1:,int(int(x)-time):int(int(x)+time)])
                        smells_trig_pc_2.append(conv_neurons_pc[1:,int(int(x)-time):int(int(x)+time)])
                        

         
                smells_trig_ob_concentration_2 = np.array(smells_trig_ob_2)
                smells_trig_pc_concentration_2 = np.array(smells_trig_pc_2)
                
            
                smells_trig_ob_concentration_2_animals.append(smells_trig_ob_concentration_2)
                smells_trig_pc_concentration_2_animals.append(smells_trig_pc_concentration_2)
                
                concentration_animals_2.append(concentration_2)
            
    
        
        
        
    #
    ob_loadings_animals_odorless.append(ob_loadings_odorless)
    pc_loadings_animals_odorless.append(pc_loadings_odorless)
    
    ob_loadings_animals_odor.append(ob_loadings_odor)
    pc_loadings_animals_odor.append(pc_loadings_odor)
    
    
    smells_trig_ob_odor_animals.append(smells_trig_ob_odor)
    smells_trig_pc_odor_animals.append(smells_trig_pc_odor)
    
    smells_trig_ob_odorless_animals.append(smells_trig_ob_odorless)
    smells_trig_pc_odorless_animals.append(smells_trig_pc_odorless)
    
    
    odorants_animals.append(odorants)
    
    #
    del units_ob, units_pc, lfp, smells_trig_ob, smells_trig_pc, conv_neurons_ob, conv_neurons_pc
    

#%%

mean_ob_proj_odor_animals = []
mean_pc_proj_odor_animals = []

mua_ob_odor_animals = []

for x in range(13):
    
    mean_ob_proj_odor_animals.append(np.mean(np.array(spontaneous_proj_trig_ob_odor_animals[x])[:],axis = 0))
    mean_pc_proj_odor_animals.append(np.mean(np.array(spontaneous_proj_trig_pc_odor_animals[x])[:],axis = 0))
    
    
    mua_ob_odor_animals.append(np.mean(smells_trig_pc_odor[x],axis = 0))
    
    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


time = np.linspace(-1000,1000,4000)
plt.figure(dpi = 300, figsize = (12,5))

signs = []
diff_response_pc = []
diff_response_ob = []

for animal in range(13):
    
    plt.subplot(3,6,animal+1)
    plt.plot(time, mean_pc_proj_odor_animals[animal])
    plt.plot(time, mean_ob_proj_odor_animals[animal])
    

    
    
    argmax = np.argmax(np.abs(mean_ob_proj_odor_animals[animal][2050:2400]))
    
    
    signs.append(mean_ob_proj_odor_animals[animal][2050+argmax]-mean_ob_proj_odor_animals[animal][2000])
    
    argmax = np.argmax(mean_pc_proj_odor_animals[animal][2000:2400])
    argmin = np.argmin(mean_pc_proj_odor_animals[animal][2000:2400])

    max_response = np.abs(mean_pc_proj_odor_animals[animal][2000:2400][argmax]-mean_pc_proj_odor_animals[animal][2000])
    min_response = np.abs(mean_pc_proj_odor_animals[animal][2000:2400][argmin]-mean_pc_proj_odor_animals[animal][2000])

    diff_response_pc.append(max_response-min_response)
    
    argmax = np.argmax(mean_ob_proj_odor_animals[animal][2000:2400])
    argmin = np.argmin(mean_ob_proj_odor_animals[animal][2000:2400])

    max_response = np.abs(mean_ob_proj_odor_animals[animal][2000:2400][argmax]-mean_ob_proj_odor_animals[animal][2000])
    min_response = np.abs(mean_ob_proj_odor_animals[animal][2000:2400][argmin]-mean_ob_proj_odor_animals[animal][2000])

    diff_response_ob.append(max_response-min_response)
    
    
    
    plt.xlim([-200,400])
    
    plt.ylim([-4,4])
    
    plt.fill_between(np.arange(0,200),-4,4,alpha = 0.1)
    plt.tight_layout()
    
    
    if animal == 12:
        
        plt.xlabel('Time from inhalation start (ms)')
        
signs = np.sign(np.array(diff_response_pc)+np.array(diff_response_ob))

   
plt.savefig('cs_odors_animals.pdf')

#%% check CS activation and correlation

mean_ob_proj_odorless_animals = []
mean_ob_proj_odor_animals = []
mean_pc_proj_odorless_animals = []
mean_pc_proj_odor_animals = []

odor_mean_ob_proj_odorless_animals = []
odor_mean_ob_proj_odor_animals = []
odor_mean_pc_proj_odorless_animals = []
odor_mean_pc_proj_odor_animals = []


from pyinform.transferentropy import transfer_entropy

t_ob_pc_proj_odor = []
t_pc_ob_proj_odor = []
t_ob_pc_proj_odorless = []
t_pc_ob_proj_odorless = []
r_odorless = []
r_odor = []

odor_t_ob_pc_proj_odor = []
odor_t_pc_ob_proj_odor = []
odor_t_ob_pc_proj_odorless = []
odor_t_pc_ob_proj_odorless = []
odor_r_odorless = []
odor_r_odor = []

sign_max_ob_odor_animals = []

for x in range(13):
    
    odor_num = len(spontaneous_proj_trig_ob_odorless_animals[x])
    
    odorless_samples = np.random.randint(0,len(spontaneous_proj_trig_ob_odorless_animals[x]),odor_num)
    
    max_ob_odor = np.max(np.abs(np.mean(spontaneous_proj_trig_ob_odorless_animals[x],axis = 0)))
    max_ob_odor_location = np.where(np.abs(np.mean(spontaneous_proj_trig_ob_odorless_animals[x],axis = 0))==max_ob_odor)[0][0]
    sign_max_ob_odor_animals.append(np.sign(np.max(spontaneous_proj_trig_ob_odorless_animals[x])))
    #sign_max_ob_odor_animals.append(np.sign(np.sum(np.mean(inh_trig_ob_proj_odorless_animals[x],axis = 0)[2000:4000])))
    sign_max_ob_odor = signs[x]
    
    mean_ob_proj_odorless_animals.append(np.mean(np.array(spontaneous_proj_trig_ob_odorless_animals[x])[:],axis = 0)*sign_max_ob_odor)
    mean_ob_proj_odor_animals.append(np.mean(spontaneous_proj_trig_ob_odor_animals[x],axis = 0)*sign_max_ob_odor)
    
    mean_pc_proj_odorless_animals.append(np.mean(np.array(spontaneous_proj_trig_pc_odorless_animals[x])[:],axis = 0)*sign_max_ob_odor)
    mean_pc_proj_odor_animals.append(np.mean(spontaneous_proj_trig_pc_odor_animals[x],axis = 0)*sign_max_ob_odor)
    
    time_start = 2000
    time_end = 2800
    
    ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odorless_animals[x])[:,time_start:time_end],axis = 0)
    pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odorless_animals[x])[:,time_start:time_end],axis = 0)
    
    r_odorless.append(stats.pearsonr(ob_cat,pc_cat)[0])
    
    t_ob_pc_proj_odorless.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
    t_pc_ob_proj_odorless.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
    
    ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odor_animals[x])[:,time_start:time_end],axis = 0)
    pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odor_animals[x])[:,time_start:time_end],axis = 0)
    
    r_odor.append(stats.pearsonr(ob_cat,pc_cat)[0])
    
    t_ob_pc_proj_odor.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
    t_pc_ob_proj_odor.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
    
    # check odor only CS 
    
    max_ob_odor = np.max(np.abs(np.mean(proj_trig_ob_odorless_animals[x],axis = 0)))
    max_ob_odor_location = np.where(np.abs(np.mean(proj_trig_ob_odorless_animals[x],axis = 0))==max_ob_odor)[0][0]
    #sign_max_ob_odor_animals.append(np.sign(np.mean(proj_trig_ob_odorless_animals[x],axis = 0)[max_ob_odor_location]))
    #sign_max_ob_odor_animals.append(np.sign(np.sum(np.mean(inh_trig_ob_proj_odorless_animals[x],axis = 0)[2000:4000])))
    sign_max_ob_odor = signs[x]
    
    odor_mean_ob_proj_odorless_animals.append(np.mean(np.array(proj_trig_ob_odorless_animals[x])[:],axis = 0)*sign_max_ob_odor)
    odor_mean_ob_proj_odor_animals.append(np.mean(proj_trig_ob_odor_animals[x],axis = 0)*sign_max_ob_odor)
    
    odor_mean_pc_proj_odorless_animals.append(np.mean(np.array(proj_trig_pc_odorless_animals[x])[:],axis = 0)*sign_max_ob_odor)
    odor_mean_pc_proj_odor_animals.append(np.mean(proj_trig_pc_odor_animals[x],axis = 0)*sign_max_ob_odor)
    
    ob_cat = np.concatenate(np.array(proj_trig_ob_odorless_animals[x])[:,time_start:time_end],axis = 0)
    pc_cat = np.concatenate(np.array(proj_trig_pc_odorless_animals[x])[:,time_start:time_end],axis = 0)
    
    odor_r_odorless.append(stats.pearsonr(ob_cat,pc_cat)[0])
    
    odor_t_ob_pc_proj_odorless.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
    odor_t_pc_ob_proj_odorless.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
    
    ob_cat = np.concatenate(np.array(proj_trig_ob_odor_animals[x])[:,time_start:time_end],axis = 0)
    pc_cat = np.concatenate(np.array(proj_trig_pc_odor_animals[x])[:,time_start:time_end],axis = 0)
    
    odor_r_odor.append(stats.pearsonr(ob_cat,pc_cat)[0])
    
    odor_t_ob_pc_proj_odor.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
    odor_t_pc_ob_proj_odor.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
    
#%%
sign_max_ob_odor_animals = []

for x in range(13):
    
    max_ob_odor = np.max(np.abs(np.mean(spontaneous_proj_trig_ob_odor_animals[x],axis = 0)[2000:2500]))
    max_ob_odor_location = np.where(np.abs(np.mean(spontaneous_proj_trig_ob_odor_animals[x],axis = 0)[2000:2500])==max_ob_odor)[0]
    
    
    sign_max_ob_odor_animals.append(np.sign(np.mean(spontaneous_proj_trig_ob_odor_animals[x],axis = 0)[2000:2500][max_ob_odor_location]))
    
    
#%%
animal = 3

plt.plot(mean_ob_proj_odor_animals[animal])
plt.plot(mean_ob_proj_odorless_animals[animal])


#%%
time = np.linspace(-1000,1000,4000)

inh_trig_ob_proj_odorless = np.mean(mean_ob_proj_odorless_animals,axis = 0)
inh_trig_ob_proj_odor = np.mean(mean_ob_proj_odor_animals,axis = 0)
inh_trig_ob_proj_odorless_error = stats.sem(mean_ob_proj_odorless_animals,axis = 0)
inh_trig_ob_proj_odor_error = stats.sem(mean_ob_proj_odor_animals,axis = 0)

inh_trig_pc_proj_odorless = np.mean(mean_pc_proj_odorless_animals,axis = 0)
inh_trig_pc_proj_odor = np.mean(mean_pc_proj_odor_animals,axis = 0)
inh_trig_pc_proj_odorless_error = stats.sem(mean_pc_proj_odorless_animals,axis = 0)
inh_trig_pc_proj_odor_error = stats.sem(mean_pc_proj_odor_animals,axis = 0)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (10,5))



gs = gridspec.GridSpec(2, 2)

plt.subplot(gs[0])
plt.plot(time,inh_trig_ob_proj_odorless, label = 'No Odor', color = 'tab:purple')    
plt.fill_between(time, inh_trig_ob_proj_odorless-inh_trig_ob_proj_odorless_error,inh_trig_ob_proj_odorless+inh_trig_ob_proj_odorless_error,alpha = 0.2, color = 'tab:purple')

plt.plot(time,inh_trig_ob_proj_odor, label = 'Odor', color = 'tab:green')    
plt.fill_between(time, inh_trig_ob_proj_odor-inh_trig_ob_proj_odor_error,inh_trig_ob_proj_odor+inh_trig_ob_proj_odor_error,alpha = 0.2, color = 'tab:green')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.legend()

plt.ylabel('OB proj amp (z)')
plt.xlim([-100,600])
#plt.ylim([-0.5,1])
t_ob,p_ob = stats.ttest_rel(mean_ob_proj_odor_animals,mean_ob_proj_odorless_animals,alternative = 'greater') 
p_ob_plot = time[np.where(p_ob<0.05)[0]]
plt.scatter(p_ob_plot,np.ones(p_ob_plot.shape[0])*1, s = 1)

plt.subplot(gs[2])
plt.plot(time,inh_trig_pc_proj_odorless, color = 'tab:purple')    #
plt.fill_between(time, inh_trig_pc_proj_odorless-inh_trig_pc_proj_odorless_error,inh_trig_pc_proj_odorless+inh_trig_pc_proj_odorless_error,alpha = 0.2, color = 'tab:purple')

plt.plot(time,inh_trig_pc_proj_odor, color = 'tab:green')    
plt.fill_between(time, inh_trig_pc_proj_odor-inh_trig_pc_proj_odor_error,inh_trig_pc_proj_odor+inh_trig_pc_proj_odor_error,alpha = 0.2, color = 'tab:green')

#plt.scatter(time, np.array(p_all)<0.05)

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('PC CSA (z)')
plt.xlabel('Time from inhalation start (ms)')
plt.xlim([-100,600])
t_pc,p_pc = stats.ttest_rel(mean_pc_proj_odor_animals,mean_pc_proj_odorless_animals,alternative = 'greater') 
p_pc_plot = time[np.where(p_pc<0.05)[0]]
plt.scatter(p_pc_plot,np.ones(p_pc_plot.shape[0])*1, s = 1)

#plt.ylim([-0.5,1])

inh_trig_ob_proj_odorless = np.mean(odor_mean_ob_proj_odorless_animals,axis = 0)
inh_trig_ob_proj_odor = np.mean(odor_mean_ob_proj_odor_animals,axis = 0)
inh_trig_ob_proj_odorless_error = stats.sem(odor_mean_ob_proj_odorless_animals,axis = 0)
inh_trig_ob_proj_odor_error = stats.sem(odor_mean_ob_proj_odor_animals,axis = 0)

inh_trig_pc_proj_odorless = np.mean(odor_mean_pc_proj_odorless_animals,axis = 0)
inh_trig_pc_proj_odor = np.mean(odor_mean_pc_proj_odor_animals,axis = 0)
inh_trig_pc_proj_odorless_error = stats.sem(odor_mean_pc_proj_odorless_animals,axis = 0)
inh_trig_pc_proj_odor_error = stats.sem(odor_mean_pc_proj_odor_animals,axis = 0)


plt.subplot(gs[1])
plt.plot(time,inh_trig_ob_proj_odorless, label = 'No Odor', color = 'tab:purple')    
plt.fill_between(time, inh_trig_ob_proj_odorless-inh_trig_ob_proj_odorless_error,inh_trig_ob_proj_odorless+inh_trig_ob_proj_odorless_error,alpha = 0.2, color = 'tab:purple')

plt.plot(time,inh_trig_ob_proj_odor, label = 'Odor', color = 'tab:green')    
plt.fill_between(time, inh_trig_ob_proj_odor-inh_trig_ob_proj_odor_error,inh_trig_ob_proj_odor+inh_trig_ob_proj_odor_error,alpha = 0.2, color = 'tab:green')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

t_ob,p_ob = stats.ttest_rel(odor_mean_ob_proj_odor_animals,odor_mean_ob_proj_odorless_animals,alternative = 'greater') 
p_ob_plot = time[np.where(p_ob<0.05)[0]]
plt.scatter(p_ob_plot,np.ones(p_ob_plot.shape[0])*1, s = 1)

plt.legend()

plt.ylabel('OB proj amp (z)')
plt.xlim([-100,600])
#plt.ylim([-0.5,1])

plt.subplot(gs[3])
plt.plot(time,inh_trig_pc_proj_odorless, color = 'tab:purple')    #
plt.fill_between(time, inh_trig_pc_proj_odorless-inh_trig_pc_proj_odorless_error,inh_trig_pc_proj_odorless+inh_trig_pc_proj_odorless_error,alpha = 0.2, color = 'tab:purple')

plt.plot(time,inh_trig_pc_proj_odor, color = 'tab:green')    
plt.fill_between(time, inh_trig_pc_proj_odor-inh_trig_pc_proj_odor_error,inh_trig_pc_proj_odor+inh_trig_pc_proj_odor_error,alpha = 0.2, color = 'tab:green')

t_pc,p_pc = stats.ttest_rel(odor_mean_pc_proj_odor_animals,odor_mean_pc_proj_odorless_animals,alternative = 'greater') 
p_pc_plot = time[np.where(p_pc<0.05)[0]]
plt.scatter(p_pc_plot,np.ones(p_pc_plot.shape[0])*1, s = 1)

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('PC CSA (z)')
plt.xlabel('Time from inhalation start (ms)')
plt.xlim([-100,600])
#plt.ylim([-0.5,1])



plt.tight_layout()

#%%

time = np.linspace(-1000,1000,4000)

diff_ob = (np.array(mean_ob_proj_odor_animals)-np.array(mean_ob_proj_odorless_animals))
diff_pc = (np.array(mean_pc_proj_odor_animals)-np.array(mean_pc_proj_odorless_animals))

t_ob,p_ob = stats.ttest_rel(mean_ob_proj_odor_animals,mean_ob_proj_odorless_animals,alternative = 'greater') 
#t_ob,p_ob = stats.ttest_1samp(diff_ob,popmean = 0,alternative = 'greater') 

p_ob_plot = time[np.where(p_ob<0.05)[0]]


t_pc,p_pc = stats.ttest_rel(mean_pc_proj_odor_animals,mean_pc_proj_odorless_animals,alternative = 'greater')
p_pc_plot = time[np.where(p_pc<0.05)[0]]



inh_trig_ob_proj_odorless = np.mean(diff_ob,axis = 0)
inh_trig_ob_proj_odorless_error = stats.sem(diff_ob,axis = 0)


inh_trig_pc_proj_odorless = np.mean(diff_pc,axis = 0)
inh_trig_pc_proj_odorless_error = stats.sem(diff_pc,axis = 0)


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (5,5))



gs = gridspec.GridSpec(2, 1)

plt.subplot(gs[0])
plt.plot(time,inh_trig_ob_proj_odorless, label = 'OB', color = 'tab:orange')    
plt.fill_between(time, inh_trig_ob_proj_odorless-inh_trig_ob_proj_odorless_error,inh_trig_ob_proj_odorless+inh_trig_ob_proj_odorless_error,alpha = 0.2, color = 'tab:orange')

plt.plot(time,inh_trig_pc_proj_odorless, color = 'tab:blue', label = 'PCx')    #
plt.fill_between(time, inh_trig_pc_proj_odorless-inh_trig_pc_proj_odorless_error,inh_trig_pc_proj_odorless+inh_trig_pc_proj_odorless_error,alpha = 0.2, color = 'tab:blue')

plt.xlim([-100,600])
plt.ylim([-0.5,2])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.scatter(p_pc_plot,np.ones(p_pc_plot.shape[0])*1.5, s = 1)
plt.scatter(p_ob_plot,np.ones(p_ob_plot.shape[0])*1.7, s = 1)

#
diff_ob = (np.array(odor_mean_ob_proj_odor_animals)-np.array(odor_mean_ob_proj_odorless_animals))
diff_pc = (np.array(odor_mean_pc_proj_odor_animals)-np.array(odor_mean_pc_proj_odorless_animals))

t_ob,p_ob = stats.ttest_rel(odor_mean_ob_proj_odor_animals,odor_mean_ob_proj_odorless_animals,alternative = 'greater') 
p_ob_plot = time[np.where(p_ob<0.05)[0]]


t_pc,p_pc = stats.ttest_rel(odor_mean_pc_proj_odor_animals,odor_mean_pc_proj_odorless_animals,alternative = 'greater') 
p_pc_plot = time[np.where(p_pc<0.05)[0]]



inh_trig_ob_proj_odorless = np.mean(diff_ob,axis = 0)
inh_trig_ob_proj_odorless_error = stats.sem(diff_ob,axis = 0)

inh_trig_pc_proj_odorless = np.mean(diff_pc,axis = 0)
inh_trig_pc_proj_odorless_error = stats.sem(diff_pc,axis = 0)




plt.ylabel('(CS1 Odor - CS1 No odor)',loc = 'top')
plt.text(-80,1.7,'Spontaneous CS')

plt.subplot(gs[1])
plt.plot(time,inh_trig_ob_proj_odorless, label = 'OB', color = 'tab:orange')    
plt.fill_between(time, inh_trig_ob_proj_odorless-inh_trig_ob_proj_odorless_error,inh_trig_ob_proj_odorless+inh_trig_ob_proj_odorless_error,alpha = 0.2, color = 'tab:orange')

plt.plot(time,inh_trig_pc_proj_odorless, color = 'tab:blue',label = 'PCx')    #
plt.fill_between(time, inh_trig_pc_proj_odorless-inh_trig_pc_proj_odorless_error,inh_trig_pc_proj_odorless+inh_trig_pc_proj_odorless_error,alpha = 0.2, color = 'tab:blue')

plt.xlim([-100,600])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.scatter(p_pc_plot,np.ones(p_pc_plot.shape[0])*1.5, s = 1)
plt.scatter(p_ob_plot,np.ones(p_ob_plot.shape[0])*1.7, s = 1)

plt.legend()

plt.text(-80,1.7,'Odor CS')

plt.ylim([-0.5,2])

plt.xlabel('Time from inhalation start (ms)')

#plt.savefig('cs_diff_odors.pdf')

#%% get sum values across time

time_max = 2400

t_ob_sum,p_ob_sum = stats.ttest_rel(np.sum(np.array(mean_ob_proj_odor_animals)[:,2000:time_max],axis = 1),np.sum(np.array(mean_ob_proj_odorless_animals)[:,2000:time_max],axis = 1),alternative = 'greater') 
t_pc_sum,p_pc_sum = stats.ttest_rel(np.sum(np.array(mean_pc_proj_odor_animals)[:,2000:time_max],axis = 1),np.sum(np.array(mean_pc_proj_odorless_animals)[:,2000:time_max],axis = 1),alternative = 'greater') 

odor_t_ob_sum,odor_p_ob_sum = stats.ttest_rel(np.sum(np.array(odor_mean_ob_proj_odor_animals)[:,2000:time_max],axis = 1),np.sum(np.array(odor_mean_ob_proj_odorless_animals)[:,2000:time_max],axis = 1),alternative = 'greater') 
odor_t_pc_sum,odor_p_pc_sum = stats.ttest_rel(np.sum(np.array(odor_mean_pc_proj_odor_animals)[:,2000:2500],axis = 1),np.sum(np.array(odor_mean_pc_proj_odorless_animals)[:,2000:time_max],axis = 1),alternative = 'greater') 


#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



plt.figure(dpi = 300, figsize = (5,6))


plt.subplot(221)

animal = 3

time_start = 2000
time_end = 3000

odor_num = len(spontaneous_proj_trig_ob_odor_animals[animal])

odorless_samples = np.random.randint(0,len(spontaneous_proj_trig_ob_odorless_animals[animal]),odor_num)

ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odorless_animals[animal])[odorless_samples,time_start:time_end:1],axis = 0)
pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odorless_animals[animal])[odorless_samples,time_start:time_end:1],axis = 0)

plt.scatter(ob_cat,pc_cat, s = 0.1,alpha = 0.1, color = 'black',rasterized = True)
trend = np.polyfit(ob_cat,pc_cat,1)
trendpoly = np.poly1d(trend) 

plt.plot(ob_cat,trendpoly(ob_cat), color = 'red', linewidth = 2, label = 'R = '+str(np.round(r_odorless[animal],decimals = 2)))
plt.legend(fontsize = 8, loc = 'lower right')
plt.xlim([-3,6])
plt.ylim([-6,6])
plt.hlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.vlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('PCx CS1')
plt.title('No Odor')




plt.subplot(222)

ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odor_animals[animal])[:,time_start:time_end:1],axis = 0)
pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odor_animals[animal])[:,time_start:time_end:1],axis = 0)

plt.scatter(ob_cat,pc_cat, s = 0.1,alpha = 0.1, color = 'black',rasterized = True)

trend = np.polyfit(ob_cat,pc_cat,1)
trendpoly = np.poly1d(trend) 

plt.plot(ob_cat,trendpoly(ob_cat), color = 'red', linewidth = 2, label = 'R = '+str(np.round(r_odor[animal],decimals = 2)))

#plt.ylabel('PCx CS1')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend(fontsize = 8, loc = 'lower right')

plt.xlim([-3,6])
plt.ylim([-4.5,6])
plt.hlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.vlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.title('Odor')

plt.subplot(223)

odor_num = len(proj_trig_ob_odor_animals[animal])

odorless_samples = np.random.randint(0,len(proj_trig_ob_odorless_animals[animal]),odor_num)

ob_cat = np.concatenate(np.array(proj_trig_ob_odorless_animals[animal])[odorless_samples,time_start:time_end:1],axis = 0)
pc_cat = np.concatenate(np.array(proj_trig_pc_odorless_animals[animal])[odorless_samples,time_start:time_end:1],axis = 0)

plt.scatter(ob_cat,pc_cat, s = 0.1,alpha = 0.1, color = 'black',rasterized = True)
trend = np.polyfit(ob_cat,pc_cat,1)
trendpoly = np.poly1d(trend) 

plt.plot(ob_cat,trendpoly(ob_cat), color = 'red', linewidth = 2, label = 'R = '+str(np.round(odor_r_odorless[animal],decimals = 2)))
plt.legend(fontsize = 8, loc = 'lower right')
plt.xlim([-3,6])
plt.ylim([-4.5,4])
plt.hlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.vlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.xlabel('OB CS1')
plt.ylabel('PCx CS1')



plt.subplot(224)

ob_cat = np.concatenate(np.array(proj_trig_ob_odor_animals[animal])[:,time_start:time_end:1],axis = 0)
pc_cat = np.concatenate(np.array(proj_trig_pc_odor_animals[animal])[:,time_start:time_end:1],axis = 0)

plt.scatter(ob_cat,pc_cat, s = 0.1,alpha = 0.1, color = 'black',rasterized = True)

trend = np.polyfit(ob_cat,pc_cat,1)
trendpoly = np.poly1d(trend) 

plt.plot(ob_cat,trendpoly(ob_cat), color = 'red', linewidth = 2, label = 'R = '+str(np.round(odor_r_odor[animal],decimals = 2)))

plt.xlabel('OB CS1')
#plt.ylabel('PCx CS1')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend(fontsize = 8, loc = 'lower right')

plt.xlim([-3,6])
plt.ylim([-4.5,4])
plt.hlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.vlines(0,-6,6,linestyles = 'dashed', color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.savefig('corr_comp.pdf')
#%%

# #%% check corr and directionalty in time

from pyinform.mutualinfo import mutual_info


t_ob_pc_proj_odor_time_animals = []
t_pc_ob_proj_odor_time_animals = []
t_ob_pc_proj_odorless_time_animals = []
t_pc_ob_proj_odorless_time_animals = []

odor_t_ob_pc_proj_odor_time_animals = []
odor_t_pc_ob_proj_odor_time_animals = []
odor_t_ob_pc_proj_odorless_time_animals = []
odor_t_pc_ob_proj_odorless_time_animals = []

r_odorless_time_animals = []
r_odor_time_animals = []
odor_r_odorless_time_animals = []
odor_r_odor_time_animals = []

window = 200

for x in range(13):
    
    t_ob_pc_proj_odor_time = []
    t_pc_ob_proj_odor_time = []
    t_ob_pc_proj_odorless_time = []
    t_pc_ob_proj_odorless_time = []
    
    odor_t_ob_pc_proj_odor_time = []
    odor_t_pc_ob_proj_odor_time = []
    odor_t_ob_pc_proj_odorless_time = []
    odor_t_pc_ob_proj_odorless_time = []
    
    
    r_odorless_time = []
    r_odor_time = []
    odor_r_odorless_time = []
    odor_r_odor_time = []
    
    
    times = np.arange(1600,3000,20)
    
    for time_start in times:
        
        # spont cs
        time_end = int(time_start+window)
        
        ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odorless_animals[x])[:,time_start:time_end],axis = 0)
        pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odorless_animals[x])[:,time_start:time_end],axis = 0)
        
        r_odorless_time.append(stats.pearsonr(ob_cat,pc_cat)[0])
        
        t_ob_pc_proj_odorless_time.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
        t_pc_ob_proj_odorless_time.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
        
        ob_cat = np.concatenate(np.array(spontaneous_proj_trig_ob_odor_animals[x])[:,time_start:time_end],axis = 0)
        pc_cat = np.concatenate(np.array(spontaneous_proj_trig_pc_odor_animals[x])[:,time_start:time_end],axis = 0)
        
        r_odor_time.append(stats.pearsonr(ob_cat,pc_cat)[0])
        
        t_ob_pc_proj_odor_time.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
        t_pc_ob_proj_odor_time.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
        
        
        # odor cs
        time_end = int(time_start+window)
        
        ob_cat = np.concatenate(np.array(proj_trig_ob_odorless_animals[x])[:,time_start:time_end],axis = 0)
        pc_cat = np.concatenate(np.array(proj_trig_pc_odorless_animals[x])[:,time_start:time_end],axis = 0)
        
        odor_r_odorless_time.append(stats.pearsonr(ob_cat,pc_cat)[0])
        
        odor_t_ob_pc_proj_odorless_time.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
        odor_t_pc_ob_proj_odorless_time.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
        
        ob_cat = np.concatenate(np.array(proj_trig_ob_odor_animals[x])[:,time_start:time_end],axis = 0)
        pc_cat = np.concatenate(np.array(proj_trig_pc_odor_animals[x])[:,time_start:time_end],axis = 0)
        
        odor_r_odor_time.append(stats.pearsonr(ob_cat,pc_cat)[0])
        
        odor_t_ob_pc_proj_odor_time.append(transfer_entropy(np.squeeze(ob_cat)>np.mean(ob_cat), np.squeeze(pc_cat)>np.mean(pc_cat),k = 20))
        odor_t_pc_ob_proj_odor_time.append(transfer_entropy(np.squeeze(pc_cat)>np.mean(pc_cat), np.squeeze(ob_cat)>np.mean(ob_cat),k = 20))
        


    t_ob_pc_proj_odor_time_animals.append(t_ob_pc_proj_odor_time)
    t_pc_ob_proj_odor_time_animals.append(t_pc_ob_proj_odor_time)
    t_ob_pc_proj_odorless_time_animals.append(t_ob_pc_proj_odorless_time)
    t_pc_ob_proj_odorless_time_animals.append(t_pc_ob_proj_odorless_time)
    
    odor_t_ob_pc_proj_odor_time_animals.append(odor_t_ob_pc_proj_odor_time)
    odor_t_pc_ob_proj_odor_time_animals.append(odor_t_pc_ob_proj_odor_time)
    odor_t_ob_pc_proj_odorless_time_animals.append(odor_t_ob_pc_proj_odorless_time)
    odor_t_pc_ob_proj_odorless_time_animals.append(odor_t_pc_ob_proj_odorless_time)
    
    
    r_odorless_time_animals.append(r_odorless_time)
    r_odor_time_animals.append(r_odor_time)

    odor_r_odorless_time_animals.append(odor_r_odorless_time)
    odor_r_odor_time_animals.append(odor_r_odor_time)
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.subplot(211)

time_plot = ((times-2000)/2)+window/2

inh_r_odor = np.mean(r_odor_time_animals,axis = 0)
inh_r_odorless = np.mean(r_odorless_time_animals,axis = 0)
error_inh_r_odor = stats.sem(r_odor_time_animals,axis = 0)
error_inh_r_odorless = stats.sem(r_odorless_time_animals,axis = 0)


plt.plot(time_plot,inh_r_odorless, label = 'No Odor', color = 'tab:purple')    
plt.fill_between(time_plot, inh_r_odorless-error_inh_r_odorless,inh_r_odorless+error_inh_r_odorless,alpha = 0.2, color = 'tab:purple')

plt.plot(time_plot,inh_r_odor, label = 'Odor', color = 'tab:green')    
plt.fill_between(time_plot, inh_r_odor-error_inh_r_odor,inh_r_odor+error_inh_r_odor,alpha = 0.2, color = 'tab:green')

plt.xlabel('Time from inhalation start (ms)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('OB-PCx CS correlation')
plt.xlim([-50,400])
plt.legend()

plt.subplot(212)

time_plot = ((times-2000)/2)+window/2

inh_r_odor = np.mean(odor_r_odor_time_animals,axis = 0)
inh_r_odorless = np.mean(odor_r_odorless_time_animals,axis = 0)
error_inh_r_odor = stats.sem(odor_r_odor_time_animals,axis = 0)
error_inh_r_odorless = stats.sem(odor_r_odorless_time_animals,axis = 0)


plt.plot(time_plot,inh_r_odorless, label = 'No Odor', color = 'tab:purple')    
plt.fill_between(time_plot, inh_r_odorless-error_inh_r_odorless,inh_r_odorless+error_inh_r_odorless,alpha = 0.2, color = 'tab:purple')

plt.plot(time_plot,inh_r_odor, label = 'Odor', color = 'tab:green')    
plt.fill_between(time_plot, inh_r_odor-error_inh_r_odor,inh_r_odor+error_inh_r_odor,alpha = 0.2, color = 'tab:green')

plt.xlabel('Time from inhalation start (ms)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('OB-PCx CS correlation')
plt.xlim([-50,400])
plt.legend()

plt.savefig('corr_traces.pdf')
#%%


#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,5))

plt.subplot(221)
plt.boxplot([r_odorless,r_odor], showfliers = False)


for x in range(13):
    plt.plot([1.2,1.8],[r_odorless[x],r_odor[x]], color = 'grey')


t_r,p_r = stats.ttest_rel(r_odorless,r_odor)

plt.xticks(ticks = [1,2], labels = [])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.05,0.55])

plt.ylabel('OB-PCx CS correlation')
plt.title('Spontaneous CS')


plt.subplot(222)
plt.boxplot([odor_r_odorless,odor_r_odor],showfliers = False)


for x in range(13):
    plt.plot([1.2,1.8],[odor_r_odorless[x],odor_r_odor[x]], color = 'grey')


t_odor_r,odor_p_r = stats.ttest_rel(odor_r_odorless,odor_r_odor)

plt.xticks(ticks = [1,2], labels = [])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.title('Odor CS')
plt.ylim([-0.05,0.55])
plt.yticks(np.arange(0,0.6,0.1),labels = [])

plt.tight_layout()

plt.subplot(223)

t_diff_odorless = np.array(t_ob_pc_proj_odorless)-np.array(t_pc_ob_proj_odorless)
t_diff_odor = np.array(t_ob_pc_proj_odor)-np.array(t_pc_ob_proj_odor)

plt.boxplot([t_diff_odorless,t_diff_odor], showfliers = False)

for x in range(13):
    plt.plot([1.2,1.8],[t_diff_odorless[x],t_diff_odor[x]], color = 'grey')

t_t,p_t = stats.ttest_rel(t_diff_odorless,t_diff_odor)

plt.xticks(ticks = [1,2], labels = ['No odor','Odor'])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylabel('TE D (OB - PCx)')
plt.ylim([-0.001,0.002])
plt.yticks(np.arange(-0.001,0.002,0.0005),labels = np.arange(-0.001,0.002,0.0005)*1000)
plt.text(0.9,0.0016,'x10-3')
# plt.title('Spontaneous CS')

plt.subplot(224)

t_diff_odorless = np.array(odor_t_ob_pc_proj_odorless)-np.array(odor_t_pc_ob_proj_odorless)
t_diff_odor = np.array(odor_t_ob_pc_proj_odor)-np.array(odor_t_pc_ob_proj_odor)

plt.boxplot([t_diff_odorless,t_diff_odor], showfliers = False)

for x in range(13):
    plt.plot([1.2,1.8],[t_diff_odorless[x],t_diff_odor[x]], color = 'grey')

t_odor_t,odor_p_t = stats.ttest_rel(t_diff_odorless,t_diff_odor)

plt.xticks(ticks = [1,2], labels = ['No odor','Odor'])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
# plt.title('Odor CS')
plt.yticks(np.arange(-0.001,0.002,0.0005),labels = [])

plt.ylim([-0.001,0.002])
plt.tight_layout()

#plt.savefig('cs_corr_dir_odors.pdf')

#%% %% check concentration dependenent correlation and dir

ob_conc1_mean = []
ob_conc2_mean = []
ob_conc3_mean = []
ob_conc4_mean = []

pc_conc1_mean = []
pc_conc2_mean = []
pc_conc3_mean = []
pc_conc4_mean = []


r_conc1 = []
r_conc2 = []
r_conc3 = []
r_conc4 = []


odor_ob_conc1_mean = []
odor_ob_conc2_mean = []
odor_ob_conc3_mean = []
odor_ob_conc4_mean = []

odor_pc_conc1_mean = []
odor_pc_conc2_mean = []
odor_pc_conc3_mean = []
odor_pc_conc4_mean = []


odor_r_conc1 = []
odor_r_conc2 = []
odor_r_conc3 = []
odor_r_conc4 = []


t_ob_pc_conc1 = []
t_ob_pc_conc2 = []
t_ob_pc_conc3 = []
t_ob_pc_conc4 = []

t_pc_ob_conc1 = []
t_pc_ob_conc2 = []
t_pc_ob_conc3 = []
t_pc_ob_conc4 = []



signs = [1,1,1,1,1,1,1,1,1,1,-1,-1,1]

for x in range(len(spontaneous_proj_trig_ob_concentration_animals)):
    
    
    time_start = 2000
    time_end = 3200
    
    
    ob_conc1_mean.append(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==0]*signs[x])
    ob_conc2_mean.append(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==1]*signs[x])
    ob_conc3_mean.append(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==2]*signs[x])
    ob_conc4_mean.append(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==3]*signs[x])
    
    pc_conc1_mean.append(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==0]*signs[x])
    pc_conc2_mean.append(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==1]*signs[x])
    pc_conc3_mean.append(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==2]*signs[x])
    pc_conc4_mean.append(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==3]*signs[x])
    
    
    ob_conc1 = np.concatenate(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==0][:,time_start:time_end])
    ob_conc2 = np.concatenate(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==1][:,time_start:time_end])
    ob_conc3 = np.concatenate(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==2][:,time_start:time_end])
    ob_conc4 = np.concatenate(np.array(spontaneous_proj_trig_ob_concentration_animals[x])[concentration_animals[x]==3][:,time_start:time_end])
    
    pc_conc1 = np.concatenate(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==0][:,time_start:time_end])
    pc_conc2 = np.concatenate(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==1][:,time_start:time_end])
    pc_conc3 = np.concatenate(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==2][:,time_start:time_end])
    pc_conc4 = np.concatenate(np.array(spontaneous_proj_trig_pc_concentration_animals[x])[concentration_animals[x]==3][:,time_start:time_end])
    
    # check correlation
    r_conc1.append(stats.pearsonr(ob_conc1,pc_conc1)[0])
    r_conc2.append(stats.pearsonr(ob_conc2,pc_conc2)[0])
    r_conc3.append(stats.pearsonr(ob_conc3,pc_conc3)[0])
    r_conc4.append(stats.pearsonr(ob_conc4,pc_conc4)[0])
    
    # check odor defined CS
    
    odor_ob_conc1_mean.append(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==0]*signs[x])
    odor_ob_conc2_mean.append(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==1]*signs[x])
    odor_ob_conc3_mean.append(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==2]*signs[x])
    odor_ob_conc4_mean.append(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==3]*signs[x])
    
    odor_pc_conc1_mean.append(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==0]*signs[x])
    odor_pc_conc2_mean.append(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==1]*signs[x])
    odor_pc_conc3_mean.append(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==2]*signs[x])
    odor_pc_conc4_mean.append(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==3]*signs[x])
    
    
    ob_conc1 = np.concatenate(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==0][:,time_start:time_end])
    ob_conc2 = np.concatenate(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==1][:,time_start:time_end])
    ob_conc3 = np.concatenate(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==2][:,time_start:time_end])
    ob_conc4 = np.concatenate(np.array(proj_trig_ob_concentration_animals[x])[concentration_animals[x]==3][:,time_start:time_end])
    
    pc_conc1 = np.concatenate(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==0][:,time_start:time_end])
    pc_conc2 = np.concatenate(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==1][:,time_start:time_end])
    pc_conc3 = np.concatenate(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==2][:,time_start:time_end])
    pc_conc4 = np.concatenate(np.array(proj_trig_pc_concentration_animals[x])[concentration_animals[x]==3][:,time_start:time_end])
    
    # check correlation
    odor_r_conc1.append(stats.pearsonr(ob_conc1,pc_conc1)[0])
    odor_r_conc2.append(stats.pearsonr(ob_conc2,pc_conc2)[0])
    odor_r_conc3.append(stats.pearsonr(ob_conc3,pc_conc3)[0])
    odor_r_conc4.append(stats.pearsonr(ob_conc4,pc_conc4)[0])
    
    
    # # check directionality
    
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,5))


gs = gridspec.GridSpec(2, 2, width_ratios = [2,1])

time = np.linspace(-1000,1000,4000)

ob_conc1 = np.mean(np.concatenate(ob_conc1_mean,axis = 0),axis = 0)
error_ob_conc1 = stats.sem(np.concatenate(ob_conc1_mean,axis = 0),axis = 0)
ob_conc2 = np.mean(np.concatenate(ob_conc2_mean,axis = 0),axis = 0)
error_ob_conc2 = stats.sem(np.concatenate(ob_conc2_mean,axis = 0),axis = 0)
ob_conc3 = np.mean(np.concatenate(ob_conc3_mean,axis = 0),axis = 0)
error_ob_conc3 = stats.sem(np.concatenate(ob_conc3_mean,axis = 0),axis = 0)
ob_conc4 = np.mean(np.concatenate(ob_conc4_mean,axis = 0),axis = 0)
error_ob_conc4 = stats.sem(np.concatenate(ob_conc4_mean,axis = 0),axis = 0)


pc_conc1 = np.mean(np.concatenate(pc_conc1_mean,axis = 0),axis = 0)
error_pc_conc1 = stats.sem(np.concatenate(pc_conc1_mean,axis = 0),axis = 0)
pc_conc2 = np.mean(np.concatenate(pc_conc2_mean,axis = 0),axis = 0)
error_pc_conc2 = stats.sem(np.concatenate(pc_conc2_mean,axis = 0),axis = 0)
pc_conc3 = np.mean(np.concatenate(pc_conc3_mean,axis = 0),axis = 0)
error_pc_conc3 = stats.sem(np.concatenate(pc_conc3_mean,axis = 0),axis = 0)
pc_conc4 = np.mean(np.concatenate(pc_conc4_mean,axis = 0),axis = 0)
error_pc_conc4 = stats.sem(np.concatenate(pc_conc4_mean,axis = 0),axis = 0)


plt.subplot(gs[0])

import cycler
n = 6
color1 = plt.cm.Oranges(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[2:]))

plt.plot(time, ob_conc1, label = '0.01 v./v.')
plt.fill_between(time, ob_conc1-error_ob_conc1,ob_conc1+error_ob_conc1, alpha = 0.2)

plt.plot(time, ob_conc2, label = '0.1 v./v.')
plt.fill_between(time, ob_conc2-error_ob_conc2,ob_conc2+error_ob_conc2, alpha = 0.2)

plt.plot(time, ob_conc3, label = '0.3 v./v.')
plt.fill_between(time, ob_conc3-error_ob_conc3,ob_conc3+error_ob_conc3, alpha = 0.2)

plt.plot(time, ob_conc4, label = '1.0 v./v.')
plt.fill_between(time, ob_conc4-error_ob_conc4,ob_conc4+error_ob_conc4, alpha = 0.2)

plt.ylim([-1,2])
plt.xlim([-100,600])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('OB CS1 Activity (z)')
plt.legend()


plt.subplot(gs[2])

import cycler
n = 6
color1 = plt.cm.Blues(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[2:]))

plt.plot(time, pc_conc1, label = '0.01 v./v.')
plt.fill_between(time, pc_conc1-error_pc_conc1,pc_conc1+error_pc_conc1, alpha = 0.2)

plt.plot(time, pc_conc2, label = '0.1 v./v.')
plt.fill_between(time, pc_conc2-error_pc_conc2,pc_conc2+error_pc_conc2, alpha = 0.2)

plt.plot(time, pc_conc3, label = '0.3 v./v.')
plt.fill_between(time, pc_conc3-error_pc_conc3,pc_conc3+error_pc_conc3, alpha = 0.2)

plt.plot(time, pc_conc4, label = '1.0 v./v.')
plt.fill_between(time, pc_conc4-error_pc_conc4,pc_conc4+error_pc_conc4, alpha = 0.2)

plt.ylim([-1,2])
plt.xlim([-100,600])

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xlabel('Time from inhalation start (ms)')
plt.ylabel('PCx CS1 Activity (z)')
plt.legend()

plt.subplot(gs[1])


mean_ob_conc1_animals = []
mean_ob_conc2_animals = []
mean_ob_conc3_animals = []
mean_ob_conc4_animals = []

mean_pc_conc1_animals = []
mean_pc_conc2_animals = []
mean_pc_conc3_animals = []
mean_pc_conc4_animals = []

for x in range(5):
    
    mean_ob_conc1_animals.append(np.mean(np.mean(ob_conc1_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc2_animals.append(np.mean(np.mean(ob_conc2_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc3_animals.append(np.mean(np.mean(ob_conc3_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc4_animals.append(np.mean(np.mean(ob_conc4_mean[x][:,2000:3200],axis = 1)))
    
 
    mean_pc_conc1_animals.append(np.mean(np.mean(pc_conc1_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc2_animals.append(np.mean(np.mean(pc_conc2_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc3_animals.append(np.mean(np.mean(pc_conc3_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc4_animals.append(np.mean(np.mean(pc_conc4_mean[x][:,2000:3200],axis = 1)))
    
    
mean_conc1_initial_mean_proj = np.mean(mean_ob_conc1_animals,axis = 0)
mean_conc2_initial_mean_proj = np.mean(mean_ob_conc2_animals,axis = 0)
mean_conc3_initial_mean_proj = np.mean(mean_ob_conc3_animals,axis = 0)
mean_conc4_initial_mean_proj = np.mean(mean_ob_conc4_animals,axis = 0)

error_conc1_initial_mean_proj = stats.sem(mean_ob_conc1_animals,axis = 0)
error_conc2_initial_mean_proj = stats.sem(mean_ob_conc2_animals,axis = 0)
error_conc3_initial_mean_proj = stats.sem(mean_ob_conc3_animals,axis = 0)
error_conc4_initial_mean_proj = stats.sem(mean_ob_conc4_animals,axis = 0)

from statsmodels.stats.anova import AnovaRM

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([mean_ob_conc1_animals,mean_ob_conc2_animals,mean_ob_conc3_animals,mean_ob_conc4_animals])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova_ob_activity = np.array(anova_table)[-1][-1]


plt.errorbar(np.arange(0,4),[mean_conc1_initial_mean_proj,mean_conc2_initial_mean_proj,mean_conc3_initial_mean_proj,mean_conc4_initial_mean_proj],[error_conc1_initial_mean_proj,error_conc2_initial_mean_proj,error_conc3_initial_mean_proj,error_conc4_initial_mean_proj],fmt = 'o', label = 'OB P = '+str(np.round(p_rmanova_ob_activity,decimals = 2)), color = 'tab:orange')


mean_conc1_initial_mean_proj = np.mean(mean_pc_conc1_animals,axis = 0)
mean_conc2_initial_mean_proj = np.mean(mean_pc_conc2_animals,axis = 0)
mean_conc3_initial_mean_proj = np.mean(mean_pc_conc3_animals,axis = 0)
mean_conc4_initial_mean_proj = np.mean(mean_pc_conc4_animals,axis = 0)

error_conc1_initial_mean_proj = stats.sem(mean_pc_conc1_animals,axis = 0)
error_conc2_initial_mean_proj = stats.sem(mean_pc_conc2_animals,axis = 0)
error_conc3_initial_mean_proj = stats.sem(mean_pc_conc3_animals,axis = 0)
error_conc4_initial_mean_proj = stats.sem(mean_pc_conc4_animals,axis = 0)

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([mean_pc_conc1_animals,mean_pc_conc2_animals,mean_pc_conc3_animals,mean_pc_conc4_animals])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova_pc_activity = np.array(anova_table)[-1][-1]


plt.errorbar(np.arange(0,4),[mean_conc1_initial_mean_proj,mean_conc2_initial_mean_proj,mean_conc3_initial_mean_proj,mean_conc4_initial_mean_proj],[error_conc1_initial_mean_proj,error_conc2_initial_mean_proj,error_conc3_initial_mean_proj,error_conc4_initial_mean_proj],fmt = 'o', label = 'PCx P = '+str(np.round(p_rmanova_pc_activity,decimals = 2)), color = 'tab:blue')

plt.ylabel('CS Actvity')
plt.xlabel('Odor Concentrarion')

plt.xticks(np.arange(0,4), labels = [0.01,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.legend(fontsize = 8)

plt.ylim([-0.5,1])
#plt.ylim([-1,1])


plt.subplot(gs[3])

mean_conc1 = np.mean(r_conc1)
mean_conc2 = np.mean(r_conc2)
mean_conc3 = np.mean(r_conc3)
mean_conc4 = np.mean(r_conc4)

correlation_mean_conc = np.hstack([mean_conc1,mean_conc2,mean_conc3,mean_conc4])

error_conc1 = stats.sem(r_conc1)
error_conc2 = stats.sem(r_conc2)
error_conc3 = stats.sem(r_conc3)
error_conc4 = stats.sem(r_conc4)

correlation_error_conc = np.hstack([error_conc1,error_conc2,error_conc3,error_conc4])

from statsmodels.stats.anova import AnovaRM

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([np.arctanh(r_conc1),np.arctanh(r_conc2),np.arctanh(r_conc3),np.arctanh(r_conc4)])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova = np.array(anova_table)[-1][-1]

plt.errorbar([1,2,3,4], correlation_mean_conc, correlation_error_conc,fmt = 'o', color = 'tab:purple', label = 'P = '+str(np.round(p_rmanova,decimals = 2)))
plt.ylabel('OB-PCx CS Correlation')
plt.xlabel('Odor Concentrarion')

plt.xticks([1,2,3,4], labels = [0.01,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend(fontsize = 8)

plt.ylim([-0.05,0.55])

plt.tight_layout()

plt.savefig('cs_conc_spont.pdf')

#%% odor defined cs

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,5))


gs = gridspec.GridSpec(2, 2, width_ratios = [2,1])


ob_conc1 = np.mean(np.concatenate(odor_ob_conc1_mean,axis = 0),axis = 0)
error_ob_conc1 = stats.sem(np.concatenate(odor_ob_conc1_mean,axis = 0),axis = 0)
ob_conc2 = np.mean(np.concatenate(odor_ob_conc2_mean,axis = 0),axis = 0)
error_ob_conc2 = stats.sem(np.concatenate(odor_ob_conc2_mean,axis = 0),axis = 0)
ob_conc3 = np.mean(np.concatenate(odor_ob_conc3_mean,axis = 0),axis = 0)
error_ob_conc3 = stats.sem(np.concatenate(odor_ob_conc3_mean,axis = 0),axis = 0)
ob_conc4 = np.mean(np.concatenate(odor_ob_conc4_mean,axis = 0),axis = 0)
error_ob_conc4 = stats.sem(np.concatenate(odor_ob_conc4_mean,axis = 0),axis = 0)


pc_conc1 = np.mean(np.concatenate(odor_pc_conc1_mean,axis = 0),axis = 0)
error_pc_conc1 = stats.sem(np.concatenate(odor_pc_conc1_mean,axis = 0),axis = 0)
pc_conc2 = np.mean(np.concatenate(odor_pc_conc2_mean,axis = 0),axis = 0)
error_pc_conc2 = stats.sem(np.concatenate(odor_pc_conc2_mean,axis = 0),axis = 0)
pc_conc3 = np.mean(np.concatenate(odor_pc_conc3_mean,axis = 0),axis = 0)
error_pc_conc3 = stats.sem(np.concatenate(odor_pc_conc3_mean,axis = 0),axis = 0)
pc_conc4 = np.mean(np.concatenate(odor_pc_conc4_mean,axis = 0),axis = 0)
error_pc_conc4 = stats.sem(np.concatenate(odor_pc_conc4_mean,axis = 0),axis = 0)


plt.subplot(gs[0])

import cycler
n = 6
color1 = plt.cm.Oranges(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[2:]))

plt.plot(time, ob_conc1, label = '0.01 v./v.')
plt.fill_between(time, ob_conc1-error_ob_conc1,ob_conc1+error_ob_conc1, alpha = 0.2)

plt.plot(time, ob_conc2, label = '0.1 v./v.')
plt.fill_between(time, ob_conc2-error_ob_conc2,ob_conc2+error_ob_conc2, alpha = 0.2)

plt.plot(time, ob_conc3, label = '0.3 v./v.')
plt.fill_between(time, ob_conc3-error_ob_conc3,ob_conc3+error_ob_conc3, alpha = 0.2)

plt.plot(time, ob_conc4, label = '1.0 v./v.')
plt.fill_between(time, ob_conc4-error_ob_conc4,ob_conc4+error_ob_conc4, alpha = 0.2)

plt.ylim([-1,3])
plt.xlim([-100,600])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('OB CS1 Activity (z)')
plt.legend()


plt.subplot(gs[2])

import cycler
n = 6
color1 = plt.cm.Blues(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[2:]))

plt.plot(time, pc_conc1, label = '0.01 v./v.')
plt.fill_between(time, pc_conc1-error_pc_conc1,pc_conc1+error_pc_conc1, alpha = 0.2)

plt.plot(time, pc_conc2, label = '0.1 v./v.')
plt.fill_between(time, pc_conc2-error_pc_conc2,pc_conc2+error_pc_conc2, alpha = 0.2)

plt.plot(time, pc_conc3, label = '0.3 v./v.')
plt.fill_between(time, pc_conc3-error_pc_conc3,pc_conc3+error_pc_conc3, alpha = 0.2)

plt.plot(time, pc_conc4, label = '1.0 v./v.')
plt.fill_between(time, pc_conc4-error_pc_conc4,pc_conc4+error_pc_conc4, alpha = 0.2)

plt.ylim([-1,3])
plt.xlim([-100,600])

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xlabel('Time from inhalation start (ms)')
plt.ylabel('PCx CS1 Activity (z)')
#plt.legend()

plt.subplot(gs[1])


mean_ob_conc1_animals = []
mean_ob_conc2_animals = []
mean_ob_conc3_animals = []
mean_ob_conc4_animals = []

mean_pc_conc1_animals = []
mean_pc_conc2_animals = []
mean_pc_conc3_animals = []
mean_pc_conc4_animals = []

for x in range(5):
    
    mean_ob_conc1_animals.append(np.mean(np.mean(odor_ob_conc1_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc2_animals.append(np.mean(np.mean(odor_ob_conc2_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc3_animals.append(np.mean(np.mean(odor_ob_conc3_mean[x][:,2000:3200],axis = 1)))
    mean_ob_conc4_animals.append(np.mean(np.mean(odor_ob_conc4_mean[x][:,2000:3200],axis = 1)))
    
 
    mean_pc_conc1_animals.append(np.mean(np.mean(odor_pc_conc1_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc2_animals.append(np.mean(np.mean(odor_pc_conc2_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc3_animals.append(np.mean(np.mean(odor_pc_conc3_mean[x][:,2000:3200],axis = 1)))
    mean_pc_conc4_animals.append(np.mean(np.mean(odor_pc_conc4_mean[x][:,2000:3200],axis = 1)))
    
    
mean_conc1_initial_mean_proj = np.mean(mean_ob_conc1_animals,axis = 0)
mean_conc2_initial_mean_proj = np.mean(mean_ob_conc2_animals,axis = 0)
mean_conc3_initial_mean_proj = np.mean(mean_ob_conc3_animals,axis = 0)
mean_conc4_initial_mean_proj = np.mean(mean_ob_conc4_animals,axis = 0)

error_conc1_initial_mean_proj = stats.sem(mean_ob_conc1_animals,axis = 0)
error_conc2_initial_mean_proj = stats.sem(mean_ob_conc2_animals,axis = 0)
error_conc3_initial_mean_proj = stats.sem(mean_ob_conc3_animals,axis = 0)
error_conc4_initial_mean_proj = stats.sem(mean_ob_conc4_animals,axis = 0)

from statsmodels.stats.anova import AnovaRM

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([mean_ob_conc1_animals,mean_ob_conc2_animals,mean_ob_conc3_animals,mean_ob_conc4_animals])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova_ob_activity = np.array(anova_table)[-1][-1]


plt.errorbar(np.arange(0,4),[mean_conc1_initial_mean_proj,mean_conc2_initial_mean_proj,mean_conc3_initial_mean_proj,mean_conc4_initial_mean_proj],[error_conc1_initial_mean_proj,error_conc2_initial_mean_proj,error_conc3_initial_mean_proj,error_conc4_initial_mean_proj],fmt = 'o', label = 'OB P = '+str(np.round(p_rmanova_ob_activity,decimals = 2)), color = 'tab:orange')

mean_conc1_initial_mean_proj = np.mean(mean_pc_conc1_animals,axis = 0)
mean_conc2_initial_mean_proj = np.mean(mean_pc_conc2_animals,axis = 0)
mean_conc3_initial_mean_proj = np.mean(mean_pc_conc3_animals,axis = 0)
mean_conc4_initial_mean_proj = np.mean(mean_pc_conc4_animals,axis = 0)

error_conc1_initial_mean_proj = stats.sem(mean_pc_conc1_animals,axis = 0)
error_conc2_initial_mean_proj = stats.sem(mean_pc_conc2_animals,axis = 0)
error_conc3_initial_mean_proj = stats.sem(mean_pc_conc3_animals,axis = 0)
error_conc4_initial_mean_proj = stats.sem(mean_pc_conc4_animals,axis = 0)

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([mean_pc_conc1_animals,mean_pc_conc2_animals,mean_pc_conc3_animals,mean_pc_conc4_animals])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova_pc_activity = np.array(anova_table)[-1][-1]


plt.errorbar(np.arange(0,4),[mean_conc1_initial_mean_proj,mean_conc2_initial_mean_proj,mean_conc3_initial_mean_proj,mean_conc4_initial_mean_proj],[error_conc1_initial_mean_proj,error_conc2_initial_mean_proj,error_conc3_initial_mean_proj,error_conc4_initial_mean_proj],fmt = 'o', label = 'PCx P = '+str(np.round(p_rmanova_pc_activity,decimals = 2)), color = 'tab:blue')

plt.ylabel('CS Actvity')
plt.xlabel('Odor Concentrarion')

plt.xticks(np.arange(0,4), labels = [0.01,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.legend(fontsize = 8)

#plt.ylim([1,8])
plt.ylim([-0.5,1])


plt.subplot(gs[3])

mean_ob_conc1_animals = []
mean_ob_conc2_animals = []
mean_ob_conc3_animals = []
mean_ob_conc4_animals = []

mean_pc_conc1_animals = []
mean_pc_conc2_animals = []
mean_pc_conc3_animals = []
mean_pc_conc4_animals = []

for x in range(5):
    
    mean_ob_conc1_animals.append((np.mean(np.argmax(odor_ob_conc1_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_ob_conc2_animals.append((np.mean(np.argmax(odor_ob_conc2_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_ob_conc3_animals.append((np.mean(np.argmax(odor_ob_conc3_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_ob_conc4_animals.append((np.mean(np.argmax(odor_ob_conc4_mean[x][:,2100:2800],axis = 1))+200)/2)
    
 
    mean_pc_conc1_animals.append((np.mean(np.argmax(odor_pc_conc1_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_pc_conc2_animals.append((np.mean(np.argmax(odor_pc_conc2_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_pc_conc3_animals.append((np.mean(np.argmax(odor_pc_conc3_mean[x][:,2100:2800],axis = 1))+200)/2)
    mean_pc_conc4_animals.append((np.mean(np.argmax(odor_pc_conc4_mean[x][:,2100:2800],axis = 1))+200)/2)
    
    
mean_conc1_initial_mean_proj = np.mean(mean_ob_conc1_animals,axis = 0)
mean_conc2_initial_mean_proj = np.mean(mean_ob_conc2_animals,axis = 0)
mean_conc3_initial_mean_proj = np.mean(mean_ob_conc3_animals,axis = 0)
mean_conc4_initial_mean_proj = np.mean(mean_ob_conc4_animals,axis = 0)

error_conc1_initial_mean_proj = stats.sem(mean_ob_conc1_animals,axis = 0)
error_conc2_initial_mean_proj = stats.sem(mean_ob_conc2_animals,axis = 0)
error_conc3_initial_mean_proj = stats.sem(mean_ob_conc3_animals,axis = 0)
error_conc4_initial_mean_proj = stats.sem(mean_ob_conc4_animals,axis = 0)

from statsmodels.stats.anova import AnovaRM

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([mean_ob_conc1_animals,mean_ob_conc2_animals,mean_ob_conc3_animals,mean_ob_conc4_animals])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova_ob_activity = np.array(anova_table)[-1][-1]


mean_conc1 = np.mean(odor_r_conc1)
mean_conc2 = np.mean(odor_r_conc2)
mean_conc3 = np.mean(odor_r_conc3)
mean_conc4 = np.mean(odor_r_conc4)

correlation_mean_conc = np.hstack([mean_conc1,mean_conc2,mean_conc3,mean_conc4])

error_conc1 = stats.sem(odor_r_conc1)
error_conc2 = stats.sem(odor_r_conc2)
error_conc3 = stats.sem(odor_r_conc3)
error_conc4 = stats.sem(odor_r_conc4)

correlation_error_conc = np.hstack([error_conc1,error_conc2,error_conc3,error_conc4])

from statsmodels.stats.anova import AnovaRM

xdata = np.repeat([0.03,0.1,0.3,1],5)
animal = np.tile([1,2,3,4,5],4)

ydata = np.hstack([np.arctanh(odor_r_conc1),np.arctanh(odor_r_conc2),np.arctanh(odor_r_conc3),np.arctanh(odor_r_conc4)])
df = pd.DataFrame({'animal': animal,'conc': xdata,'amp': ydata})
anova_table = AnovaRM(data=df, depvar='amp', subject='animal', within=['conc']).fit().anova_table
p_rmanova = np.array(anova_table)[-1][-1]

plt.errorbar([1,2,3,4], correlation_mean_conc, correlation_error_conc,fmt = 'o', color = 'tab:purple', label = 'P = '+str(np.round(p_rmanova,decimals = 2)))
plt.ylabel('OB-PCx CS Correlation')
plt.xlabel('Odor Concentrarion')

plt.xticks([1,2,3,4], labels = [0.01,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend(fontsize = 8)

plt.ylim([-0.05,0.55])

plt.tight_layout()

plt.savefig('cs_conc_odors.pdf')

#%%

#%%

cs_pair = 0

r_pc,p_pc = stats.pearsonr(np.squeeze(np.concatenate(pc_loadings_animals_odorless)[:,cs_pair]),np.squeeze(np.concatenate(pc_loadings_animals_odor)[:,cs_pair]))
r_ob,p_ob = stats.pearsonr(np.squeeze(np.concatenate(ob_loadings_animals_odorless)[:,cs_pair]),np.squeeze(np.concatenate(ob_loadings_animals_odor)[:,cs_pair]))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,3))

#plt.subplot(212)
sns.regplot(np.concatenate(pc_loadings_animals_odorless)[:,cs_pair],np.concatenate(pc_loadings_animals_odor)[:,cs_pair],x_bins = np.linspace(-1,1,20), label = 'PCx '+'r = '+str(np.round(r_pc,decimals = 2)))
#plt.scatter(np.concatenate(pc_loadings_odorless_animals),np.concatenate(pc_loadings_odor_animals),s = 2,)

sns.regplot(np.concatenate(ob_loadings_animals_odorless)[:,cs_pair],np.concatenate(ob_loadings_animals_odor)[:,cs_pair],x_bins = np.linspace(-1,1,20),  label = 'OB '+'r = '+str(np.round(r_ob,decimals = 2)))
#plt.scatter(np.concatenate(ob_loadings_odorless_animals),np.concatenate(ob_loadings_odor_animals),s = 2)

plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.linspace(-1,1,10),np.zeros(10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.zeros(10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend(fontsize = 6)
plt.ylabel('CS1 Weights (Odor)')
plt.xlabel('CS1 Weights (No Odor)')

#%%

cs_pair = 0

r_pc,p_pc = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_animals_odorless))[:,cs_pair]),np.abs(np.squeeze(np.concatenate(pc_loadings_animals_odor))[:,cs_pair]))
r_ob,p_ob = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_animals_odorless))[:,cs_pair]),np.abs(np.squeeze(np.concatenate(ob_loadings_animals_odor))[:,cs_pair]))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,3))

#plt.subplot(212)
sns.regplot(np.abs(np.concatenate(pc_loadings_animals_odorless))[:,cs_pair],np.abs(np.concatenate(pc_loadings_animals_odor))[:,cs_pair],x_bins = np.linspace(0,1,10), label = 'PCx '+'r = '+str(np.round(r_pc,decimals = 2)))
#plt.scatter(np.concatenate(pc_loadings_odorless_animals),np.concatenate(pc_loadings_odor_animals),s = 2,)

sns.regplot(np.abs(np.concatenate(ob_loadings_animals_odorless))[:,cs_pair],np.abs(np.concatenate(ob_loadings_animals_odor))[:,cs_pair],x_bins = np.linspace(0,1,10),  label = 'OB '+'r = '+str(np.round(r_ob,decimals = 2)))
#plt.scatter(np.concatenate(ob_loadings_odorless_animals),np.concatenate(ob_loadings_odor_animals),s = 2)

plt.plot(np.linspace(0,1,10),np.linspace(0,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.linspace(0,1,10),np.zeros(10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.zeros(10),np.linspace(0,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend(fontsize = 6)
plt.ylabel('CS1 Weights (Odor)')
plt.xlabel('CS1 Weights (No Odor)')

#%%
ob_angle = []
pc_angle = []

for x in range(13):

    ob_angle.append(scipy.linalg.subspace_angles(ob_loadings_animals_odor[x][:,0:1],ob_loadings_animals_odorless[x][:,0:1]))
    pc_angle.append(scipy.linalg.subspace_angles(pc_loadings_animals_odor[x][:,0:1],pc_loadings_animals_odorless[x][:,0:1]))

ax = plt.subplot(111,polar = True)

ax.hist(np.squeeze(pc_angle),bins = np.linspace(-1*np.pi, np.pi,50))
ax.hist(np.squeeze(ob_angle),bins = np.linspace(-1*np.pi, np.pi,50))


#%%

x = 12

sns.regplot(ob_loadings_animals_odor[x][:,0:1],ob_loadings_animals_odorless[x][:,0:1])

#%%


import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


ob_angle = []
pc_angle = []


for x in range(13):

    ob_angle.append(angle(ob_loadings_animals_odor[x][:,0:1],ob_loadings_animals_odorless[x][:,0:1]))
    pc_angle.append(angle(pc_loadings_animals_odor[x][:,0:1],pc_loadings_animals_odorless[x][:,0:1]))
  

#
ax = plt.subplot(111,polar = True)


ax.hist(np.squeeze(pc_angle),bins = np.linspace(-1*np.pi, np.pi,30))
ax.hist(np.squeeze(ob_angle),bins = np.linspace(-1*np.pi, np.pi,30))

#%%

ob_r2 = []
pc_r2 = []


for x in range(13):

    ob_r2.append(stats.pearsonr(np.squeeze(ob_loadings_animals_odor[x][:,0:1]),np.squeeze(ob_loadings_animals_odorless[x][:,0:1]))[0]**2)
    pc_r2.append(stats.pearsonr(np.squeeze(pc_loadings_animals_odor[x][:,0:1]),np.squeeze(pc_loadings_animals_odorless[x][:,0:1]))[0]**2)
  


#%% decode odor identity from CS activity

odor_accuracy_knn_ob_dimensions_animals = []
odor_accuracy_knn_pc_dimensions_animals = []


for animal in range(13):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 1000
    
    odorants_sorted = odorants_animals[animal]

    
    
    smells_trig_ob_cat = np.concatenate(smells_trig_ob_odor_animals[animal][:,:,0:time].T,axis = 1)
    smells_trig_pc_cat = np.concatenate(smells_trig_pc_odor_animals[animal][:,:,0:time].T,axis = 1)
    

    n_components = 15
    
    
    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_odor_animals[animal][:,:,0:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_odor_animals[animal][:,:,0:time],axis = -1)
    



    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_mua = []
        accuracy_knn_pc_mua = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
          
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    odor_accuracy_knn_ob_dimensions_animals.append(accuracy_knn_ob_dimensions)
    odor_accuracy_knn_pc_dimensions_animals.append(accuracy_knn_pc_dimensions)
    

# decode odor identity from spontanous CS activity

spont_odor_accuracy_knn_ob_dimensions_animals = []
spont_odor_accuracy_knn_pc_dimensions_animals = []


for animal in range(13):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 1000
    
    odorants_sorted = odorants_animals[animal]

    
    
    smells_trig_ob_cat = np.concatenate(smells_trig_ob_odor_animals[animal][:,:,0:time].T,axis = 1)
    smells_trig_pc_cat = np.concatenate(smells_trig_pc_odor_animals[animal][:,:,0:time].T,axis = 1)
    

    n_components = 15
    
    
    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_odor_animals[animal][:,:,0:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_odor_animals[animal][:,:,0:time],axis = -1)
    



    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_mua = []
        accuracy_knn_pc_mua = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
          
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    spont_odor_accuracy_knn_ob_dimensions_animals.append(accuracy_knn_ob_dimensions)
    spont_odor_accuracy_knn_pc_dimensions_animals.append(accuracy_knn_pc_dimensions)


    
#%% check concentration

concentration_accuracy_knn_ob_dimensions_animals = []
concentration_accuracy_knn_pc_dimensions_animals = []

concentration_accuracy_knn_ob_dimensions_animals_shuffled = []
concentration_accuracy_knn_pc_dimensions_animals_shuffled = []

concentration_accuracy_knn_ob_units_animals = []
concentration_accuracy_knn_pc_units_animals = []

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals[animal]

    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000, tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000, tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals.append(accuracy_knn_pc_dimensions)
    

#

spont_concentration_accuracy_knn_ob_dimensions_animals = []
spont_concentration_accuracy_knn_pc_dimensions_animals = []

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals[animal]

    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000, tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000, tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    spont_concentration_accuracy_knn_ob_dimensions_animals.append(accuracy_knn_ob_dimensions)
    spont_concentration_accuracy_knn_pc_dimensions_animals.append(accuracy_knn_pc_dimensions)
    


#%% check concentration sinlge odorant 1

concentration_accuracy_knn_ob_dimensions_animals_1 = []
concentration_accuracy_knn_pc_dimensions_animals_1 = []


for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals_1[animal]

    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals_1.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals_1.append(accuracy_knn_pc_dimensions)
    

#

spont_concentration_accuracy_knn_ob_dimensions_animals_1 = []
spont_concentration_accuracy_knn_pc_dimensions_animals_1 = []

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals_1[animal]

    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    spont_concentration_accuracy_knn_ob_dimensions_animals_1.append(accuracy_knn_ob_dimensions)
    spont_concentration_accuracy_knn_pc_dimensions_animals_1.append(accuracy_knn_pc_dimensions)
    
# check concentration sinlge odorant 2

concentration_accuracy_knn_ob_dimensions_animals_2 = []
concentration_accuracy_knn_pc_dimensions_animals_2 = []

concentration_accuracy_knn_ob_dimensions_animals_shuffled = []
concentration_accuracy_knn_pc_dimensions_animals_shuffled = []

concentration_accuracy_knn_ob_units_animals = []
concentration_accuracy_knn_pc_units_animals = []

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals_2[animal]

    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals_2.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals_2.append(accuracy_knn_pc_dimensions)
    

#

spont_concentration_accuracy_knn_ob_dimensions_animals_2 = []
spont_concentration_accuracy_knn_pc_dimensions_animals_2 = []

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,2000:time],axis = -1)

    odorants_sorted = concentration_animals_2[animal]

    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,16):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        from sklearn.neighbors import KNeighborsClassifier
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-10,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    spont_concentration_accuracy_knn_ob_dimensions_animals_2.append(accuracy_knn_ob_dimensions)
    spont_concentration_accuracy_knn_pc_dimensions_animals_2.append(accuracy_knn_pc_dimensions)
    
#%% compare odor and odorless example decoding

import cycler 

animal = 0
time_start = 0
time_end = 2000

odorants_repeated = odorants_animals[animal]

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



#

scatter_size = 30
plt.figure(dpi = 300, figsize = (10,10))


n = 8


ob_loadings = ob_loadings_animals_odorless[animal]
pc_loadings = pc_loadings_animals_odorless[animal]


odorants_repeated = odorants_animals[animal]

    
smells_trig_ob_cat = np.sum(smells_trig_ob_odor_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_odor_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]



#scatter_size = 30


ax2 = plt.subplot(223, projection = '3d')

color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

for x in np.unique(odorants):
    
    times = odorants_repeated == x
    
    x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], num_std_dev=1.95)

    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size)
    ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

# ax2.view_init(elev= 0., azim =90)
# ax2.view_init(elev= 15., azim = 80)
ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('OB')

ax2.set_xlim([-7000,5000])
ax2.set_ylim([-7000,7000])
ax2.set_zlim([-1000,3000])


ax2 = plt.subplot(224, projection = '3d')

color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

for x in np.unique(odorants):
    
    times = odorants_repeated == x
    
    x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size)
    ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')
#


#
ob_loadings = ob_loadings_animals_odor[animal]
pc_loadings = pc_loadings_animals_odor[animal]


odorants_repeated = odorants_animals[animal]

    
smells_trig_ob_cat = np.sum(smells_trig_ob_odor_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_odor_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]

# add concentration



ax2 = plt.subplot(221, projection = '3d')

color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

for x in np.unique(odorants):
    
    times = odorants_repeated == x
    
    x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], num_std_dev=1.95)

    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size)
    ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

# ax2.view_init(elev= 0., azim =90)
# ax2.view_init(elev= 15., azim = 80)
ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('OB')

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_xlim([-7000,2000])
ax2.set_ylim([-3000,3000])
ax2.set_zlim([-2000,7000])


ax2 = plt.subplot(222, projection = '3d')

color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

for x in np.unique(odorants):
    
    times = odorants_repeated == x
    
    x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size)
    ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


scatter_size = 50
plt.figure(dpi = 300, figsize = (20,5))

time_start = 2000
time_end = 4000

animal = 0


ax2 = plt.subplot(141, projection = '3d')

ob_loadings = ob_loadings_animals_odorless[animal]
pc_loadings = pc_loadings_animals_odorless[animal]

odorants_repeated = concentration_animals_1[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj_conc = []
pc_proj_conc = []


for x in range(n_components):
    
    ob_proj_conc.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj_conc.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj_conc).T
pc_proj = np.array(pc_proj_conc).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]


n = 5
color1 = plt.cm.Purples(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    
    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')



odorants_repeated = concentration_animals_2[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]



n = 5
color1 = plt.cm.Greens(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('OB')


#

ax2 = plt.subplot(142, projection = '3d')


odorants_repeated = concentration_animals_1[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj_conc = []
pc_proj_conc = []


for x in range(n_components):
    
    ob_proj_conc.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj_conc.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj_conc).T
pc_proj = np.array(pc_proj_conc).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]


n = 5
color1 = plt.cm.Purples(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')



odorants_repeated = concentration_animals_2[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]



n = 5
color1 = plt.cm.Greens(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')


ax2 = plt.subplot(143, projection = '3d')

ob_loadings = ob_loadings_animals_odor[animal]
pc_loadings = pc_loadings_animals_odor[animal]

odorants_repeated = concentration_animals_1[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj_conc = []
pc_proj_conc = []


for x in range(n_components):
    
    ob_proj_conc.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj_conc.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj_conc).T
pc_proj = np.array(pc_proj_conc).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]


n = 5
color1 = plt.cm.Purples(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')



odorants_repeated = concentration_animals_2[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]



n = 5
color1 = plt.cm.Greens(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(ob_proj[times,0],ob_proj[times,2],ob_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('OB')


#

ax2 = plt.subplot(144, projection = '3d')


odorants_repeated = concentration_animals_1[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj_conc = []
pc_proj_conc = []


for x in range(n_components):
    
    ob_proj_conc.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj_conc.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj_conc).T
pc_proj = np.array(pc_proj_conc).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]


n = 5
color1 = plt.cm.Purples(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')



odorants_repeated = concentration_animals_2[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,time_start:time_end],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T

#

ob_proj = ob_proj[:,[0,2,4,3]]
pc_proj = pc_proj[:,[0,2,4,3]]



n = 5
color1 = plt.cm.Greens(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[1:,:]))

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    #x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], num_std_dev=1.95)

    ax2.scatter(pc_proj[times,0],pc_proj[times,2],pc_proj[times,1], alpha = 1,s = scatter_size, rasterized = True)
    #ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 80)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_title('PCx')


plt.savefig('odor_clusters_conc.pdf')

#%%
mean_accuracy_knn_pc_dimensions = np.mean(np.array(odor_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)
error_accuracy_knn_pc_dimensions = stats.sem(np.array(odor_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)

mean_accuracy_knn_ob_dimensions = np.mean(np.array(odor_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)
error_accuracy_knn_ob_dimensions = stats.sem(np.array(odor_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)


mean_accuracy_knn_pc_dimensions = mean_accuracy_knn_pc_dimensions-0.16
mean_accuracy_knn_ob_dimensions = mean_accuracy_knn_ob_dimensions-0.16

mean_accuracy_knn_pc_dimensions = mean_accuracy_knn_pc_dimensions/0.84
mean_accuracy_knn_ob_dimensions = mean_accuracy_knn_ob_dimensions/0.84
error_accuracy_knn_pc_dimensions = error_accuracy_knn_pc_dimensions/0.84
error_accuracy_knn_ob_dimensions = error_accuracy_knn_ob_dimensions/0.84


spont_mean_accuracy_knn_pc_dimensions = np.mean(np.array(spont_odor_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)
spont_error_accuracy_knn_pc_dimensions = stats.sem(np.array(spont_odor_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)

spont_mean_accuracy_knn_ob_dimensions = np.mean(np.array(spont_odor_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)
spont_error_accuracy_knn_ob_dimensions = stats.sem(np.array(spont_odor_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)


spont_mean_accuracy_knn_pc_dimensions = spont_mean_accuracy_knn_pc_dimensions-0.16
spont_mean_accuracy_knn_ob_dimensions = spont_mean_accuracy_knn_ob_dimensions-0.16

spont_mean_accuracy_knn_pc_dimensions = spont_mean_accuracy_knn_pc_dimensions/0.84
spont_mean_accuracy_knn_ob_dimensions = spont_mean_accuracy_knn_ob_dimensions/0.84
spont_error_accuracy_knn_pc_dimensions = spont_error_accuracy_knn_pc_dimensions/0.84
spont_error_accuracy_knn_ob_dimensions = spont_error_accuracy_knn_ob_dimensions/0.84


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,6))

plt.subplot(223)

plt.plot(np.arange(1,16),mean_accuracy_knn_pc_dimensions, label = 'Odor CS', color = 'tab:blue')
plt.fill_between(np.arange(1,16), mean_accuracy_knn_pc_dimensions-error_accuracy_knn_pc_dimensions,mean_accuracy_knn_pc_dimensions+error_accuracy_knn_pc_dimensions, alpha = 0.2, color = 'tab:blue')

plt.plot(np.arange(1,16),spont_mean_accuracy_knn_pc_dimensions,'--', label = 'Spont CS', color = 'tab:blue')
plt.fill_between(np.arange(1,16), spont_mean_accuracy_knn_pc_dimensions-spont_error_accuracy_knn_pc_dimensions,spont_mean_accuracy_knn_pc_dimensions+spont_error_accuracy_knn_pc_dimensions, alpha = 0.2, color = 'grey')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#plt.title('PCx')
plt.ylabel('Accuracy Delta \n(% real - % chance)/(100-% chance)', multialignment='center')
plt.xlabel('# CS Dimensions')

plt.yticks(ticks = np.arange(0,1,0.2), labels = np.arange(0,100,20))
plt.xticks(np.arange(2,16,2))

plt.legend(ncol = 1,fontsize = 10)

plt.subplot(221)

plt.plot(np.arange(1,16),mean_accuracy_knn_ob_dimensions, label = 'Odor CS', color = 'tab:orange')
plt.fill_between(np.arange(1,16), mean_accuracy_knn_ob_dimensions-error_accuracy_knn_ob_dimensions,mean_accuracy_knn_ob_dimensions+error_accuracy_knn_ob_dimensions, alpha = 0.2, color = 'tab:orange')


plt.plot(np.arange(1,16),spont_mean_accuracy_knn_ob_dimensions,'--', label = 'Spont CS', color = 'tab:orange')
plt.fill_between(np.arange(1,16), spont_mean_accuracy_knn_ob_dimensions-spont_error_accuracy_knn_ob_dimensions,spont_mean_accuracy_knn_ob_dimensions+spont_error_accuracy_knn_ob_dimensions, alpha = 0.2, color = 'grey')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Accuracy Delta \n(% real - % chance)/(100-% chance)', multialignment='center')

plt.ylim([0,0.7])
plt.yticks(ticks = np.arange(0,1,0.2), labels = np.arange(0,100,20))
plt.xticks(np.arange(2,16,2))



# concentration

mean_accuracy_knn_pc_dimensions = np.mean(np.array(concentration_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)
error_accuracy_knn_pc_dimensions = stats.sem(np.array(concentration_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)

mean_accuracy_knn_ob_dimensions = np.mean(np.array(concentration_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)
error_accuracy_knn_ob_dimensions = stats.sem(np.array(concentration_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)


mean_accuracy_knn_pc_dimensions = mean_accuracy_knn_pc_dimensions-0.25
mean_accuracy_knn_ob_dimensions = mean_accuracy_knn_ob_dimensions-0.25

mean_accuracy_knn_pc_dimensions = mean_accuracy_knn_pc_dimensions/0.75
mean_accuracy_knn_ob_dimensions = mean_accuracy_knn_ob_dimensions/0.75
error_accuracy_knn_pc_dimensions = error_accuracy_knn_pc_dimensions/0.75
error_accuracy_knn_ob_dimensions = error_accuracy_knn_ob_dimensions/0.75


spont_mean_accuracy_knn_pc_dimensions = np.mean(np.array(spont_concentration_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)
spont_error_accuracy_knn_pc_dimensions = stats.sem(np.array(spont_concentration_accuracy_knn_pc_dimensions_animals)[:,:],axis = 0)

spont_mean_accuracy_knn_ob_dimensions = np.mean(np.array(spont_concentration_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)
spont_error_accuracy_knn_ob_dimensions = stats.sem(np.array(spont_concentration_accuracy_knn_ob_dimensions_animals)[:,:],axis = 0)


spont_mean_accuracy_knn_pc_dimensions = spont_mean_accuracy_knn_pc_dimensions-0.25
spont_mean_accuracy_knn_ob_dimensions = spont_mean_accuracy_knn_ob_dimensions-0.25

spont_mean_accuracy_knn_pc_dimensions = spont_mean_accuracy_knn_pc_dimensions/0.75
spont_mean_accuracy_knn_ob_dimensions = spont_mean_accuracy_knn_ob_dimensions/0.75
spont_error_accuracy_knn_pc_dimensions = spont_error_accuracy_knn_pc_dimensions/0.75
spont_error_accuracy_knn_ob_dimensions = spont_error_accuracy_knn_ob_dimensions/0.75

plt.subplot(224)

plt.plot(np.arange(1,16),mean_accuracy_knn_pc_dimensions, label = 'Odor CS', color = 'tab:blue')
plt.fill_between(np.arange(1,16), mean_accuracy_knn_pc_dimensions-error_accuracy_knn_pc_dimensions,mean_accuracy_knn_pc_dimensions+error_accuracy_knn_pc_dimensions, alpha = 0.2, color = 'tab:blue')

plt.plot(np.arange(1,16),spont_mean_accuracy_knn_pc_dimensions,'--', label = 'Spont CS', color = 'tab:blue')
plt.fill_between(np.arange(1,16), spont_mean_accuracy_knn_pc_dimensions-spont_error_accuracy_knn_pc_dimensions,spont_mean_accuracy_knn_pc_dimensions+spont_error_accuracy_knn_pc_dimensions, alpha = 0.2, color = 'grey')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('# CS Dimensions')

plt.yticks(ticks = np.arange(0,1,0.2), labels = np.arange(0,100,20))
plt.xticks(np.arange(2,16,2))
plt.ylim([0,0.8])

plt.legend(ncol = 1,fontsize = 10)

plt.subplot(222)

plt.plot(np.arange(1,16),mean_accuracy_knn_ob_dimensions, label = 'Odor CS', color = 'tab:orange')
plt.fill_between(np.arange(1,16), mean_accuracy_knn_ob_dimensions-error_accuracy_knn_ob_dimensions,mean_accuracy_knn_ob_dimensions+error_accuracy_knn_ob_dimensions, alpha = 0.2, color = 'tab:orange')


plt.plot(np.arange(1,16),spont_mean_accuracy_knn_ob_dimensions,'--', label = 'Spont CS', color = 'tab:orange')
plt.fill_between(np.arange(1,16), spont_mean_accuracy_knn_ob_dimensions-spont_error_accuracy_knn_ob_dimensions,spont_mean_accuracy_knn_ob_dimensions+spont_error_accuracy_knn_ob_dimensions, alpha = 0.2, color = 'grey')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylabel('Accuracy Delta \n(% real - % chance)/(100-% chance)', multialignment='center')

plt.ylim([0,0.7])
plt.yticks(ticks = np.arange(0,1,0.2), labels = np.arange(0,100,20))
plt.xticks(np.arange(2,16,2))

plt.tight_layout()

#plt.savefig('cs_decoding_comp.pdf')

#%% check statistics of decoding

# identity
odor_accuracy_knn_pc_dimensions_animals_norm = np.array(odor_accuracy_knn_pc_dimensions_animals)-0.16
odor_accuracy_knn_ob_dimensions_animals_norm = np.array(odor_accuracy_knn_ob_dimensions_animals)-0.16

odor_accuracy_knn_pc_dimensions_animals_norm = odor_accuracy_knn_pc_dimensions_animals_norm/0.84
odor_accuracy_knn_ob_dimensions_animals_norm = odor_accuracy_knn_ob_dimensions_animals_norm/0.84

spont_odor_accuracy_knn_pc_dimensions_animals_norm = np.array(spont_odor_accuracy_knn_pc_dimensions_animals)-0.16
spont_odor_accuracy_knn_ob_dimensions_animals_norm = np.array(spont_odor_accuracy_knn_ob_dimensions_animals)-0.16

spont_odor_accuracy_knn_pc_dimensions_animals_norm = spont_odor_accuracy_knn_pc_dimensions_animals_norm/0.84
spont_odor_accuracy_knn_ob_dimensions_animals_norm = spont_odor_accuracy_knn_ob_dimensions_animals_norm/0.84


# concentration
concentration_accuracy_knn_pc_dimensions_animals_norm = np.array(concentration_accuracy_knn_pc_dimensions_animals)-0.25
concentration_accuracy_knn_ob_dimensions_animals_norm = np.array(concentration_accuracy_knn_ob_dimensions_animals)-0.25

concentration_accuracy_knn_pc_dimensions_animals_norm = concentration_accuracy_knn_pc_dimensions_animals_norm/0.75
concentration_accuracy_knn_ob_dimensions_animals_norm = concentration_accuracy_knn_ob_dimensions_animals_norm/0.75

spont_concentration_accuracy_knn_pc_dimensions_animals_norm = np.array(spont_concentration_accuracy_knn_pc_dimensions_animals)-0.25
spont_concentration_accuracy_knn_ob_dimensions_animals_norm = np.array(spont_concentration_accuracy_knn_ob_dimensions_animals)-0.25

spont_concentration_accuracy_knn_pc_dimensions_animals_norm = spont_concentration_accuracy_knn_pc_dimensions_animals_norm/0.75
spont_concentration_accuracy_knn_ob_dimensions_animals_norm = spont_concentration_accuracy_knn_ob_dimensions_animals_norm/0.75


import statsmodels.api as sm
from statsmodels.formula.api import ols


# pc odor
pc_odor = np.reshape(odor_accuracy_knn_pc_dimensions_animals_norm,[13*15])
pc_conc = np.reshape(concentration_accuracy_knn_pc_dimensions_animals_norm,[5*15])

data = np.hstack([pc_odor,pc_conc])
dim = np.hstack([np.tile(np.arange(0,15),13),np.tile(np.arange(0,15),5)])
    
decoding = np.hstack([np.repeat('identity',15*13),np.repeat('concentration',15*5)])

dataframe = pd.DataFrame({'decoding': decoding,
                          'dim': dim,
                          'data': data})

# Performing two-way ANOVA
model = ols('data ~ C(decoding) + C(dim) +\
C(decoding):C(dim)',data=dataframe).fit()

result_pc_odor = sm.stats.anova_lm(model, type=2)
  
print(result_pc_odor)

# pc spont
pc_odor = np.reshape(spont_odor_accuracy_knn_pc_dimensions_animals_norm,[13*15])
pc_conc = np.reshape(spont_concentration_accuracy_knn_pc_dimensions_animals_norm,[5*15])

data = np.hstack([pc_odor,pc_conc])
dim = np.hstack([np.tile(np.arange(0,15),13),np.tile(np.arange(0,15),5)])
    
decoding = np.hstack([np.repeat('identity',15*13),np.repeat('concentration',15*5)])

dataframe = pd.DataFrame({'decoding': decoding,
                          'dim': dim,
                          'data': data})

# Performing two-way ANOVA
model = ols('data ~ C(decoding) + C(dim) +\
C(decoding):C(dim)',data=dataframe).fit()

result_pc_spont = sm.stats.anova_lm(model, type=2)
  
print(result_pc_spont)

# ob odor
pc_odor = np.reshape(odor_accuracy_knn_ob_dimensions_animals_norm,[13*15])
pc_conc = np.reshape(concentration_accuracy_knn_ob_dimensions_animals_norm,[5*15])

data = np.hstack([pc_odor,pc_conc])
dim = np.hstack([np.tile(np.arange(0,15),13),np.tile(np.arange(0,15),5)])
    
decoding = np.hstack([np.repeat('identity',15*13),np.repeat('concentration',15*5)])

dataframe = pd.DataFrame({'decoding': decoding,
                          'dim': dim,
                          'data': data})

# Performing two-way ANOVA
model = ols('data ~ C(decoding) + C(dim) +\
C(decoding):C(dim)',data=dataframe).fit()

result_ob_odor = sm.stats.anova_lm(model, type=2)
  
print(result_ob_odor)

# ob spont
pc_odor = np.reshape(spont_odor_accuracy_knn_ob_dimensions_animals_norm,[13*15])
pc_conc = np.reshape(spont_concentration_accuracy_knn_ob_dimensions_animals_norm,[5*15])

data = np.hstack([pc_odor,pc_conc])
dim = np.hstack([np.tile(np.arange(0,15),13),np.tile(np.arange(0,15),5)])
    
decoding = np.hstack([np.repeat('identity',15*13),np.repeat('concentration',15*5)])

dataframe = pd.DataFrame({'decoding': decoding,
                          'dim': dim,
                          'data': data})

# Performing two-way ANOVA
model = ols('data ~ C(decoding) + C(dim) +\
C(decoding):C(dim)',data=dataframe).fit()

result_ob_spont = sm.stats.anova_lm(model, type=2)
  
print(result_ob_spont)

#%%

dim = -1

plt.boxplot([np.array(odor_accuracy_knn_ob_dimensions_animals_norm)[:,dim],np.array(concentration_accuracy_knn_ob_dimensions_animals_norm)[:,dim]])
# plt.boxplot([np.array(odor_accuracy_knn_pc_dimensions_animals)[:,dim],np.array(concentration_accuracy_knn_pc_dimensions_animals)[:,dim]])


#%% check CS actviity for each odor


animal = 0
time = 1000

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import cycler

odorants_repeated = odorants_animals[animal]


ob_loadings = ob_loadings_animals_odor[animal]
pc_loadings = pc_loadings_animals_odor[animal]


smells_trig_ob_cat = np.sum(smells_trig_ob_odor_animals[animal][:,:,0:time],axis = -1)
smells_trig_pc_cat = np.sum(smells_trig_pc_odor_animals[animal][:,:,0:time],axis = -1)


ob_proj = []
pc_proj = []


for x in range(n_components):
    
    ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
    pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
    
    
ob_proj = np.array(ob_proj).T
pc_proj = np.array(pc_proj).T


#%%

ob_proj_odor = []
pc_proj_odor = []

for x in np.unique(odorants_repeated):
    
    times = odorants_repeated == x
    
    ob_proj_odor.append(np.mean(ob_proj[times,:],axis = 0))
    pc_proj_odor.append(np.mean(pc_proj[times,:],axis = 0))
    

#%% check correlation between cs activity during different odors

time = 1000

corr_pc_spont_cs = []
corr_ob_spont_cs = []

p_pc_spont_cs = []
p_ob_spont_cs = []

for animal in range(13):
    
    
    odorants_repeated = odorants_animals[animal]
    
    
    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]
    
    
    smells_trig_ob_cat = np.mean(smells_trig_ob_odor_animals[animal][:,:,0:time],axis = -1)
    smells_trig_pc_cat = np.mean(smells_trig_pc_odor_animals[animal][:,:,0:time],axis = -1)
    
    
    ob_proj = []
    pc_proj = []
    
    
    for x in range(n_components):
        
        ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
        pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
        
        
    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    
    
    
    ob_proj_odor = []
    pc_proj_odor = []
    
    for x in np.unique(odorants_repeated):
        
        times = odorants_repeated == x
        
        ob_proj_odor.append(np.mean(ob_proj[times,:],axis = 0))
        pc_proj_odor.append(np.mean(pc_proj[times,:],axis = 0))
        
        
    
    for x in range(5):
        
        for y in np.arange(x+1,6):
            
        
            r,p = stats.pearsonr(pc_proj_odor[x],pc_proj_odor[y])
            corr_pc_spont_cs.append(r)
            p_pc_spont_cs.append(p)
            
            
            r,p = stats.pearsonr(ob_proj_odor[x],ob_proj_odor[y])
            corr_ob_spont_cs.append(r)
            p_ob_spont_cs.append(p)
            
#
corr_pc_odor_cs = []
corr_ob_odor_cs = []

p_pc_odor_cs = []
p_ob_odor_cs = []

for animal in range(13):
    
    
    odorants_repeated = odorants_animals[animal]
    
    
    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]
    
    
    smells_trig_ob_cat = np.mean(smells_trig_ob_odor_animals[animal][:,:,0:time],axis = -1)
    smells_trig_pc_cat = np.mean(smells_trig_pc_odor_animals[animal][:,:,0:time],axis = -1)
    
    
    ob_proj = []
    pc_proj = []
    
    
    for x in range(n_components):
        
        ob_proj.append(np.mean(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1))
        pc_proj.append(np.mean(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1))
        
        
    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    
    
    
    ob_proj_odor = []
    pc_proj_odor = []
    
    for x in np.unique(odorants_repeated):
        
        times = odorants_repeated == x
        
        ob_proj_odor.append(np.mean(ob_proj[times,:],axis = 0))
        pc_proj_odor.append(np.mean(pc_proj[times,:],axis = 0))
        
        
    
    for x in range(5):
        
        for y in np.arange(x+1,6):
        
            r,p = stats.pearsonr(pc_proj_odor[x],pc_proj_odor[y])
            corr_pc_odor_cs.append(r)
            p_pc_odor_cs.append(p)
            
            
            r,p = stats.pearsonr(ob_proj_odor[x],ob_proj_odor[y])
            corr_ob_odor_cs.append(r)
            p_ob_odor_cs.append(p)
            
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (5,6))

plt.subplot(211)
sns.histplot(corr_pc_spont_cs,bins = 100, element="step", cumulative = True, label = 'PCx')
sns.histplot(corr_ob_spont_cs,bins = 100, color = 'Tab:orange', element="step", cumulative = True, label = 'OB')
plt.xlim([0,1])
plt.legend(loc = 'upper left')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.title('Spontaneous CS')

plt.subplot(212)
sns.histplot(corr_pc_odor_cs,bins = 100, element="step", cumulative = True)
sns.histplot(corr_ob_odor_cs,bins = 100, color = 'Tab:orange', element="step", cumulative = True)
plt.xlim([0,1])
plt.xlabel('CS Activity Correlation (acorss odors)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.title('Odor CS')

plt.tight_layout()

plt.savefig('cs_correlation_odors.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,2))

plt.subplot(122)

mean_corr_pc_odor_cs_animals = np.mean(np.reshape(np.array(p_pc_odor_cs),[13,15]),axis = 1) 
mean_corr_ob_odor_cs_animals = np.mean(np.reshape(np.array(p_ob_odor_cs),[13,15]),axis = 1) 


plt.boxplot([mean_corr_ob_odor_cs_animals,mean_corr_pc_odor_cs_animals], showfliers = False, widths = 0.3)

plt.scatter(np.ones(13),mean_corr_ob_odor_cs_animals, color = 'tab:orange', s = 20)
plt.scatter(np.ones(13)+1,mean_corr_pc_odor_cs_animals, color = 'tab:blue', s = 20)
plt.xticks(ticks = [1,2],labels = ['OB','PCx'])
plt.xlim([0.7,2.3])
plt.hlines(0.05,0.7,2.3, color = 'black')
plt.yscale('log')
plt.ylim([0,0.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.title('Odor CS')


plt.subplot(121)

mean_corr_pc_odor_cs_animals = np.mean(np.reshape(np.array(p_pc_spont_cs),[13,15]),axis = 1) 
mean_corr_ob_odor_cs_animals = np.mean(np.reshape(np.array(p_ob_spont_cs),[13,15]),axis = 1) 


plt.boxplot([mean_corr_ob_odor_cs_animals,mean_corr_pc_odor_cs_animals], showfliers = False, widths = 0.3)

plt.scatter(np.ones(13),mean_corr_ob_odor_cs_animals, color = 'tab:orange', s = 20)
plt.scatter(np.ones(13)+1,mean_corr_pc_odor_cs_animals, color = 'tab:blue', s = 20)
plt.xticks(ticks = [1,2],labels = ['OB','PCx'])
plt.xlim([0.7,2.3])
plt.hlines(0.05,0.7,2.3, color = 'black')
plt.yscale('log')
plt.ylim([0,0.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.title('Spontaneous CS')
plt.ylabel('Average corr p-value')

plt.tight_layout()

plt.savefig('cs_correlation_p_odors.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


odor_names = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace']

plt.figure(dpi = 300, figsize = (6,8))

plt.subplot(621)
plt.plot(np.arange(1,16), ob_proj_odor[0],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[0],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[0])
plt.title('OB CS activity (z-scored)')

plt.subplot(623)
plt.plot(np.arange(1,16), ob_proj_odor[1],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[1],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[1])

plt.subplot(625)
plt.plot(np.arange(1,16), ob_proj_odor[2],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[2],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[2])

plt.subplot(627)
plt.plot(np.arange(1,16), ob_proj_odor[3],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[3],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[3])

plt.subplot(629)
plt.plot(np.arange(1,16), ob_proj_odor[4],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[4],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[5])

plt.subplot(6,2,11)
plt.plot(np.arange(1,16), ob_proj_odor[5],'o',color = 'tab:orange')
plt.vlines(np.arange(1,16), ob_proj_odor[5],'o',color = 'tab:orange')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-1,1.8])
plt.ylabel(odor_names[5])


# PCx

plt.subplot(622)
plt.plot(np.arange(1,16), pc_proj_odor[0],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[0],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])
plt.title('OB PCx activity (z-scored)')

plt.subplot(624)
plt.plot(np.arange(1,16), pc_proj_odor[1],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[1],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])

plt.subplot(626)
plt.plot(np.arange(1,16), pc_proj_odor[2],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[2],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])

plt.subplot(628)
plt.plot(np.arange(1,16), pc_proj_odor[3],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[3],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])

plt.subplot(6,2,10)
plt.plot(np.arange(1,16), pc_proj_odor[4],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[4],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])

plt.subplot(6,2,12)
plt.plot(np.arange(1,16), pc_proj_odor[5],'o',color = 'tab:blue')
plt.vlines(np.arange(1,16), pc_proj_odor[5],'o',color = 'tab:blue')
plt.hlines(0,1,15, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([-0.4,0.3])

plt.xlabel('CS pair #',loc = 'right')
plt.tight_layout()

plt.savefig('cs_activity_odors.pdf')

#%%

#%% check concentration sinlge odorant 1

concentration_accuracy_knn_ob_dimensions_animals_odor = []
concentration_accuracy_knn_pc_dimensions_animals_odor = []


from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 4000
    
    
    smells_trig_ob_cat_1 = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_1 = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,2000:time],axis = -1)

    smells_trig_ob_cat_2 = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_2 = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,2000:time],axis = -1)

    
    smells_trig_ob_cat = np.vstack([smells_trig_ob_cat_1,smells_trig_ob_cat_2])
    smells_trig_pc_cat = np.vstack([smells_trig_pc_cat_1,smells_trig_pc_cat_2])

    #
    odorants_sorted = np.hstack([np.repeat(0,smells_trig_ob_cat_1.shape[0]),np.repeat(1,smells_trig_ob_cat_2.shape[0])])
    
    #odorants_sorted = concentration_animals_1[animal]
    
    #

    ob_loadings = ob_loadings_animals_odor[animal]
    pc_loadings = pc_loadings_animals_odor[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,15):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        # pc_proj_dim = pc_proj_dim[:,np.newaxis]
        # ob_proj_dim = ob_proj_dim[:,np.newaxis]
        
        
        
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB()
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #kernel = 1.0 * RBF(1.0)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB(
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals_odor.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals_odor.append(accuracy_knn_pc_dimensions)
    
    

# check concentration sinlge odorant 1

concentration_accuracy_knn_ob_dimensions_animals_odorless = []
concentration_accuracy_knn_pc_dimensions_animals_odorless = []


from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 4000
    
    
    smells_trig_ob_cat_1 = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_1 = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,2000:time],axis = -1)

    smells_trig_ob_cat_2 = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_2 = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,2000:time],axis = -1)

    
    smells_trig_ob_cat = np.vstack([smells_trig_ob_cat_1,smells_trig_ob_cat_2])
    smells_trig_pc_cat = np.vstack([smells_trig_pc_cat_1,smells_trig_pc_cat_2])

    #
    odorants_sorted = np.hstack([np.repeat(0,smells_trig_ob_cat_1.shape[0]),np.repeat(1,smells_trig_ob_cat_2.shape[0])])
    
    #odorants_sorted = concentration_animals_1[animal]
    
    #

    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(1,15):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,:dimensions]
        ob_proj_dim = ob_proj[:,:dimensions]
        
        # pc_proj_dim = pc_proj_dim[:,np.newaxis]
        # ob_proj_dim = ob_proj_dim[:,np.newaxis]
        
        
        
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB()
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #kernel = 1.0 * RBF(1.0)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB(
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals_odorless.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals_odorless.append(accuracy_knn_pc_dimensions)
    


#%%
plt.bar(np.arange(0,15),np.mean(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0))
plt.errorbar(np.arange(0,15),np.mean(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0),stats.sem(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0))
#%%

plt.errorbar(np.arange(0,14),np.mean(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0),stats.sem(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0))
plt.errorbar(np.arange(0,14),np.mean(concentration_accuracy_knn_pc_dimensions_animals_odorless,axis = 0),stats.sem(concentration_accuracy_knn_pc_dimensions_animals_odorless,axis = 0))

#%%
plt.errorbar(np.arange(0,14),np.mean(concentration_accuracy_knn_ob_dimensions_animals_odor,axis = 0),stats.sem(concentration_accuracy_knn_ob_dimensions_animals_odor,axis = 0))
plt.errorbar(np.arange(0,14),np.mean(concentration_accuracy_knn_ob_dimensions_animals_odorless,axis = 0),stats.sem(concentration_accuracy_knn_ob_dimensions_animals_odorless,axis = 0))




#%%
concentration_accuracy_knn_ob_dimensions_animals_odorless = []
concentration_accuracy_knn_pc_dimensions_animals_odorless = []


from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

for animal in range(5):

    print(animal)
    
    
    accuracy_knn_ob_dimensions = []
    accuracy_knn_pc_dimensions = []
    
    accuracy_knn_ob_dimensions_shuffled = []
    accuracy_knn_pc_dimensions_shuffled = []
    
    time = 3000
    
    
    smells_trig_ob_cat_1 = np.sum(smells_trig_ob_concentration_1_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_1 = np.sum(smells_trig_pc_concentration_1_animals[animal][:,:,2000:time],axis = -1)

    smells_trig_ob_cat_2 = np.sum(smells_trig_ob_concentration_2_animals[animal][:,:,2000:time],axis = -1)
    smells_trig_pc_cat_2 = np.sum(smells_trig_pc_concentration_2_animals[animal][:,:,2000:time],axis = -1)

    
    smells_trig_ob_cat = np.vstack([smells_trig_ob_cat_1,smells_trig_ob_cat_2])
    smells_trig_pc_cat = np.vstack([smells_trig_pc_cat_1,smells_trig_pc_cat_2])

    #
    odorants_sorted = np.hstack([np.repeat(1,smells_trig_ob_cat_1.shape[0]),np.repeat(2,smells_trig_ob_cat_2.shape[0])])
    
    #odorants_sorted = concentration_animals_1[animal]
    
    #

    ob_loadings = ob_loadings_animals_odorless[animal]
    pc_loadings = pc_loadings_animals_odorless[animal]

    n_components = 15

    ob_proj = []
    pc_proj = []

    for x in range(n_components):
        
        ob_proj.append(stats.zscore(np.sum(smells_trig_ob_cat*ob_loadings[:,x][:,np.newaxis].T,axis = 1)))
        pc_proj.append(stats.zscore(np.sum(smells_trig_pc_cat*pc_loadings[:,x][:,np.newaxis].T,axis = 1)))

    ob_proj = np.array(ob_proj).T
    pc_proj = np.array(pc_proj).T
    
    for dimensions in np.arange(0,15):
    
        #print(dimensions)
        
        pc_proj_dim = pc_proj[:,dimensions]
        ob_proj_dim = ob_proj[:,dimensions]
        
        pc_proj_dim = pc_proj_dim[:,np.newaxis]
        ob_proj_dim = ob_proj_dim[:,np.newaxis]
        
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        
        accuracy_knn_ob = []
        accuracy_knn_pc = []
        
        accuracy_knn_ob_units = []
        accuracy_knn_pc_units = []
        
        accuracy_knn_ob_shuffled = []
        accuracy_knn_pc_shuffled = []
        
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        loo.get_n_splits(ob_proj_dim)

        
        for i, (train_index, test_index) in enumerate(loo.split(ob_proj_dim)):
        
            # CS actvity
        
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB()
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #kernel = 1.0 * RBF(1.0)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(ob_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_ob.append(sgd_clf.score(ob_proj_dim[test_index,:], odorants_sorted[test_index]))
            
            
            #sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000000,loss = 'hinge', tol=1e-5,alpha = 0.1, random_state = 1234))
            #sgd_clf = DecisionTreeClassifier(random_state=0)
            #sgd_clf = GaussianNB()
            sgd_clf = KNeighborsClassifier(n_neighbors=5)
            #sgd_clf = MLPClassifier(random_state=1, max_iter=10000)
            #sgd_clf = GaussianProcessClassifier(random_state=0)
            
            sgd_clf.fit(pc_proj_dim[train_index,:], odorants_sorted[train_index])
            accuracy_knn_pc.append(sgd_clf.score(pc_proj_dim[test_index,:], odorants_sorted[test_index]))
            
        
        
        accuracy_knn_ob = np.mean(accuracy_knn_ob,axis = 0)
        accuracy_knn_pc = np.mean(accuracy_knn_pc,axis = 0)
        
        accuracy_knn_ob_dimensions.append(accuracy_knn_ob)
        accuracy_knn_pc_dimensions.append(accuracy_knn_pc)
        
        
    
    concentration_accuracy_knn_ob_dimensions_animals_odorless.append(accuracy_knn_ob_dimensions)
    concentration_accuracy_knn_pc_dimensions_animals_odorless.append(accuracy_knn_pc_dimensions)

#%%

#%%

plt.subplot(121)
plt.stem(np.mean(concentration_accuracy_knn_pc_dimensions_animals_odor,axis = 0),bottom = 0.5)
#plt.stem(np.mean(concentration_accuracy_knn_pc_dimensions_animals_odorless,axis = 0),bottom = 0.5)

plt.subplot(122)
plt.stem(np.mean(concentration_accuracy_knn_ob_dimensions_animals_odor,axis = 0))
plt.stem(np.mean(concentration_accuracy_knn_ob_dimensions_animals_odorless,axis = 0))



#%%

plt.scatter(pc_proj[0:60,0],pc_proj[0:60,3])
plt.scatter(pc_proj[60:,0],pc_proj[60:,3])



#%%

signs = [1,1,1,1,1,1,1,1,1,1,-1,-1,1]

animal = 3

plt.subplot(211)

plt.stem(ob_loadings_animals_odorless[animal][:,0])
print(np.max(ob_loadings_animals_odorless[animal][:,0]),np.min(ob_loadings_animals_odorless[animal][:,0]))

plt.subplot(212)

plt.stem(pc_loadings_animals_odorless[animal][:,0])
print(np.max(pc_loadings_animals_odorless[animal][:,0]),np.min(pc_loadings_animals_odorless[animal][:,0]))



#%%

data = np.random.randn(1000)
f,p = signal.periodogram(data, scaling = 'spectrum')

print(np.sum(p))
print(np.var(data))

