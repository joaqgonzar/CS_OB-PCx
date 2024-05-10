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
from statsmodels.tsa.stattools import grangercausalitytests


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

#%% general information 

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset/'


names_decimated_ipsi = ['PFx_TeLC-Thy_lfp_150627-1_.npz',
                   'PFx_TeLC-Thy_lfp_151213-1_.npz',
                   'PFx_TeLC-Thy_lfp_151116-3_.npz',  
                   'PFx_TeLC-Thy_lfp_151213-2_.npz',
                   'PFx_TeLC-Thy_lfp_151116-4_.npz',  
                   'PFx_TeLC-Thy_lfp_151215-1_.npz',
                   'PFx_TeLC-Thy_lfp_151118-1_.npz', 
                   'PFx_TeLC-Thy_lfp_151220-2_.npz',
                   'PFx_TeLC-Thy_lfp_151118-3_.npz', 
                   'PFx_TeLC-Thy_lfp_151220-3_.npz',
                   'PFx_TeLC-Thy_lfp_151120-2_.npz',
                   'PFx_TeLC-Thy_lfp_151222-1_.npz',
                   'PFx_TeLC-Thy_lfp_151120-3_.npz',
                   'PFx_TeLC-Thy_lfp_151222-2_.npz']

os.chdir(directory+'/TeLC-Thy_simul/processed')

exp_data = pd.read_csv('ExperimentCatalog_THY1-TeLC.txt', sep=" ", header=None)

pcx = np.where(exp_data[3]=='P')[0]
ob = np.where(exp_data[3]=='B')[0]
ipsi = np.where(exp_data[4]=='T')[0]

# get name and banks of recordings
mice = np.intersect1d(pcx,ipsi)
full_names = exp_data[0][mice]

names = []
for x in full_names:
    names.append(x[-14:-6])
       
pc_bank = []
for x in full_names:
    pc_bank.append(x[-5:])

names_banks_pcx = np.vstack([names, pc_bank])

# get name and banks of recordings
mice = np.intersect1d(ob,ipsi)
full_names = exp_data[0][mice]

names = []
for x in full_names:
    names.append(x[-14:-6])
    
ob_bank = []
for x in full_names:
    ob_bank.append(x[-5:])

names_banks_ob = np.vstack([names, ob_bank])

laser_intensities_first = np.array(exp_data[1][mice])
laser_intensities_last = np.array(exp_data[2][mice])

laser_packet = 10 # number of laser presentations per intensity
#

ob_loadings_animals_telc = []
pc_loadings_animals_telc = []

c_ob_proj_animals_telc = []
c_pc_proj_animals_telc = []
c_ob_pc_proj_animals_telc = []
p_ob_proj_animals_telc = []
p_pc_proj_animals_telc = []
p_resp_animals_telc = []

ob_comm_resp_phase_animals_telc = []
pc_comm_resp_phase_animals_telc = []
ob_mua_resp_phase_animals_telc = []
pc_mua_resp_phase_animals_telc = []

ob_neurons_resp_phase_animals_telc = []
pc_neurons_resp_phase_animals_telc = []

ob_mi_neurons_animals_telc = []
pc_mi_neurons_animals_telc = []

c_ob_mua_animals_telc = []
c_pc_mua_animals_telc = []
c_ob_pc_mua_animals_telc = []
p_ob_mua_animals_telc = []
p_pc_mua_animals_telc = []

ccg_ob_pc_proj_animals_telc = []
lags_ob_pc_animals_telc = []
ccg_ob_pc_mua_animals_telc = []
gc_ob_pc_proj_animals_telc = []
gc_pc_ob_proj_animals_telc = []
gc_ob_pc_mua_animals_telc = []
gc_pc_ob_mua_animals_telc = []

inh_trig_ob_proj_animals_telc = []
inh_trig_pc_proj_animals_telc = []
inh_trig_ob_mua_animals_telc = []
inh_trig_pc_mua_animals_telc = []

mean_ob_proj_laser_animals_telc = []
mean_pc_proj_laser_animals_telc = []

ob_proj_laser_animals_telc = []
pc_proj_laser_animals_telc = []
ob_mua_laser_animals_telc = []
pc_mua_laser_animals_telc = []

laser_intensities_animals_telc = []

phase_units_ob_animals_telc = []
phase_units_pc_animals_telc = []

ob_units_laser_animals_telc = []
pc_units_laser_animals_telc = []

# loop through animals 

for index, name in enumerate(names):
    
    print(name)
    
    # get recordings
    
    # get laser data
    os.chdir(directory+'/TeLC-Thy_simul/processed/'+name)
    bank = names_banks_pcx[1,np.where(names_banks_pcx[0,:] == name)[0][0]]
    laser_data = scipy.io.loadmat(name+'_'+bank+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    
    srate = 2000
    laser_times = laser_data*srate
    
    os.chdir(directory+'/TeLC-Thy_simul/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index][-1])+'_st.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index][-1])+'_st.mat')['SpikeTimes']['tsec']
    
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    srate_resp = 2000
    
    
    # load samples around laser onset
    laser_samples = []
    for x in laser_times:
        laser_samples.append(np.arange(int(x-300),int(x+2700)))
    
    laser_samples = np.concatenate(laser_samples)
    
    
    
    #
    
    lenta_BandWidth = 2
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
        
    os.chdir(directory+'/TeLC-Thy_simul/processed/'+name)
    inh_start = scipy.io.loadmat(name+'_bank1.mat')['efd']['PREX'][0][0][0]*srate_resp
    inh_start = np.squeeze(inh_start)


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
    
    # load PCx lfp
    
    srate = 2000
    
    units_ob = units_ob[:,0:recording_time]
    units_pc = units_pc[:,0:recording_time]
    resp = resp[0:recording_time]
    
    #
    
    all_samples = np.arange(0,recording_time)
    
    samples_non_laser = np.setdiff1d(all_samples,laser_samples)
    
    inh_start_non_laser = np.setdiff1d(inh_start,laser_samples)
    
    
    
    # check laser stimulations
    
    intensity_first = laser_intensities_first[index]
    intenstity_last = laser_intensities_last[index]
    intenisities_power_series = np.repeat(np.arange(int(intensity_first),int(intenstity_last)+1),laser_packet)
    
    srate_resample = 2000
    kernel = signal.gaussian(int(0.1*srate_resample),5) 

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
        
        
        
    
    X_all = np.array(conv_neurons_ob)[1:,:].T
    Y_all = np.array(conv_neurons_pc)[1:,:].T
    X_all = stats.zscore(X_all,axis = 0)
    Y_all = stats.zscore(Y_all,axis = 0)
    
    ob_mua_all = stats.zscore(conv_neurons_ob[0])
    pc_mua_all = stats.zscore(conv_neurons_pc[0])
    
    
    ob_proj_laser = []
    pc_proj_laser = []
    ob_mua_laser = []
    pc_mua_laser = []
    ob_units_laser = []
    pc_units_laser = []
    
    for x in laser_times:
        
        ob_units_laser.append(X_all[int(x-300):int(x+2700),:])
        pc_units_laser.append(Y_all[int(x-300):int(x+2700),:])
        
    
    # get cca for the laser stim only
    
    X = np.concatenate(ob_units_laser,axis = 0)
    Y = np.concatenate(pc_units_laser,axis = 0)
    
    X_z = stats.zscore(X,axis = 0)
    Y_z = stats.zscore(Y,axis = 0)
    
    cca = CCA(n_components=1,max_iter=100000,tol = 1e-12)
    cca.fit(X_z[:,:], Y_z[:,:])
    
    ob_loadings = cca.x_rotations_
    pc_loadings = cca.y_rotations_
    
    
    
    # save results 

    ob_loadings_animals_telc.append(ob_loadings)
    pc_loadings_animals_telc.append(pc_loadings)
    
    ob_proj_laser_animals_telc.append(ob_proj_laser)
    pc_proj_laser_animals_telc.append(pc_proj_laser)
    ob_mua_laser_animals_telc.append(ob_mua_laser)
    pc_mua_laser_animals_telc.append(pc_mua_laser)
    
    
    laser_intensities_animals_telc.append(intenisities_power_series)
    
    ob_units_laser_animals_telc.append(ob_units_laser)
    pc_units_laser_animals_telc.append(pc_units_laser)
    

#%% check THY animals 

#directory = '/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset'


names = ['150610','150624','150701','150707','150709']

os.chdir(directory+'/THY')

exp_data = pd.read_csv('ExperimentCatalog_THY1.txt', sep=" ", header=None)


pcx = exp_data[6:11]
ob = exp_data[0:6]

laser_intensities_first = np.array(pcx[1])
laser_intensities_last = np.array(pcx[2])

laser_packet = 10 # number of laser presentations per intensity

ob_bank = [2,1,2,1,1]
pc_bank = [1,2,1,2,2]

    
#

ob_loadings_animals_thy = []
pc_loadings_animals_thy = []

c_ob_proj_animals_thy = []
c_pc_proj_animals_thy = []
c_ob_pc_proj_animals_thy = []
p_ob_proj_animals_thy = []
p_pc_proj_animals_thy = []
p_resp_animals_thy = []

ob_comm_resp_phase_animals_thy = []
pc_comm_resp_phase_animals_thy = []
ob_mua_resp_phase_animals_thy = []
pc_mua_resp_phase_animals_thy = []

ob_neurons_resp_phase_animals_thy = []
pc_neurons_resp_phase_animals_thy = []

ob_mi_neurons_animals_thy = []
pc_mi_neurons_animals_thy = []

c_ob_mua_animals_thy = []
c_pc_mua_animals_thy = []
c_ob_pc_mua_animals_thy = []
p_ob_mua_animals_thy = []
p_pc_mua_animals_thy = []

ccg_ob_pc_proj_animals_thy = []
lags_ob_pc_animals_thy = []
ccg_ob_pc_mua_animals_thy = []
gc_ob_pc_proj_animals_thy = []
gc_pc_ob_proj_animals_thy = []
gc_ob_pc_mua_animals_thy = []
gc_pc_ob_mua_animals_thy = []

inh_trig_ob_proj_animals_thy = []
inh_trig_pc_proj_animals_thy = []
inh_trig_ob_mua_animals_thy = []
inh_trig_pc_mua_animals_thy = []

mean_ob_proj_laser_animals_thy = []
mean_pc_proj_laser_animals_thy = []

ob_proj_laser_animals_thy = []
pc_proj_laser_animals_thy = []
ob_mua_laser_animals_thy = []
pc_mua_laser_animals_thy = []


laser_intensities_animals_thy = []

phase_units_ob_animals_thy = []
phase_units_pc_animals_thy = []
   
ob_units_laser_animals_thy = []
pc_units_laser_animals_thy = []

# loop through animals 

for index, name in enumerate(names):
    
    print(name)
    
    # get recordings
    
    # get laser data
    os.chdir(directory+'/THY/processed/'+name)
    #bank = names_banks_pcx[1,np.where(names_banks_pcx[0,:] == name)[0][0]]
    laser_data = scipy.io.loadmat(name+'_'+'bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    
    srate = 2000
    laser_times = laser_data*srate
    
    os.chdir(directory+'/THY/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index])+'_st.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index])+'_st.mat')['SpikeTimes']['tsec']
    
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    srate_resp = 2000
    
    #
    
    # load samples around laser onset
    laser_samples = []
    for x in laser_times:
        laser_samples.append(np.arange(int(x-300),int(x+2700)))
    
    laser_samples = np.concatenate(laser_samples)
    
    
    
    #
    
    lenta_BandWidth = 2
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
        
    os.chdir(directory+'/THY/processed/'+name)
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate_resp
    inh_start = np.squeeze(inh_start)


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
    
    # load PCx lfp
    
    srate = 2000
    
    units_ob = units_ob[:,0:recording_time]
    units_pc = units_pc[:,0:recording_time]
    resp = resp[0:recording_time]
    
    #
    
    all_samples = np.arange(0,recording_time)
    
    samples_non_laser = np.setdiff1d(all_samples,laser_samples)
    
    inh_start_non_laser = np.setdiff1d(inh_start,laser_samples)
    
    
    
    # check laser stimulations
    
    intensity_first = laser_intensities_first[index]
    intenstity_last = laser_intensities_last[index]
    intenisities_power_series = np.repeat(np.arange(int(intensity_first),int(intenstity_last)+1),laser_packet)
    
    srate_resample = 2000
    kernel = signal.gaussian(int(0.1*srate_resample),5) 
    
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
        
    
    X_all = np.array(conv_neurons_ob)[1:,:].T
    Y_all = np.array(conv_neurons_pc)[1:,:].T
    X_all = stats.zscore(X_all,axis = 0)
    Y_all = stats.zscore(Y_all,axis = 0)
    
    
    ob_proj_laser = []
    pc_proj_laser = []
    ob_mua_laser = []
    pc_mua_laser = []
    ob_units_laser = []
    pc_units_laser = []
    
    for x in laser_times:
        
        ob_units_laser.append(X_all[int(x-300):int(x+2700),:])
        pc_units_laser.append(Y_all[int(x-300):int(x+2700),:])
        
    
    # get cca for the laser stim only
    
    X = np.concatenate(ob_units_laser,axis = 0)
    Y = np.concatenate(pc_units_laser,axis = 0)
    
    X_z = stats.zscore(X,axis = 0)
    Y_z = stats.zscore(Y,axis = 0)
    
    cca = CCA(n_components=1,max_iter=100000,tol = 1e-12)
    cca.fit(X_z[:,:], Y_z[:,:])
    
    ob_loadings = cca.x_rotations_
    pc_loadings = cca.y_rotations_
    
    #
    
    
    
    # save results 

    ob_loadings_animals_thy.append(ob_loadings)
    pc_loadings_animals_thy.append(pc_loadings)
    
    
    ob_proj_laser_animals_thy.append(ob_proj_laser)
    pc_proj_laser_animals_thy.append(pc_proj_laser)
    
    ob_mua_laser_animals_thy.append(ob_mua_laser)
    pc_mua_laser_animals_thy.append(pc_mua_laser)
    
    ob_units_laser_animals_thy.append(ob_units_laser)
    pc_units_laser_animals_thy.append(pc_units_laser)
    
    
    
    laser_intensities_animals_thy.append(intenisities_power_series)
    
#%%

names = ['150610','150624','150701','150707','150709']

os.chdir(directory+'/THY')

exp_data = pd.read_csv('ExperimentCatalog_THY1.txt', sep=" ", header=None)


pcx = exp_data[6:11]
ob = exp_data[0:6]

laser_intensities_first = np.array(pcx[1])
laser_intensities_last = np.array(pcx[2])

laser_packet = 10 # number of laser presentations per intensity

ob_bank = [2,1,2,1,1]
pc_bank = [1,2,1,2,2]

    

ob_neurons_resp_phase_animals_thy = []
pc_neurons_resp_phase_animals_thy = []

ob_mi_neurons_animals_thy = []
pc_mi_neurons_animals_thy = []


# loop through animals 

for index, name in enumerate(names):
    
    print(name)
    
    # get recordings
    
    # get laser data
    os.chdir(directory+'/THY/processed/'+name)
    #bank = names_banks_pcx[1,np.where(names_banks_pcx[0,:] == name)[0][0]]
    laser_data = scipy.io.loadmat(name+'_'+'bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    
    srate = 2000
    laser_times = laser_data*srate
    
    os.chdir(directory+'/THY/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index])+'_st.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index])+'_st.mat')['SpikeTimes']['tsec']
    
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    srate_resp = 2000
    
    
    # load samples around laser onset
    laser_samples = []
    for x in laser_times:
        laser_samples.append(np.arange(int(x-300),int(x+2700)))
    
    laser_samples = np.concatenate(laser_samples)
    
    
    
    #
    
    lenta_BandWidth = 2
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
        
    os.chdir(directory+'/THY/processed/'+name)
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate_resp
    inh_start = np.squeeze(inh_start)


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
    
    # load PCx lfp
    
    srate = 2000
    
    units_ob = units_ob[:,0:recording_time]
    units_pc = units_pc[:,0:recording_time]
    resp = resp[0:recording_time]
    
    #
    
    all_samples = np.arange(0,recording_time)
    
    samples_non_laser = np.setdiff1d(all_samples,laser_samples)
    
    inh_start_non_laser = np.setdiff1d(inh_start,laser_samples)
    
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
    
    # perform canonical correlation analysis 
    
    X = np.array(conv_neurons_ob)[1:,samples_non_laser].T
    Y = np.array(conv_neurons_pc)[1:,samples_non_laser].T
    
    
    
    resp = resp[samples_non_laser]
    faselenta = faselenta[samples_non_laser] 
    
    # get resp spiking phases
    units_ob_non_laser = units_ob[:,samples_non_laser]
    units_pc_non_laser = units_pc[:,samples_non_laser]
    
    
    phase_units_ob = []
    for x in range(units_ob_non_laser.shape[0]-1):
        spike_times_unit = np.where(units_ob_non_laser[x+1,:]>0)[0]
        phase_units_ob.append(faselenta[spike_times_unit])
    
    phase_units_pc = []
    for x in range(units_pc_non_laser.shape[0]-1):
        spike_times_unit = np.where(units_pc_non_laser[x+1,:]>0)[0]
        phase_units_pc.append(faselenta[spike_times_unit])
        

    # check single neuron sync to resp

    X = np.array(conv_neurons_ob)[1:,samples_non_laser].T
    Y = np.array(conv_neurons_pc)[1:,samples_non_laser].T
    
    numbin = 300
    
    max_entrop = np.log(numbin)

    ob_neurons_resp_phase = []
    pc_neurons_resp_phase = []

    ob_mi_neurons = []
    pc_mi_neurons = []
    
    

    for x in range(X.shape[1]):
        
        hist = phase_amp_hist(X[:,x],faselenta,numbin)
        ob_neurons_resp_phase.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        ob_mi_neurons.append((max_entrop-entrop)/max_entrop)
        

    for x in range(Y.shape[1]):    
        
        hist = phase_amp_hist(Y[:,x],faselenta,numbin)
        pc_neurons_resp_phase.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        pc_mi_neurons.append((max_entrop-entrop)/max_entrop)
        
    

    ob_neurons_resp_phase_animals_thy.append(ob_neurons_resp_phase)
    pc_neurons_resp_phase_animals_thy.append(pc_neurons_resp_phase)
    
    ob_mi_neurons_animals_thy.append(ob_mi_neurons)
    pc_mi_neurons_animals_thy.append(pc_mi_neurons)
    
#%% check how laser stimulation looks for cca weights

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,3))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",['m','white','c'])

gs = gridspec.GridSpec(2, 2,height_ratios = [77,111], width_ratios = [1,2])

plt.subplot(gs[0])

sorted_loadings = np.sort(np.squeeze(np.concatenate(ob_loadings_animals_thy)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,np.concatenate(ob_loadings_animals_thy).shape[0]])
plt.xticks(ticks = [-1,0,1], labels = [])

plt.subplot(gs[1])
ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals_thy)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)#/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_animals_thy)))
#load_sort = np.argsort(ob_neurons_resp_phase_norm[:,0])

plt.imshow(ob_neurons_resp_phase_norm[load_sort,:],cmap = cmap,interpolation = None,aspect = 'auto',vmin = -1, vmax = 1)

plt.ylim([0,len(np.concatenate(ob_mi_neurons_animals_thy))])
#plt.xticks(ticks = np.arange(16,53,6),labels = [],rotation = 30)
#plt.xlim([16,52])
plt.xlim([245,245+600])
plt.xticks(ticks = np.arange(245,250+600,100),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)

#plt.title('OB')

plt.subplot(gs[2])

sorted_loadings = np.sort(np.squeeze(np.concatenate(pc_loadings_animals_thy)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,np.concatenate(pc_loadings_animals_thy).shape[0]])
plt.xlabel('CS Weight')

plt.subplot(gs[3])

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_thy)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)#/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]

load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_animals_thy)))
#load_sort = np.argsort(pc_neurons_resp_phase_norm[:,0])

plt.imshow(pc_neurons_resp_phase_norm[load_sort,:],cmap = cmap,interpolation = None,aspect = 'auto',vmin = -1, vmax = 1)
#plt.colorbar()
plt.ylim([0,len(np.concatenate(pc_mi_neurons_animals_thy))])

#plt.title('PCx')
#plt.xticks(ticks = np.arange(16,53,6),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)

plt.xlim([245,245+600])
plt.xticks(ticks = np.arange(245,250+600,100),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)


plt.tight_layout()
#plt.colorbar(location = 'bottom')
plt.xlabel('Respiration Phase (deg)')


plt.xlabel('Respiration Coupling')

#%%



#plt.savefig('telc_sup_raster.pdf')

#%%

phase_vector = np.arange(300,400)

plt.subplot(111)

plt.xlabel('CCA Weight')

sns.regplot(np.concatenate(pc_loadings_animals_thy),np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1),x_bins = np.linspace(np.min(np.concatenate(pc_loadings_animals_thy)),np.max(np.concatenate(pc_loadings_animals_thy)),10) , color = 'tab:blue')

plt.ylim([-1.3,1.1])
plt.ylim([-1.2,1.2])
plt.xlim([-1.2,1.2])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
r_pc, p_pc = stats.pearsonr(np.squeeze(np.concatenate(pc_loadings_animals_thy)),np.squeeze(np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1)))

plt.text(-0.4,-1.15, s = 'R = '+str(np.round(r_pc,decimals = 3))+'  p = '+str(np.round(p_pc,decimals = 3)))

#plt.title('PCx')

plt.xlabel('CCA Weight')


#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,6))

phase_vector = np.arange(18,24)

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_thy)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)


plt.subplot(211)

loadings_cat = np.concatenate(pc_loadings_animals_thy)
load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_animals_thy)))

top_percentile_pc = np.quantile(np.squeeze(np.concatenate(pc_loadings_animals_thy)),0.95)
top_percentile_pc = 0.6

top_pc_neurons = np.squeeze(np.concatenate(pc_loadings_animals_thy))>top_percentile_pc

low_pc_neurons = np.squeeze(np.concatenate(pc_loadings_animals_thy))<top_percentile_pc


mean_high_pc = np.mean(pc_neurons_resp_phase_norm[top_pc_neurons,:],axis = 0)
error_high_pc = stats.sem(pc_neurons_resp_phase_norm[top_pc_neurons,:],axis = 0)

mean_low_pc = np.mean(pc_neurons_resp_phase_norm[low_pc_neurons,:],axis = 0)
error_low_pc = stats.sem(pc_neurons_resp_phase_norm[low_pc_neurons,:],axis = 0)
      

plt.plot(mean_high_pc,color = 'tab:blue', label = '>0.6 CCA')
plt.fill_between(np.arange(0,54), mean_high_pc-error_high_pc,mean_high_pc+error_high_pc, alpha = 0.2,color = 'tab:blue')

plt.plot(mean_low_pc,'--',color = 'tab:blue',label = '<0.6 CCA')
plt.fill_between(np.arange(0,54), mean_low_pc-error_low_pc,mean_low_pc+error_low_pc, alpha = 0.2,color = 'grey')

plt.legend(fontsize = 6)
plt.xlim([16,52])
plt.ylim([-1.5,2])
#plt.ylim([0.01,0.032])
plt.xlabel('Respiration phase (deg)')
#plt.title('PCx')
plt.xticks(ticks = np.arange(16,53,6),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#

plt.subplot(212)

plt.xlabel('CCA Weight')

sns.regplot(np.concatenate(pc_loadings_animals_thy),np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1),x_bins = np.linspace(np.min(np.concatenate(pc_loadings_animals_thy)),np.max(np.concatenate(pc_loadings_animals_thy)),10) , color = 'tab:blue')
plt.ylim([-1.3,1.1])
plt.ylim([-1.2,1.2])
plt.xlim([-1.2,1.2])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
r_pc, p_pc = stats.pearsonr(np.squeeze(np.concatenate(pc_loadings_animals_thy)),np.squeeze(np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1)))

plt.plot(np.linspace(-1.6,1.6,100),np.linspace(-1.6,1.6,100), linestyle = 'dashed', color = 'black', alpha = 0.5)

plt.text(-0.4,-1.15, s = 'R = '+str(np.round(r_pc,decimals = 3))+'  p = '+str(np.round(p_pc,decimals = 3)))

#plt.title('PCx')

plt.xlabel('CCA Weight')
plt.tight_layout()

#plt.savefig('telc_sup_regplot.pdf')

#%%

neg_activity_ob = []
pos_activity_ob = []

neg_activity_pc = []
pos_activity_pc = []

mean_activity_proj_ob = []
mean_activity_proj_pc = []

for x in range(len(ob_proj_laser_animals_thy)):
    
    #laser_intensities_animals_thy[0]
    
    ob_laser = np.array(ob_units_laser_animals_thy[x]) 
    non_firing_neurons = np.unique(np.where(np.sum(ob_laser,axis = 1) == 0)[1])
    neurons = np.arange(0,ob_laser.shape[2])
    firing_neurons = np.setdiff1d(neurons,non_firing_neurons)
    ob_activity = stats.zscore(ob_laser[:,:,firing_neurons],axis = 1)
    ob_activity = ob_laser
    weighted_ob_activity = ob_activity#*laser_intensities_animals_thy[x][:,np.newaxis,np.newaxis]
    activity_proj_ob = np.sum(weighted_ob_activity*np.squeeze(ob_loadings_animals_thy[x][:])[np.newaxis,np.newaxis,:],axis = -1)
    mean_activity_proj_ob.append(np.mean(activity_proj_ob,axis = 0))
    
    ob_activity = ob_laser
    weighted_ob_activity = ob_activity#*laser_intensities_animals_thy[x][:,np.newaxis,np.newaxis]
    
    mean_activity_ob = np.mean(weighted_ob_activity,axis = 0)
    neg_neurons_ob = np.squeeze(np.argmin(ob_loadings_animals_thy[x]))
    pos_neurons_ob = np.squeeze(np.argmax(ob_loadings_animals_thy[x]))
    neg_activity_ob.append(mean_activity_ob[:,neg_neurons_ob])
    pos_activity_ob.append(mean_activity_ob[:,pos_neurons_ob])
    
    pc_laser = np.array(pc_units_laser_animals_thy[x])
    non_firing_neurons = np.unique(np.where(np.sum(pc_laser,axis = 1) == 0)[1])
    neurons = np.arange(0,pc_laser.shape[2])
    firing_neurons = np.setdiff1d(neurons,non_firing_neurons)
    pc_activity = stats.zscore(pc_laser[:,:,firing_neurons],axis = 1)
    pc_activity = pc_laser
    weighted_pc_activity = pc_activity#*laser_intensities_animals_thy[x][:,np.newaxis,np.newaxis]
    activity_proj_pc = np.sum(weighted_pc_activity*np.squeeze(pc_loadings_animals_thy[x][:])[np.newaxis,np.newaxis,:],axis = -1)
    mean_activity_proj_pc.append(np.mean(activity_proj_pc,axis = 0))
    
    
    pc_activity = pc_laser[:,:,:]
    weighted_pc_activity = pc_activity#*laser_intensities_animals_thy[x][:,np.newaxis,np.newaxis]
    
    mean_activity_pc = np.mean(weighted_pc_activity,axis = 0)
    neg_neurons_pc = np.squeeze(np.argmin(pc_loadings_animals_thy[x]))
    pos_neurons_pc = np.squeeze(np.argmax(pc_loadings_animals_thy[x]))
    neg_activity_pc.append(mean_activity_pc[:,neg_neurons_pc])
    pos_activity_pc.append(mean_activity_pc[:,pos_neurons_pc])


neg_activity_ob_telc = []
pos_activity_ob_telc = []

neg_activity_pc_telc = []
pos_activity_pc_telc = []

mean_activity_proj_ob_telc = []
mean_activity_proj_pc_telc = []

for x in range(len(ob_proj_laser_animals_telc)):
    
    #laser_intensities_animals_telc[0]
    
    ob_laser = np.array(ob_units_laser_animals_telc[x]) 
    non_firing_neurons = np.unique(np.where(np.sum(ob_laser,axis = 1) == 0)[1])
    neurons = np.arange(0,ob_laser.shape[2])
    firing_neurons = np.setdiff1d(neurons,non_firing_neurons)
    ob_activity = stats.zscore(ob_laser[:,:,firing_neurons],axis = 1)
    ob_activity = ob_laser
    weighted_ob_activity = ob_activity#*laser_intensities_animals_telc[x][:,np.newaxis,np.newaxis]
    activity_proj_ob = np.sum(weighted_ob_activity*np.squeeze(ob_loadings_animals_telc[x][:])[np.newaxis,np.newaxis,:],axis = -1)
    mean_activity_proj_ob_telc.append(np.mean(activity_proj_ob,axis = 0))
    
    ob_activity = ob_laser
    weighted_ob_activity = ob_activity#*laser_intensities_animals_telc[x][:,np.newaxis,np.newaxis]
    
    mean_activity_ob = np.mean(weighted_ob_activity,axis = 0)
    neg_neurons_ob = np.squeeze(np.argmin(ob_loadings_animals_telc[x]))
    pos_neurons_ob = np.squeeze(np.argmax(ob_loadings_animals_telc[x]))
    neg_activity_ob_telc.append(mean_activity_ob[:,neg_neurons_ob])
    pos_activity_ob_telc.append(mean_activity_ob[:,pos_neurons_ob])
    
    pc_laser = np.array(pc_units_laser_animals_telc[x])
    non_firing_neurons = np.unique(np.where(np.sum(pc_laser,axis = 1) == 0)[1])
    neurons = np.arange(0,pc_laser.shape[2])
    firing_neurons = np.setdiff1d(neurons,non_firing_neurons)
    pc_activity = stats.zscore(pc_laser[:,:,firing_neurons],axis = 1)
    pc_activity = pc_laser
    weighted_pc_activity = pc_activity#*laser_intensities_animals_telc[x][:,np.newaxis,np.newaxis]
    activity_proj_pc = np.sum(weighted_pc_activity*np.squeeze(pc_loadings_animals_telc[x][:])[np.newaxis,np.newaxis,:],axis = -1)
    mean_activity_proj_pc_telc.append(np.mean(activity_proj_pc,axis = 0))
    
    
    pc_activity = pc_laser[:,:,:]
    weighted_pc_activity = pc_activity#*laser_intensities_animals_telc[x][:,np.newaxis,np.newaxis]
    
    mean_activity_pc = np.mean(weighted_pc_activity,axis = 0)
    neg_neurons_pc = np.squeeze(np.argmin(pc_loadings_animals_telc[x]))
    pos_neurons_pc = np.squeeze(np.argmax(pc_loadings_animals_telc[x]))
    neg_activity_pc_telc.append(mean_activity_pc[:,neg_neurons_pc])
    pos_activity_pc_telc.append(mean_activity_pc[:,pos_neurons_pc])
    
#%% measure time difference between ob and pcx

time_laser = np.arange(-150,1350,0.5) 

time_peak_ob = np.argmax(np.array(mean_activity_proj_ob),axis = 1)
time_peak_pc = np.argmax(np.array(mean_activity_proj_pc),axis = 1)
    
s,p = stats.ttest_rel(time_laser[time_peak_ob],time_laser[time_peak_pc])

# check difference between + and -PCx cells

time_peak_neg = np.argmax(np.array(neg_activity_pc),axis = 1)
time_peak_pos = np.argmax(np.array(pos_activity_pc),axis = 1)
    
s,p = stats.ttest_rel(time_laser[time_peak_neg],time_laser[time_peak_pos])

time_peak_neg = np.argmax(np.array(neg_activity_pc_telc),axis = 1)
time_peak_pos = np.argmax(np.array(pos_activity_pc_telc),axis = 1)
    
s,p = stats.ttest_rel(time_laser[time_peak_neg],time_laser[time_peak_pos])

# check difference between + and - OB cells

peak_neg = np.max(np.array(neg_activity_ob)[:,300:350],axis = 1)
peak_pos = np.max(np.array(pos_activity_ob)[:,300:350],axis = 1)
    
s,p = stats.ttest_rel(peak_neg,peak_pos)

peak_neg = np.max(np.array(neg_activity_ob_telc)[:,300:350],axis = 1)
peak_pos = np.max(np.array(pos_activity_ob_telc)[:,300:350],axis = 1)
    
s,p = stats.ttest_rel(peak_neg,peak_pos)

#%%


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

time_laser = np.arange(-150,1350,0.5) 

plt.figure(dpi = 300, figsize = (8,6))

plt.subplot(321)
mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    

plt.plot(time_laser[280:400],mean_ob[280:400],label = 'OB', color = 'tab:orange')
plt.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')

plt.ylabel('Firing Rate (z)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,16])
plt.xlim(-10,50)
plt.title('Control')
plt.legend()


mean_pos_ob = np.mean(pos_activity_ob,axis = 0)    
mean_neg_ob = np.mean(neg_activity_ob,axis = 0)    
error_pos_ob = stats.sem(pos_activity_ob,axis = 0)    
error_neg_ob = stats.sem(neg_activity_ob,axis = 0)    

mean_pos_pc = np.mean(pos_activity_pc,axis = 0)    
mean_neg_pc = np.mean(neg_activity_pc,axis = 0)    
error_pos_pc = stats.sem(pos_activity_pc,axis = 0)    
error_neg_pc = stats.sem(neg_activity_pc,axis = 0)


plt.subplot(323)

plt.plot(time_laser[280:400],mean_pos_ob[280:400],label = 'PCx', color = 'c')
plt.fill_between(time_laser[280:400], mean_pos_ob[280:400]-error_pos_ob[280:400],mean_pos_ob[280:400]+error_pos_ob[280:400],alpha = 0.2, color = 'c')

plt.plot(time_laser[280:400],mean_neg_ob[280:400],label = 'PCx', color = 'm')
plt.fill_between(time_laser[280:400], mean_neg_ob[280:400]-error_neg_ob[280:400],mean_neg_ob[280:400]+error_neg_ob[280:400],alpha = 0.2, color = 'm')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Firing Rate (z)')
plt.ylim([0,10])

plt.xlim(-10,50)

plt.subplot(325)

plt.plot(time_laser[280:400],mean_pos_pc[280:400],label = '+ CS Weight', color = 'c')
plt.fill_between(time_laser[280:400], mean_pos_pc[280:400]-error_pos_pc[280:400],mean_pos_pc[280:400]+error_pos_pc[280:400],alpha = 0.2, color = 'c')

plt.plot(time_laser[280:400],mean_neg_pc[280:400],label = '- CS Weight', color = 'm')
plt.fill_between(time_laser[280:400], mean_neg_pc[280:400]-error_neg_pc[280:400],mean_neg_pc[280:400]+error_neg_pc[280:400],alpha = 0.2, color = 'm')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from laser onset (ms)')
plt.ylabel('Firing Rate (z)')
plt.xlim(-10,50)
plt.ylim([0,10])
plt.legend()


plt.subplot(322)
mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    

plt.plot(time_laser[280:400],mean_ob[280:400],label = 'OB', color = 'tab:orange')
plt.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,16])
plt.xlim(-10,50)
plt.legend()
plt.title('TeLC')


mean_pos_ob = np.mean(pos_activity_ob_telc,axis = 0)    
mean_neg_ob = np.mean(neg_activity_ob_telc,axis = 0)    
error_pos_ob = stats.sem(pos_activity_ob_telc,axis = 0)    
error_neg_ob = stats.sem(neg_activity_ob_telc,axis = 0)    

mean_pos_pc = np.mean(pos_activity_pc_telc,axis = 0)    
mean_neg_pc = np.mean(neg_activity_pc_telc,axis = 0)    
error_pos_pc = stats.sem(pos_activity_pc_telc,axis = 0)    
error_neg_pc = stats.sem(neg_activity_pc_telc,axis = 0)


plt.subplot(324) 

plt.plot(time_laser[280:400],mean_pos_ob[280:400],label = 'PCx', color = 'c')
plt.fill_between(time_laser[280:400], mean_pos_ob[280:400]-error_pos_ob[280:400],mean_pos_ob[280:400]+error_pos_ob[280:400],alpha = 0.2, color = 'c')

plt.plot(time_laser[280:400],mean_neg_ob[280:400],label = 'PCx', color = 'm')
plt.fill_between(time_laser[280:400], mean_neg_ob[280:400]-error_neg_ob[280:400],mean_neg_ob[280:400]+error_neg_ob[280:400],alpha = 0.2, color = 'm')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.ylim([0,10])

plt.xlim(-10,50)

plt.subplot(326)

plt.plot(time_laser[280:400],mean_pos_pc[280:400],label = '+ CS Weight', color = 'c')
plt.fill_between(time_laser[280:400], mean_pos_pc[280:400]-error_pos_pc[280:400],mean_pos_pc[280:400]+error_pos_pc[280:400],alpha = 0.2, color = 'c')

plt.plot(time_laser[280:400],mean_neg_pc[280:400],label = '- CS Weight', color = 'm')
plt.fill_between(time_laser[280:400], mean_neg_pc[280:400]-error_neg_pc[280:400],mean_neg_pc[280:400]+error_neg_pc[280:400],alpha = 0.2, color = 'm')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from laser onset (ms)')
plt.xlim(-10,50)
plt.ylim([0,10])
plt.legend()

#plt.savefig('laser_activity.pdf')


#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

time_laser = np.arange(-150,1350,0.5) 

plt.figure(dpi = 300, figsize = (4,4))

plt.subplot(211)
mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    

plt.plot(time_laser[0:2600],mean_ob[0:2600],label = 'OB', color = 'tab:orange')
plt.fill_between(time_laser[0:2600], mean_ob[0:2600]-error_ob[0:2600],mean_ob[0:2600]+error_ob[0:2600],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[0:2600],mean_pc[0:2600],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[0:2600], mean_pc[0:2600]-error_pc[0:2600],mean_pc[0:2600]+error_pc[0:2600],alpha = 0.2, color = 'tab:blue')



plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([-0.4,1.5])

plt.subplot(212)
mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    

plt.plot(time_laser[0:2600],mean_ob[0:2600],label = 'OB', color = 'tab:orange')
plt.fill_between(time_laser[0:2600], mean_ob[0:2600]-error_ob[0:2600],mean_ob[0:2600]+error_ob[0:2600],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[0:2600],mean_pc[0:2600],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[0:2600], mean_pc[0:2600]-error_pc[0:2600],mean_pc[0:2600]+error_pc[0:2600],alpha = 0.2, color = 'tab:blue')



plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([-0.4,1.5])

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

time_laser = np.arange(-150,1350,0.5) 

plt.figure(dpi = 300, figsize = (3,6))

ax1 = plt.subplot(311)
mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    


plt.plot(time_laser[0:2600],mean_ob[0:2600],label = 'THY', color = 'tab:orange')
plt.fill_between(time_laser[0:2600], mean_ob[0:2600]-error_ob[0:2600],mean_ob[0:2600]+error_ob[0:2600],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[0:2600],mean_pc[0:2600],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[0:2600], mean_pc[0:2600]-error_pc[0:2600],mean_pc[0:2600]+error_pc[0:2600],alpha = 0.2, color = 'tab:blue')

plt.xlim([-100,1100])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([-0.4,1.5])
plt.ylabel('CSA (z-scored)')
plt.yscale('symlog')
plt.legend()

axins = inset_axes(ax1, width=1, height=0.6,loc = 'upper center')

plt.plot(time_laser[280:400],mean_ob[280:400],label = 'OB', color = 'tab:orange')
plt.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')

plt.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')

plt.ylim([0,16])
plt.xlim(-10,30)



ax1 = plt.subplot(312)

mean_ob = np.mean(mean_activity_proj_ob,axis = 0)       
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    

plt.plot(time_laser[0:2600],mean_ob[0:2600],label = 'THY', color = 'tab:orange')
plt.fill_between(time_laser[0:2600], mean_ob[0:2600]-error_ob[0:2600],mean_ob[0:2600]+error_ob[0:2600],alpha = 0.2, color = 'tab:orange')



mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    

plt.plot(time_laser[0:2600],mean_ob[0:2600],label = 'Telc', color = 'grey')
plt.fill_between(time_laser[0:2600], mean_ob[0:2600]-error_ob[0:2600],mean_ob[0:2600]+error_ob[0:2600],alpha = 0.2, color = 'grey')

plt.legend()

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([-0.4,1.5])
plt.ylabel('CSA (z-scored)')
plt.yscale('symlog')

plt.xlim([-100,1100])

axins = inset_axes(ax1, width=1, height=0.6,loc = 'upper center')
mean_ob = np.mean(mean_activity_proj_ob,axis = 0)       
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    

plt.plot(time_laser[280:400],mean_ob[280:400],label = 'THY', color = 'tab:orange')
plt.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')



mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    

plt.plot(time_laser[280:400],mean_ob[280:400],label = 'THY', color = 'grey')
plt.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'grey')

plt.xlim([-10,30])
plt.ylim([0,16])

ax2 = plt.subplot(313)

mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    

plt.plot(time_laser[0:2600],mean_pc[0:2600],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[0:2600], mean_pc[0:2600]-error_pc[0:2600],mean_pc[0:2600]+error_pc[0:2600],alpha = 0.2, color = 'tab:blue')



mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    

plt.plot(time_laser[0:2600],mean_pc[0:2600],label = 'PCx', color = 'grey')
plt.fill_between(time_laser[0:2600], mean_pc[0:2600]-error_pc[0:2600],mean_pc[0:2600]+error_pc[0:2600],alpha = 0.2, color = 'grey')



plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([-0.4,1.5])
plt.yscale('symlog')
plt.xlim([-100,1100])

plt.xlabel('Time from laser onset (ms)')
plt.ylabel('CSA (z-scored)')

axins = inset_axes(ax2, width=1, height=0.6,loc = 'upper center')


mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    

plt.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
plt.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')



mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    

plt.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'grey')
plt.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'grey')

plt.xlim([-10,30])
plt.ylim([0,16])

plt.savefig('telc_laser.pdf')
#%%

time_laser = np.arange(-150,1350,0.5) 

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1], 'wspace':0.05}, dpi = 300, figsize = (4,4))

mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)    


ax1.plot(time_laser[280:400],mean_ob[280:400],label = 'OB', color = 'tab:orange')
ax1.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')

ax1.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
ax1.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')

ax1.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


ax2.plot(time_laser[400:2700],mean_ob[400:2700],label = 'OB', color = 'tab:orange')
ax2.fill_between(time_laser[400:2700], mean_ob[400:2700]-error_ob[400:2700],mean_ob[400:2700]+error_ob[400:2700],alpha = 0.2, color = 'tab:orange')

ax2.plot(time_laser[400:2700],mean_pc[400:2700],label = 'PCx', color = 'tab:blue')
ax2.fill_between(time_laser[400:2700], mean_pc[400:2700]-error_pc[400:2700],mean_pc[400:2700]+error_pc[400:2700],alpha = 0.2, color = 'tab:blue')


ax1.set_xlim(-10, 50)  # outliers only
ax2.set_xlim(50, 1200)  # most of the data

# hide the spines between ax and ax2
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)

d = 2  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)


# plot the same data on both axes

mean_ob = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    


ax3.plot(time_laser[280:400],mean_ob[280:400],label = 'OB', color = 'tab:orange')
ax3.fill_between(time_laser[280:400], mean_ob[280:400]-error_ob[280:400],mean_ob[280:400]+error_ob[280:400],alpha = 0.2, color = 'tab:orange')

ax3.plot(time_laser[280:400],mean_pc[280:400],label = 'PCx', color = 'tab:blue')
ax3.fill_between(time_laser[280:400], mean_pc[280:400]-error_pc[280:400],mean_pc[280:400]+error_pc[280:400],alpha = 0.2, color = 'tab:blue')

ax3.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


ax4.plot(time_laser[400:2700],mean_ob[400:2700],label = 'OB', color = 'tab:orange')
ax4.fill_between(time_laser[400:2700], mean_ob[400:2700]-error_ob[400:2700],mean_ob[400:2700]+error_ob[400:2700],alpha = 0.2, color = 'tab:orange')

ax4.plot(time_laser[400:2700],mean_pc[400:2700],label = 'PCx', color = 'tab:blue')
ax4.fill_between(time_laser[400:2700], mean_pc[400:2700]-error_pc[400:2700],mean_pc[400:2700]+error_pc[400:2700],alpha = 0.2, color = 'tab:blue')


# zoom-in / limit the view to different portions of the data
ax3.set_xlim(-10, 50)  # outliers only
ax4.set_xlim(50, 1200)  # most of the data

# hide the spines between ax and ax2
ax3.spines.right.set_visible(False)
ax4.spines.left.set_visible(False)

ax3.tick_params(labeltop=False)  # don't put tick labels at the top

d = 2  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax3.plot([1, 1], [0, 1], transform=ax3.transAxes, **kwargs)
ax4.plot([0, 0], [0, 1], transform=ax4.transAxes, **kwargs)


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


#%% check all laser intesities


neg_activity_ob_intensity = []
pos_activity_ob_intensity = []

neg_activity_pc_intensity = []
pos_activity_pc_intensity = []

mean_activity_proj_ob_intensity = []
mean_activity_proj_pc_intensity = []

unique_laser_intensity_session = []

for x in range(len(ob_proj_laser_animals_thy)):
    
    laser_intensity = laser_intensities_animals_thy[x]
    
    unique_laser_intensity = np.unique(laser_intensity)
    unique_laser_intensity_session.append(unique_laser_intensity)
    
    ob_laser = np.array(ob_units_laser_animals_thy[x]) 
    pc_laser = np.array(pc_units_laser_animals_thy[x]) 
    
    
    mean_activity_proj_ob = []
    mean_activity_proj_pc = []
    
    neg_activity_ob = []
    pos_activity_ob = []
    neg_activity_pc = []
    pos_activity_pc = []
    
    for y in [2, 3, 4, 5, 6, 7, 8, 9]: 
        
        indexes = laser_intensity == y
    
        activity_proj_ob = np.sum(ob_laser[indexes,:,:]*np.squeeze(ob_loadings_animals_thy[x][:])[np.newaxis,np.newaxis,:],axis = -1)
        mean_activity_proj_ob.append(np.mean(activity_proj_ob,axis = 0))
    
        mean_activity_ob = np.mean(ob_laser[indexes,:,:],axis = 0)
        neg_neurons_ob = np.squeeze(np.argmin(ob_loadings_animals_thy[x]))
        pos_neurons_ob = np.squeeze(np.argmax(ob_loadings_animals_thy[x]))
        neg_activity_ob.append(mean_activity_ob[:,neg_neurons_ob])
        pos_activity_ob.append(mean_activity_ob[:,pos_neurons_ob])
        
        activity_proj_pc = np.sum(pc_laser[indexes,:,:]*np.squeeze(pc_loadings_animals_thy[x][:])[np.newaxis,np.newaxis,:],axis = -1)
        mean_activity_proj_pc.append(np.mean(activity_proj_pc,axis = 0))
    
        mean_activity_pc = np.mean(pc_laser[indexes,:,:],axis = 0)
        neg_neurons_pc = np.squeeze(np.argmin(pc_loadings_animals_thy[x]))
        pos_neurons_pc = np.squeeze(np.argmax(pc_loadings_animals_thy[x]))
        neg_activity_pc.append(mean_activity_pc[:,neg_neurons_pc])
        pos_activity_pc.append(mean_activity_pc[:,pos_neurons_pc])
  
    
     
    neg_activity_ob_intensity.append(neg_activity_ob)
    pos_activity_ob_intensity.append(pos_activity_ob)

    neg_activity_pc_intensity.append(neg_activity_pc)
    pos_activity_pc_intensity.append(pos_activity_pc)

    mean_activity_proj_ob_intensity.append(mean_activity_proj_ob)
    mean_activity_proj_pc_intensity.append(mean_activity_proj_pc)
      
# telc
    
neg_activity_ob_intensity_telc = []
pos_activity_ob_intensity_telc = []

neg_activity_pc_intensity_telc = []
pos_activity_pc_intensity_telc = []

mean_activity_proj_ob_intensity_telc = []
mean_activity_proj_pc_intensity_telc = []

unique_laser_intensity_session_telc = []

for x in range(len(ob_proj_laser_animals_telc)):
    
    laser_intensity = laser_intensities_animals_telc[x]
    
    unique_laser_intensity = np.unique(laser_intensity)
    unique_laser_intensity_session_telc.append(unique_laser_intensity)
    
    ob_laser = np.array(ob_units_laser_animals_telc[x]) 
    pc_laser = np.array(pc_units_laser_animals_telc[x]) 
    
    
    mean_activity_proj_ob = []
    mean_activity_proj_pc = []
    
    neg_activity_ob = []
    pos_activity_ob = []
    neg_activity_pc = []
    pos_activity_pc = []
    
    for y in [2, 3, 4, 5, 6, 7, 8, 9]: 
        
        indexes = laser_intensity == y
    
        activity_proj_ob = np.sum(ob_laser[indexes,:,:]*np.squeeze(ob_loadings_animals_telc[x][:])[np.newaxis,np.newaxis,:],axis = -1)
        mean_activity_proj_ob.append(np.mean(activity_proj_ob,axis = 0))
    
        mean_activity_ob = np.mean(ob_laser[indexes,:,:],axis = 0)
        neg_neurons_ob = np.squeeze(np.argmin(ob_loadings_animals_telc[x]))
        pos_neurons_ob = np.squeeze(np.argmax(ob_loadings_animals_telc[x]))
        neg_activity_ob.append(mean_activity_ob[:,neg_neurons_ob])
        pos_activity_ob.append(mean_activity_ob[:,pos_neurons_ob])
        
        activity_proj_pc = np.sum(pc_laser[indexes,:,:]*np.squeeze(pc_loadings_animals_telc[x][:])[np.newaxis,np.newaxis,:],axis = -1)
        mean_activity_proj_pc.append(np.mean(activity_proj_pc,axis = 0))
    
        mean_activity_pc = np.mean(pc_laser[indexes,:,:],axis = 0)
        neg_neurons_pc = np.squeeze(np.argmin(pc_loadings_animals_telc[x]))
        pos_neurons_pc = np.squeeze(np.argmax(pc_loadings_animals_telc[x]))
        neg_activity_pc.append(mean_activity_pc[:,neg_neurons_pc])
        pos_activity_pc.append(mean_activity_pc[:,pos_neurons_pc])
  
    
     
    neg_activity_ob_intensity_telc.append(neg_activity_ob)
    pos_activity_ob_intensity_telc.append(pos_activity_ob)

    neg_activity_pc_intensity_telc.append(neg_activity_pc)
    pos_activity_pc_intensity_telc.append(pos_activity_pc)

    mean_activity_proj_ob_intensity_telc.append(mean_activity_proj_ob)
    mean_activity_proj_pc_intensity_telc.append(mean_activity_proj_pc)
      

#%% comm subspace actity 

#  check stats

s_ob_initial_animals = []
p_ob_initial_animals = []
s_pc_initial_animals = []
p_pc_initial_animals = []

s_ob_sustained_animals = []
p_ob_sustained_animals = []
s_pc_sustained_animals = []
p_pc_sustained_animals = []

df = []

time_sustained = np.arange(1300,2300)
time_initial = np.arange(300,350)

for x in range(8):
    
    # initial 
    
    time = time_initial
    
    ob_thy = np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,x,time],axis = -1)
    ob_telc = np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,x,time],axis = -1)
    
    s_ob_initial,p_ob_initial = stats.ttest_ind(ob_thy,ob_telc, nan_policy='omit',equal_var = False)
    s_ob_initial_animals.append(s_ob_initial)
    p_ob_initial_animals.append(p_ob_initial)
    
    pc_thy = np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,x,time],axis = -1)
    pc_telc = np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,x,time],axis = -1)
    
    s_pc_initial,p_pc_initial = stats.ttest_ind(pc_thy,pc_telc, nan_policy='omit',equal_var = False)
    s_pc_initial_animals.append(s_pc_initial)
    p_pc_initial_animals.append(p_pc_initial)
    
    
    # sustained
    
    
    time = time_sustained
    
    ob_thy = np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,x,time],axis = -1)
    ob_telc = np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,x,time],axis = -1)
    
    
    s_ob_sustained,p_ob_sustained = stats.ttest_ind(ob_thy,ob_telc, nan_policy='omit',equal_var = False)
    s_ob_sustained_animals.append(s_ob_sustained)
    p_ob_sustained_animals.append(p_ob_sustained)
    
    pc_thy = np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,x,time],axis = -1)
    pc_telc = np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,x,time],axis = -1)
    
    s_pc_sustained,p_pc_sustained = stats.ttest_ind(pc_thy,pc_telc, nan_policy='omit',equal_var = False)
    s_pc_sustained_animals.append(s_pc_sustained)
    p_pc_sustained_animals.append(p_pc_sustained)
    
    df.append(ob_thy.shape[0]-1+ob_telc.shape[0]-1)
    
    
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
    
plt.figure(dpi = 300, figsize = (8,6))

plt.subplot(221)
time = time_initial

laser_series =  np.array([0,1,5,10,20,30,40,50])


mean_proj_ob_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')


mean_proj_ob_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_ob_intensity,error_proj_ob_intensity,fmt = '-o', color = 'tab:orange', label = 'Control')
plt.errorbar(laser_series,mean_proj_ob_intensity_telc,error_proj_ob_intensity_telc,fmt = '-s', color = 'tab:orange',markerfacecolor = 'white', label = 'TeLC')

plt.scatter(laser_series[np.array(p_ob_initial_animals)<0.05],15*np.ones(np.sum(np.array(p_ob_initial_animals)<0.05)))
plt.ylim([-0.2,20])
plt.ylabel('Initial CSA (0-25 ms)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend()
plt.xticks(np.arange(0,60,10))
plt.title('OB')

plt.subplot(222)


mean_proj_pc_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_pc_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_pc_intensity,error_proj_pc_intensity,fmt = '-o', color = 'tab:blue', label = 'Control')
plt.errorbar(laser_series,mean_proj_pc_intensity_telc,error_proj_pc_intensity_telc,fmt = '-s', color = 'tab:blue',markerfacecolor = 'white', label = 'TeLC')

plt.scatter(laser_series[np.array(p_pc_initial_animals)<0.05],15*np.ones(np.sum(np.array(p_pc_initial_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,20])
plt.xticks(np.arange(0,60,10))
#plt.ylabel('PCx CSA (z)')
plt.title('PCx')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.subplot(223)
time = time_sustained



mean_proj_ob_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_ob_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')


plt.errorbar(laser_series,mean_proj_ob_intensity,error_proj_ob_intensity,fmt = '-o', color = 'tab:orange', label = 'Control')
plt.errorbar(laser_series,mean_proj_ob_intensity_telc,error_proj_ob_intensity_telc,fmt = '-s', color = 'tab:orange',markerfacecolor = 'white', label = 'TeLC')
plt.scatter(laser_series[np.array(p_ob_sustained_animals)<0.05],1.5*np.ones(np.sum(np.array(p_ob_sustained_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,2])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(np.arange(0,60,10))
plt.ylabel('Sustained CSA (500-1000 ms)')

plt.subplot(224)


mean_proj_pc_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_pc_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_pc_intensity,error_proj_pc_intensity,fmt = '-o', color = 'tab:blue', label = 'Control')
plt.errorbar(laser_series,mean_proj_pc_intensity_telc,error_proj_pc_intensity_telc,fmt = '-s', color = 'tab:blue',markerfacecolor = 'white', label = 'TeLC')
plt.scatter(laser_series[np.array(p_pc_sustained_animals)<0.05],1.5*np.ones(np.sum(np.array(p_pc_sustained_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,2])
plt.xticks(np.arange(0,60,10))
plt.xlabel('Laser Intensity (mW/mm2)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#plt.savefig('laser_intensoty.pdf')

#%% use peak activity instead of initial average 

#  check stats

s_ob_initial_animals = []
p_ob_initial_animals = []
s_pc_initial_animals = []
p_pc_initial_animals = []

s_ob_sustained_animals = []
p_ob_sustained_animals = []
s_pc_sustained_animals = []
p_pc_sustained_animals = []

df = []

time_sustained = np.arange(1300,2300)
time_initial = np.arange(300,350)

for x in range(8):
    
    # initial 
    
    time = time_initial
    
    ob_thy = np.nanmax(np.array(mean_activity_proj_ob_intensity)[:,x,time],axis = -1)
    ob_telc = np.nanmax(np.array(mean_activity_proj_ob_intensity_telc)[:,x,time],axis = -1)
    
    s_ob_initial,p_ob_initial = stats.ttest_ind(ob_thy,ob_telc, nan_policy='omit',equal_var = False)
    s_ob_initial_animals.append(s_ob_initial)
    p_ob_initial_animals.append(p_ob_initial)
    
    pc_thy = np.nanmax(np.array(mean_activity_proj_pc_intensity)[:,x,time],axis = -1)
    pc_telc = np.nanmax(np.array(mean_activity_proj_pc_intensity_telc)[:,x,time],axis = -1)
    
    s_pc_initial,p_pc_initial = stats.ttest_ind(pc_thy,pc_telc, nan_policy='omit',equal_var = False)
    s_pc_initial_animals.append(s_pc_initial)
    p_pc_initial_animals.append(p_pc_initial)
    
    # sustained
    
    
    time = time_sustained
    
    ob_thy = np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,x,time],axis = -1)
    ob_telc = np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,x,time],axis = -1)
    
    
    s_ob_sustained,p_ob_sustained = stats.ttest_ind(ob_thy,ob_telc, nan_policy='omit',equal_var = False)
    s_ob_sustained_animals.append(s_ob_sustained)
    p_ob_sustained_animals.append(p_ob_sustained)
    
    pc_thy = np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,x,time],axis = -1)
    pc_telc = np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,x,time],axis = -1)
    
    s_pc_sustained,p_pc_sustained = stats.ttest_ind(pc_thy,pc_telc, nan_policy='omit',equal_var = False)
    s_pc_sustained_animals.append(s_pc_sustained)
    p_pc_sustained_animals.append(p_pc_sustained)
    
    df.append(ob_thy.shape[0]-1+ob_telc.shape[0]-1)
    
    
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
    
plt.figure(dpi = 300, figsize = (8,6))

plt.subplot(221)
time = time_initial

laser_series =  np.array([0,1,5,10,20,30,40,50])


mean_proj_ob_intensity = np.nanmean(np.nanmax(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity = stats.sem(np.nanmax(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')


mean_proj_ob_intensity_telc = np.nanmean(np.nanmax(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity_telc = stats.sem(np.nanmax(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_ob_intensity,error_proj_ob_intensity,fmt = '-o', color = 'tab:orange', label = 'Control')
plt.errorbar(laser_series,mean_proj_ob_intensity_telc,error_proj_ob_intensity_telc,fmt = '-s', color = 'tab:orange',markerfacecolor = 'white', label = 'TeLC')

plt.scatter(laser_series[np.array(p_ob_initial_animals)<0.05],25*np.ones(np.sum(np.array(p_ob_initial_animals)<0.05)))
plt.ylim([-0.2,40])
plt.ylabel('Peak CSA')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend()
plt.xticks(np.arange(0,60,10))
plt.title('OB')

plt.subplot(222)


mean_proj_pc_intensity = np.nanmean(np.nanmax(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity = stats.sem(np.nanmax(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_pc_intensity_telc = np.nanmean(np.nanmax(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity_telc = stats.sem(np.nanmax(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_pc_intensity,error_proj_pc_intensity,fmt = '-o', color = 'tab:blue', label = 'Control')
plt.errorbar(laser_series,mean_proj_pc_intensity_telc,error_proj_pc_intensity_telc,fmt = '-s', color = 'tab:blue',markerfacecolor = 'white', label = 'TeLC')

plt.scatter(laser_series[np.array(p_pc_initial_animals)<0.05],25*np.ones(np.sum(np.array(p_pc_initial_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,40])
plt.xticks(np.arange(0,60,10))
#plt.ylabel('PCx CSA (z)')
plt.title('PCx')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.subplot(223)
time = time_sustained



mean_proj_ob_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_ob_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_ob_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')


plt.errorbar(laser_series,mean_proj_ob_intensity,error_proj_ob_intensity,fmt = '-o', color = 'tab:orange', label = 'Control')
plt.errorbar(laser_series,mean_proj_ob_intensity_telc,error_proj_ob_intensity_telc,fmt = '-s', color = 'tab:orange',markerfacecolor = 'white', label = 'TeLC')
plt.scatter(laser_series[np.array(p_ob_sustained_animals)<0.05],1.5*np.ones(np.sum(np.array(p_ob_sustained_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,2])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(np.arange(0,60,10))
plt.ylabel('Sustained CSA (500-1000 ms)')

plt.subplot(224)


mean_proj_pc_intensity = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_proj_pc_intensity_telc = np.nanmean(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_proj_pc_intensity_telc = stats.sem(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_proj_pc_intensity,error_proj_pc_intensity,fmt = '-o', color = 'tab:blue', label = 'Control')
plt.errorbar(laser_series,mean_proj_pc_intensity_telc,error_proj_pc_intensity_telc,fmt = '-s', color = 'tab:blue',markerfacecolor = 'white', label = 'TeLC')
plt.scatter(laser_series[np.array(p_pc_sustained_animals)<0.05],1.5*np.ones(np.sum(np.array(p_pc_sustained_animals)<0.05)), marker = '*', color = 'black')

plt.ylim([-0.2,2])
plt.xticks(np.arange(0,60,10))
plt.xlabel('Laser Intensity (mW/mm2)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.savefig('laser_intensoty.pdf')

#%% check two-way anova for each case


ob_thy = np.concatenate(np.nanmax(np.array(mean_activity_proj_ob_intensity)[:,:,:],axis = -1))
ob_telc = np.concatenate(np.nanmax(np.array(mean_activity_proj_ob_intensity_telc)[:,:,:],axis = -1))

#
# Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
  

response = np.hstack([ob_thy,ob_telc])
laser_intensity = np.hstack([np.tile(laser_series,5),np.tile(laser_series,15)])

data_type = np.hstack([np.repeat('THY',ob_thy.shape[0]),np.repeat('TeLC',ob_telc.shape[0])])


dataframe = pd.DataFrame({'Type': data_type,
                          'Intensity': laser_intensity,
                          'Response': response})

#animal = np.tile([1,2,3,4,5,6],4)

# Performing two-way ANOVA
model = ols('Response ~ C(Type) + C(Intensity) +\
C(Type):C(Intensity)',
            data=dataframe).fit()
result = sm.stats.anova_lm(model, type=2)
  
# Print the result
print(result)

#%% pc

pc_thy = np.concatenate(np.nanmax(np.array(mean_activity_proj_pc_intensity)[:,:,:],axis = -1))
pc_telc = np.concatenate(np.nanmax(np.array(mean_activity_proj_pc_intensity_telc)[:,:,:],axis = -1))

#
# Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
  

response = np.hstack([pc_thy,pc_telc])
laser_intensity = np.hstack([np.tile(laser_series,5),np.tile(laser_series,15)])

data_type = np.hstack([np.repeat('THY',ob_thy.shape[0]),np.repeat('TeLC',ob_telc.shape[0])])


dataframe = pd.DataFrame({'Type': data_type,
                          'Intensity': laser_intensity,
                          'Response': response})

#animal = np.tile([1,2,3,4,5,6],4)

# Performing two-way ANOVA
model = ols('Response ~ C(Type) + C(Intensity) +\
C(Type):C(Intensity)',
            data=dataframe).fit()
result = sm.stats.anova_lm(model, type=2)
  
# Print the result
print(result)

#%% ob sustauned 

time = time_sustained
ob_thy = np.concatenate(np.nanmean(np.array(mean_activity_proj_ob_intensity)[:,:,time],axis = -1))
ob_telc = np.concatenate(np.nanmean(np.array(mean_activity_proj_ob_intensity_telc)[:,:,time],axis = -1))

#
# Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
  

response = np.hstack([ob_thy,ob_telc])
laser_intensity = np.hstack([np.tile(laser_series,5),np.tile(laser_series,15)])

data_type = np.hstack([np.repeat('THY',ob_thy.shape[0]),np.repeat('TeLC',ob_telc.shape[0])])


dataframe = pd.DataFrame({'Type': data_type,
                          'Intensity': laser_intensity,
                          'Response': response})

#animal = np.tile([1,2,3,4,5,6],4)

# Performing two-way ANOVA
model = ols('Response ~ C(Type) + C(Intensity) +\
C(Type):C(Intensity)',
            data=dataframe).fit()
result = sm.stats.anova_lm(model, type=2)
  
# Print the result
print(result)

#%% pc sustauned 

time = time_sustained
pc_thy = np.concatenate(np.nanmean(np.array(mean_activity_proj_pc_intensity)[:,:,time],axis = -1))
pc_telc = np.concatenate(np.nanmean(np.array(mean_activity_proj_pc_intensity_telc)[:,:,time],axis = -1))

#
# Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
  

response = np.hstack([pc_thy,pc_telc])
laser_intensity = np.hstack([np.tile(laser_series,5),np.tile(laser_series,15)])

data_type = np.hstack([np.repeat('THY',ob_thy.shape[0]),np.repeat('TeLC',ob_telc.shape[0])])


dataframe = pd.DataFrame({'Type': data_type,
                          'Intensity': laser_intensity,
                          'Response': response})

#animal = np.tile([1,2,3,4,5,6],4)

# Performing two-way ANOVA
model = ols('Response ~ C(Type) + C(Intensity) +\
C(Type):C(Intensity)',
            data=dataframe).fit()
result = sm.stats.anova_lm(model, type=2)
  
# Print the result
print(result)

#%%

plt.figure(dpi = 300, figsize = (6,6))
plt.subplot(221)
time = np.arange(300,315)

laser_series =  [0,1,5,10,20,30,40,50]


mean_neg_ob_intensity = np.nanmean(np.nanmean(np.array(neg_activity_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_neg_ob_intensity = stats.sem(np.nanmean(np.array(neg_activity_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_pos_ob_intensity = np.nanmean(np.nanmean(np.array(pos_activity_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_pos_ob_intensity = stats.sem(np.nanmean(np.array(pos_activity_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_neg_ob_intensity,error_neg_ob_intensity,fmt = '-o', color = 'm', label = '- CS Weight')
plt.errorbar(laser_series,mean_pos_ob_intensity,error_pos_ob_intensity,fmt = '-o', color = 'c', label = '+ CS Weight')

plt.ylim([-0.1,14])
plt.ylabel('OB Activity (z)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend()
plt.xticks(np.arange(0,60,10))

plt.subplot(223)


mean_neg_pc_intensity = np.nanmean(np.nanmean(np.array(neg_activity_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_neg_pc_intensity = stats.sem(np.nanmean(np.array(neg_activity_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_pos_pc_intensity = np.nanmean(np.nanmean(np.array(pos_activity_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_pos_pc_intensity = stats.sem(np.nanmean(np.array(pos_activity_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_neg_pc_intensity,error_neg_pc_intensity,fmt = '-o' ,color = 'm')
plt.errorbar(laser_series,mean_pos_pc_intensity,error_pos_pc_intensity,fmt = '-o', color = 'c')

plt.ylim([-0.1,14])
plt.xticks(np.arange(0,60,10))
plt.xlabel('Laser Intensity (mW/mm2)')
plt.ylabel('PCx Activity (z)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


plt.figure(dpi = 300, figsize = (3,6))
plt.subplot(211)
time = np.arange(500,2300)

laser_series =  [0,1,5,10,20,30,40,50]


mean_pos_ob_intensity = np.nanmean(np.nanmean(np.array(pos_activity_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_pos_ob_intensity = stats.sem(np.nanmean(np.array(pos_activity_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_pos_ob_intensity_telc = np.nanmean(np.nanmean(np.array(pos_activity_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_pos_ob_intensity_telc = stats.sem(np.nanmean(np.array(pos_activity_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_pos_ob_intensity,error_pos_ob_intensity, color = 'black', label = '+ CS Control')
plt.errorbar(laser_series,mean_pos_ob_intensity_telc,error_pos_ob_intensity_telc, color = 'tab:green', label = '+ CS TeLC')

plt.ylim([-0.2,2])
plt.ylabel('OB Activity (z)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend()
plt.xticks(np.arange(0,60,10))

plt.subplot(212)


mean_pos_pc_intensity = np.nanmean(np.nanmean(np.array(pos_activity_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_pos_pc_intensity = stats.sem(np.nanmean(np.array(pos_activity_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_pos_pc_intensity_telc = np.nanmean(np.nanmean(np.array(pos_activity_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_pos_pc_intensity_telc = stats.sem(np.nanmean(np.array(pos_activity_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_pos_pc_intensity,error_pos_pc_intensity,color = 'black')
plt.errorbar(laser_series,mean_pos_pc_intensity_telc,error_pos_pc_intensity_telc, color = 'tab:green')

plt.ylim([-0.2,2])
plt.xticks(np.arange(0,60,10))
plt.xlabel('Laser Intensity (mW/mm2)')
plt.ylabel('PCx Activity (z)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


#%%

plt.figure(dpi = 300, figsize = (3,6))
plt.subplot(211)
time = np.arange(500,2300)

laser_series =  [0,1,5,10,20,30,40,50]


mean_neg_ob_intensity = np.nanmean(np.nanmean(np.array(neg_activity_ob_intensity)[:,:,time],axis = -1),axis = 0)
error_neg_ob_intensity = stats.sem(np.nanmean(np.array(neg_activity_ob_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_neg_ob_intensity_telc = np.nanmean(np.nanmean(np.array(neg_activity_ob_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_neg_ob_intensity_telc = stats.sem(np.nanmean(np.array(neg_activity_ob_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_neg_ob_intensity,error_neg_ob_intensity, color = 'black', label = '- CS Control')
plt.errorbar(laser_series,mean_neg_ob_intensity_telc,error_neg_ob_intensity_telc, color = 'tab:green', label = '- CS TeLC')

plt.ylim([-0.2,2])
plt.ylabel('OB Activity (z)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend()
plt.xticks(np.arange(0,60,10))

plt.subplot(212)


mean_neg_pc_intensity = np.nanmean(np.nanmean(np.array(neg_activity_pc_intensity)[:,:,time],axis = -1),axis = 0)
error_neg_pc_intensity = stats.sem(np.nanmean(np.array(neg_activity_pc_intensity)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

mean_neg_pc_intensity_telc = np.nanmean(np.nanmean(np.array(neg_activity_pc_intensity_telc)[:,:,time],axis = -1),axis = 0)
error_neg_pc_intensity_telc = stats.sem(np.nanmean(np.array(neg_activity_pc_intensity_telc)[:,:,time],axis = -1),axis = 0,nan_policy='omit')

plt.errorbar(laser_series,mean_neg_pc_intensity,error_neg_pc_intensity,color = 'black')
plt.errorbar(laser_series,mean_neg_pc_intensity_telc,error_neg_pc_intensity_telc, color = 'tab:green')

plt.ylim([-0.2,0.5])
plt.xticks(np.arange(0,60,10))
plt.xlabel('Laser Intensity (mW/mm2)')
plt.ylabel('PCx Activity (z)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#%%

plt.figure(dpi = 300, figsize = (4,5))

plt.subplot(211)

mean_ob = np.mean(mean_activity_proj_ob,axis = 0)    
mean_pc = np.mean(mean_activity_proj_pc,axis = 0)    
error_ob = stats.sem(mean_activity_proj_ob,axis = 0)    
error_pc = stats.sem(mean_activity_proj_pc,axis = 0)  


mean_ob_telc = np.mean(mean_activity_proj_ob_telc,axis = 0)    
mean_pc_telc = np.mean(mean_activity_proj_pc_telc,axis = 0)    
error_ob_telc = stats.sem(mean_activity_proj_ob_telc,axis = 0)    
error_pc_telc = stats.sem(mean_activity_proj_pc_telc,axis = 0)    

plt.plot(time_laser[0:3000],mean_ob_telc[0:3000],label = 'TeLC', color = 'tab:green')
plt.fill_between(time_laser[0:3000], mean_ob_telc[0:3000]-error_ob_telc[0:3000],mean_ob_telc[0:3000]+error_ob_telc[0:3000],alpha = 0.2, color = 'tab:green')

plt.plot(time_laser[0:3000],mean_ob[0:3000],label = 'Control', color = 'black')
plt.fill_between(time_laser[0:3000], mean_ob[0:3000]-error_ob[0:3000],mean_ob[0:3000]+error_ob[0:3000],alpha = 0.2, color = 'black')

plt.ylabel('OB Firing Rate (Hz)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([0,400])
#plt.xlim(-10,200)
plt.legend()


plt.subplot(212)

plt.plot(time_laser[0:3000],mean_pc_telc[0:3000],label = 'TeLC', color = 'tab:green')
plt.fill_between(time_laser[0:3000], mean_pc_telc[0:3000]-error_pc_telc[0:3000],mean_pc_telc[0:3000]+error_pc_telc[0:3000],alpha = 0.2, color = 'tab:green')

plt.plot(time_laser[0:3000],mean_pc[0:3000],label = 'Control', color = 'black')
plt.fill_between(time_laser[0:3000], mean_pc[0:3000]-error_pc[0:3000],mean_pc[0:3000]+error_pc[0:3000],alpha = 0.2, color = 'black')

plt.ylabel('PCx Firing Rate (Hz)')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([0,400])
#plt.xlim(-10,200)
plt.legend()

#%%
time = np.arange(300,310)

plt.boxplot([np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc)[:,time],axis = 1)])

s_final_pc, p_final_pc = stats.ttest_ind(np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc)[:,time],axis = 1), equal_var = False)



#%%
plt.figure(dpi = 300)

time = np.arange(300,330)

plt.subplot(221)
plt.boxplot([np.mean(np.array(mean_activity_proj_ob)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_ob_telc)[:,time],axis = 1)])

s_inital_ob, p_initial_ob = stats.ttest_ind(np.mean(np.array(mean_activity_proj_ob)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_ob_telc)[:,time],axis = 1))


plt.subplot(222)
plt.boxplot([np.mean(np.array(mean_activity_proj_pc)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_pc_telc)[:,time],axis = 1)])

s_inital_pc, p_initial_pc = stats.ttest_ind(np.mean(np.array(mean_activity_proj_pc)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_pc_telc)[:,time],axis = 1), equal_var = False)

time = np.arange(350,2300)

plt.subplot(223)
plt.boxplot([np.mean(np.array(mean_activity_proj_ob)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_ob_telc)[:,time],axis = 1)])

s_final_ob, p_final_ob = stats.ttest_ind(np.mean(np.array(mean_activity_proj_ob)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_ob_telc)[:,time],axis = 1))


plt.subplot(224)
plt.boxplot([np.mean(np.array(mean_activity_proj_pc)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_pc_telc)[:,time],axis = 1)])

s_final_pc, p_final_pc = stats.ttest_ind(np.mean(np.array(mean_activity_proj_pc)[:,time],axis = 1),np.mean(np.array(mean_activity_proj_pc_telc)[:,time],axis = 1), equal_var = False)

#%%

plt.figure(dpi = 300)

time = np.arange(300,330)

plt.subplot(221)
plt.boxplot([np.mean(np.array(pos_activity_ob)[:,time],axis = 1),np.mean(np.array(pos_activity_ob_telc)[:,time],axis = 1)])

s_inital_ob, p_initial_ob = stats.ttest_ind(np.mean(np.array(pos_activity_ob)[:,time],axis = 1),np.mean(np.array(pos_activity_ob_telc)[:,time],axis = 1))


plt.subplot(222)
plt.boxplot([np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(pos_activity_pc_telc)[:,time],axis = 1)])

s_inital_pc, p_initial_pc = stats.ttest_ind(np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(pos_activity_pc_telc)[:,time],axis = 1), equal_var = False)

time = np.arange(350,2300)

plt.subplot(223)
plt.boxplot([np.mean(np.array(pos_activity_ob)[:,time],axis = 1),np.mean(np.array(pos_activity_ob_telc)[:,time],axis = 1)])

s_final_ob, p_final_ob = stats.ttest_ind(np.mean(np.array(pos_activity_ob)[:,time],axis = 1),np.mean(np.array(pos_activity_ob_telc)[:,time],axis = 1))


plt.subplot(224)
plt.boxplot([np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(pos_activity_pc_telc)[:,time],axis = 1)])

s_final_pc, p_final_pc = stats.ttest_ind(np.mean(np.array(pos_activity_pc)[:,time],axis = 1),np.mean(np.array(pos_activity_pc_telc)[:,time],axis = 1), equal_var = False)

#%%
plt.figure(dpi = 300)

time = np.arange(300,330)

plt.subplot(221)
plt.boxplot([np.mean(np.array(neg_activity_ob)[:,time],axis = 1),np.mean(np.array(neg_activity_ob_telc)[:,time],axis = 1)])

s_inital_ob, p_initial_ob = stats.ttest_ind(np.mean(np.array(neg_activity_ob)[:,time],axis = 1),np.mean(np.array(neg_activity_ob_telc)[:,time],axis = 1))


plt.subplot(222)
plt.boxplot([np.mean(np.array(neg_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc_telc)[:,time],axis = 1)])

s_inital_pc, p_initial_pc = stats.ttest_ind(np.mean(np.array(neg_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc_telc)[:,time],axis = 1), equal_var = False)

time = np.arange(350,2300)

plt.subplot(223)
plt.boxplot([np.mean(np.array(neg_activity_ob)[:,time],axis = 1),np.mean(np.array(neg_activity_ob_telc)[:,time],axis = 1)])

s_final_ob, p_final_ob = stats.ttest_ind(np.mean(np.array(neg_activity_ob)[:,time],axis = 1),np.mean(np.array(neg_activity_ob_telc)[:,time],axis = 1))


plt.subplot(224)
plt.boxplot([np.mean(np.array(neg_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc_telc)[:,time],axis = 1)])

s_final_pc, p_final_pc = stats.ttest_ind(np.mean(np.array(neg_activity_pc)[:,time],axis = 1),np.mean(np.array(neg_activity_pc_telc)[:,time],axis = 1), equal_var = False)


#%%

plt.subplot(121)
sns.kdeplot(np.squeeze(np.concatenate(ob_loadings_animals_thy)))
sns.kdeplot(np.squeeze(np.concatenate(ob_loadings_animals_telc)))

plt.subplot(122)
sns.kdeplot(np.squeeze(np.concatenate(pc_loadings_animals_thy)))
sns.kdeplot(np.squeeze(np.concatenate(pc_loadings_animals_telc)))


