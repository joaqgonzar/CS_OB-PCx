#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:05:23 2021

@author: joaquin
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

#%% define pac function

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


#%% experiment data 

names_anh = ['141208-1','141208-2','160819','160820','170608','170609','170614','170618','170621','170622']

anh_onset = [6,10,0,0,1,3,3,2,2,0]

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'


#%% get recording times for anh 

names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

os.chdir(directory)


exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)

awake_times = np.array([names,np.array(exp_data[2][1:14])])

ob_bank = exp_data[0][1:14]
pfx_bank = exp_data[0][14:]

recording_times = []
non_awake_trials_animals = []

for name in names:
    
    print(name)
    os.chdir(directory+'/Simul/processed/'+name)
    
    trial_num = scipy.io.loadmat(name+'_bank2_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0].shape[0]
        
    FT = int(awake_times[1,int(np.where(name == awake_times[0,:])[0][0])])
    
    #
    awake_trials = []
    non_awake_trials = []
    for x in range(trial_num):
        awake_trials.append(scipy.io.loadmat(name+'_bank2_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0][x][0][0:FT])
        non_awake_trials.append(scipy.io.loadmat(name+'_bank2_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0][x][0][FT:])
        
    # get max awake trial among awake_trials 
    trial = []
    for x in range(len(awake_trials)):
        if len(awake_trials[x])>0:
           trial.append(awake_trials[x])      
                
    max_wake_time = np.max(trial)
    recording_times.append(int(np.round(max_wake_time)))
    non_awake_trials_animals.append(non_awake_trials)

#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:18:27 2024

@author: pcanalisis2
"""

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
from pyinform.transferentropy import transfer_entropy


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


#%%
directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset/'

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

ob_bank = np.array(ob_bank)[[0,1,3,4,5,6,8,9,11,12]]
pc_bank = np.array(pc_bank)[[0,1,3,4,5,6,8,9,11,12]]

#%%

ob_loadings_anh_animals = []
pc_loadings_anh_animals = []

ob_loadings_awake_animals = []
pc_loadings_awake_animals = []

r_proj_awake_animals = []
r_mua_wake_animals = []

r_proj_anh_animals = []
r_mua_anh_animals = []


ob_comm_resp_phase_animals_awake = []
pc_comm_resp_phase_animals_awake = []

ob_neurons_resp_phase_animals_awake = []
pc_neurons_resp_phase_animals_awake = []

ob_mi_neurons_animals_awake = []
pc_mi_neurons_animals_awake = []

ob_comm_resp_phase_animals_anh = []
pc_comm_resp_phase_animals_anh = []

ob_neurons_resp_phase_animals_anh = []
pc_neurons_resp_phase_animals_anh = []

ob_mi_neurons_animals_anh = []
pc_mi_neurons_animals_anh = []

p_ob_proj_awake_animals = []
p_pc_proj_awake_animals = []
p_ob_proj_anh_animals = []
p_pc_proj_anh_animals = []
c_awake_animals = []
c_anh_animals = []
c_ob_proj_awake_animals = []
c_pc_proj_awake_animals = []
p_resp_awake_animals = []
c_ob_proj_anh_animals = []
c_pc_proj_anh_animals = []
p_resp_anh_animals = []

r_proj_awake_anh_animals = []

for index,name in enumerate(names_anh):
    
    
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')

    print(name)

    
    # get respiration
    
    os.chdir(directory+'/Simul/processed/'+name)
    
    srate_resp = 2000
    start_anh = recording_times[index]*srate_resp
    
    
    start_time = int(anh_onset[index]*60*srate_resp)
    anh_duration = int(start_time+30*60*srate_resp)
    
    
    #
    os.chdir(directory+'/Simul/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index])+'.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index])+'.mat')['SpikeTimes']['tsec']

    resp = scipy.io.loadmat(name+'.mat')['RRR']
    srate_resp = 2000
    
    
    ob_spikes = []
    for x in range(len(spike_times_ob)):
        ob_spikes.append(spike_times_ob[int(x)][0])

    units_ob = get_spike_matrix(ob_spikes)


    # get pc neurons 

    pc_spikes = []
    for x in range(len(spike_times_pc)):
        pc_spikes.append(spike_times_pc[int(x)][0])

    units_pc = get_spike_matrix(pc_spikes)

    
    
    # load PCx lfp
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')
    lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    recording_time = lfp.shape[1]
    srate = 2000
    
    units_ob_awake = units_ob[:,0:recording_time]
    units_pc_awake = units_pc[:,0:recording_time]
    resp_awake = resp[0:recording_time]
    
    resp_anh = np.squeeze(resp[start_anh:start_anh + 30*60*srate_resp])
    units_ob_anh = units_ob[:,start_anh:start_anh + 30*60*srate_resp]
    units_pc_anh = units_pc[:,start_anh:start_anh + 30*60*srate_resp]
    
    
    # compute CCA anhestesia 
    
    # convolve spike trains

    srate_resample = 2000
    kernel = signal.gaussian(int(0.1*srate_resample),20) 

    conv_neurons_ob_anh = []
    for x in range(units_ob.shape[0]):
        conv = signal.convolve(np.squeeze(units_ob_anh[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_ob_anh.append(conv)
        
    conv_neurons_pc_anh = []
    for x in range(units_pc.shape[0]):
        conv = signal.convolve(np.squeeze(units_pc_anh[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_pc_anh.append(conv)
    
    # perform canonical correlation analysis 
    
    X = np.array(conv_neurons_ob_anh)[1:,:].T
    Y = np.array(conv_neurons_pc_anh)[1:,:].T
    
    firing_neurons_ob_anh = np.sum(X,axis = 0)>0
    firing_neurons_pc_anh = np.sum(Y,axis = 0)>0
    
    X = stats.zscore(X[:,firing_neurons_ob_anh],axis = 0)
    Y = stats.zscore(Y[:,firing_neurons_pc_anh],axis = 0)
    
    lenght = X.shape[0]-4000
    
    x = 50 
    
    start_x = 0+2000
    start_y = x+2000
    end_x = start_x+lenght
    end_y = start_y+lenght
    
    cca = CCA(n_components=1,max_iter=100000,tol = 1e-12)
    cca.fit(X[start_x:end_x,:], Y[start_y:end_y,:])
    #ob_proj, pc_proj = cca.transform(X, Y)

    params = cca.get_params()
    
    ob_loadings_anh = cca.x_weights_
    pc_loadings_anh = cca.y_weights_
    
    # compute CCA awake 
    
    # convolve spike trains

    srate_resample = 2000
    kernel = signal.gaussian(int(0.1*srate_resample),20) 

    conv_neurons_ob_awake = []
    for x in range(units_ob.shape[0]):
        conv = signal.convolve(np.squeeze(units_ob_awake[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_ob_awake.append(conv)
        
    conv_neurons_pc_awake = []
    for x in range(units_pc.shape[0]):
        conv = signal.convolve(np.squeeze(units_pc_awake[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_pc_awake.append(conv)
    
    # perform canonical correlation analysis 
    
    X = np.array(conv_neurons_ob_awake)[1:,:].T
    Y = np.array(conv_neurons_pc_awake)[1:,:].T
    
    X = stats.zscore(X[:,firing_neurons_ob_anh],axis = 0)
    Y = stats.zscore(Y[:,firing_neurons_pc_anh],axis = 0)
    
    lenght = X.shape[0]-4000
    
    x = 50 
    
    start_x = 0+2000
    start_y = x+2000
    end_x = start_x+lenght
    end_y = start_y+lenght
    
    cca = CCA(n_components=1,max_iter=100000,tol = 1e-12)
    cca.fit(X[start_x:end_x,:], Y[start_y:end_y,:])

    params = cca.get_params()
    
    ob_loadings_awake = cca.x_weights_
    pc_loadings_awake = cca.y_weights_
    
    
    
    # compare correlations
    
    X = np.array(conv_neurons_ob_awake)[1:,:].T
    Y = np.array(conv_neurons_pc_awake)[1:,:].T
    
    ob_proj_awake = stats.zscore(np.sum(X[:,firing_neurons_ob_anh]*ob_loadings_awake.T,axis = 1),axis = 0)
    pc_proj_awake = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh]*pc_loadings_awake.T,axis = 1),axis = 0)
    
    ob_mua_awake = stats.zscore(np.sum(X[:,firing_neurons_ob_anh],axis = 1),axis = 0)
    pc_mua_awake = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh],axis = 1),axis = 0)
    
    X = np.array(conv_neurons_ob_anh)[1:,:].T
    Y = np.array(conv_neurons_pc_anh)[1:,:].T
    
    ob_proj_anh = stats.zscore(np.sum(X[:,firing_neurons_ob_anh]*ob_loadings_anh.T,axis = 1),axis = 0)
    pc_proj_anh = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh]*pc_loadings_anh.T,axis = 1),axis = 0)
    
    ob_mua_anh = stats.zscore(np.sum(X[:,firing_neurons_ob_anh],axis = 1),axis = 0)
    pc_mua_anh = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh],axis = 1),axis = 0)
    
    r_proj_awake = stats.pearsonr(ob_proj_awake,pc_proj_awake)
    r_mua_wake = stats.pearsonr(ob_mua_awake,pc_mua_awake)
    r_proj_anh = stats.pearsonr(ob_proj_anh,pc_proj_anh)
    r_mua_anh = stats.pearsonr(ob_mua_anh,pc_mua_anh)
    
    X = np.array(conv_neurons_ob_anh)[1:,:].T
    Y = np.array(conv_neurons_pc_anh)[1:,:].T

    ob_proj_awake_anh = stats.zscore(np.sum(X[:,firing_neurons_ob_anh]*ob_loadings_awake.T,axis = 1),axis = 0)
    pc_proj_awake_anh = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh]*pc_loadings_awake.T,axis = 1),axis = 0)

    
    r_proj_awake_anh = stats.pearsonr(ob_proj_awake_anh,pc_proj_awake_anh)


    
    ob_loadings_anh_animals.append(ob_loadings_anh)
    pc_loadings_anh_animals.append(pc_loadings_anh)
    
    ob_loadings_awake_animals.append(ob_loadings_awake)
    pc_loadings_awake_animals.append(pc_loadings_awake)
    
    r_proj_awake_animals.append(r_proj_awake)
    r_mua_wake_animals.append(r_mua_wake)
    
    r_proj_anh_animals.append(r_proj_anh)
    r_mua_anh_animals.append(r_mua_anh)
    
    r_proj_awake_anh_animals.append(r_proj_awake_anh)
    
    
    # run spectral analysis
    
    f,p_ob_proj_awake = signal.welch(np.squeeze(ob_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_pc_proj_awake = signal.welch(np.squeeze(pc_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_awake = signal.coherence(np.squeeze(pc_proj_awake),np.squeeze(ob_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)


    f,p_ob_proj_anh = signal.welch(np.squeeze(ob_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_pc_proj_anh = signal.welch(np.squeeze(pc_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_anh = signal.coherence(np.squeeze(pc_proj_anh),np.squeeze(ob_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)

    f,c_ob_proj_awake = signal.coherence(np.squeeze(ob_proj_awake),np.squeeze(resp_awake), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_pc_proj_awake = signal.coherence(np.squeeze(pc_proj_awake),np.squeeze(resp_awake), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_resp_awake = signal.welch(np.squeeze(stats.zscore(resp_awake)), fs = 2000, nperseg = 2000, nfft = 20000)
    
    f,c_ob_proj_anh = signal.coherence(np.squeeze(ob_proj_anh),np.squeeze(resp_anh), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_pc_proj_anh = signal.coherence(np.squeeze(pc_proj_anh),np.squeeze(resp_anh), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_resp_anh = signal.welch(np.squeeze(stats.zscore(resp_anh)), fs = 2000, nperseg = 2000, nfft = 20000)
    
    numbin = 300
    
    lenta_BandWidth = 8
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp_awake),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta_awake = np.angle(analytic_signal)
    
    
    ob_comm_resp_phase_awake = phase_amp_hist(ob_proj_awake,faselenta_awake,numbin)
    pc_comm_resp_phase_awake = phase_amp_hist(pc_proj_awake,faselenta_awake,numbin)
    
    PhaseFreq=eegfilt(np.squeeze(resp_anh),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta_anh = np.angle(analytic_signal)
    
    
    ob_comm_resp_phase_anh = phase_amp_hist(ob_proj_anh,faselenta_anh,numbin)
    pc_comm_resp_phase_anh = phase_amp_hist(pc_proj_anh,faselenta_anh,numbin)
    
    
    numbin = 300
    
    max_entrop = np.log(numbin)

    ob_neurons_resp_phase_awake = []
    pc_neurons_resp_phase_awake = []

    ob_mi_neurons_awake = []
    pc_mi_neurons_awake = []
    
    X = np.array(conv_neurons_ob_awake)[1:,:].T
    Y = np.array(conv_neurons_pc_awake)[1:,:].T
    
    X = X[:,firing_neurons_ob_anh]
    Y = Y[:,firing_neurons_pc_anh]
    
    for x in range(X.shape[1]):
        
        hist = phase_amp_hist(X[:,x],faselenta_awake,numbin)
        ob_neurons_resp_phase_awake.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        ob_mi_neurons_awake.append((max_entrop-entrop)/max_entrop)
        

    for x in range(Y.shape[1]):    
        
        hist = phase_amp_hist(Y[:,x],faselenta_awake,numbin)
        pc_neurons_resp_phase_awake.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        pc_mi_neurons_awake.append((max_entrop-entrop)/max_entrop)
        
    
    # anh
    numbin = 300
    
    max_entrop = np.log(numbin)

    ob_neurons_resp_phase_anh = []
    pc_neurons_resp_phase_anh = []

    ob_mi_neurons_anh = []
    pc_mi_neurons_anh = []
    
    X = np.array(conv_neurons_ob_anh)[1:,:].T
    Y = np.array(conv_neurons_pc_anh)[1:,:].T
    
    X = X[:,firing_neurons_ob_anh]
    Y = Y[:,firing_neurons_pc_anh]

    for x in range(X.shape[1]):
        
        hist = phase_amp_hist(X[:,x],faselenta_anh,numbin)
        ob_neurons_resp_phase_anh.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        ob_mi_neurons_anh.append((max_entrop-entrop)/max_entrop)
        

    for x in range(Y.shape[1]):    
        
        hist = phase_amp_hist(Y[:,x],faselenta_anh,numbin)
        pc_neurons_resp_phase_anh.append(hist)
        prob = hist/np.sum(hist)
        prob = prob[prob>0]
        entrop = -1*np.sum(np.log(prob)*prob)
        pc_mi_neurons_anh.append((max_entrop-entrop)/max_entrop)
    
    
    
    ob_comm_resp_phase_animals_awake.append(ob_comm_resp_phase_awake)
    pc_comm_resp_phase_animals_awake.append(pc_comm_resp_phase_awake)
    
    ob_neurons_resp_phase_animals_awake.append(ob_neurons_resp_phase_awake)
    pc_neurons_resp_phase_animals_awake.append(pc_neurons_resp_phase_awake)
    
    ob_mi_neurons_animals_awake.append(ob_mi_neurons_awake)
    pc_mi_neurons_animals_awake.append(pc_mi_neurons_awake)
    
    ob_comm_resp_phase_animals_anh.append(ob_comm_resp_phase_anh)
    pc_comm_resp_phase_animals_anh.append(pc_comm_resp_phase_anh)
    
    ob_neurons_resp_phase_animals_anh.append(ob_neurons_resp_phase_anh)
    pc_neurons_resp_phase_animals_anh.append(pc_neurons_resp_phase_anh)
    
    ob_mi_neurons_animals_anh.append(ob_mi_neurons_anh)
    pc_mi_neurons_animals_anh.append(pc_mi_neurons_anh)
    
    
    p_ob_proj_awake_animals.append(p_ob_proj_awake)
    p_pc_proj_awake_animals.append(p_pc_proj_awake)
    p_ob_proj_anh_animals.append(p_ob_proj_anh)
    p_pc_proj_anh_animals.append(p_pc_proj_anh)
    c_awake_animals.append(c_awake)
    c_anh_animals.append(c_anh)
    c_ob_proj_awake_animals.append(c_ob_proj_awake)
    c_pc_proj_awake_animals.append(c_pc_proj_awake)
    p_resp_awake_animals.append(p_resp_awake)
    c_ob_proj_anh_animals.append(c_ob_proj_anh)
    c_pc_proj_anh_animals.append(c_pc_proj_anh)
    p_resp_anh_animals.append(p_resp_anh)
    

     
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (5,3))

plt.subplot(121)
plt.boxplot([np.array(r_proj_awake_animals)[:,0],np.array(r_proj_awake_anh_animals)[:,0]],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[np.array(r_proj_awake_animals)[:,0][x],np.array(r_proj_awake_anh_animals)[:,0][x]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['CS1 Awake','CS1 Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.ylim([-0.3,0.7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_awake,p_awake = stats.ttest_rel(np.array(r_proj_awake_animals)[:,0],np.array(r_proj_awake_anh_animals)[:,0])
plt.ylabel('OB-PCx Correlation')

plt.subplot(122)

plt.boxplot([np.array(r_proj_awake_animals)[:,0]-np.array(r_mua_wake_animals)[:,0],np.array(r_proj_anh_animals)[:,0]-np.array(r_mua_anh_animals)[:,0]],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[np.array(r_proj_awake_animals)[:,0][x]-np.array(r_mua_wake_animals)[:,0][x],np.array(r_proj_anh_animals)[:,0][x]-np.array(r_mua_anh_animals)[:,0][x]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['Awake','Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.ylim([-0.3,0.7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_anh,p_anh = stats.ttest_rel(np.array(r_proj_awake_animals)[:,0]-np.array(r_mua_wake_animals)[:,0],np.array(r_proj_anh_animals)[:,0]-np.array(r_mua_anh_animals)[:,0])
plt.ylabel('CS1 Correlation - MUA Correlation')
plt.tight_layout()

#plt.savefig('CS1_corr_anhestesia.pdf')

s_anh_mua,p_anh_mua = stats.ttest_rel(np.array(r_mua_wake_animals)[:,0],np.array(r_mua_anh_animals)[:,0])

#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (8,3))

plt.subplot(131)
plt.boxplot([np.array(r_proj_awake_animals)[:,0],np.array(r_proj_awake_anh_animals)[:,0]],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[np.array(r_proj_awake_animals)[:,0][x],np.array(r_proj_awake_anh_animals)[:,0][x]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['CS1 Awake','CS1 Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.ylim([-0.3,0.7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_awake,p_awake = stats.ttest_rel(np.array(r_proj_awake_animals)[:,0],np.array(r_proj_awake_anh_animals)[:,0])
plt.ylabel('OB-PCx Correlation')
plt.title('AwAke CS1')

plt.subplot(132)

plt.boxplot([np.array(r_mua_wake_animals)[:,0],np.array(r_mua_anh_animals)[:,0]],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[np.array(r_mua_wake_animals)[:,0][x],np.array(r_mua_anh_animals)[:,0][x]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['Awake','Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.ylim([-0.3,0.7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_anh,p_anh = stats.ttest_rel(np.array(r_proj_awake_animals)[:,0]-np.array(r_mua_wake_animals)[:,0],np.array(r_proj_anh_animals)[:,0]-np.array(r_mua_anh_animals)[:,0])
plt.title('MUA')

plt.subplot(133)

plt.boxplot([np.array(r_proj_awake_animals)[:,0]-np.array(r_mua_wake_animals)[:,0],np.array(r_proj_anh_animals)[:,0]-np.array(r_mua_anh_animals)[:,0]],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[np.array(r_proj_awake_animals)[:,0][x]-np.array(r_mua_wake_animals)[:,0][x],np.array(r_proj_anh_animals)[:,0][x]-np.array(r_mua_anh_animals)[:,0][x]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['Awake','Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.ylim([-0.3,0.7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_anh,p_anh = stats.ttest_rel(np.array(r_proj_awake_animals)[:,0]-np.array(r_mua_wake_animals)[:,0],np.array(r_proj_anh_animals)[:,0]-np.array(r_mua_anh_animals)[:,0])
plt.title('CS1 - MUA')
plt.tight_layout()

plt.savefig('CS1_corr_anhestesia.pdf')

s_anh_mua,p_anh_mua = stats.ttest_rel(np.array(r_mua_wake_animals)[:,0],np.array(r_mua_anh_animals)[:,0])



    
#%%

sum_weights_pc_awake = []
sum_weights_pc_anh = []

sum_weights_ob_awake = []
sum_weights_ob_anh = []

for x in range(len(ob_loadings_awake_animals)):
    
    sum_weights_pc_awake.append(np.sum(pc_loadings_awake_animals[x]))
    sum_weights_pc_anh.append(np.sum(pc_loadings_anh_animals[x]))
    
    sum_weights_ob_awake.append(np.sum(ob_loadings_awake_animals[x]))
    sum_weights_ob_anh.append(np.sum(ob_loadings_anh_animals[x]))
    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.figure(dpi = 300, figsize = (6,4))

ax1 = plt.subplot(211)
sns.kdeplot(np.squeeze(np.concatenate(ob_loadings_awake_animals)),shade = True, color = 'black', label = 'Awake')
sns.kdeplot(np.squeeze(np.concatenate(ob_loadings_anh_animals)),shade = True, color = 'tab:purple', label = 'Anhestesia')
#plt.xlabel('CS1 Weights')
plt.ylabel('OB Density')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend(loc = 'upper left')
plt.xlim([-1,1])

axins = inset_axes(ax1, width=1, height=0.9,borderpad = 1.2)


plt.boxplot([sum_weights_ob_awake,sum_weights_ob_anh],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[sum_weights_ob_awake[x],sum_weights_ob_anh[x]], color = 'grey')
    
plt.ylabel('CS1 Weights Sum', fontsize = 8)
plt.xticks(ticks = [1,2], labels = ['Awake','Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')



ax1 = plt.subplot(212)
sns.kdeplot(np.squeeze(np.concatenate(pc_loadings_awake_animals)),shade = True, color = 'black', label = 'Awake')
sns.kdeplot(np.squeeze(np.concatenate(pc_loadings_anh_animals)),shade = True, color = 'tab:purple', label = 'Anh')
plt.xlabel('CS1 Weights')
plt.ylabel('PCx Density')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.xlim([-1,1])


axins = inset_axes(ax1, width=1, height=0.9,borderpad = 1.2)

plt.boxplot([sum_weights_pc_awake,sum_weights_pc_anh],widths = 0.2, showfliers=False)

for x in range(10):
    plt.plot([1.2,1.8],[sum_weights_pc_awake[x],sum_weights_pc_anh[x]], color = 'grey')
    
plt.ylabel('CS1 Weights Sum', fontsize = 8)
plt.xticks(ticks = [1,2], labels = ['Awake','Anesthesia'], fontsize = 8)

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.tight_layout()
    
#plt.savefig('cs_weights_dist_anh.pdf')
#%%

s_ob, p_ob = stats.kstest(np.squeeze(np.concatenate(ob_loadings_awake_animals)),np.squeeze(np.concatenate(ob_loadings_anh_animals)))
s_pc, p_pc = stats.kstest(np.squeeze(np.concatenate(pc_loadings_awake_animals)),np.squeeze(np.concatenate(pc_loadings_anh_animals)))



#%%

cs_pair = 0

r_pc,p_pc = stats.pearsonr(np.squeeze(np.concatenate(pc_loadings_awake_animals)[:,cs_pair]),np.squeeze(np.concatenate(pc_loadings_anh_animals)[:,cs_pair]))
r_ob,p_ob = stats.pearsonr(np.squeeze(np.concatenate(ob_loadings_awake_animals)[:,cs_pair]),np.squeeze(np.concatenate(ob_loadings_anh_animals)[:,cs_pair]))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,2))

gs = gridspec.GridSpec(1, 2, width_ratios = [1,1],wspace = 0.2)

plt.subplot(gs[0])
sns.regplot(np.concatenate(pc_loadings_awake_animals),np.concatenate(pc_loadings_anh_animals),x_bins = np.linspace(-1,1,20), label = 'PCx '+'r = '+str(np.round(r_pc,decimals = 2)))
#plt.scatter(np.concatenate(pc_loadings_odorless_animals),np.concatenate(pc_loadings_odor_animals),s = 2,)

sns.regplot(np.concatenate(ob_loadings_awake_animals),np.concatenate(ob_loadings_anh_animals),x_bins = np.linspace(-1,1,20),  label = 'OB '+'r = '+str(np.round(r_ob,decimals = 2)))
#plt.scatter(np.concatenate(ob_loadings_odorless_animals),np.concatenate(ob_loadings_odor_animals),s = 2)

plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.linspace(-1,1,10),np.zeros(10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.zeros(10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend(fontsize = 6)
plt.ylabel('CS1 Weights (Awake)')
plt.xlabel('CS1 Weights (Anesthesia)')

#%%
ax = plt.subplot(gs[1],polar = True)

ob_angle = []
pc_angle = []

for x in range(10):

    ob_angle.append(scipy.linalg.subspace_angles(ob_loadings_awake_animals[x],ob_loadings_anh_animals[x]))
    pc_angle.append(scipy.linalg.subspace_angles(pc_loadings_awake_animals[x],pc_loadings_anh_animals[x]))
      

ax.hist(np.squeeze(ob_angle),bins = np.linspace(-1*np.pi, np.pi,50))
ax.hist(np.squeeze(pc_angle),bins = np.linspace(-1*np.pi, np.pi,50))

#plt.savefig('corr_weights_anh.pdf')

#%%
import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


#%%
ob_angle = []
pc_angle = []


for x in range(10):

    ob_angle.append(angle(ob_loadings_awake_animals[x],ob_loadings_anh_animals[x]))
    pc_angle.append(angle(pc_loadings_awake_animals[x],pc_loadings_anh_animals[x]))
      

#
ax = plt.subplot(gs[1],polar = True)

ax.hist(np.squeeze(ob_angle),bins = np.linspace(-1*np.pi, np.pi,30))
ax.hist(np.squeeze(pc_angle),bins = np.linspace(-1*np.pi, np.pi,30))
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,3))


sns.regplot(np.concatenate(pc_loadings_awake_animals),np.concatenate(pc_loadings_anh_animals),x_bins = np.linspace(-1,1,20), label = 'PCx '+'r = '+str(np.round(r_pc,decimals = 2)))
#plt.scatter(np.concatenate(pc_loadings_odorless_animals),np.concatenate(pc_loadings_odor_animals),s = 2,)

sns.regplot(np.concatenate(ob_loadings_awake_animals),np.concatenate(ob_loadings_anh_animals),x_bins = np.linspace(-1,1,20),  label = 'OB '+'r = '+str(np.round(r_ob,decimals = 2)))
#plt.scatter(np.concatenate(ob_loadings_odorless_animals),np.concatenate(ob_loadings_odor_animals),s = 2)

plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.linspace(-1,1,10),np.zeros(10), color = 'black',linewidth = 0.8, linestyle = 'dashed')
plt.plot(np.zeros(10),np.linspace(-1,1,10), color = 'black',linewidth = 0.8, linestyle = 'dashed')


plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.legend(fontsize = 6)
plt.xlabel('CS1 Weights (Awake)')
plt.ylabel('CS1 Weights (Anesthesia)')

#plt.savefig('cs_weight_corr.pdf')

#%% respiratory entrainment

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(dpi = 300, figsize = (6,9))

plt.subplot(211)

#
ob_comm_resp_phase = np.mean(ob_comm_resp_phase_animals_awake,axis = 0)
ob_comm_resp_phase = np.hstack([ob_comm_resp_phase,ob_comm_resp_phase,ob_comm_resp_phase])

error_ob_comm_resp_phase = 1*np.std(ob_comm_resp_phase_animals_awake,axis = 0)/np.sqrt(len(ob_comm_resp_phase_animals_awake))
error_ob_comm_resp_phase = np.hstack([error_ob_comm_resp_phase,error_ob_comm_resp_phase,error_ob_comm_resp_phase])

pc_comm_resp_phase = np.mean(pc_comm_resp_phase_animals_awake,axis = 0)
pc_comm_resp_phase = np.hstack([pc_comm_resp_phase,pc_comm_resp_phase,pc_comm_resp_phase])

error_pc_comm_resp_phase = 1*np.std(pc_comm_resp_phase_animals_awake,axis = 0)/np.sqrt(len(pc_comm_resp_phase_animals_awake))
error_pc_comm_resp_phase = np.hstack([error_pc_comm_resp_phase,error_pc_comm_resp_phase,error_pc_comm_resp_phase])


plt.plot(np.linspace(0,1080,900)-295,ob_comm_resp_phase,label = 'OB', color = 'tab:orange')
plt.fill_between(np.linspace(0,1080,900)-295, ob_comm_resp_phase-error_ob_comm_resp_phase, ob_comm_resp_phase+error_ob_comm_resp_phase,alpha = 0.2, color = 'tab:orange')
plt.plot(np.linspace(0,1080,900)-295,pc_comm_resp_phase,label = 'PCx', color = 'tab:blue')
plt.fill_between(np.linspace(0,1080,900)-295, pc_comm_resp_phase-error_pc_comm_resp_phase, pc_comm_resp_phase+error_pc_comm_resp_phase,alpha = 0.2, color = 'tab:blue')

plt.legend(ncol = 2, loc = 'lower left')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


plt.ylabel('CS1 (z-score)')
plt.xlabel('Respiration Phase (deg)')
#plt.ylim([-0.5,0.5])

plt.xlim([0,720])


plt.subplot(212)


ob_comm_resp_phase = np.mean(ob_comm_resp_phase_animals_anh,axis = 0)
ob_comm_resp_phase = np.hstack([ob_comm_resp_phase,ob_comm_resp_phase,ob_comm_resp_phase])

error_ob_comm_resp_phase = 1*np.std(ob_comm_resp_phase_animals_anh,axis = 0)/np.sqrt(len(ob_comm_resp_phase_animals_awake))
error_ob_comm_resp_phase = np.hstack([error_ob_comm_resp_phase,error_ob_comm_resp_phase,error_ob_comm_resp_phase])

pc_comm_resp_phase = np.mean(pc_comm_resp_phase_animals_anh,axis = 0)
pc_comm_resp_phase = np.hstack([pc_comm_resp_phase,pc_comm_resp_phase,pc_comm_resp_phase])

error_pc_comm_resp_phase = 1*np.std(pc_comm_resp_phase_animals_anh,axis = 0)/np.sqrt(len(pc_comm_resp_phase_animals_awake))
error_pc_comm_resp_phase = np.hstack([error_pc_comm_resp_phase,error_pc_comm_resp_phase,error_pc_comm_resp_phase])


plt.plot(np.linspace(0,1080,900)-295,ob_comm_resp_phase,label = 'OB', color = 'tab:orange')
plt.fill_between(np.linspace(0,1080,900)-295, ob_comm_resp_phase-error_ob_comm_resp_phase, ob_comm_resp_phase+error_ob_comm_resp_phase,alpha = 0.2, color = 'tab:orange')
plt.plot(np.linspace(0,1080,900)-295,pc_comm_resp_phase,label = 'PCx', color = 'tab:blue')
plt.fill_between(np.linspace(0,1080,900)-295, pc_comm_resp_phase-error_pc_comm_resp_phase, pc_comm_resp_phase+error_pc_comm_resp_phase,alpha = 0.2, color = 'tab:blue')

plt.legend(ncol = 2, loc = 'lower left')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.ylabel('CS1 (z-score)')
plt.xlabel('Respiration Phase (deg)')

plt.xlim([0,720])

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,6))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",['m','white','c'])

gs = gridspec.GridSpec(2, 3, width_ratios = [0.8,2,0.8], height_ratios=[np.concatenate(ob_loadings_awake_animals).shape[0],np.concatenate(pc_loadings_awake_animals).shape[0]])

plt.subplot(gs[0])

sorted_loadings = np.sort(np.squeeze(np.concatenate(ob_loadings_awake_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,np.concatenate(ob_loadings_awake_animals).shape[0]])
plt.yticks([])

plt.subplot(gs[1])

ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals_awake)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_awake_animals)))

plt.imshow(ob_neurons_resp_phase_norm[load_sort,:], cmap = cmap,aspect = 'auto',vmin = -1, vmax = 1)


plt.ylim([0,np.concatenate(ob_loadings_awake_animals).shape[0]])
#plt.xticks(ticks = np.arange(16,53,6),labels = [],rotation = 30)
plt.xticks(ticks = np.arange(245,250+600,100),labels = [],rotation = 30)
plt.xlim([245,245+600])

plt.yticks([])

plt.subplot(gs[2])


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_awake_animals)))
plt.barh(np.arange(0,load_sort.shape[0]),np.squeeze(np.concatenate(ob_mi_neurons_animals_awake)[load_sort]), color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,np.concatenate(ob_loadings_awake_animals).shape[0]])
#plt.xscale('log')
plt.xlim([0,0.1])

plt.yticks([])

plt.subplot(gs[3])

sorted_loadings = np.sort(np.squeeze(np.concatenate(pc_loadings_awake_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.yticks([])

plt.xlim([-1,1])
plt.xlabel('CCA Weight')
plt.ylim([0,np.concatenate(pc_loadings_awake_animals).shape[0]])

plt.subplot(gs[4])

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_awake)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)

load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_awake_animals)))

plt.imshow(pc_neurons_resp_phase_norm[load_sort,:],aspect = 'auto',vmin = -1, vmax = 1, cmap = cmap)
#plt.colorbar()
plt.ylim([0,np.concatenate(pc_loadings_awake_animals).shape[0]])
plt.yticks([])
plt.xlim([245,245+600])
#plt.title('PCx')
plt.xticks(ticks = np.arange(245,250+600,100),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)
plt.tight_layout()
#plt.colorbar(location = 'bottom')
plt.xlabel('Respiration Phase (deg)')

plt.subplot(gs[5])


load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_awake_animals)))
plt.barh(np.arange(0,load_sort.shape[0]),np.squeeze(np.concatenate(pc_mi_neurons_animals_awake)[load_sort]), color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,np.concatenate(pc_loadings_awake_animals).shape[0]])
plt.yticks([])
plt.xlabel('Resp MI')
#plt.xscale('log')
plt.xlim([0,0.1])


#%% anhestesia


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,6))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",['m','white','c'])

gs = gridspec.GridSpec(2, 3, width_ratios = [0.8,2,0.8], height_ratios=[np.concatenate(ob_loadings_awake_animals).shape[0],np.concatenate(pc_loadings_awake_animals).shape[0]])

plt.subplot(gs[0])

sorted_loadings = np.sort(np.squeeze(np.concatenate(ob_loadings_anh_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,np.concatenate(ob_loadings_anh_animals).shape[0]])
plt.yticks([])

plt.subplot(gs[1])

ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals_anh)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_anh_animals)))

plt.imshow(ob_neurons_resp_phase_norm[load_sort,:], cmap = cmap,aspect = 'auto',vmin = -1, vmax = 1)


plt.ylim([0,np.concatenate(ob_loadings_anh_animals).shape[0]])
#plt.xticks(ticks = np.arange(16,53,6),labels = [],rotation = 30)
plt.xticks(ticks = np.arange(245,250+600,100),labels = [],rotation = 30)
plt.xlim([245,245+600])

plt.yticks([])

plt.subplot(gs[2])


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_anh_animals)))
plt.barh(np.arange(0,load_sort.shape[0]),np.squeeze(np.concatenate(ob_mi_neurons_animals_anh)[load_sort]), color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,np.concatenate(ob_loadings_anh_animals).shape[0]])
#plt.xscale('log')
plt.xlim([0,0.1])


plt.yticks([])


plt.subplot(gs[3])

sorted_loadings = np.sort(np.squeeze(np.concatenate(pc_loadings_anh_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.yticks([])

plt.xlim([-1,1])
plt.xlabel('CCA Weight')
plt.ylim([0,np.concatenate(pc_loadings_anh_animals).shape[0]])

plt.subplot(gs[4])

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_anh)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)

load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_anh_animals)))

plt.imshow(pc_neurons_resp_phase_norm[load_sort,:],aspect = 'auto',vmin = -1, vmax = 1, cmap = cmap)
#plt.colorbar()
plt.ylim([0,np.concatenate(pc_loadings_anh_animals).shape[0]])
plt.yticks([])
plt.xlim([245,245+600])
#plt.title('PCx')
plt.xticks(ticks = np.arange(245,250+600,100),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)
plt.tight_layout()
#plt.colorbar(location = 'bottom')
plt.xlabel('Respiration Phase (deg)')

plt.subplot(gs[5])


load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_anh_animals)))
plt.barh(np.arange(0,load_sort.shape[0]),np.squeeze(np.concatenate(pc_mi_neurons_animals_anh)[load_sort]), color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylim([0,np.concatenate(pc_loadings_anh_animals).shape[0]])
plt.yticks([])
plt.xlabel('Resp MI')
#plt.xscale('log')
plt.xlim([0,0.1])

#%%

plt.figure(dpi = 300, figsize = (3,5))

plt.subplot(211)

sns.regplot(x = np.abs(np.concatenate(ob_loadings_awake_animals)),y = np.concatenate(ob_mi_neurons_animals_awake),x_bins = np.linspace(0,1.2,10) , color = 'tab:orange')
sns.regplot(x = np.abs(np.concatenate(ob_loadings_anh_animals)),y = np.concatenate(ob_mi_neurons_animals_anh),x_bins = np.linspace(0,1.2,10) , color = 'grey')

plt.ylabel('Respiratory Coupling')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([-0.01,0.36])
plt.xlim([-0.1,1])
plt.yscale('log')
plt.ylim([0.003,0.5])

r_ob_anh, p_ob_anh = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_anh_animals))),np.squeeze(np.concatenate(ob_mi_neurons_animals_anh)))
r_ob_awake, p_ob_awake = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_awake_animals))),np.squeeze(np.concatenate(ob_mi_neurons_animals_awake)))



plt.subplot(212)

sns.regplot(x = np.abs(np.concatenate(pc_loadings_awake_animals)),y = np.concatenate(pc_mi_neurons_animals_awake),x_bins = np.linspace(0,1.2,10) , color = 'tab:blue')
sns.regplot(x = np.abs(np.concatenate(pc_loadings_anh_animals)),y = np.concatenate(pc_mi_neurons_animals_anh),x_bins = np.linspace(0,1.2,10) , color = 'grey')

plt.xlabel('Abs CS1 Weight')
plt.ylabel('Respiratory Coupling')
#plt.ylabel('Respiratory Modulation Index')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-0.1,1])
plt.yscale('log')
plt.ylim([0.003,0.5])

r_pc_anh, p_pc_anh = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_anh_animals))),np.squeeze(np.concatenate(pc_mi_neurons_animals_anh)))
r_pc_awake, p_pc_awake = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_awake_animals))),np.squeeze(np.concatenate(pc_mi_neurons_animals_awake)))


plt.tight_layout()

#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (10,5))

gs = gridspec.GridSpec(2, 2, width_ratios = [1,0.4])


plt.subplot(gs[0])
sns.histplot(np.concatenate(ob_mi_neurons_animals_awake), color = 'tab:orange',bins = np.geomspace(0.0001,1,50), label = 'OB Awake')
sns.histplot(np.concatenate(ob_mi_neurons_animals_anh), color = 'grey',bins = np.geomspace(0.0001,1,50), label = 'OB Anesthesia')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
s_mi_ob, p_mi_ob = stats.ttest_rel(np.squeeze(np.concatenate(ob_mi_neurons_animals_awake)),np.squeeze(np.concatenate(ob_mi_neurons_animals_anh)))
plt.text(0.1,20,'p = ' + str(np.round(p_mi_ob,decimals = 5)))
plt.vlines(np.mean(np.concatenate(ob_mi_neurons_animals_awake)),0,30, color = 'tab:orange')
plt.vlines(np.mean(np.concatenate(ob_mi_neurons_animals_anh)),0,30, color = 'grey')

plt.subplot(gs[1])
sns.regplot(x = np.abs(np.concatenate(ob_loadings_awake_animals)),y = np.concatenate(ob_mi_neurons_animals_awake),x_bins = np.linspace(0,0.8,5),robust = True, truncate = True, color = 'tab:orange')
sns.regplot(x = np.abs(np.concatenate(ob_loadings_anh_animals)),y = np.concatenate(ob_mi_neurons_animals_anh),x_bins = np.linspace(0,0.8,5),robust = True, truncate = True,n_boot = 1000 ,color = 'grey')

plt.ylabel('Respiratory Coupling')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.ylim([-0.01,0.36])
plt.xlim([-0.1,1.1])
plt.yscale('log')
plt.ylim([0.001,0.8])

r_ob_anh, p_ob_anh = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_anh_animals))),np.squeeze(np.concatenate(ob_mi_neurons_animals_anh)))
r_ob_awake, p_ob_awake = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_awake_animals))),np.squeeze(np.concatenate(ob_mi_neurons_animals_awake)))

plt.text(-0.05,0.4,'R (W) = ' + str(np.round(r_ob_awake,decimals = 2)) +', p = '+ str(np.round(p_ob_awake,decimals = 1)))
plt.text(-0.05,0.2,'R (Anh) = ' + str(np.round(r_ob_anh,decimals = 2)) +', p = '+ str(np.round(p_ob_anh,decimals = 1)))

plt.subplot(gs[2])
sns.histplot(np.concatenate(pc_mi_neurons_animals_awake), color = 'tab:blue',bins = np.geomspace(0.0001,1,50), label = 'PCx Awake')
sns.histplot(np.concatenate(pc_mi_neurons_animals_anh), color = 'grey',bins = np.geomspace(0.0001,1,50), label = 'PCx Anesthesia')
plt.legend()
plt.xscale('log')
plt.xlabel('Respiratory Coupling (a.u.)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
s_mi_pc, p_mi_pc = stats.ttest_rel(np.squeeze(np.concatenate(pc_mi_neurons_animals_awake)),np.squeeze(np.concatenate(pc_mi_neurons_animals_anh)))
plt.text(0.1,50,'p = ' + str(np.round(p_mi_pc,decimals = 5)))
plt.vlines(np.mean(np.concatenate(pc_mi_neurons_animals_awake)),0,70, color = 'tab:blue')
plt.vlines(np.mean(np.concatenate(pc_mi_neurons_animals_anh)),0,70, color = 'grey')


plt.subplot(gs[3])
sns.regplot(x = np.abs(np.concatenate(pc_loadings_awake_animals)),y = np.concatenate(pc_mi_neurons_animals_awake),x_bins = np.linspace(0,0.8,5),robust = True, truncate = True , color = 'tab:blue')
sns.regplot(x = np.abs(np.concatenate(pc_loadings_anh_animals)),y = np.concatenate(pc_mi_neurons_animals_anh),x_bins = np.linspace(0,0.5,5),robust = True, truncate = True , color = 'grey')

plt.xlabel('Abs CS1 Weight')
plt.ylabel('Respiratory Coupling')
#plt.ylabel('Respiratory Modulation Index')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-0.1,1.1])
plt.yscale('log')
plt.ylim([0.003,0.8])

r_pc_anh, p_pc_anh = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_anh_animals))),np.squeeze(np.concatenate(pc_mi_neurons_animals_anh)))
r_pc_awake, p_pc_awake = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_awake_animals))),np.squeeze(np.concatenate(pc_mi_neurons_animals_awake)))

plt.text(-0.05,0.4,'R (W) = ' + str(np.round(r_pc_awake,decimals = 2)) +', p = '+ str(np.round(p_pc_awake,decimals = 1)))
plt.text(-0.05,0.2,'R (Anh) = ' + str(np.round(r_pc_anh,decimals = 2)) +', p = '+ str(np.round(p_pc_anh,decimals = 1)))

plt.tight_layout()

#plt.savefig('coupling_anh.pdf')


#%%

phase_pref_ob_anh = np.linspace(-1*np.pi, np.pi,300)[np.argmax(np.concatenate(ob_neurons_resp_phase_animals_anh),axis = 1)]
phase_pref_pc_anh = np.linspace(-1*np.pi, np.pi,300)[np.argmax(np.concatenate(pc_neurons_resp_phase_animals_anh),axis = 1)]
phase_pref_ob_awake = np.linspace(-1*np.pi, np.pi,300)[np.argmax(np.concatenate(ob_neurons_resp_phase_animals_awake),axis = 1)]
phase_pref_pc_awake = np.linspace(-1*np.pi, np.pi,300)[np.argmax(np.concatenate(pc_neurons_resp_phase_animals_awake),axis = 1)]


plt.figure(dpi = 300)
ax = plt.subplot(211,polar = True)

ax.hist(np.squeeze(phase_pref_pc_awake),bins = np.linspace(-1*np.pi, np.pi,20))
ax.hist(np.squeeze(phase_pref_ob_awake),bins = np.linspace(-1*np.pi, np.pi,20))

plt.title('Awake')

ax = plt.subplot(212,polar = True)

ax.hist(np.squeeze(phase_pref_pc_anh),bins = np.linspace(-1*np.pi, np.pi,20))
ax.hist(np.squeeze(phase_pref_ob_anh),bins = np.linspace(-1*np.pi, np.pi,20))

plt.title('Anesthesia')
plt.tight_layout()

#%%

sns.regplot(np.concatenate(pc_loadings_awake_animals),np.rad2deg(np.squeeze(phase_pref_pc_awake))+180, x_bins = np.linspace(-1,1,25))
sns.regplot(np.concatenate(pc_loadings_anh_animals),np.rad2deg(np.squeeze(phase_pref_pc_anh))+180, x_bins = np.linspace(-1,1,25))



#%%

high_cs_pc_anh = np.concatenate(pc_loadings_anh_animals)>np.quantile(np.concatenate(pc_loadings_anh_animals),0.75)
low_cs_pc_anh = np.concatenate(pc_loadings_anh_animals)<np.quantile(np.concatenate(pc_loadings_anh_animals),0.25)

high_cs_pc_awake = np.concatenate(pc_loadings_awake_animals)>np.quantile(np.concatenate(pc_loadings_awake_animals),0.75)
low_cs_pc_awake = np.concatenate(pc_loadings_awake_animals)<np.quantile(np.concatenate(pc_loadings_awake_animals),0.25)

plt.figure(dpi = 300)
ax = plt.subplot(211,polar = True)

ax.hist(np.squeeze(phase_pref_pc_awake)[np.squeeze(high_cs_pc_awake)],bins = np.linspace(-1*np.pi, np.pi,30), color = 'tab:blue')
ax.hist(np.squeeze(phase_pref_pc_awake)[np.squeeze(low_cs_pc_awake)],bins = np.linspace(-1*np.pi, np.pi,30), color = 'grey')

plt.title('Awake')

ax = plt.subplot(212,polar = True)

ax.hist(np.squeeze(phase_pref_pc_anh)[np.squeeze(high_cs_pc_anh)],bins = np.linspace(-1*np.pi, np.pi,30), color = 'tab:blue')
ax.hist(np.squeeze(phase_pref_pc_anh)[np.squeeze(low_cs_pc_anh)],bins = np.linspace(-1*np.pi, np.pi,30), color = 'grey')

plt.title('Anesthesia')

plt.tight_layout()
#%%

high_cs_pc_anh = np.concatenate(pc_loadings_anh_animals)>np.quantile(np.concatenate(pc_loadings_anh_animals),0.99)
low_cs_pc_anh = np.concatenate(pc_loadings_anh_animals)<np.quantile(np.concatenate(pc_loadings_anh_animals),0.99)

high_cs_pc_awake = np.concatenate(pc_loadings_awake_animals)>np.quantile(np.concatenate(pc_loadings_awake_animals),0.99)
low_cs_pc_awake = np.concatenate(pc_loadings_awake_animals)<np.quantile(np.concatenate(pc_loadings_awake_animals),0.99)


pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_awake)
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)#/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]


mean_high = np.mean(pc_neurons_resp_phase_norm[np.squeeze(high_cs_pc_awake),:],axis = 0)
mean_high = np.hstack([mean_high,mean_high])

error_high = stats.sem(pc_neurons_resp_phase_norm[np.squeeze(high_cs_pc_awake),:],axis = 0)
error_high = np.hstack([error_high,error_high])

mean_low = np.mean(pc_neurons_resp_phase_norm[np.squeeze(low_cs_pc_awake),:],axis = 0)
mean_low = np.hstack([mean_low,mean_low])

error_low = stats.sem(pc_neurons_resp_phase_norm[np.squeeze(low_cs_pc_awake),:],axis = 0)
error_low = np.hstack([error_low,error_low])



plt.figure(dpi = 300, figsize = (3,4))
plt.subplot(211)
plt.plot(np.linspace(0,720,600),mean_high, color = 'tab:blue')
plt.fill_between(np.linspace(0,720,600), mean_high-error_high, mean_high+error_high, alpha = 0.2)

plt.plot(np.linspace(0,720,600),mean_low, color = 'grey')
plt.fill_between(np.linspace(0,720,600), mean_low-error_low, mean_low+error_low, alpha = 0.2, color = 'grey')


pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals_anh)
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)#/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]


mean_high = np.mean(pc_neurons_resp_phase_norm[np.squeeze(high_cs_pc_awake),:],axis = 0)
mean_high = np.hstack([mean_high,mean_high])

error_high = stats.sem(pc_neurons_resp_phase_norm[np.squeeze(high_cs_pc_awake),:],axis = 0)
error_high = np.hstack([error_high,error_high])

mean_low = np.mean(pc_neurons_resp_phase_norm[np.squeeze(low_cs_pc_awake),:],axis = 0)
mean_low = np.hstack([mean_low,mean_low])

error_low = stats.sem(pc_neurons_resp_phase_norm[np.squeeze(low_cs_pc_awake),:],axis = 0)
error_low = np.hstack([error_low,error_low])

plt.subplot(212)
plt.plot(np.linspace(0,720,600),mean_high, color = 'tab:blue')
plt.fill_between(np.linspace(0,720,600), mean_high-error_high, mean_high+error_high, alpha = 0.2)

plt.plot(np.linspace(0,720,600),mean_low, color = 'grey')
plt.fill_between(np.linspace(0,720,600), mean_low-error_low, mean_low+error_low, alpha = 0.2, color = 'grey')


#%%

ax = plt.subplot(212,polar = True)

ax.hist(np.squeeze(phase_pref_pc_anh)[np.squeeze(high_cs_pc_anh)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'tab:blue')
ax.hist(np.squeeze(phase_pref_pc_anh)[np.squeeze(low_cs_pc_anh)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'grey')
plt.title('Anesthesia')


high_cs_ob_anh = np.concatenate(ob_loadings_anh_animals)>np.quantile(np.concatenate(ob_loadings_anh_animals),0.75)
low_cs_ob_anh = np.concatenate(ob_loadings_anh_animals)<np.quantile(np.concatenate(ob_loadings_anh_animals),0.25)

high_cs_ob_awake = np.concatenate(ob_loadings_awake_animals)>np.quantile(np.concatenate(ob_loadings_awake_animals),0.75)
low_cs_ob_awake = np.concatenate(ob_loadings_awake_animals)<np.quantile(np.concatenate(ob_loadings_awake_animals),0.25)

#%%
plt.figure(dpi = 300)
ax = plt.subplot(211,polar = True)

ax.hist(np.squeeze(phase_pref_ob_awake)[np.squeeze(high_cs_ob_awake)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'tab:blue')
ax.hist(np.squeeze(phase_pref_ob_awake)[np.squeeze(low_cs_ob_awake)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'grey')

plt.title('Awake')

ax = plt.subplot(212,polar = True)

ax.hist(np.squeeze(phase_pref_ob_anh)[np.squeeze(high_cs_ob_anh)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'tab:blue')
ax.hist(np.squeeze(phase_pref_ob_anh)[np.squeeze(low_cs_ob_anh)],bins = np.linspace(-1*np.pi, np.pi,50), color = 'grey')
plt.title('Anesthesia')

#%%

start = 280000
end = start + 4000
window = np.arange(start,end)

plt.figure(dpi = 300, figsize = (5,5))

plt.subplot(211)
plt.plot(pc_proj_awake[window])
plt.plot(ob_proj_awake[window])
plt.box(False)
plt.xticks([])
plt.yticks([])
plt.ylim([-10,10])
plt.title('Awake')



start = 13000
end = start + 4000
window = np.arange(start,end)

plt.subplot(212)
plt.plot(pc_proj_awake_anh[window])
plt.plot(ob_proj_awake_anh[window])
plt.box(False)
plt.xticks([])
plt.yticks([])
plt.ylim([-5,15])
plt.title('Anesthesia')

plt.savefig('traces_cs_anh.pdf')
#%%
# check synchronization to resp

f,p_ob_proj_awake = signal.welch(np.squeeze(ob_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)
f,p_pc_proj_awake = signal.welch(np.squeeze(pc_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)
f,c_awake = signal.coherence(np.squeeze(pc_proj_awake),np.squeeze(ob_proj_awake), fs = 2000, nperseg = 2000, nfft = 20000)


f,p_ob_proj_anh = signal.welch(np.squeeze(ob_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)
f,p_pc_proj_anh = signal.welch(np.squeeze(pc_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)
f,c_anh = signal.coherence(np.squeeze(pc_proj_anh),np.squeeze(ob_proj_anh), fs = 2000, nperseg = 2000, nfft = 20000)

#%%

plt.figure(dpi = 300, figsize = (6,3))

plt.subplot(131)

plt.plot(f,p_ob_proj_awake/np.sum(p_ob_proj_awake))

plt.plot(f,p_ob_proj_anh/np.sum(p_ob_proj_anh))

plt.xlim([0,10])

plt.subplot(132)

plt.plot(f,p_pc_proj_awake/np.sum(p_pc_proj_awake))

plt.plot(f,p_pc_proj_anh/np.sum(p_pc_proj_anh))

plt.xlim([0,10])

plt.subplot(133)

plt.plot(f,c_awake)
plt.plot(f,c_anh)

plt.xlim([0,10])
#f,p_resp = signal.welch(np.squeeze(stats.zscore(resp)), fs = 2000, nperseg = 2000, nfft = 20000)


#%%


X = np.array(conv_neurons_ob_anh)[1:,:].T
Y = np.array(conv_neurons_pc_anh)[1:,:].T

ob_proj_awake_anh = stats.zscore(np.sum(X[:,firing_neurons_ob_anh]*ob_loadings_awake.T,axis = 1),axis = 0)
pc_proj_awake_anh = stats.zscore(np.sum(Y[:,firing_neurons_pc_anh]*pc_loadings_awake.T,axis = 1),axis = 0)


ob_comm_resp_phase_awake_anh = phase_amp_hist(ob_proj_awake_anh,faselenta_anh,numbin)
pc_comm_resp_phase_awake_anh = phase_amp_hist(pc_proj_awake_anh,faselenta_anh,numbin)

#%%

plt.plot(pc_comm_resp_phase_awake)
plt.plot(pc_comm_resp_phase_anh)
plt.plot(pc_comm_resp_phase_awake_anh)

#%%
r_proj_awake = stats.pearsonr(ob_proj_awake,pc_proj_awake)


#%%



