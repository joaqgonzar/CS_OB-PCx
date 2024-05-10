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

#%% general information 

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
    
#%%

ob_loadings_animals = []
pc_loadings_animals = []

c_ob_proj_animals = []
c_pc_proj_animals = []
c_ob_pc_proj_animals = []
p_ob_proj_animals = []
p_pc_proj_animals = []
p_resp_animals = []

ob_comm_resp_phase_animals = []
pc_comm_resp_phase_animals = []
ob_mua_resp_phase_animals = []
pc_mua_resp_phase_animals = []

ob_neurons_resp_phase_animals = []
pc_neurons_resp_phase_animals = []

ob_mi_neurons_animals = []
pc_mi_neurons_animals = []

c_ob_mua_animals = []
c_pc_mua_animals = []
c_ob_pc_mua_animals = []
p_ob_mua_animals = []
p_pc_mua_animals = []

ccg_ob_pc_proj_animals = []
lags_ob_pc_animals = []
ccg_ob_pc_mua_animals = []
gc_ob_pc_proj_animals = []
gc_pc_ob_proj_animals = []
gc_ob_pc_mua_animals = []
gc_pc_ob_mua_animals = []

gc_ob_resp_proj_animals = []
gc_resp_ob_proj_animals = []
gc_pc_resp_proj_animals = []
gc_resp_pc_proj_animals = []

inh_trig_ob_proj_animals = []
inh_trig_pc_proj_animals = []
inh_trig_ob_mua_animals = []
inh_trig_pc_mua_animals = []

ccg_ob_proj_resp_animals = []
ccg_pc_proj_resp_animals = []

r_phase_animals = []

t_ob_pc_proj_phase_animals = []
t_pc_ob_proj_phase_animals = []

t_ob_pc_proj_time_animals = []
t_pc_ob_proj_time_animals = []

r_proj_animals = []
r_mua_animals = []

mi_proj_animals = []
mi_mua_animals = []

t_ob_pc_proj_binned_downsample_animals = []
t_pc_ob_proj_binned_downsample_animals = []


# loop through animals 

for index, name in enumerate(names):
    
    print(name)
    
    # get recordings
    
    os.chdir(directory+'/Simul/processed/'+name)
    
    spike_times_ob = mat73.loadmat(name+'_bank'+str(ob_bank[index])+'.mat')['SpikeTimes']['tsec']
    spike_times_pc = mat73.loadmat(name+'_bank'+str(pc_bank[index])+'.mat')['SpikeTimes']['tsec']
    
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    srate_resp = 2000
    
    lenta_BandWidth = 2
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp),srate_resp,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
        
    
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
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')
    lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    recording_time = lfp.shape[1]
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
    
    resp = resp[odorless_mask]
    faselenta = faselenta[odorless_mask]
    
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
    
    X = np.array(conv_neurons_ob)[1:,odorless_mask].T
    Y = np.array(conv_neurons_pc)[1:,odorless_mask].T
    X = stats.zscore(X,axis = 0)
    Y = stats.zscore(Y,axis = 0)
    
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
    
    ob_loadings = cca.x_weights_
    pc_loadings = cca.y_weights_
    

    X = np.array(conv_neurons_ob)[1:,odorless_mask].T
    Y = np.array(conv_neurons_pc)[1:,odorless_mask].T
    
    ob_proj = stats.zscore(np.sum(X*ob_loadings.T,axis = 1),axis = 0)
    pc_proj = stats.zscore(np.sum(Y*pc_loadings.T,axis = 1),axis = 0)
    
    
    # check synchronization to resp
    
    f,c_ob_proj = signal.coherence(np.squeeze(ob_proj),np.squeeze(resp), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_pc_proj = signal.coherence(np.squeeze(pc_proj),np.squeeze(resp), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_ob_pc_proj = signal.coherence(np.squeeze(pc_proj),np.squeeze(ob_proj), fs = 2000, nperseg = 2000, nfft = 20000)

    f,p_ob_proj = signal.welch(np.squeeze(ob_proj), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_pc_proj = signal.welch(np.squeeze(pc_proj), fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_resp = signal.welch(np.squeeze(stats.zscore(resp)), fs = 2000, nperseg = 2000, nfft = 20000)
    
    numbin = 300
    
    ob_comm_resp_phase = phase_amp_hist(ob_proj,faselenta,numbin)
    pc_comm_resp_phase = phase_amp_hist(pc_proj,faselenta,numbin)
    
    ob_mua_resp_phase = phase_amp_hist(stats.zscore(conv_neurons_ob[0][odorless_mask]),faselenta,numbin)
    pc_mua_resp_phase = phase_amp_hist(stats.zscore(conv_neurons_pc[0][odorless_mask]),faselenta,numbin)
    
    
    # check inhalation triggered average
    
    inh_trig_ob_proj = []
    inh_trig_pc_proj = []
    inh_trig_ob_mua = []
    inh_trig_pc_mua = []
    
    X_all = np.array(conv_neurons_ob)[1:,:].T
    Y_all = np.array(conv_neurons_pc)[1:,:].T
    
    ob_proj_all = stats.zscore(np.sum(X_all*ob_loadings.T,axis = 1),axis = 0)
    pc_proj_all = stats.zscore(np.sum(Y_all*pc_loadings.T,axis = 1),axis = 0)
    
    
    mua_ob_z_score = stats.zscore(np.sum(X_all,axis = 1),axis = 0)
    mua_pc_z_score = stats.zscore(np.sum(Y_all,axis = 1),axis = 0)
    
    for inhalation in inh_no_odor:
        
        if ob_proj[int(inhalation-1000):int(inhalation+1000)].shape[0] == 2000:
            inh_trig_ob_proj.append(ob_proj_all[int(inhalation-1000):int(inhalation+1000)])
            inh_trig_pc_proj.append(pc_proj_all[int(inhalation-1000):int(inhalation+1000)])
            inh_trig_ob_mua.append(mua_ob_z_score[int(inhalation-1000):int(inhalation+1000)])
            inh_trig_pc_mua.append(mua_pc_z_score[int(inhalation-1000):int(inhalation+1000)])
    
    inh_trig_ob_proj = np.mean(inh_trig_ob_proj,axis = 0)
    inh_trig_pc_proj = np.mean(inh_trig_pc_proj,axis = 0)
    inh_trig_ob_mua = np.mean(inh_trig_ob_mua,axis = 0)
    inh_trig_pc_mua = np.mean(inh_trig_pc_mua,axis = 0)
    
    # compute correlaition as function of phase
    
    numbin = 300

    fase_lenta = faselenta

    position=np.zeros(numbin) # this variable will get the beginning (not the center) of each phase bin (in rads)
    winsize = 2*np.pi/numbin # bin de fase

    position = []
    for j in np.arange(1,numbin+1):
        position.append(-np.pi+(j-1)*winsize)
        


    nbin=numbin 
    pc_proj_phase = []
    ob_proj_phase = []
    r_phase = []

    for j in np.arange(0,nbin):  
        boolean_array = np.logical_and(fase_lenta >=  position[j], fase_lenta < position[j]+winsize)
        I = np.where(boolean_array)[0]
        pc_proj_phase.append(pc_proj[I])
        ob_proj_phase.append(ob_proj[I])
        r_phase.append(stats.pearsonr(ob_proj[I],pc_proj[I])[0])

    # check single neuron sync to resp

    X = np.array(conv_neurons_ob)[1:,odorless_mask].T
    Y = np.array(conv_neurons_pc)[1:,odorless_mask].T
    
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
        
    
    
    X = np.array(conv_neurons_ob)[1:,odorless_mask].T
    Y = np.array(conv_neurons_pc)[1:,odorless_mask].T
    
    ob_mua = stats.zscore(np.sum(X,axis = 1),axis = 0)
    pc_mua = stats.zscore(np.sum(Y,axis = 1),axis = 0)
    
    # compare coherence between max corr proj and average multiunit

    f,c_ob_mua = signal.coherence(ob_mua,np.squeeze(resp), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_pc_mua = signal.coherence(pc_mua,np.squeeze(resp), fs = 2000, nperseg = 2000, nfft = 20000)
    f,c_ob_pc_mua = signal.coherence(ob_mua,pc_mua, fs = 2000, nperseg = 2000, nfft = 20000)

    f,p_ob_mua = signal.welch(ob_mua, fs = 2000, nperseg = 2000, nfft = 20000)
    f,p_pc_mua = signal.welch(pc_mua, fs = 2000, nperseg = 2000, nfft = 20000)
    
    
    ccg_ob_pc_proj = signal.correlate(np.squeeze(ob_proj), np.squeeze(pc_proj),mode = 'same',method = 'fft')
    lags_ob_pc = signal.correlation_lags(ob_proj.size, pc_proj.size, mode = 'same')
    ccg_ob_pc_mua = signal.correlate(ob_mua, pc_mua,mode = 'same',method = 'fft') 
    
    ccg_ob_proj_resp = signal.correlate(np.squeeze(ob_proj), np.squeeze(resp),mode = 'same',method = 'fft')
    ccg_pc_proj_resp = signal.correlate(np.squeeze(pc_proj), np.squeeze(resp),mode = 'same',method = 'fft')
    
    
    from pyinform.transferentropy import transfer_entropy
    
    ob_proj_downsample = ob_proj[::1]
    pc_proj_downsample = pc_proj[::1]
    
    ob_mua_downsample = ob_mua[::1]
    pc_mua_downsample = pc_mua[::1]
    
    t_ob_pc_proj = transfer_entropy(np.squeeze(ob_proj_downsample)>0, np.squeeze(pc_proj_downsample)>0,k = 20)
    t_pc_ob_proj = transfer_entropy(np.squeeze(pc_proj_downsample)>0, np.squeeze(ob_proj_downsample)>0,k = 20)
    
    t_ob_pc_mua = transfer_entropy(ob_mua_downsample>0, pc_mua_downsample>0,k = 20)
    t_pc_ob_mua = transfer_entropy(pc_mua_downsample>0, ob_mua_downsample>0,k = 20)
    
    # check csa against resp
    
    ob_proj_downsample = ob_proj[::50]
    pc_proj_downsample = pc_proj[::50]
    
    resp_z = stats.zscore(resp)
    resp_downsample = resp_z[::50]
    
    mean_resp = np.mean(resp_downsample)
    mean_pc = np.mean(pc_proj_downsample)
    mean_ob = np.mean(ob_proj_downsample)
    
    t_ob_resp_proj = transfer_entropy(np.squeeze(ob_proj_downsample)>mean_ob, np.squeeze(resp_downsample)>mean_resp,k = 20)
    t_resp_ob_proj = transfer_entropy(np.squeeze(resp_downsample)>mean_resp, np.squeeze(ob_proj_downsample)>mean_ob,k = 20)
    
    t_pc_resp_proj = transfer_entropy(np.squeeze(pc_proj_downsample)>mean_pc, np.squeeze(resp_downsample)>mean_resp,k = 20)
    t_resp_pc_proj = transfer_entropy(np.squeeze(resp_downsample)>mean_resp, np.squeeze(pc_proj_downsample)>mean_pc,k = 20)
    
    
    
    
    # compare correlation values between CS and MUA
    
    r_proj = stats.pearsonr(ob_proj,pc_proj)
    r_mua = stats.pearsonr(ob_mua,pc_mua)
    
    from pyinform.mutualinfo import mutual_info
    
    mi_proj = mutual_info(np.squeeze(ob_proj)>0, np.squeeze(pc_proj)>0)
    mi_mua = mutual_info(np.squeeze(ob_mua)>0, np.squeeze(pc_mua)>0)
    
    r_proj_animals.append(r_proj)
    r_mua_animals.append(r_mua)
    
    mi_proj_animals.append(mi_proj)
    mi_mua_animals.append(mi_mua)
    
    


    
    # save results 
    
    r_phase_animals.append(r_phase)

    ob_loadings_animals.append(ob_loadings)
    pc_loadings_animals.append(pc_loadings)
    
    c_ob_proj_animals.append(c_ob_proj)
    c_pc_proj_animals.append(c_pc_proj)
    c_ob_pc_proj_animals.append(c_ob_pc_proj)
    p_ob_proj_animals.append(p_ob_proj)
    p_pc_proj_animals.append(p_pc_proj)
    p_resp_animals.append(p_resp)
    
    ob_comm_resp_phase_animals.append(ob_comm_resp_phase)
    pc_comm_resp_phase_animals.append(pc_comm_resp_phase)
    ob_mua_resp_phase_animals.append(ob_mua_resp_phase)
    pc_mua_resp_phase_animals.append(pc_mua_resp_phase)
    
    
    ob_neurons_resp_phase_animals.append(ob_neurons_resp_phase)
    pc_neurons_resp_phase_animals.append(pc_neurons_resp_phase)
    
    ob_mi_neurons_animals.append(ob_mi_neurons)
    pc_mi_neurons_animals.append(pc_mi_neurons)
    
    c_ob_mua_animals.append(c_ob_mua)
    c_pc_mua_animals.append(c_pc_mua)
    c_ob_pc_mua_animals.append(c_ob_pc_mua)
    p_ob_mua_animals.append(p_ob_mua)
    p_pc_mua_animals.append(p_pc_mua)
    
    ccg_ob_pc_proj_animals.append(ccg_ob_pc_proj)
    lags_ob_pc_animals.append(lags_ob_pc)
    ccg_ob_pc_mua_animals.append(ccg_ob_pc_mua)
    
    ccg_ob_proj_resp_animals.append(ccg_ob_proj_resp)
    ccg_pc_proj_resp_animals.append(ccg_pc_proj_resp)
    
    inh_trig_ob_proj_animals.append(inh_trig_ob_proj)
    inh_trig_pc_proj_animals.append(inh_trig_pc_proj)
    inh_trig_ob_mua_animals.append(inh_trig_ob_mua)
    inh_trig_pc_mua_animals.append(inh_trig_pc_mua)
    
    
    gc_ob_pc_proj_animals.append(t_ob_pc_proj)
    gc_pc_ob_proj_animals.append(t_pc_ob_proj)
    gc_ob_pc_mua_animals.append(t_ob_pc_mua)
    gc_pc_ob_mua_animals.append(t_pc_ob_mua)
    
    gc_ob_resp_proj_animals.append(t_ob_resp_proj)
    gc_resp_ob_proj_animals.append(t_resp_ob_proj)
    gc_pc_resp_proj_animals.append(t_pc_resp_proj)
    gc_resp_pc_proj_animals.append(t_resp_pc_proj)
    
    
    
    ob_proj_downsample = ob_proj[::1]
    pc_proj_downsample = pc_proj[::1]
    
    
    phase_ob = np.angle(signal.hilbert(ob_proj_downsample))
    phase_pc = np.angle(signal.hilbert(pc_proj_downsample))
    
    phase_diff = np.exp(1j*(phase_pc-phase_ob))
    
    
    numbin = 300

    fase_lenta = faselenta

    position=np.zeros(numbin) # this variable will get the beginning (not the center) of each phase bin (in rads)
    winsize = 2*np.pi/numbin # bin de fase

    position = []
    for j in np.arange(1,numbin+1):
        position.append(-np.pi+(j-1)*winsize)
        


    #
    nbin=numbin 
    t_ob_pc_proj_phase = []
    t_pc_ob_proj_phase = []
    r_phase = []
    resp_phase_bins = []
    
    pc_proj_phase = []
    ob_proj_phase = []
    
    phase_diff_phase = []
    
    phase_points_total = []
    for j in np.arange(0,nbin):  
        boolean_array = np.logical_and(fase_lenta >=  position[j], fase_lenta < position[j]+winsize)
        I = np.where(boolean_array)[0]
        pc_proj_phase.append(pc_proj_downsample[I])
        ob_proj_phase.append(ob_proj_downsample[I])
        
        resp_phase_bins.append(np.mean(resp[I]))
        
        t_ob_pc_proj_phase.append(transfer_entropy(np.squeeze(ob_proj_downsample[I])>0, np.squeeze(pc_proj_downsample[I])>0,k = 20))
        t_pc_ob_proj_phase.append(transfer_entropy(np.squeeze(pc_proj_downsample[I])>0, np.squeeze(ob_proj_downsample[I])>0,k = 20))
        
        r_phase.append(stats.pearsonr(ob_proj[I],pc_proj[I])[0])
        
        phase_points_total.append(I.shape[0])
        
        phase_diff_phase.append(np.mean(phase_diff[I]))
        
        
        
    #
    t_ob_pc_proj_phase_animals.append(np.array(t_ob_pc_proj_phase))
    t_pc_ob_proj_phase_animals.append(np.array(t_pc_ob_proj_phase))
    
     
#%% plot results


#%% corr mua cs


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (4,3))

plt.boxplot([np.array(r_mua_animals)[:,0],np.array(r_proj_animals)[:,0]],positions = [1,2], widths = 0.2, showfliers = False)

for x in range(13):
    plt.plot([1.2,1.8],[r_mua_animals[x][0],r_proj_animals[x][0]], color = 'grey')
    
plt.ylabel('OB-PCx Correlation')
plt.xticks(ticks = [1,2],labels = ['MUA','CS1'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0.8,2.2])
s_r,p_r = stats.ttest_rel(np.array(r_mua_animals)[:,0],np.array(r_proj_animals)[:,0])

#plt.savefig('corr_cs_mua.pdf')



#%% respiratory entrainment

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(dpi = 300, figsize = (6,9))

plt.subplot(321)

signs = [1,1,1,1,1,1,1,1,1,1,-1,-1,1]

pc_comm_resp_phase_flip = []
ob_comm_resp_phase_flip = []

for x in range(13):
    
    pc_comm_resp_phase_flip.append(np.array(pc_comm_resp_phase_animals[x])*signs[x])
    ob_comm_resp_phase_flip.append(np.array(ob_comm_resp_phase_animals[x])*signs[x])
    
pc_comm_resp_phase_flip = np.array(pc_comm_resp_phase_flip)
ob_comm_resp_phase_flip = np.array(ob_comm_resp_phase_flip)
   
#
ob_comm_resp_phase = np.mean(ob_comm_resp_phase_flip,axis = 0)
ob_comm_resp_phase = np.hstack([ob_comm_resp_phase,ob_comm_resp_phase,ob_comm_resp_phase])

error_ob_comm_resp_phase = 1*np.std(ob_comm_resp_phase_flip,axis = 0)/np.sqrt(len(ob_comm_resp_phase_animals))
error_ob_comm_resp_phase = np.hstack([error_ob_comm_resp_phase,error_ob_comm_resp_phase,error_ob_comm_resp_phase])

pc_comm_resp_phase = np.mean(pc_comm_resp_phase_flip,axis = 0)
pc_comm_resp_phase = np.hstack([pc_comm_resp_phase,pc_comm_resp_phase,pc_comm_resp_phase])

error_pc_comm_resp_phase = 1*np.std(pc_comm_resp_phase_flip,axis = 0)/np.sqrt(len(pc_comm_resp_phase_animals))
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


norm_ob = np.array(p_ob_proj_animals)[:,0:100]/np.sum(np.array(p_ob_proj_animals)[:,0:100],axis = 1)[:,np.newaxis]
norm_pc = np.array(p_pc_proj_animals)[:,0:100]/np.sum(np.array(p_pc_proj_animals)[:,0:100],axis = 1)[:,np.newaxis]
norm_resp = np.array(p_resp_animals)[:,0:100]/np.sum(np.array(p_resp_animals)[:,0:100],axis = 1)[:,np.newaxis]


mean_p_ob_proj = np.mean(norm_ob,axis = 0)
error_p_ob_proj = np.std(norm_ob,axis = 0)/np.sqrt(len(norm_ob))

mean_p_pc_proj = np.mean(norm_pc,axis = 0)
error_p_pc_proj = np.std(norm_pc,axis = 0)/np.sqrt(len(norm_pc))

mean_p_resp = np.mean(norm_resp,axis = 0)
error_p_resp = np.std(norm_resp,axis = 0)/np.sqrt(len(norm_resp))


ax2 = plt.subplot(322)

plt.plot(f[0:100],mean_p_pc_proj,label = 'PCx CAD')
plt.fill_between(f[0:100], mean_p_pc_proj-error_p_pc_proj,mean_p_pc_proj+error_p_pc_proj,alpha = 0.2)

plt.plot(f[0:100],mean_p_ob_proj,label = 'OB CAD')
plt.fill_between(f[0:100], mean_p_ob_proj-error_p_ob_proj,mean_p_ob_proj+error_p_ob_proj,alpha = 0.2)

plt.plot(f[0:100],mean_p_resp,label = 'Resp', color = 'black')
plt.fill_between(f[0:100], mean_p_resp-error_p_resp,mean_p_resp+error_p_resp, color = 'black',alpha = 0.2)

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#plt.legend()

plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Power')

plt.ylim([0,0.032])

plt.xlim([0,10])

plt.yticks(np.arange(0,0.04,0.01))

axins = inset_axes(ax2, width=0.6, height=0.9)


plt.boxplot([np.array(gc_resp_ob_proj_animals)-np.array(gc_ob_resp_proj_animals),np.array(gc_resp_pc_proj_animals)-np.array(gc_pc_resp_proj_animals)],positions = [1,2], widths = 0.3, showfliers = False)

plt.xticks(ticks = [1,2],labels = ['OB','PCx'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Resp leads')
plt.ylim([-0.02,0.16])
plt.hlines(0,0.7,2.3, color = 'black',linestyles='dashed')
plt.xlim([0.7,2.3])
plt.box()


mean_c_ob_proj = np.mean(c_ob_proj_animals,axis = 0)[0:100]
error_c_ob_proj = np.std(c_ob_proj_animals,axis = 0)[0:100]/np.sqrt(len(c_ob_proj_animals))

mean_c_pc_proj = np.mean(c_pc_proj_animals,axis = 0)[0:100]
error_c_pc_proj = np.std(c_pc_proj_animals,axis = 0)[0:100]/np.sqrt(len(c_pc_proj_animals))

mean_c_ob_mua = np.mean(c_ob_mua_animals,axis = 0)[0:100]
error_c_ob_mua = np.std(c_ob_mua_animals,axis = 0)[0:100]/np.sqrt(len(c_ob_pc_proj_animals))

mean_c_pc_mua = np.mean(c_pc_mua_animals,axis = 0)[0:100]
error_c_pc_mua = np.std(c_pc_mua_animals,axis = 0)[0:100]/np.sqrt(len(c_ob_pc_proj_animals))


ax5 = plt.subplot(323)

plt.plot(f[0:100],mean_c_ob_proj,label = 'OB CAD-Resp', color = 'tab:orange')
plt.fill_between(f[0:100], mean_c_ob_proj-error_c_ob_proj,mean_c_ob_proj+error_c_ob_proj,alpha = 0.2, color = 'tab:orange')

plt.plot(f[0:100],mean_c_ob_mua,'--',label = 'OB MUA-Resp', color = 'tab:orange')
plt.fill_between(f[0:100], mean_c_ob_mua-error_c_ob_mua,mean_c_ob_mua+error_c_ob_mua,alpha = 0.1, color = 'tab:orange')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#plt.legend()

plt.xlabel('Frequency (Hz)')
plt.ylabel('OB-Resp Coherence')
plt.ylim([0,0.58])
plt.yticks(np.arange(0,0.6,0.1))


plt.xlim([0,10])

peak_freq = np.argmax(mean_c_ob_proj)

plt.scatter(peak_freq*0.1,mean_c_ob_mua[peak_freq],s=30,facecolors='none',color = 'tab:orange',label = 'OB MUA')
plt.scatter(peak_freq*0.1,mean_c_ob_proj[peak_freq],s=30,color = 'tab:orange',label = 'OB CAD')

axins = inset_axes(ax5, width=0.6, height=0.9)

plt.boxplot([np.array(c_ob_mua_animals)[:,peak_freq],np.array(c_ob_proj_animals)[:,peak_freq]])
for x in range(13):
    plt.plot([1,2],[c_ob_mua_animals[x][peak_freq],c_ob_proj_animals[x][peak_freq]], color = 'grey', alpha = 0.5)

plt.xticks(ticks = [1,2],labels = ['MUA','CAD'])
plt.xlim([0.7,2.3])
plt.box()
plt.yticks(np.arange(0.1,0.8,0.1),fontsize = 6)

s_c_ob,p_c_ob = stats.ttest_rel(np.array(c_ob_mua_animals)[:,peak_freq],np.array(c_ob_proj_animals)[:,peak_freq])

ax6 = plt.subplot(324)

plt.plot(f[0:100],mean_c_pc_proj,label = 'PCx CAD-Resp', color = 'tab:blue')
plt.fill_between(f[0:100], mean_c_pc_proj-error_c_pc_proj,mean_c_pc_proj+error_c_pc_proj,alpha = 0.2, color = 'tab:blue')


#plt.plot(f,mean_c_ob_mua,'--',label = 'OB MUA-Resp', color = 'tab:blue')
plt.plot(f[0:100],mean_c_pc_mua,'--',label = 'PCx MUA-Resp',color = 'tab:blue')
plt.fill_between(f[0:100], mean_c_pc_mua-error_c_pc_mua,mean_c_pc_mua+error_c_pc_mua,alpha = 0.1,color = 'tab:blue')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

#plt.legend()

plt.xlabel('Frequency (Hz)')
plt.ylabel('PCx-Resp Coherence')
plt.ylim([0,0.58])
plt.xlim([0,10])
plt.yticks(np.arange(0,0.6,0.1))

peak_freq = np.argmax(mean_c_pc_proj)

plt.scatter(peak_freq*0.1,mean_c_pc_mua[peak_freq],s=30,facecolors='none',color = 'tab:blue',label = 'OB MUA')
plt.scatter(peak_freq*0.1,mean_c_pc_proj[peak_freq],s=30,color = 'tab:blue',label = 'OB CAD')

axins = inset_axes(ax6, width=0.6, height=0.9)

plt.boxplot([np.array(c_pc_mua_animals)[:,peak_freq],np.array(c_pc_proj_animals)[:,peak_freq]], showfliers = False)
for x in range(13):
    plt.plot([1,2],[c_pc_mua_animals[x][peak_freq],c_pc_proj_animals[x][peak_freq]], color = 'grey', alpha = 0.5)

plt.xticks(ticks = [1,2],labels = ['MUA','CAD'])
plt.xlim([0.7,2.3])
plt.box()
plt.yticks(np.arange(0.1,0.6,0.1),fontsize = 6)

s_c_pc,p_c_pc = stats.ttest_rel(np.array(c_pc_mua_animals)[:,peak_freq],np.array(c_pc_proj_animals)[:,peak_freq])


plt.subplot(325)


mean_r_phase = np.mean(np.array(r_phase_animals),axis = 0)
mean_r_phase = np.hstack([mean_r_phase,mean_r_phase,mean_r_phase])

error_r_phase = stats.sem(np.array(r_phase_animals),axis = 0)
error_r_phase = np.hstack([error_r_phase,error_r_phase,error_r_phase])


plt.plot(np.linspace(0,1080,900)-295,mean_r_phase,color = 'purple')
plt.fill_between(np.linspace(0,1080,900)-295, mean_r_phase-error_r_phase, mean_r_phase+error_r_phase,color = 'purple',alpha = 0.2)



#plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
#plt.xticks(ticks = np.arange(32,105,12),labels = np.round(np.arange(0,730,120)).astype(int),rotation = 45)
plt.xlim([0,720])
plt.ylabel('OB-PCx CSA Correlation')

plt.ylim([0.06,0.28])
plt.xlabel('Respiration phase (deg)')

ax4 = plt.subplot(326)

mean_coher_mua = np.mean(c_ob_pc_mua_animals,axis = 0)[0:100]
error_coher_mua = np.std(c_ob_pc_mua_animals,axis = 0)[0:100]/np.sqrt(len(c_ob_pc_mua_animals))

mean_coher_proj = np.mean(c_ob_pc_proj_animals,axis = 0)[0:100]
error_coher_proj = np.std(c_ob_pc_proj_animals,axis = 0)[0:100]/np.sqrt(len(c_ob_pc_proj_animals))


plt.plot(f[0:100],mean_coher_mua,'--',label = 'MUA', color = 'purple')
plt.fill_between(f[0:100], mean_coher_mua-error_coher_mua, mean_coher_mua+error_coher_mua,alpha = 0.2, color = 'plum')

plt.plot(f[0:100],mean_coher_proj,label = 'CAD', color = 'purple')
plt.fill_between(f[0:100], mean_coher_proj-error_coher_proj, mean_coher_proj+error_coher_proj,alpha = 0.2, color = 'purple')


plt.xlabel('Frequency (Hz)')
plt.ylabel('OB-PCx Coherence')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,10])
plt.ylim([0,0.58])
plt.yticks(np.arange(0,0.6,0.1))
#plt.legend()

peak_freq = np.argmax(mean_coher_proj)

plt.scatter(peak_freq*0.1,mean_coher_mua[peak_freq],s=30,facecolors='none',color = 'purple',label = 'OB MUA')
plt.scatter(peak_freq*0.1,mean_coher_proj[peak_freq],s=30,color = 'purple',label = 'OB CAD')

axins = inset_axes(ax4, width=0.6, height=0.9)
plt.boxplot([np.array(c_ob_pc_mua_animals)[:,peak_freq],np.array(c_ob_pc_proj_animals)[:,peak_freq]], showfliers = False)

for x in range(13):
    plt.plot([1,2],[c_ob_pc_mua_animals[x][peak_freq],c_ob_pc_proj_animals[x][peak_freq]], color = 'grey', alpha = 0.5)

plt.xticks(ticks = [1,2],labels = ['MUA','CAD'])
plt.xlim([0.7,2.3])
plt.box()
plt.yticks(np.arange(0.1,0.7,0.1),fontsize = 6)

s_c_ob_pc,p_c_ob_pc = stats.ttest_rel(np.array(c_ob_pc_mua_animals)[:,peak_freq],np.array(c_ob_pc_proj_animals)[:,peak_freq])



plt.tight_layout()

plt.savefig('CSA_Resp_quant.pdf')


#%% directionality with resp


ccg_ob_proj_resp_plot = []
ccg_pc_proj_resp_plot = []

for x in range(len(names)):  
    
    zero_lag = np.where(lags_ob_pc_animals[x] == 0)[0]
    ccg_ob_proj_resp_plot.append(ccg_ob_proj_resp_animals[x][int(zero_lag-1000):int(zero_lag+1000)]/np.max(ccg_ob_proj_resp_animals[x]))
    ccg_pc_proj_resp_plot.append(ccg_pc_proj_resp_animals[x][int(zero_lag-1000):int(zero_lag+1000)]/np.max(ccg_pc_proj_resp_animals[x]))
    

    
    
mean_ccg_ob_pc_proj = np.mean(ccg_pc_proj_resp_plot, axis = 0)
mean_ccg_ob_pc_mua = np.mean(ccg_ob_proj_resp_plot, axis = 0)
error_ccg_ob_pc_proj = np.std(ccg_pc_proj_resp_plot, axis = 0)/np.sqrt(13)
error_ccg_ob_pc_mua = np.std(ccg_ob_proj_resp_plot, axis = 0)/np.sqrt(13)

    
lags = np.arange(-1000,1000,1)/2    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

    
plt.figure(dpi = 300, figsize = (6,3))

gs = gridspec.GridSpec(1, 2, width_ratios = [4,1])

plt.subplot(gs[0])

plt.plot(lags,mean_ccg_ob_pc_proj, label = 'PCx')
plt.fill_between(lags,mean_ccg_ob_pc_proj-error_ccg_ob_pc_proj,mean_ccg_ob_pc_proj+error_ccg_ob_pc_proj, alpha = 0.2)

plt.plot(lags,mean_ccg_ob_pc_mua, label = 'OB')
plt.fill_between(lags,mean_ccg_ob_pc_mua-error_ccg_ob_pc_mua,mean_ccg_ob_pc_mua+error_ccg_ob_pc_mua, alpha = 0.2)

plt.legend()
plt.xlim([500,-500])
plt.vlines(0,-1.52,1.2, color = 'black', linewidth = 0.8)
plt.ylim([-1.5,0.8])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Cross Correlation')
plt.xlabel('Lags (ms)')
plt.title('<- Resp Leads / CS Leads ->')




plt.subplot(gs[1])

plt.boxplot([np.array(gc_resp_ob_proj_animals)-np.array(gc_ob_resp_proj_animals),np.array(gc_resp_pc_proj_animals)-np.array(gc_pc_resp_proj_animals)],positions = [1,2], widths = 0.3, showfliers = False)

plt.xticks(ticks = [1,2],labels = ['OB','PCx'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('TE D (Resp-CS)')
plt.ylim([-0.02,0.08])
plt.hlines(0,0.7,2.3, color = 'black',linestyles='dashed')
plt.xlim([0.7,2.3])


plt.tight_layout()

#plt.savefig('resp_leads.pdf')

t_ob, p_ob = stats.ttest_rel(np.array(gc_resp_ob_proj_animals),np.array(gc_ob_resp_proj_animals))
t_pc, p_pc = stats.ttest_rel(np.array(gc_resp_pc_proj_animals),np.array(gc_pc_resp_proj_animals))


#%% cs mua cross correlogram

ccg_ob_pc_proj_plot = []
ccg_ob_pc_mua_plot = []

for x in range(len(names)):  
    
    zero_lag = np.where(lags_ob_pc_animals[x] == 0)[0]
    ccg_ob_pc_proj_plot.append(ccg_ob_pc_proj_animals[x][int(zero_lag-1000):int(zero_lag+1000)]/np.max(ccg_ob_pc_proj_animals[x]))
    ccg_ob_pc_mua_plot.append(ccg_ob_pc_mua_animals[x][int(zero_lag-1000):int(zero_lag+1000)]/np.max(ccg_ob_pc_mua_animals[x]))
    
    
mean_ccg_ob_pc_proj = np.mean(ccg_ob_pc_proj_plot, axis = 0)
mean_ccg_ob_pc_mua = np.mean(ccg_ob_pc_mua_plot, axis = 0)
error_ccg_ob_pc_proj = np.std(ccg_ob_pc_proj_plot, axis = 0)/np.sqrt(13)
error_ccg_ob_pc_mua = np.std(ccg_ob_pc_mua_plot, axis = 0)/np.sqrt(13)

    
lags = np.arange(-1000,1000,1)/2    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

    
plt.figure(dpi = 300, figsize = (6,6))

gs = gridspec.GridSpec(2, 2)

plt.subplot(gs[0:2])

plt.plot(lags,mean_ccg_ob_pc_proj, label = 'Proj')
plt.fill_between(lags,mean_ccg_ob_pc_proj-error_ccg_ob_pc_proj,mean_ccg_ob_pc_proj+error_ccg_ob_pc_proj, alpha = 0.2)

plt.plot(lags,mean_ccg_ob_pc_mua, label = 'Mua')
plt.fill_between(lags,mean_ccg_ob_pc_mua-error_ccg_ob_pc_mua,mean_ccg_ob_pc_mua+error_ccg_ob_pc_mua, alpha = 0.2)

plt.legend()
plt.xlim([-500,500])
plt.vlines(0,-0.52,1.2, color = 'black', linewidth = 0.8)
plt.ylim([-0.52,1.2])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Cross Correlation')
plt.xlabel('Lags (ms)')
plt.title('<- OB Leads / PCx Leads ->')


plt.subplot(gs[2])

plt.boxplot([np.array(gc_ob_pc_proj_animals),np.array(gc_pc_ob_proj_animals)],positions = [1,2], widths = 0.2, showfliers = False)

for x in range(13):
    plt.plot([1.2,1.8],[gc_ob_pc_proj_animals[x],gc_pc_ob_proj_animals[x]], color = 'grey')
    
plt.ylabel('Transfer Entropy')
plt.xticks(ticks = [1,2],labels = ['OB-PCx','PCx-OB'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0.8,2.2])
s_proj,p_proj = stats.ttest_rel(np.array(gc_ob_pc_proj_animals),np.array(gc_pc_ob_proj_animals))
plt.title('Can Corr Proj')
#plt.text(1.5,12,'p = '+ str(np.round(p_proj,decimals = 3)),fontsize = 12)
plt.yscale('log')
plt.ylim([1e-6,1e-2])

plt.subplot(gs[3])



plt.boxplot([np.array(gc_ob_pc_mua_animals),np.array(gc_pc_ob_mua_animals)],positions = [1,2], widths = 0.2, showfliers = False)

for x in range(13):
    plt.plot([1.2,1.8],[gc_ob_pc_mua_animals[x],gc_pc_ob_mua_animals[x]], color = 'grey')
    
    
#plt.ylabel('G-Causality Mua')
plt.xticks(ticks = [1,2],labels = ['OB-PCx','PCx-OB'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0.8,2.2])
plt.ylim([1e-6,1e-2])

s_mua,p_mua = stats.ttest_rel(np.array(gc_ob_pc_mua_animals),np.array(gc_pc_ob_mua_animals))
plt.title('MUA')
#plt.text(1.5,0.000075,'p = '+ str(np.round(p_mua,decimals = 3)),fontsize = 12)
plt.yscale('log')

plt.tight_layout()


#%% directionallity cs

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,4))

plt.boxplot([np.array(gc_ob_pc_proj_animals)-np.array(gc_pc_ob_proj_animals),np.array(gc_ob_pc_mua_animals)-np.array(gc_pc_ob_mua_animals)],positions = [1,1.5], widths = 0.2, showfliers = False)

plt.xticks(ticks = [1,1.5],labels = ['CAD','MUA'])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('PCx leads       OB leads')
plt.ylim([-0.0012,0.0015])
plt.hlines(0,0.6,1.9, color = 'black',linestyles='dashed')
plt.xlim([0.8,1.7])
plt.title('Information Transfer Sense')


s_mua,p_mua = stats.ttest_rel(np.array(gc_ob_pc_mua_animals),np.array(gc_pc_ob_mua_animals))

s_proj,p_proj = stats.ttest_rel(np.array(gc_ob_pc_proj_animals),np.array(gc_pc_ob_proj_animals))

plt.savefig('dir.pdf')



#%% directionallity phase



import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,4))

t_ob_pc_proj_phase_diff = np.mean(np.array(t_ob_pc_proj_phase_animals)-np.array(t_pc_ob_proj_phase_animals),axis = 0)
error_t_ob_pc_proj_phase_diff = stats.sem(np.array(t_ob_pc_proj_phase_animals)-np.array(t_pc_ob_proj_phase_animals),axis = 0)

t_ob_pc_proj_phase_diff = np.hstack([t_ob_pc_proj_phase_diff,t_ob_pc_proj_phase_diff,t_ob_pc_proj_phase_diff])
error_t_ob_pc_proj_phase_diff = np.hstack([error_t_ob_pc_proj_phase_diff,error_t_ob_pc_proj_phase_diff,error_t_ob_pc_proj_phase_diff])


phase_vector = np.linspace(0,1080,300*3)
plt.plot(phase_vector-300,t_ob_pc_proj_phase_diff, color = 'black')
plt.fill_between(phase_vector-300, t_ob_pc_proj_phase_diff-error_t_ob_pc_proj_phase_diff, t_ob_pc_proj_phase_diff+error_t_ob_pc_proj_phase_diff, alpha = 0.2, color = 'black')

plt.xlim([0,600])

plt.ylabel('CS Transder Entropy Delta (OB-PCx)')
plt.xlabel('Respiration phase (deg)')

plt.plot(phase_vector-300,np.hstack([np.array(resp_phase_bins),np.array(resp_phase_bins),np.array(resp_phase_bins)])*0.00001, label = 'Resp')

plt.legend()
plt.hlines(0,0,600, linestyles = 'dashed', color = 'black')

t_phase = []
h_phase = []
p_phase = []

for x in range(np.array(t_ob_pc_proj_phase_animals).shape[1]):
    
    t,p = stats.ttest_rel(np.array(t_ob_pc_proj_phase_animals)[:,x],np.array(t_pc_ob_proj_phase_animals)[:,x])
    
    t_phase.append(t)
    h_phase.append(p<0.05)
    p_phase.append(p)
    

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'bothplt.savefig('dir_phase.pdf')

# get stats

min_t = t_phase[87]
min_p = p_phase[87]


max_t = t_phase[264]
max_p = p_phase[264]


#%% corr phase example


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,3))
gs = gridspec.GridSpec(2, 1, height_ratios = [2,1])


resp_average = phase_amp_hist(resp,faselenta,numbin = 300)
resp_average = np.hstack([resp_average,resp_average,resp_average])

resp_bin = []

ob_proj_phase_rep = np.hstack([ob_proj_phase,ob_proj_phase,ob_proj_phase])
pc_proj_phase_rep = np.hstack([pc_proj_phase,pc_proj_phase,pc_proj_phase])

plt.subplot(gs[0])
for index, x in enumerate(np.arange(0,300*3,20)):
    
    
    plt.scatter(ob_proj_phase_rep[x]+index*10,pc_proj_phase_rep[x], s = 0.001, alpha = 0.2, color = 'black',rasterized = True)
    plt.yticks([])
    
    
    plt.vlines(index*10,-6,6, color = 'grey', linestyle = 'dashed')
    
    
    r = stats.pearsonr(ob_proj_phase_rep[x],pc_proj_phase_rep[x])[0]
    
    resp_bin.append(resp_average[x])
    
    
    trend = np.polyfit(ob_proj_phase_rep[x]+index*10,pc_proj_phase_rep[x],1)
    trendpoly = np.poly1d(trend) 
    plt.plot(ob_proj_phase_rep[x]+index*10,trendpoly(ob_proj_phase_rep[x]+index*10), color = 'red')


plt.box()


plt.xlim([130,340])

#
plt.subplot(gs[1])


resp_average = phase_amp_hist(resp,faselenta,numbin = 300)

resp_average = np.hstack([resp_average,resp_average,resp_average])

plt.plot(np.linspace(0,360*3,300*3),resp_average[::1], color = 'black')
#plt.xticks(ticks = np.arange(0,360,10)[::3])
plt.xlim([0,360])
plt.box()

plt.xlim([300,800])

plt.yticks([])

plt.savefig('resp_correlation.pdf')



#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,6))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",['m','white','c'])

gs = gridspec.GridSpec(2, 2, width_ratios = [0.8,2], height_ratios=[np.concatenate(ob_loadings_animals).shape[0],np.concatenate(pc_loadings_animals).shape[0]])

plt.subplot(gs[0])

sorted_loadings = np.sort(np.squeeze(np.concatenate(ob_loadings_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,320])
plt.yticks([])

plt.subplot(gs[1])

ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_animals)))

plt.imshow(ob_neurons_resp_phase_norm[load_sort,:], cmap = cmap,aspect = 'auto',vmin = -1, vmax = 1)

plt.ylim([0,320])
#plt.xticks(ticks = np.arange(16,53,6),labels = [],rotation = 30)
plt.yticks([])
#plt.xlim([16,52])
#plt.title('OB')


plt.subplot(gs[2])

sorted_loadings = np.sort(np.squeeze(np.concatenate(pc_loadings_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.yticks([])

plt.xlim([-1,1])
plt.xlabel('CCA Weight')
plt.ylim([0,849])

plt.subplot(gs[3])

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)

load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_animals)))

plt.imshow(pc_neurons_resp_phase_norm[load_sort,:],aspect = 'auto',vmin = -1, vmax = 1, cmap = cmap)
#plt.colorbar()
plt.ylim([0,849])
plt.yticks([])
#plt.xlim([16,52])
#plt.title('PCx')
plt.xticks(ticks = np.arange(16,53,6),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)
plt.tight_layout()
#plt.colorbar(location = 'bottom')
plt.xlabel('Respiration Phase (deg)')

#plt.savefig('all_units_cmap.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,6))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",['m','white','c'])

gs = gridspec.GridSpec(2, 2, width_ratios = [0.8,2], height_ratios=[np.concatenate(ob_loadings_animals).shape[0],np.concatenate(pc_loadings_animals).shape[0]])

plt.subplot(gs[0])

sorted_loadings = np.sort(np.squeeze(np.concatenate(ob_loadings_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-1,1])
plt.ylim([0,320])
plt.yticks([])

plt.subplot(gs[1])

ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)


load_sort = np.argsort(np.squeeze(np.concatenate(ob_loadings_animals)))

plt.imshow(ob_neurons_resp_phase_norm[load_sort,:], cmap = cmap,aspect = 'auto',vmin = -1, vmax = 1)

plt.ylim([0,320])
#plt.xticks(ticks = np.arange(16,53,6),labels = [],rotation = 30)
plt.xticks(ticks = np.arange(245,250+600,100),labels = [],rotation = 30)
plt.xlim([245,245+600])

plt.yticks([])


#plt.xlim([16,52])
#plt.title('OB')

plt.subplot(gs[2])

sorted_loadings = np.sort(np.squeeze(np.concatenate(pc_loadings_animals)))
plt.barh(np.arange(0,sorted_loadings.shape[0]),sorted_loadings,  color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.yticks([])

plt.xlim([-1,1])
plt.xlabel('CCA Weight')
plt.ylim([0,849])

plt.subplot(gs[3])

pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)

load_sort = np.argsort(np.squeeze(np.concatenate(pc_loadings_animals)))

plt.imshow(pc_neurons_resp_phase_norm[load_sort,:],aspect = 'auto',vmin = -1, vmax = 1, cmap = cmap)
#plt.colorbar()
plt.ylim([0,849])
plt.yticks([])
plt.xlim([245,245+600])
#plt.title('PCx')
plt.xticks(ticks = np.arange(245,250+600,100),labels = np.round(np.arange(0,740,120)).astype(int),rotation = 30)
plt.tight_layout()
#plt.colorbar(location = 'bottom')
plt.xlabel('Respiration Phase (deg)')

#plt.savefig('all_units_cmap.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,5))

phase_vector = np.arange(20,28)

ob_neurons_resp_phase = np.concatenate(ob_neurons_resp_phase_animals)
ob_neurons_resp_phase = np.hstack([ob_neurons_resp_phase,ob_neurons_resp_phase,ob_neurons_resp_phase])
ob_neurons_resp_phase_norm = ob_neurons_resp_phase/np.sum(ob_neurons_resp_phase,axis = 1)[:,np.newaxis]
ob_neurons_resp_phase_norm = stats.zscore(ob_neurons_resp_phase,axis = 1)


plt.xlabel('CCA Weight')

plt.subplot(221)
#plt.scatter(np.mean(ob_neurons_resp_phase_norm[:,phase_vector],axis = 1),np.concatenate(ob_loadings_animals_thy), s = 2, alpha = 0.5, color = 'black')
sns.regplot(np.abs(np.concatenate(ob_loadings_animals)),np.concatenate(ob_mi_neurons_animals),x_bins = np.linspace(0,1.2,10) , color = 'tab:orange')
# plt.xlim([-0.6,1])
# plt.ylim([0,0.04])
#plt.ylim([-0.02,0.35])
plt.xlim([-0.2,1.1])
plt.ylim([0.003,0.5])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
r_ob, p_ob = stats.pearsonr(np.abs(np.squeeze(np.concatenate(ob_loadings_animals))),np.squeeze(np.concatenate(ob_mi_neurons_animals)))


plt.ylabel('Respiratory coupling')
plt.yscale('log')
#plt.text(-0.6,-1.15, s = 'R = '+str(np.round(r_ob,decimals = 3))+'  p = '+str(np.round(p_ob,decimals = 3)))

#plt.title('OB')


pc_neurons_resp_phase = np.concatenate(pc_neurons_resp_phase_animals)
pc_neurons_resp_phase = np.hstack([pc_neurons_resp_phase,pc_neurons_resp_phase,pc_neurons_resp_phase])
pc_neurons_resp_phase_norm = pc_neurons_resp_phase/np.sum(pc_neurons_resp_phase,axis = 1)[:,np.newaxis]
pc_neurons_resp_phase_norm = stats.zscore(pc_neurons_resp_phase,axis = 1)


plt.xlabel('Abs CS1 Weights')

plt.subplot(222)
#plt.scatter(np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1),np.concatenate(pc_loadings_animals_thy), s = 2, alpha = 0.5, color = 'black')
sns.regplot(np.abs(np.concatenate(pc_loadings_animals)),np.concatenate(pc_mi_neurons_animals),x_bins = np.linspace(0,1.2,10) , color = 'tab:blue')
#sns.regplot(np.concatenate(pc_loadings_animals),np.mean(pc_neurons_resp_phase_norm[:,phase_vector],axis = 1),x_bins = np.linspace(-1,1,12) , color = 'tab:blue')
plt.xlim([-0.2,1.1])
plt.ylim([0.003,0.5])
# plt.xlim([-0.6,1])
# plt.ylim([0,0.04])
#plt.ylim([-1.2,1])
#plt.ylim([-0.02,0.35])
plt.yscale('log')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
r_pc, p_pc = stats.pearsonr(np.abs(np.squeeze(np.concatenate(pc_loadings_animals))),np.squeeze(np.concatenate(pc_mi_neurons_animals)))

#plt.plot(np.linspace(-1.3,1.1,100),np.linspace(-1.3,1.1,100), linestyle = 'dashed', color = 'black', alpha = 0.5)

#plt.text(-0.4,-1.15, s = 'R = '+str(np.round(r_pc,decimals = 3))+'  p = '+str(np.round(p_pc,decimals = 3)))

#plt.title('PCx')

plt.xlabel('Abs CS1 Weights')
plt.tight_layout()

plt.savefig('resp_coup.pdf')

#%%

np.savez('pc_spike_resp_phase_control.npz',pc_neurons_resp_phase_animals=pc_neurons_resp_phase_animals)
np.savez('pc_loadings_control.npz',pc_loadings_animals=pc_loadings_animals)

