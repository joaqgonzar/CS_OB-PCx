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
from statsmodels.multivariate.cancorr import CanCorr
from scipy.stats import pearsonr

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

directory = '/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset'

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

ob_loadings_lags_animals = []
pc_loadings_lags_animals = []    
r_lags_animals = []

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

    # cca = CCA(n_components=1)
    # cca.fit(X, Y)
    # ob_proj, pc_proj = cca.transform(X, Y)

    # params = cca.get_params()
    
    # ob_loadings = cca.x_loadings_
    # pc_loadings = cca.y_loadings_
    
    # lag time-series to get cross correlograms
    
    
    lenght = X.shape[0]-4000
    
    r_lags = []
    ob_loadings_lags = []
    pc_loadings_lags = []
    
    for x in np.arange(-200,320,10):
        
        print(x)
    
        start_x = 0+2000
        start_y = x+2000
        end_x = start_x+lenght
        end_y = start_y+lenght
        
        cca = CCA(n_components=1,max_iter=100000,tol = 1e-12)
        #cca = CCA(n_components=1)
        cca.fit(X[start_x:end_x,:], Y[start_y:end_y,:])
        
        ob_loadings = cca.x_rotations_
        pc_loadings = cca.y_rotations_
        
        ob_proj = np.sum(X[start_x:end_x,:]*ob_loadings.T,axis = 1)
        pc_proj = np.sum(Y[start_y:end_y,:]*pc_loadings.T,axis = 1)
        
        #ob_proj, pc_proj = cca.transform(X[start_x:end_x,:], Y[start_y:end_y,:])
        ob_loadings_lags.append(cca.x_loadings_)
        pc_loadings_lags.append(cca.y_loadings_)
        
        r = pearsonr(np.squeeze(ob_proj),np.squeeze(pc_proj))[0]
        r_lags.append(r)
        
    ob_loadings_lags_animals.append(ob_loadings_lags)    
    pc_loadings_lags_animals.append(pc_loadings_lags)    
    r_lags_animals.append(r_lags)
    
    #%% save results 
    
    np.savez('can_corr_lags_new_params.npz',ob_loadings_lags_animals = ob_loadings_lags_animals, pc_loadings_lags_animals= pc_loadings_lags_animals, r_lags_animals = r_lags_animals)
    
    #%%
    
    can_corr_lags = np.load('can_corr_lags_new_params.npz', allow_pickle=(True))
    
    ob_loadings_lags_animals = can_corr_lags['ob_loadings_lags_animals']
    pc_loadings_lags_animals = can_corr_lags['pc_loadings_lags_animals']
    r_lags_animals = can_corr_lags['r_lags_animals']
    
    #%%
    fig = plt.figure(dpi = 300, figsize = (3,5)) 
    gs = gridspec.GridSpec(3, 1,height_ratios = [1,4,3])

    animal = 12

    pc_loadings_lags = np.squeeze(np.array(pc_loadings_lags_animals[animal]))
    ob_loadings_lags = np.squeeze(np.array(ob_loadings_lags_animals[animal]))
    
    plt.subplot(gs[2])
    plt.plot(np.arange(-200,320,20)/2,r_lags_animals[animal])
    plt.xlim([-100,150])
    plt.xlabel('Lags (ms)')
    plt.vlines(0,0.15,0.25, color = 'black', linestyles='dashed')
    #plt.ylim([0.15,0.25])
    plt.xticks(np.arange(-100,160,25), rotation = 60)
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.ylabel('OB-PCx Can Corr')
    
    plt.subplot(gs[0])
    plt.imshow(ob_loadings_lags.T,aspect = 'auto', cmap = 'bwr')
    plt.ylabel('OB Cells')
    plt.xlim([0,pc_loadings_lags.shape[0]-1])
    plt.xticks(np.arange(0,pc_loadings_lags.shape[0],2.5),labels = [])
    plt.vlines(10,0,ob_loadings_lags.shape[1]-1, color = 'black', linestyles='dashed')
    
    plt.subplot(gs[1])
    plt.imshow(pc_loadings_lags.T,aspect = 'auto', cmap = 'bwr')
    plt.ylabel('PCx Cells')
    plt.xlim([0,pc_loadings_lags.shape[0]-1])
    plt.xticks(np.arange(0,pc_loadings_lags.shape[0],2.5),labels = [])
    plt.vlines(10,0,pc_loadings_lags.shape[1]-1, color = 'black', linestyles='dashed')
    
    fig.align_ylabels()
    
    plt.tight_layout()
    
    
    #%%
    
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    fig = plt.figure(dpi = 300, figsize = (4,3)) 
    
    mean_r_lags = np.mean(r_lags_animals,axis = 0)
    error_r_lags = np.std(r_lags_animals,axis = 0)/np.sqrt(len(names))
    
    
    plt.plot(np.arange(-200,320,10)/2,mean_r_lags, color = 'black')
    plt.fill_between(np.arange(-200,320,10)/2,mean_r_lags-error_r_lags,mean_r_lags+error_r_lags, color = 'grey', alpha = 0.2)
    plt.xlim([-100,150])
    plt.xlabel('Corr Lag (ms, OB reference)')
    plt.vlines(0,0.15,0.4, color = 'black', linestyles='dashed')
    plt.ylim([0.24,0.4])
    plt.xticks(np.arange(-100,160,25), rotation = 45)
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.ylabel('Canonical Correlation')
    plt.title('OB-PCx')
    
    #%% r2 
    
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    fig = plt.figure(dpi = 300, figsize = (4,3)) 
    
    mean_r_lags = np.mean(r_lags_animals**2,axis = 0)
    error_r_lags = np.std(r_lags_animals**2,axis = 0)/np.sqrt(len(names))
    
    
    plt.plot(np.arange(-200,320,20)/2,mean_r_lags, color = 'black')
    plt.fill_between(np.arange(-200,320,20)/2,mean_r_lags-error_r_lags,mean_r_lags+error_r_lags, color = 'grey', alpha = 0.2)
    plt.xlim([-100,150])
    plt.xlabel('Corr Lag (ms, OB reference)')
    #plt.vlines(0,0.15,0.4, color = 'black', linestyles='dashed')
    #plt.ylim([0.24,0.4])
    plt.xticks(np.arange(-100,160,25), rotation = 45)
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.ylabel('Canonical Correlation')
    plt.title('OB-PCx')
    
    #plt.savefig('cancorr_lags.pdf')
    
    #%%
    fig = plt.figure(dpi = 300, figsize = (9,3)) 
    
    mean_r_lags = np.mean(r_lags_animals,axis = 0)
    error_r_lags = np.std(r_lags_animals,axis = 0)/np.sqrt(len(names))
    
    plt.subplot(131)
    
    plt.plot(np.arange(-200,320,20)/2,mean_r_lags, color = 'black')
    plt.fill_between(np.arange(-200,320,20)/2,mean_r_lags-error_r_lags,mean_r_lags+error_r_lags, color = 'grey', alpha = 0.2)
    plt.xlim([-100,150])
    plt.xlabel('Corr Lag (ms, OB reference)')
    plt.vlines(0,0.15,0.4, color = 'black', linestyles='dashed')
    plt.ylim([0.24,0.4])
    plt.xticks(np.arange(-100,160,25), rotation = 45)
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.ylabel('Canonical Correlation')
    plt.title('OB-PCx')
    
    plt.subplot(132)
    plt.scatter(np.concatenate(ob_loadings_lags_animals[:,10]),np.concatenate(ob_loadings_lags_animals[:,13]),s = 2, color ='tab:orange')
    plt.plot(np.linspace(-1,1,100),np.linspace(-1,1,100), color = 'black')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.xticks(np.arange(-0.5,0.7,0.2), rotation = 45)
    plt.xlabel('Can Corr Loadings (0 ms)')
    plt.ylabel('Can Corr Loadings (25 ms)')
    plt.title('OB')
    plt.xlim([-0.5,0.65])
    plt.ylim([-0.5,0.65])
    r = pearsonr(np.squeeze(np.concatenate(ob_loadings_lags_animals[:,10])),np.squeeze(np.concatenate(ob_loadings_lags_animals[:,13])))
    plt.text(-0.45,0.5,'Corr = '+str(np.round(r[0],decimals = 2)))
    plt.text(-0.45,0.38,r'$p < 1x10^{-72}$')
    
    
    plt.subplot(133)
    plt.scatter(np.concatenate(pc_loadings_lags_animals[:,10]),np.concatenate(pc_loadings_lags_animals[:,13]),s = 2)
    plt.plot(np.linspace(-1,1,100),np.linspace(-1,1,100), color = 'black')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.xticks(np.arange(-0.6,0.7,0.2), rotation = 45)
    plt.xlabel('Can Corr Loadings (0 ms)')
    plt.title('PCx')
    plt.xlim([-0.6,0.65])
    plt.ylim([-0.6,0.65])
    r = pearsonr(np.squeeze(np.concatenate(pc_loadings_lags_animals[:,10])),np.squeeze(np.concatenate(pc_loadings_lags_animals[:,13])))
    plt.text(-0.45,0.5,'Corr = '+str(np.round(r[0],decimals = 2)))
    plt.text(-0.45,0.38,r'$p < 1x10^{-260}$')
    plt.tight_layout()

        
    #%%
    # plt.scatter(ob_loadings_lags_animals[:,])
    # plt.scatter(ob_loadings_lags_animals[:,])
    #%% check loadings at peak
    
    
    
    max_corr_lag = np.argmax(r_lags_animals,axis = 1)
    
    loadings_max = []
    loadings_zero = []
    loading_diff_pc = []
    loading_diff_ob = []
    
    
    
    for index, x in enumerate(max_corr_lag):
        loadings_max = np.array(pc_loadings_lags_animals[index])[13,:]
        loadings_zero = np.array(pc_loadings_lags_animals[index])[10,:]
        loading_diff_pc.append(loadings_zero-loadings_max)
        
        loadings_max = np.array(ob_loadings_lags_animals[index])[13,:]
        loadings_zero = np.array(ob_loadings_lags_animals[index])[10,:]
        loading_diff_ob.append(loadings_zero-loadings_max)
     
    hist_ob = np.histogram(np.squeeze(np.concatenate(loading_diff_ob)), bins = np.linspace(-1,1,100))[0]
    hist_pc = np.histogram(np.squeeze(np.concatenate(loading_diff_pc)), bins = np.linspace(-1,1,100))[0]
        
    #%%    
    
    loading_diff_ob_rand_all = []
    loading_diff_pc_rand_all = []
    
    for y in range(100):
        
        loading_diff_ob_rand = []
        loading_diff_pc_rand = []
        
        for index, x in enumerate(max_corr_lag):
            
            list_int = np.arange(0,26)
            int1 = np.random.choice(list_int)
            new_list = np.setdiff1d(list_int,int1)
            int2 = np.random.choice(new_list)
                
            loadings_max = np.array(ob_loadings_lags_animals[index])[int1,:]
            loadings_zero = np.array(ob_loadings_lags_animals[index])[int2,:]
            loading_diff_ob_rand.append(loadings_zero-loadings_max)
            
            loadings_max = np.array(pc_loadings_lags_animals[index])[int1,:]
            loadings_zero = np.array(pc_loadings_lags_animals[index])[int2,:]
            loading_diff_pc_rand.append(loadings_zero-loadings_max)
            
        hist_rand_ob = np.histogram(np.squeeze(np.concatenate(loading_diff_ob_rand)), bins = np.linspace(-1,1,100))[0]
        hist_rand_pc = np.histogram(np.squeeze(np.concatenate(loading_diff_pc_rand)), bins = np.linspace(-1,1,100))[0]
        
            
        loading_diff_ob_rand_all.append(hist_rand_ob)
        loading_diff_pc_rand_all.append(hist_rand_pc)    
        
    
    
    
    #%%
    fig = plt.figure(dpi = 300, figsize = (6,4)) 
    
    s,p_ob = scipy.stats.ttest_1samp(np.squeeze(np.concatenate(loading_diff_ob)), 0)
    s,p_pc = scipy.stats.ttest_1samp(np.squeeze(np.concatenate(loading_diff_pc)), 0)
    
    plt.subplot(221)
    sns.histplot(np.squeeze(np.concatenate(loading_diff_ob)),stat = 'probability',label = '0 vs. 25 ms', kde = True,bins = 100, color = 'grey')
    #sns.kdeplot(np.squeeze(np.concatenate(loading_diff_ob_rand_all)),cumulative = True,color = 'red', label = 'p='+str(np.round(p_ob, decimals = 2)))
    
    plt.vlines(0,0,0.15, color = 'black', linestyles='dashed')
    plt.xlim([-1,1])
    #plt.ylim([0,40])
    plt.ylabel('Probability')
    #plt.xlabel('Corr Loading Difference (O vs. 25 ms)')
    plt.title('OB')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.legend(fontsize = 7, loc = 'upper left')
    
    
    plt.subplot(222)
    sns.histplot(np.squeeze(np.concatenate(loading_diff_pc)),stat = 'probability',label = '0 vs. 25 ms',kde = True,bins = 100, color = 'grey')
    #sns.kdeplot(np.squeeze(np.concatenate(loading_diff_pc_rand_all)),color = 'red', label = 'p='+str(np.round(p_ob, decimals = 2)))
    plt.vlines(0,0,0.25, color = 'black', linestyles='dashed')
    plt.xlim([-1,1])
    #plt.ylim([0,20])
    plt.ylabel(None)
    #plt.xlabel('Corr Loading Difference (O vs. 25 ms)',loc = 'right')
    plt.title('PCx')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    #plt.legend(fontsize = 8)
    
    fig.text(0.5, -0.02, 'Canonical Correlation Loading Difference', ha='center')
    
    
    plt.subplot(223)  
    plt.plot(np.linspace(-1,1,99),np.cumsum(hist_ob)/np.sum(hist_ob), label = '0 vs. 25 ms', color = 'black')
    plt.plot(np.linspace(-1,1,99),np.cumsum(np.mean(loading_diff_ob_rand_all,axis = 0))/np.sum(np.mean(loading_diff_ob_rand_all,axis = 0)),color = 'red',  label = 'Random')  
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.legend(fontsize = 7)
    plt.ylabel('Probability')
    plt.xlim([-1,1])
    plt.text(0.3,0.3,s = 'p<1e-17')
    
    plt.subplot(224)
    plt.plot(np.linspace(-1,1,99),np.cumsum(hist_pc)/np.sum(hist_pc), label = '0 vs. 25 ms', color = 'black')
    plt.plot(np.linspace(-1,1,99),np.cumsum(np.mean(loading_diff_pc_rand_all,axis = 0))/np.sum(np.mean(loading_diff_pc_rand_all,axis = 0)),color = 'red',  label = 'Random')  
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.xlim([-1,1])
    plt.text(0.3,0.3,s = 'p<1e-21')
    
    s,p_ob = stats.kstest(hist_ob,np.mean(loading_diff_ob_rand_all,axis = 0))
    s,p_pc = stats.kstest(hist_pc,np.mean(loading_diff_pc_rand_all,axis = 0))
    plt.tight_layout()

    #%%
    fig = plt.figure(dpi = 300, figsize = (5,5)) 
    
    max_corr_lag = np.argmax(r_lags_animals,axis = 1)
    
    index = 0
    plt.subplot(211)
    plt.stem(np.array(pc_loadings_lags_animals[index])[max_corr_lag[index],:])
    plt.title('Max Corr Lag')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    
    plt.subplot(212)
    plt.stem(np.array(pc_loadings_lags_animals[index])[10,:])
    plt.title('Zero Lag')
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    
    
    plt.tight_layout()
    
    #%%
    
    
    fig = plt.figure(dpi = 300, figsize = (3,7)) 
    gs = gridspec.GridSpec(3, 1,height_ratios = [2,4,2])

    animal = 12

    pc_loadings_lags = np.squeeze(np.concatenate(pc_loadings_lags_animals,axis = 1))
    ob_loadings_lags = np.squeeze(np.concatenate(ob_loadings_lags_animals,axis = 1))
    order_pc = np.argsort(np.sum(pc_loadings_lags, axis = 0))
    order_ob = np.argsort(np.sum(ob_loadings_lags, axis = 0))
    
    plt.subplot(gs[2])
    mean_r_lags = np.mean(r_lags_animals,axis = 0)
    error_r_lags = np.std(r_lags_animals,axis = 0)/np.sqrt(len(names))
    plt.plot(np.arange(-200,320,20)/2,mean_r_lags, color = 'black')
    plt.fill_between(np.arange(-200,320,20)/2,mean_r_lags-error_r_lags,mean_r_lags+error_r_lags, color = 'grey', alpha = 0.2)
    plt.xlim([-100,150])
    plt.xlabel('Corr Lag (ms, OB reference)')
    plt.vlines(0,0.15,0.4, color = 'black', linestyles='dashed')
    plt.ylim([0.24,0.4])
    plt.xticks(np.arange(-100,160,25), rotation = 45)
    plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    plt.ylabel('OB-PCx Can Corr')
    
    
    plt.subplot(gs[0])
    plt.imshow(ob_loadings_lags[:,order_ob].T, aspect = 'auto', cmap = 'bwr', vmin = -0.5, vmax = 0.5)
    plt.ylabel('OB Cells')
    plt.xlim([0,pc_loadings_lags.shape[0]-1])
    plt.xticks(np.arange(0,pc_loadings_lags.shape[0],2.5),labels = [])
    plt.vlines(10,0,np.concatenate(ob_loadings_lags_animals,axis = 1).shape[1]-1, color = 'black', linestyles='dashed')
    plt.colorbar(location = 'top', pad = 0.2, label = 'Can Corr Loading')
    
    plt.subplot(gs[1])
    plt.imshow(pc_loadings_lags[:,order_pc].T, aspect = 'auto', cmap = 'bwr', vmin = -0.5, vmax = 0.5)
    plt.ylabel('PCx Cells (sorted by loading)')
    plt.xlim([0,pc_loadings_lags.shape[0]-1])
    plt.xticks(np.arange(0,pc_loadings_lags.shape[0],2.5),labels = [])
    plt.vlines(10,0,np.concatenate(pc_loadings_lags_animals,axis = 1).shape[1]-1, color = 'black', linestyles='dashed')
    
    fig.align_ylabels()
    
    plt.tight_layout()
    
    
