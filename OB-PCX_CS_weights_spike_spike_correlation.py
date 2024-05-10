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


def event_correlation(data1,data2,window,num_bins):
    
    # two sets of units, data1 works as trigger for data2 spikes
    
    event_hist = []
    
    for unit_1 in range(data1.shape[0]-1): # loop over neurons in data1
    
        event_hist_1 = []
        for unit_2 in range(data2.shape[0]-1): # loop over neurons in data2
            
            # trig spikes2 by spikes1
            trig_spike_times = []
            
            for x in np.where(data1[unit_1+1,:] == 1)[0]:
        
                if data2[unit_2+1,int(x-window):int(x+window)].shape[0] == window*2:
                    trig_spike_times.append(np.where(data2[unit_2+1,int(x-window):int(x+window)]==1)[0])
        
            trig_spike_times = np.concatenate(trig_spike_times,axis = 0)
            event_hist_1.append(np.histogram(trig_spike_times,bins = num_bins,range = [0,1000])[0])
        
        event_hist.append(event_hist_1)
    
    bins = np.histogram(trig_spike_times,bins = num_bins,range = [0,1000])[1][0:-1]
    
    return(event_hist,bins)

#%% general information 

directory = '/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/'

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

event_hist_ob_pc_animals = []
bins_animals = []
ob_loadings_animals = []
pc_loadings_animals = []
    
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

    params = cca.get_params()
    
    ob_loadings = cca.x_rotations_
    pc_loadings = cca.y_rotations_

    
    del conv_neurons_ob,conv_neurons_pc, X, Y, resp, lfp
    

    
    #  check event correlation
    
    window = 500
    num_bins = 100
   
    event_hist_ob_pc, bins = event_correlation(units_ob, units_pc, window, num_bins)
    
    ob_unit = np.argmax(ob_loadings)
    
    event_hist = event_hist_ob_pc[ob_unit]
    
    
    plt.figure(dpi = 300, figsize = (16,8))
    
    loadings_sorted = np.argsort(np.squeeze(pc_loadings))[::-1]
    
    for x in range(6):
        
        event_hist_plot = event_hist[loadings_sorted[x]]
        
        plt.subplot(2,6,x+1)
        plt.bar(bins,event_hist_plot/np.sum(event_hist_plot),width = window*2/num_bins, color = 'black')
        plt.vlines(500,0,0.018, linewidth = 1, color = 'black', linestyles = 'dashed')
        plt.ylim([0.005,0.018])
        plt.yticks([])
        plt.xticks(ticks = [0,500,1000], labels = [-250,0,250])
        #plt.xlim([400,600])

    
    loadings_sorted = np.argsort(np.squeeze(pc_loadings))
    
    for x in range(6):
        
        event_hist_plot = event_hist[loadings_sorted[x]]
        
        plt.subplot(2,6,x+7)
        plt.bar(bins,event_hist_plot/np.sum(event_hist_plot),width = window*2/num_bins, color = 'black')
        plt.vlines(500,0,0.018, linewidth = 1, color = 'black', linestyles = 'dashed')
        plt.ylim([0.005,0.018])
        plt.yticks([])
        plt.xticks(ticks = [0,500,1000], labels = [-250,0,250])
  

    event_hist_ob_pc_animals.append(event_hist_ob_pc)
    bins_animals.append(bins)
    ob_loadings_animals.append(ob_loadings)
    pc_loadings_animals.append(pc_loadings)
    
    
#%% save results

np.savez('spike_event_correlation.npz', event_hist_ob_pc_animals = event_hist_ob_pc_animals, bins_animals = bins_animals, ob_loadings_animals = ob_loadings_animals, pc_loadings_animals = pc_loadings_animals)
    
    
#%%

os.chdir('/home/pcanalisis2/Desktop/figures_cca')
spike_correlation = np.load('spike_event_correlation.npz', allow_pickle = True)    
event_hist_ob_pc_animals = spike_correlation['event_hist_ob_pc_animals']
bins = spike_correlation['bins_animals']
ob_loadings_animals = spike_correlation['ob_loadings_animals']
pc_loadings_animals = spike_correlation['pc_loadings_animals']



#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

window = 500
num_bins = 100

animal = 3
event_hist_ob_pc = event_hist_ob_pc_animals[animal]
bins = np.arange(0,1000,10)
ob_loadings = ob_loadings_animals[animal]
pc_loadings = pc_loadings_animals[animal]

ob_unit = np.argmax(ob_loadings)

event_hist = event_hist_ob_pc[ob_unit]

#
plt.figure(dpi = 300, figsize = (15,6))

loadings_sorted = np.argsort(np.squeeze(pc_loadings))[::-1]

num_neurons = 10

for x in range(num_neurons):
    
    event_hist_plot = event_hist[loadings_sorted[x]]
    
    plt.subplot(2,num_neurons,x+1)
    plt.bar(bins,stats.zscore(event_hist_plot)+2.5,width = window*2/num_bins, color = 'black')
    plt.vlines(500,0,5.5, linewidth = 1, color = 'black', linestyles = 'dashed')
    #plt.ylim([0.0,0.025])
    plt.yticks(np.arange(0,5.5,2.5), labels = [])
    plt.xticks(ticks = [0,500,1000], labels = [])
    plt.xlim([0,1000])
    plt.ylim([0,5.5])

    if x == 0:
        plt.yticks(np.arange(0,5.5,2.5), labels = np.arange(0,5.5,2.5)-2.5)
        plt.ylabel('Correlation')
    #plt.xlim([400,600])


loadings_sorted = np.argsort(np.squeeze(pc_loadings))

for x in range(num_neurons):
    
    event_hist_plot = event_hist[loadings_sorted[x]]
    
    plt.subplot(2,num_neurons,x+num_neurons+1)
    plt.bar(bins,stats.zscore(event_hist_plot)+2.5,width = window*2/num_bins, color = 'black')
    plt.vlines(500,0,5.5, linewidth = 1, color = 'black', linestyles = 'dashed')
    #plt.ylim([0.0,0.025])
    plt.yticks(np.arange(0,5.5,2.5), labels = [])
    plt.xticks(ticks = [0,500,1000], labels = [])
    plt.xlim([0,1000])
    plt.ylim([0,5.5])

    if x == 0:
        plt.yticks(np.arange(0,5.5,2.5), labels = np.arange(0,5.5,2.5)-2.5)
        plt.ylabel('Correlation')
    

#plt.savefig('spike_correaltion.pdf')    

#%%

peak_top = []
peak_bottom = []

activity_top_animals = []
activity_bottom_animals = []
norm_hists_top_animals = []
norm_hists_bottom_animals = []

peak_activity_top = []
peak_activity_bottom = []


for x in range(len(event_hist_ob_pc_animals)):
    
    event_hist_ob_pc = event_hist_ob_pc_animals[x]
    
    ob_loadings = ob_loadings_animals[x]
    pc_loadings = pc_loadings_animals[x]
    
    ob_loadings_sorted = np.argsort(np.squeeze(ob_loadings))
    pc_loadings_sorted = np.argsort(np.squeeze(pc_loadings))
    
    activity_top = []
    activity_bottom = []
    norm_hists_top = []
    norm_hists_bottom = []
    
    
    # top ob with top pc
    
    units_n = 10
    for y in range(units_n):
    
        ob_unit = ob_loadings_sorted[-1*y]
        hists = np.array(event_hist_ob_pc[ob_unit])[pc_loadings_sorted[-units_n:]]
        peak_top.append(bins[np.argmax(hists,axis = 1)]/2-250)
        norm_hists = hists/np.sum(hists,axis = 1)[:,np.newaxis]
        norm_hists = stats.zscore(hists,axis = 1)
        activity_top.append(norm_hists[:,55])
        norm_hists_top.append(np.mean(norm_hists,axis = 0))
        
        
        ob_unit = ob_loadings_sorted[-1*y]
        hists = np.array(event_hist_ob_pc[ob_unit])[pc_loadings_sorted[:units_n]]
        peak_bottom.append(bins[np.argmin(hists,axis = 1)]/2-250)
        norm_hists = hists/np.sum(hists,axis = 1)[:,np.newaxis]
        norm_hists = stats.zscore(hists,axis = 1)
        activity_bottom.append(norm_hists[:,55])
        norm_hists_bottom.append(np.mean(norm_hists,axis = 0))
    
    peak_activity_top.append(np.argmax(norm_hists_top,axis = 1))
    peak_activity_bottom.append(np.argmin(norm_hists_bottom,axis = 1))
    
    activity_top_animals.append(np.mean(activity_top))
    activity_bottom_animals.append(np.mean(activity_bottom))
    norm_hists_top_animals.append(np.mean(norm_hists_top,axis = 0))
    norm_hists_bottom_animals.append(np.mean(norm_hists_bottom,axis = 0))
    
    
#%% use percentiles

bins = np.arange(0,1000,10)

peak_top = []
peak_bottom = []

activity_top_animals = []
activity_bottom_animals = []
norm_hists_top_animals = []
norm_hists_bottom_animals = []

peak_activity_top = []
peak_activity_bottom = []

peak_bottom_animals = []
peak_top_animals = []

for x in range(len(event_hist_ob_pc_animals)):
    
    event_hist_ob_pc = event_hist_ob_pc_animals[x]
    
    ob_loadings = ob_loadings_animals[x]
    pc_loadings = pc_loadings_animals[x]
    
    ob_loadings_sorted = np.argsort(np.squeeze(ob_loadings))
    pc_loadings_sorted = np.argsort(np.squeeze(pc_loadings))
    
    pc_neg_weight = np.where(pc_loadings<np.quantile(pc_loadings,0.2))[0]
    pc_pos_weight = np.where(pc_loadings>np.quantile(pc_loadings,0.8))[0]
    
    ob_pos_weight = np.where(ob_loadings>np.quantile(ob_loadings,0.8))[0]
    
    activity_top = []
    activity_bottom = []
    norm_hists_top = []
    norm_hists_bottom = []
    
    
    
    # top ob with top pc
    
    peak_bottom_session = []
    peak_top_session = []
    
    units_n = 10
    for y in range(ob_pos_weight.shape[0]):
    
        #ob_unit = ob_loadings_sorted[-1*y]
        ob_unit = ob_pos_weight[y]
        hists = np.array(event_hist_ob_pc[ob_unit])[pc_pos_weight]
        peak_top.append(bins[np.argmax(hists,axis = 1)]/2-250)
        
        #norm_hists = hists/np.sum(hists,axis = 1)[:,np.newaxis]
        norm_hists = stats.zscore(hists[:,:],axis = 1)
        activity_top.append(norm_hists[:,55])
        norm_hists_top.append(np.mean(norm_hists,axis = 0))
        peak_top_session.append(bins[np.argmax(norm_hists,axis = 1)]/2-250)
        
        ob_unit = ob_loadings_sorted[-1*y]
        hists = np.array(event_hist_ob_pc[ob_unit])[pc_neg_weight]
        peak_bottom.append(bins[np.argmin(hists,axis = 1)]/2-250)
        
        hists_non_zero = np.sum(hists,axis = 1) > 0
        #norm_hists = hists[hists_non_zero,:]/np.sum(hists[hists_non_zero,:],axis = 1)[:,np.newaxis]
        norm_hists = stats.zscore(hists[:,:],axis = 1)
        activity_bottom.append(norm_hists[:,55])
        norm_hists_bottom.append(np.mean(norm_hists,axis = 0))
        peak_bottom_session.append(bins[np.argmin(norm_hists,axis = 1)]/2-250)
    
    peak_activity_top.append(np.argmax(norm_hists_top,axis = 1))
    peak_activity_bottom.append(np.argmin(norm_hists_bottom,axis = 1))
    
    activity_top_animals.append(np.nanmean(activity_top))
    activity_bottom_animals.append(np.nanmean(activity_bottom))
    norm_hists_top_animals.append(np.nanmean(norm_hists_top,axis = 0))
    norm_hists_bottom_animals.append(np.nanmean(norm_hists_bottom,axis = 0))
    
    peak_bottom_animals.append(np.mean(peak_bottom_session))
    peak_top_animals.append(np.mean(peak_top_session))
    

        

#%%


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (14,4))

gs = gridspec.GridSpec(1, 3, width_ratios = [3,1,1])

plt.subplot(gs[0])

mean_hists_top = np.nanmean(norm_hists_top_animals,axis = 0)
mean_hists_bottom = np.nanmean(norm_hists_bottom_animals,axis = 0)
error_hists_top = stats.sem(norm_hists_top_animals,axis = 0)
error_hists_bottom = stats.sem(norm_hists_bottom_animals,axis = 0)


plt.plot((bins/2)-250,mean_hists_top, color = 'c', label = '+ CCA Weight')
plt.fill_between((bins/2)-250, mean_hists_top-error_hists_top, mean_hists_top+error_hists_top,alpha = 0.2, color = 'c')

plt.plot((bins/2)-250,mean_hists_bottom, color = 'm', label = '- CCA Weight')
plt.fill_between((bins/2)-250, mean_hists_bottom-error_hists_bottom, mean_hists_bottom+error_hists_bottom,alpha = 0.2, color = 'm')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
#plt.vlines(0,0.008,0.013, linestyles = 'dashed', color = 'black')

plt.vlines(0,0.-0.75,1, linestyles = 'dashed', color = 'black', alpha = 0.3)

#plt.ylim([0.0085,0.0125])
plt.xlim([-250,250])
plt.ylabel('OB-PCx Spike Correlation (z-scored)')
plt.legend()
plt.xticks(ticks = np.arange(-200,220,50))

top_peak = bins[np.argmax(norm_hists_top_animals,axis = 1)]/2-250
bottom_peak = bins[np.argmin(norm_hists_bottom_animals,axis = 1)]/2-250

plt.vlines(np.mean(top_peak),0,0.0125, color = 'c')
plt.vlines(np.mean(bottom_peak),0,0.0125, color = 'm')

plt.xlabel('Time from OB spike (ms)')
    # top ob with bottom
    
plt.subplot(gs[1])
plt.boxplot([activity_bottom_animals,activity_top_animals],widths = 0.2, showfliers=False)

for x in range(len(activity_bottom_animals)):
    plt.plot([1.2,1.8],[activity_bottom_animals[x],activity_top_animals[x]], color = 'grey')
    
plt.ylabel('Correlation at 25ms (z-scored)')
plt.xticks(ticks = [1,2], labels = ['+ CCA Weight','- CCA Weight'])

plt.xlim([0.8,2.2])
#plt.ylim([0,20])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

t,p = stats.ttest_rel(activity_bottom_animals,activity_top_animals)
plt.tight_layout()

#plt.savefig('spike_event_stats.pdf')


plt.subplot(gs[2])
top_peak = bins[np.argmax(np.array(norm_hists_top_animals)[:,50:],axis = 1)]/2
bottom_peak = bins[np.argmin(np.array(norm_hists_bottom_animals)[:,50:],axis = 1)]/2

plt.boxplot([top_peak,bottom_peak],widths = 0.2, showfliers=False)

for x in range(len(bottom_peak)):
    plt.plot([1.2,1.8],[top_peak[x],bottom_peak[x]], color = 'grey')


t,p = stats.ttest_rel(top_peak,bottom_peak)

plt.xticks(ticks = [1,2], labels = ['+ CCA Weight','- CCA Weight'])
plt.xlim([0.8,2.2])
plt.ylim([-1,42])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylabel('Peak Excitation/Inhibition (ms)')

plt.tight_layout()

#plt.savefig('spike_correlation_metric.pdf')

#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,6))

gs = gridspec.GridSpec(3, 2, width_ratios = [3,1], height_ratios = [2,1,1])

plt.subplot(gs[0])

mean_hists_top = np.nanmean(norm_hists_top_animals,axis = 0)
mean_hists_bottom = np.nanmean(norm_hists_bottom_animals,axis = 0)
error_hists_top = stats.sem(norm_hists_top_animals,axis = 0)
error_hists_bottom = stats.sem(norm_hists_bottom_animals,axis = 0,nan_policy = 'omit')


plt.plot((bins/2)-250,mean_hists_top, color = 'c', label = '+ CCA Weight')
plt.fill_between((bins/2)-250, mean_hists_top-error_hists_top, mean_hists_top+error_hists_top,alpha = 0.2, color = 'c')

plt.plot((bins/2)-250,mean_hists_bottom, color = 'm', label = '- CCA Weight')
plt.fill_between((bins/2)-250, mean_hists_bottom-error_hists_bottom, mean_hists_bottom+error_hists_bottom,alpha = 0.2, color = 'm')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
#plt.vlines(0,0.008,0.013, linestyles = 'dashed', color = 'black')

# plt.vlines(25,0.-0.75,1, linestyles = 'dashed', color = 'black', alpha = 0.3)

#plt.ylim([0.0085,0.0125])
plt.xlim([-250,250])
plt.xticks(np.arange(-200,210,50))
plt.ylabel('OB-PCx Spike Correlation (z-scored)')
plt.legend()
plt.xticks(ticks = np.arange(-200,220,100), labels = [])

top_peak = bins[np.argmax(norm_hists_top_animals,axis = 1)]/2-250
bottom_peak = bins[np.argmin(norm_hists_bottom_animals,axis = 1)]/2-250

    # top ob with bottom
    
plt.subplot(gs[1])
plt.boxplot([activity_bottom_animals,activity_top_animals],widths = 0.2, showfliers=False)

for x in range(len(activity_bottom_animals)):
    plt.plot([1.2,1.8],[activity_bottom_animals[x],activity_top_animals[x]], color = 'grey')
    
plt.ylabel('Correlation at 25ms (z-scored)')
plt.xticks(ticks = [1,2], labels = []
           )
plt.xlim([0.8,2.2])
#plt.ylim([0,20])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

t,p = stats.ttest_rel(activity_bottom_animals,activity_top_animals)
plt.tight_layout()

#plt.savefig('spike_event_stats.pdf')


plt.subplot(gs[2])
sns.histplot(np.concatenate(peak_top), kde = True,color = 'c', bins = np.linspace(-250,250,50),  label = '+ CCA Weight')
#sns.histplot(np.concatenate(peak_bottom),kde = True, color = 'm', bins = 100, label = '- CCA Weight')
plt.xlim([-250,250])
plt.ylim([0,190])
#plt.xlabel('Time from OB spike (ms)')
plt.xticks(ticks = np.arange(-200,220,100), labels = [])
#plt.title('Peak activity histogram')

plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

t,p = stats.kstest(np.concatenate(peak_top),np.concatenate(peak_bottom))

plt.subplot(gs[4])
#sns.histplot(np.concatenate(peak_top), kde = True,color = 'c', bins = 100, label = '+ CCA Weight')
sns.histplot(np.concatenate(peak_bottom),kde = True, color = 'm', bins = np.linspace(-250,250,50), label = '- CCA Weight')
plt.xlim([-250,250])
plt.ylim([0,190])
plt.xlabel('Time from OB spike (ms)')
#plt.title('Peak activity histogram')
plt.xticks(ticks = np.arange(-200,220,100))
plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


plt.subplot(gs[1:,1])
top_peak = bins[np.argmax(np.array(norm_hists_top_animals)[:,50:60],axis = 1)]/2
bottom_peak = bins[np.argmin(np.array(norm_hists_bottom_animals)[:,50:60],axis = 1)]/2

plt.boxplot([top_peak,bottom_peak],widths = 0.2, showfliers=False)

for x in range(len(bottom_peak)):
    plt.plot([1.2,1.8],[top_peak[x],bottom_peak[x]], color = 'grey')


t,p = stats.ttest_rel(top_peak,bottom_peak)

plt.xticks(ticks = [1,2], labels = ['+ CCA Weight','- CCA Weight'])
plt.xlim([0.8,2.2])
#plt.ylim([-1,42])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylabel('Peak Excitation/Inhibition (ms)')

plt.tight_layout()
#plt.savefig('time_comp_peak.pdf')

#%%

mean_hists_top = np.mean(norm_hists_top_animals,axis = 0)
mean_hists_bottom = np.mean(norm_hists_bottom_animals,axis = 0)*-1 
error_hists_top = stats.sem(norm_hists_top_animals,axis = 0)
error_hists_bottom = stats.sem(norm_hists_bottom_animals,axis = 0)*-1 


plt.plot((bins/2)-250,mean_hists_top)
plt.fill_between((bins/2)-250, mean_hists_top-error_hists_top, mean_hists_top+error_hists_top,alpha = 0.2)

plt.plot((bins/2)-250,mean_hists_bottom)
plt.fill_between((bins/2)-250, mean_hists_bottom-error_hists_bottom, mean_hists_bottom+error_hists_bottom,alpha = 0.2)

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

#%%



    
