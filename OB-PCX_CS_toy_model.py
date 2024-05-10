#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:43:27 2023

@author: pcanalisis2
"""

# toy model for testing communication subspace


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
from statsmodels.multivariate.cancorr import CanCorr

#%% activating selected groups 

communicating_neurons_X = np.arange(0,5)
non_communicating_neurons_X = np.arange(5,10)
communicating_neurons_Y = np.arange(5,10)
non_communicating_neurons_Y = np.arange(0,5)


#X = np.random.randn(100,10)
X = np.random.randint(0,30,size = [10000,10])
Y_correlated = np.random.randint(0,30,size = [10000,10])

# activation_times = [10,30,70]r
activation_times = np.random.randint(0,10000,size = 500)

activation_times_non_com = np.random.randint(0,10000,size = 500)
activation_times_non_com_y = np.random.randint(0,10000,size = 500)

# activation_times_non_com = [20,50,90]
# activation_times_non_com_y = [2,41,66]

for x in activation_times:
    X[x,communicating_neurons_X] = 50
    Y_correlated[x,communicating_neurons_Y] = 50
    
for x in activation_times_non_com:
    X[x,non_communicating_neurons_X] = 50
 
for x in activation_times_non_com_y:
    Y_correlated[x,non_communicating_neurons_Y] = 50    
#Y_uncorrelated = np.random.randn(100,10)

cca = CCA(n_components=1)
cca.fit(X, Y_correlated)

X_proj, Y_proj = cca.transform(X, Y_correlated)

params = cca.get_params()

X_loadings = cca.x_loadings_
Y_loadings = cca.y_loadings_

r,p = stats.pearsonr(np.squeeze(X_proj),np.squeeze(Y_proj))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,5))

gs = gridspec.GridSpec(3, 2)

plt.subplot(321)
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])


plt.subplot(323)
plt.ylabel('Y Neurons')
plt.imshow(Y_correlated.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])

plt.subplot(325)

plt.plot(np.mean(X,axis = 1)+30)
plt.ylabel('X MUA')
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,100])

#plt.plot(Y_proj)
plt.plot(np.mean(Y_correlated,axis = 1))
plt.ylabel('Y MUA')
plt.xticks(ticks = np.arange(0,101,20))
plt.xlabel('Time')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.tight_layout()
plt.xlim([0,100])

#plt.savefig('toy_example1.pdf')

plt.subplot(322)
plt.stem(X_loadings,orientation = 'horizontal')
#plt.title('Communication subspace loadings')
plt.ylabel('X Laodings')
#plt.ylim([10,-1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(-0,2), labels = [])
plt.xlim([-0.5,1])

plt.subplot(324)
plt.stem(Y_loadings,orientation = 'horizontal')
plt.ylabel('Y Loadings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(-0,2), labels = [])
plt.xlim([-0.5,1])
#plt.xlabel('Neuron #')


plt.subplot(326)
#plt.title('Communication subspace projection')
plt.plot(X_proj+10, label = 'X')
plt.ylabel('Can Corr Proj')

#plt.subplot(gs[3])
plt.plot(Y_proj, label = 'Y')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,120,20))
plt.legend()
plt.xlabel('Time')
plt.xlim([0,100])
#
plt.tight_layout()


#plt.savefig('toy_example1.pdf')

#%% activating X but not projecting


communicating_neurons_X = np.arange(0,5)
non_communicating_neurons_X = np.arange(5,10)
communicating_neurons_Y = np.arange(5,10)
non_communicating_neurons_Y = np.arange(0,5)


#X = np.random.randn(100,10)
X = np.random.randint(0,30,size = [10000,10])
#Y_correlated = np.random.randint(0,30,size = [100,10])

activation_times_non_com = np.random.randint(0,10000,size = 500)
activation_times_non_com_y = np.random.randint(0,10000,size = 500)

for x in activation_times:
    X[x,communicating_neurons_X] = 50
    
for x in activation_times_non_com:
    X[x,non_communicating_neurons_X] = 50
 
Y_uncorrelated = np.random.randint(0,30,size = [10000,10])
    
for x in activation_times_non_com_y:
    Y_uncorrelated[x,non_communicating_neurons_Y] = 50  

cca = CCA(n_components=1)
cca.fit(X, Y_uncorrelated)

X_proj, Y_proj = cca.transform(X, Y_uncorrelated)

params = cca.get_params()

X_loadings = cca.x_loadings_
Y_loadings = cca.y_loadings_


r,p = stats.pearsonr(np.squeeze(X_proj),np.squeeze(Y_proj))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,5))

gs = gridspec.GridSpec(3, 2)

plt.subplot(321)
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])


plt.subplot(323)
plt.ylabel('Y Neurons')
plt.imshow(Y_uncorrelated.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])

plt.subplot(325)

plt.plot(np.mean(X,axis = 1)+30)
plt.ylabel('X MUA')
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,100])

#plt.plot(Y_proj)
plt.plot(np.mean(Y_uncorrelated,axis = 1))
plt.ylabel('Y MUA')
plt.xticks(ticks = np.arange(0,101,20))
plt.xlabel('Time')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.tight_layout()
plt.xlim([0,100])

#plt.savefig('toy_example1.pdf')

plt.subplot(322)
plt.stem(X_loadings,orientation = 'horizontal')
#plt.title('Communication subspace loadings')
plt.ylabel('X Laodings')
#plt.ylim([10,-1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(-0,2), labels = [])
plt.xlim([-0.5,1])

plt.subplot(324)
plt.stem(Y_loadings,orientation = 'horizontal')
plt.ylabel('Y Loadings')
#plt.ylim([10,-1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(-0,2), labels = [])
plt.xlim([-0.5,1])
#plt.xlabel('Neuron #')


plt.subplot(326)
#plt.title('Communication subspace projection')
plt.plot(X_proj+10, label = 'X')
plt.ylabel('Can Corr Proj')

#plt.subplot(gs[3])
plt.plot(Y_proj, label = 'Y')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,120,20))
plt.legend()
plt.xlabel('Time')
plt.xlim([0,100])
#
plt.tight_layout()


#plt.savefig('toy_example2.pdf')

#%% three independent linear combinations example

num_components = 3

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(num_components,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

for x in range(10):
    
    X = np.random.randint(0,30,size = [2000,10])
    Y = np.random.randint(0,30,size = [2000,10])
    
    #% Make 3 neurons in Y depend on 5 neurons in X
    
    Y[:,0] = X[:,0:num_components].dot(v[:,0])
    Y[:,4] = X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,8] = X[:,num_components*2:num_components*3].dot(v[:,2])
    
    # Y[:,0] = X[:,0:num_components].dot(v[:,0])
    # Y[:,1] = X[:,0:num_components].dot(v[:,1])
    # Y[:,2] = X[:,0:num_components].dot(v[:,2])
    
    
    #cca = CCA(n_components=3,max_iter=100000,tol = 1e-12)
    cca = CCA(n_components=3)
    
    cca.fit(X, Y)
    
    X_proj, Y_proj = cca.transform(X, Y)
    
    params = cca.get_params()
    
    X_loadings = cca.x_loadings_
    Y_loadings = cca.y_loadings_
    
    # cancorr_stats = CanCorr(X,Y).corr_test().stats
    
    # canon_corrs = cancorr_stats['Canonical Correlation']
    # p_values = cancorr_stats['Pr > F']
    
    load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])
    
    f_load_x.append(X_loadings[:,load_order[0]])
    s_load_x.append(X_loadings[:,load_order[1]])
    t_load_x.append(X_loadings[:,load_order[2]])
    
    f_load_y.append(Y_loadings[:,load_order[0]])
    s_load_y.append(Y_loadings[:,load_order[1]])
    t_load_y.append(Y_loadings[:,load_order[2]])
    
    
X_loadings = np.vstack([np.mean(f_load_x,axis = 0),np.mean(s_load_x,axis = 0),np.mean(t_load_x,axis = 0)]).T
Y_loadings = np.vstack([np.mean(f_load_y,axis = 0),np.mean(s_load_y,axis = 0),np.mean(t_load_y,axis = 0)]).T
    
#%%

load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(figsize = (10,4))


plt.subplot(231)
plt.stem(X_loadings[:,load_order[0]])
plt.ylim([-1.1,1.1])
plt.ylabel('X Loadings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y1 = '+str(round_weigths[0,0])+'X1 +'+str(round_weigths[1,0])+'X2 +'+str(round_weigths[2,0])+'X3'
plt.title(title,fontsize = 10)


plt.subplot(232)
plt.stem(X_loadings[:,load_order[1]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y5 = '+str(round_weigths[0,1])+'X4 +'+str(round_weigths[1,1])+'X5 +'+str(round_weigths[2,1])+'X6'
plt.title(title,fontsize = 10)


plt.subplot(233)
plt.stem(X_loadings[:,load_order[2]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y9 = '+str(round_weigths[0,2])+'X7 +'+str(round_weigths[1,2])+'X8 +'+str(round_weigths[2,2])+'X9'
plt.title(title,fontsize = 10)

plt.subplot(234)
plt.stem(Y_loadings[:,load_order[0]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.ylabel('Y Loadings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')

plt.subplot(235)
plt.stem(Y_loadings[:,load_order[1]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
#plt.ylabel('Y Laodings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')

plt.subplot(236)
plt.stem(Y_loadings[:,load_order[2]])    
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])#
#plt.ylabel('Y Laodings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')    
    
#plt.savefig('Linear_combo_loads.pdf')

#%% plot linear combination example
    
    
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
    
plt.figure(dpi = 300, figsize = (4,3))

plt.subplot(211)
#plt.title('X spike trains')

plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xlim([0,100])
#plt.colorbar()
plt.ylabel('X Neurons')
plt.title('3 Independent Linear Combinations')

plt.subplot(212)
#plt.title('Y spike trains')
plt.imshow(Y.T,aspect = 'auto', interpolation = None)
plt.xlim([0,100])
#plt.colorbar()
plt.ylabel('Y Neurons')
plt.xlabel('Time')

plt.tight_layout()

plt.savefig('Linear_combo_example.pdf')

#%% compare to pca


from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)
X_loadings_pca = pca.components_

pca = PCA(n_components=1)
pca.fit(Y)
Y_loadings_pca = pca.components_

pca = PCA(n_components=1)
pca.fit(np.concatenate([X,Y],axis = 1))
XY_loadings_pca = pca.components_


fig = plt.figure(dpi = 300, figsize = (8,5))

gs = fig.add_gridspec(2,2)

plt.subplot(gs[0])
plt.stem(np.squeeze(X_loadings_pca), label = 'X')
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])
plt.legend()
plt.ylabel('PCA Loadings')

plt.subplot(gs[1])
plt.stem(np.squeeze(Y_loadings_pca), label = 'Y')
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])
plt.legend()

plt.subplot(gs[2:])
plt.stem(np.squeeze(XY_loadings_pca), label = 'Joint X-Y')
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,20), labels = np.arange(1,21))
plt.xlabel('Neuron #')
plt.legend()
plt.ylabel('PCA Loadings')

#%% vary number of independent combinations

num_components = 10

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(10,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

corr_indepenent = []
p_indepenent = []
    
num_independent_combos = 11
for x in range(num_independent_combos):
    
    X = np.random.randint(0,30,size = [50000,10])
    Y = np.random.randint(0,30,size = [50000,10])
    
    #% Make 3 neurons in Y depend on 5 neurons in X
    
    for y in range(x):
    
        Y[:,y] = X[:,0:num_components].dot(v[:,y])
    
    cancorr_stats = CanCorr(X,Y).corr_test().stats
    
    canon_corrs = cancorr_stats['Canonical Correlation']
    p_values = cancorr_stats['Pr > F']
    
    corr_indepenent.append(np.array(canon_corrs))
    p_indepenent.append(np.array(p_values))
    
#%%

#plt.plot(np.array(corr_indepenent))
import matplotlib.colors as colors

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,3))


plt.subplot(121)

corr_image = np.array(corr_indepenent).astype(float)
plt.imshow(corr_image,aspect = 'auto', cmap = 'Greys', norm=colors.LogNorm(vmin = 0.001, vmax = 1))
plt.ylabel('# Independent Combinations')
plt.xlabel('# Canonical Pairs')
plt.title('Pearson Correlation')
plt.yticks(ticks = np.arange(0,11))
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))

plt.colorbar()

# plt.plot(np.array(corr_indepenent).T)
# plt.ylabel('Pearson Correlation')
# plt.xlabel('Canonical Pairs')


plt.subplot(122)
p_image = np.array(p_indepenent).astype(float)
plt.imshow(p_image+1e-13,aspect = 'auto', cmap = 'Greys_r', norm=colors.LogNorm(vmin = 0.0005, vmax = 1))
plt.colorbar()
#plt.ylabel('# Independent Combinations')
plt.xlabel('# Canonical Pairs')
plt.title('Signigicance (p-value)')
plt.yticks(ticks = np.arange(0,11),labels = [])
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))

# plt.plot(np.array(p_indepenent).T)
# plt.ylabel('p-value')
# plt.xlabel('Canonical Pairs')

plt.tight_layout()

plt.savefig('Linear_combo_matrices.pdf')


#%% communication with lags 

communicating_neurons_X = np.arange(0,5)
communicating_neurons_Y = np.arange(5,10)

X = np.random.randint(0,30,size = [200,10])
Y_correlated = np.random.randint(0,30,size = [200,10])


activation_times = [10,30,70]

for x in activation_times:
    X[x,communicating_neurons_X] = 100
    Y_correlated[x+2,communicating_neurons_Y] = 100
    
#Y_uncorrelated = np.random.randn(100,10)

cca = CCA(n_components=1)
cca.fit(X, Y_correlated)

X_proj, Y_proj = cca.transform(X, Y_correlated)

params = cca.get_params()

X_loadings = cca.x_loadings_
Y_loadings = cca.y_loadings_

r,p = stats.pearsonr(np.squeeze(X_proj),np.squeeze(Y_proj))



import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,4))

gs = gridspec.GridSpec(3, 2,height_ratios = [1.5,1.5,1])

plt.subplot(gs[0])
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])
plt.title('CCA (Raw Spike Trains)')

plt.subplot(gs[2])
plt.ylabel('Y Neurons')
plt.imshow(Y_correlated.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])

plt.subplot(gs[4])

#plt.plot(np.mean(X,axis = 1))
plt.plot(X_proj, label = 'X')
plt.ylabel('CCA Proj')

plt.plot(Y_proj, label = 'Y')
plt.ylabel('CCA Proj')
plt.xticks(ticks = np.arange(0,101,20))
plt.xlabel('Time')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.tight_layout()
plt.xlim([0,100])
plt.ylim([-1,7])
plt.legend(fontsize = 8, ncol = 2)

#  convolution

# convolve spike trains

kernel = signal.gaussian(10,2)

conv_neurons_X = []
conv_neurons_Y = []

for x in range(X.shape[1]):
    conv = signal.convolve(np.squeeze(X[:,x]), kernel,mode = 'same') 
    conv_neurons_X.append(conv)
    conv = signal.convolve(np.squeeze(Y_correlated[:,x]), kernel,mode = 'same') 
    conv_neurons_Y.append(conv)

X = np.array(conv_neurons_X).T
Y_correlated = np.array(conv_neurons_Y).T    
    
cca = CCA(n_components=1)
cca.fit(X, Y_correlated)

X_proj, Y_proj = cca.transform(X, Y_correlated)

r_conv,p_conv = stats.pearsonr(np.squeeze(X_proj),np.squeeze(Y_proj))

params = cca.get_params()

X_loadings = cca.x_loadings_
Y_loadings = cca.y_loadings_


plt.subplot(gs[1])
#plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])
plt.title('CCA (Smoothed Spike Trains)')

plt.subplot(gs[3])
#plt.ylabel('Y Neurons')
plt.imshow(Y_correlated.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])

plt.subplot(gs[5])

#plt.plot(np.mean(X,axis = 1))
plt.plot(X_proj)
#plt.ylabel('CCA Proj')

#plt.plot(np.mean(Y_correlated,axis = 1))
plt.plot(Y_proj)
#plt.ylabel('CCA Proj')
plt.xticks(ticks = np.arange(0,101,20))
plt.xlabel('Time')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.tight_layout()
plt.xlim([0,100])
plt.ylim([-1,7])

#plt.savefig('lags.pdf')


#%%

communicating_neurons_X = np.arange(0,5)
communicating_neurons_Y = np.arange(5,10)

X = np.random.randint(0,30,size = [200,10])
Y_correlated = np.random.randint(0,30,size = [200,10])


activation_times = [10,30,70]

for x in activation_times:
    X[x,communicating_neurons_X] = 100
    Y_correlated[x+2,communicating_neurons_Y] = 100
    
#Y_uncorrelated = np.random.randn(100,10)

cca = CCA(n_components=1)
cca.fit(X, Y_correlated)

X_proj, Y_proj = cca.transform(X, Y_correlated)

params = cca.get_params()

X_loadings = cca.x_loadings_
Y_loadings = cca.y_loadings_

# # convolve spike trains

# kernel = signal.gaussian(10,2)

# conv_neurons_X = []
# conv_neurons_Y = []

# for x in range(X.shape[1]):
#     conv = signal.convolve(np.squeeze(X[:,x]), kernel,mode = 'same') 
#     conv_neurons_X.append(conv)
#     conv = signal.convolve(np.squeeze(Y_correlated[:,x]), kernel,mode = 'same') 
#     conv_neurons_Y.append(conv)

# X = np.array(conv_neurons_X).T
# Y_correlated = np.array(conv_neurons_Y).T    
    
# cca = CCA(n_components=1)
# cca.fit(X, Y_correlated)

# X_proj, Y_proj = cca.transform(X, Y_correlated)

# params = cca.get_params()

# X_loadings = cca.x_loadings_
# Y_loadings = cca.y_loadings_


lenght = X.shape[0]-20

canon_corrs_lags = []

for x in np.arange(-10,10):
    
    start_x = 0+10
    start_y = x+10
    end_x = start_x+lenght
    end_y = start_y+lenght
    
    cancorr_stats = CanCorr(X[start_x:end_x,:],Y_correlated[start_y:end_y,:]).corr_test().stats
    
    canon_corrs_lags.append(cancorr_stats['Canonical Correlation'][0])


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,4))

plt.plot(np.arange(-10,10),canon_corrs_lags,'-o', color = 'black')
plt.xticks(np.arange(-10,10))
plt.xlabel('Lags (a.u.)')
plt.ylabel('Canonical Correlation')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.vlines(0,0.3,1, linestyle = 'dashed', color = 'grey')
plt.ylim([0.3,1])
plt.xticks(np.arange(-10,10,2))

#plt.savefig('lags_corr.pdf')


#%% three independent linear combinations example

num_components = 3

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(num_components,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

times_odor1 = np.arange(0,200,10)
times_odor2 = np.arange(0,200,10)+4
times_odor3 = np.arange(0,200,10)+8

x_cs_activation_odor1 = []
y_cs_activation_odor1 = []

x_cs_activation_odor2 = []
y_cs_activation_odor2 = []

x_cs_activation_odor3 = []
y_cs_activation_odor3 = []

x_r1_simuls = []
x_r2_simuls = []
x_r3_simuls = []

x_p1_simuls = []
x_p2_simuls = []
x_p3_simuls = []

y_r1_simuls = []
y_r2_simuls = []
y_r3_simuls = []

y_p1_simuls = []
y_p2_simuls = []
y_p3_simuls = []

for x in range(100):
    
    print(x)
    
    X = np.random.randint(0,30,size = [200,10])
    Y = np.random.randint(0,30,size = [200,10])
    
    X[times_odor1,0:num_components] = 200
    X[times_odor2,num_components:num_components*2] = 200
    X[times_odor3,num_components*2:num_components*3] = 200
    
    
    #% Make 3 neurons in Y depend on 5 neurons in X
    
    Y[:,0] = X[:,0:num_components].dot(v[:,0])
    Y[:,1] = X[:,0:num_components].dot(v[:,0])
    Y[:,2] = X[:,0:num_components].dot(v[:,0])
    
    Y[:,3] = X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,4] = X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,5] = X[:,num_components:num_components*2].dot(v[:,1])
    
    Y[:,6] = X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,7] = X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,8] = X[:,num_components*2:num_components*3].dot(v[:,2])
    
    
    # Y[:,0] = X[:,0:num_components].dot(v[:,0])
    # Y[:,1] = X[:,0:num_components].dot(v[:,1])
    # Y[:,2] = X[:,0:num_components].dot(v[:,2])
    
    
    #
    
    cca = CCA(n_components=3,tol = 1e-12, max_iter=1000000)
    # cca = CCA(n_components=3)
    
    cca.fit(X, Y)
    
    X_proj, Y_proj = cca.transform(X, Y)
    
    params = cca.get_params()
    
    X_loadings = cca.x_weights_
    Y_loadings = cca.y_weights_
    
    # cancorr_stats = CanCorr(X,Y).corr_test().stats
    
    # canon_corrs = cancorr_stats['Canonical Correlation']
    # p_values = cancorr_stats['Pr > F']
    
    load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])
    
    f_load_x.append(X_loadings[:,load_order[0]])
    s_load_x.append(X_loadings[:,load_order[1]])
    t_load_x.append(X_loadings[:,load_order[2]])
    
    f_load_y.append(Y_loadings[:,load_order[0]])
    s_load_y.append(Y_loadings[:,load_order[1]])
    t_load_y.append(Y_loadings[:,load_order[2]])
    
    #load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:5])
    
    
    load_order_example = load_order
    
    X_proj = X_proj[:,load_order]
    Y_proj = Y_proj[:,load_order]
    
    
    x_cs_activation_odor1.append(np.mean(X_proj[times_odor1,:],axis = 0))
    y_cs_activation_odor1.append(np.mean(Y_proj[times_odor1,:],axis = 0))
    
    x_cs_activation_odor2.append(np.mean(X_proj[times_odor2,:],axis = 0))
    y_cs_activation_odor2.append(np.mean(Y_proj[times_odor2,:],axis = 0))
    
    x_cs_activation_odor3.append(np.mean(X_proj[times_odor3,:],axis = 0))
    y_cs_activation_odor3.append(np.mean(Y_proj[times_odor3,:],axis = 0))
    
    
    x_r1,x_p1 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor2,:],axis = 0))
    x_r2,x_p2 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))
    x_r3,x_p3 = stats.pearsonr(np.mean(X_proj[times_odor2,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))

    y_r1,y_p1 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor2,:],axis = 0))
    y_r2,y_p2 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))
    y_r3,y_p3 = stats.pearsonr(np.mean(Y_proj[times_odor2,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))

    x_r1_simuls.append(x_r1)
    x_r2_simuls.append(x_r2)
    x_r3_simuls.append(x_r3)
    
    x_p1_simuls.append(x_p1)
    x_p2_simuls.append(x_p2)
    x_p3_simuls.append(x_p3)
    
    y_r1_simuls.append(y_r1)
    y_r2_simuls.append(y_r2)
    y_r3_simuls.append(y_r3)
    
    y_p1_simuls.append(y_p1)
    y_p2_simuls.append(y_p2)
    y_p3_simuls.append(y_p3)
    
X_loadings = np.vstack([np.mean(f_load_x,axis = 0),np.mean(s_load_x,axis = 0),np.mean(t_load_x,axis = 0)]).T
Y_loadings = np.vstack([np.mean(f_load_y,axis = 0),np.mean(s_load_y,axis = 0),np.mean(t_load_y,axis = 0)]).T
    


#%%
load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(figsize = (10,4))


plt.subplot(231)
plt.stem(X_loadings[:,load_order[0]])
plt.ylim([-1.1,1.1])
plt.ylabel('X Loadings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y1 = '+str(round_weigths[0,0])+'X1 +'+str(round_weigths[1,0])+'X2 +'+str(round_weigths[2,0])+'X3'
plt.title(title,fontsize = 10)


plt.subplot(232)
plt.stem(X_loadings[:,load_order[1]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y5 = '+str(round_weigths[0,1])+'X4 +'+str(round_weigths[1,1])+'X5 +'+str(round_weigths[2,1])+'X6'
plt.title(title,fontsize = 10)


plt.subplot(233)
plt.stem(X_loadings[:,load_order[2]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = [])

round_weigths = np.round(v,decimals = 2)
title = 'Y9 = '+str(round_weigths[0,2])+'X7 +'+str(round_weigths[1,2])+'X8 +'+str(round_weigths[2,2])+'X9'
plt.title(title,fontsize = 10)

plt.subplot(234)
plt.stem(Y_loadings[:,load_order[0]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.ylabel('Y Loadings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')

plt.subplot(235)
plt.stem(Y_loadings[:,load_order[1]])
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])
#plt.ylabel('Y Laodings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')

plt.subplot(236)
plt.stem(Y_loadings[:,load_order[2]])    
plt.ylim([-1.1,1.1])
plt.ylim([-1.1,1.1])#
#plt.ylabel('Y Laodings')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('Neuron #')    
    
#plt.savefig('Linear_combo_loads_odor.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,3))


plt.subplot(121)
sns.histplot(np.concatenate([y_r1_simuls,y_r2_simuls,y_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'Y')
sns.histplot(np.concatenate([x_r1_simuls,x_r2_simuls,x_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'X',color = 'tab:orange')
plt.xlabel('CS Activity Correlation (acorss odors)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend()

plt.xlim([-1,1])

plt.subplot(122)

plt.boxplot([np.concatenate([y_p1_simuls,y_p2_simuls,y_p3_simuls]),np.concatenate([x_p1_simuls,x_p2_simuls,x_p3_simuls])], showfliers = False, widths = 0.3)

plt.scatter(np.ones(300),np.concatenate([y_p1_simuls,y_p2_simuls,y_p3_simuls]), color = 'tab:orange', s = 20)
plt.scatter(np.ones(300)+1,np.concatenate([x_p1_simuls,x_p2_simuls,x_p3_simuls]), color = 'tab:blue', s = 20)
plt.xticks(ticks = [1,2],labels = ['X','Y'])
plt.xlim([0.7,2.3])
plt.hlines(0.05,0.7,2.3, color = 'black')
plt.yscale('log')
#plt.ylim([0,0.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylabel('Average corr p-value')

plt.tight_layout()

plt.savefig('correlation_cs_odors_toy.pdf')
#%%

num_components = 3

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(num_components,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

times_odor1 = np.arange(0,200,10)
times_odor2 = np.arange(0,200,10)+4
times_odor3 = np.arange(0,200,10)+8

x_cs_activation_odor1 = []
y_cs_activation_odor1 = []

x_cs_activation_odor2 = []
y_cs_activation_odor2 = []

x_cs_activation_odor3 = []
y_cs_activation_odor3 = []


for x in range(1):
    
    print(x)
    
    X = np.random.randint(0,30,size = [200,10])
    Y = np.random.randint(0,30,size = [200,10])
    
    X[times_odor1,0:num_components] = 200
    X[times_odor2,num_components:num_components*2] = 200
    X[times_odor3,num_components*2:num_components*3] = 200
    
    
    #% Make 3 neurons in Y depend on 5 neurons in X
    
    Y[:,0] = X[:,0:num_components].dot(v[:,0])
    Y[:,1] = X[:,0:num_components].dot(v[:,0])
    Y[:,2] = X[:,0:num_components].dot(v[:,0])
    
    Y[:,3] = X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,4] = X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,5] = X[:,num_components:num_components*2].dot(v[:,1])
    
    Y[:,6] = X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,7] = X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,8] = X[:,num_components*2:num_components*3].dot(v[:,2])
    

    cca = CCA(n_components=3,tol = 1e-12, max_iter=1000000)
    # cca = CCA(n_components=3)
    
    cca.fit(X, Y)
    
    X_proj, Y_proj = cca.transform(X, Y)
    
    params = cca.get_params()
    
    X_loadings = cca.x_weights_
    Y_loadings = cca.y_weights_
    
    
    load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])
    
    f_load_x.append(X_loadings[:,load_order[0]])
    s_load_x.append(X_loadings[:,load_order[1]])
    t_load_x.append(X_loadings[:,load_order[2]])
    
    f_load_y.append(Y_loadings[:,load_order[0]])
    s_load_y.append(Y_loadings[:,load_order[1]])
    t_load_y.append(Y_loadings[:,load_order[2]])
    
    load_order_example = load_order
    
    X_proj = X_proj[:,load_order]
    Y_proj = Y_proj[:,load_order]
    
    
    x_cs_activation_odor1.append(np.mean(X_proj[times_odor1,:],axis = 0))
    y_cs_activation_odor1.append(np.mean(Y_proj[times_odor1,:],axis = 0))
    
    x_cs_activation_odor2.append(np.mean(X_proj[times_odor2,:],axis = 0))
    y_cs_activation_odor2.append(np.mean(Y_proj[times_odor2,:],axis = 0))
    
    x_cs_activation_odor3.append(np.mean(X_proj[times_odor3,:],axis = 0))
    y_cs_activation_odor3.append(np.mean(Y_proj[times_odor3,:],axis = 0))
    
    
X_loadings = np.vstack([np.mean(f_load_x,axis = 0),np.mean(s_load_x,axis = 0),np.mean(t_load_x,axis = 0)]).T
Y_loadings = np.vstack([np.mean(f_load_y,axis = 0),np.mean(s_load_y,axis = 0),np.mean(t_load_y,axis = 0)]).T
    

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,4))

plt.subplot(221)
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])


plt.subplot(223)
plt.ylabel('Y Neurons')
plt.imshow(Y.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,100])



plt.subplot(222)

plt.plot(X_proj[:,load_order[0]]+10)
plt.plot(X_proj[:,load_order[1]]+5)
plt.plot(X_proj[:,load_order[2]]+0)
plt.xlim([0,100])

plt.subplot(224)

plt.plot(Y_proj[:,load_order[0]]+16)
plt.plot(Y_proj[:,load_order[1]]+8)
plt.plot(Y_proj[:,load_order[2]]+0)
plt.xlim([0,100])

#plt.savefig('Linear_combo_odor_trace.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,4))

plt.subplot(321)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])

plt.subplot(323)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])

plt.subplot(325)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])

#


plt.subplot(322)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])

plt.subplot(324)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])

plt.subplot(326)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])


plt.tight_layout()

#plt.savefig('cs_odor_activation.pdf')


#%%

plt.figure(dpi = 300, figsize = (3,6))

ax2 = plt.subplot(211, projection = '3d')

    
ax2.scatter(X_proj[times_odor1,0],X_proj[times_odor1,2],X_proj[times_odor1,1], alpha = 1,s = 10)
ax2.scatter(X_proj[times_odor2,0],X_proj[times_odor2,2],X_proj[times_odor2,1], alpha = 1,s = 10)
ax2.scatter(X_proj[times_odor3,0],X_proj[times_odor3,2],X_proj[times_odor3,1], alpha = 1,s = 10)

ax2.view_init(elev= 15., azim = 40)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('X')

ax2 = plt.subplot(212, projection = '3d')

    
ax2.scatter(Y_proj[times_odor1,0],Y_proj[times_odor1,2],Y_proj[times_odor1,1], alpha = 1,s = 10)
ax2.scatter(Y_proj[times_odor2,0],Y_proj[times_odor2,2],Y_proj[times_odor2,1], alpha = 1,s = 10)
ax2.scatter(Y_proj[times_odor3,0],Y_proj[times_odor3,2],Y_proj[times_odor3,1], alpha = 1,s = 10)

ax2.view_init(elev= 15., azim = 40)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('Y')

#%% make uncorrelated example

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





num_components = 3

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(num_components,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

v = np.array([[1,1,1],[1,1,1],[1,1,1]])

times_odor1 = np.arange(0,200,10)
times_odor2 = np.arange(0,200,10)+4
times_odor3 = np.arange(0,200,10)+8

x_cs_activation_odor1 = []
y_cs_activation_odor1 = []

x_cs_activation_odor2 = []
y_cs_activation_odor2 = []

x_cs_activation_odor3 = []
y_cs_activation_odor3 = []

x_r1_simuls = []
x_r2_simuls = []
x_r3_simuls = []

x_p1_simuls = []
x_p2_simuls = []
x_p3_simuls = []

y_r1_simuls = []
y_r2_simuls = []
y_r3_simuls = []

y_p1_simuls = []
y_p2_simuls = []
y_p3_simuls = []


for x in range(100):
    
    print(x)
    
    X = np.random.randint(0,30,size = [200,10])
    Y = np.random.randint(0,30,size = [200,10])
    
    comb1 = [1,5,8]
    comb2 = [2,3,9]
    comb3 = [4,7,6]
    
    
    activation = 30
    X[times_odor1,0:num_components] = X[times_odor1,0:num_components]+np.ones(3)*activation+np.random.randn(3)*5
    X[times_odor2,num_components:num_components*2] = X[times_odor2,num_components:num_components*2]+np.ones(3)*activation+np.random.randn(3)*5
    X[times_odor3,num_components*2:num_components*3] = X[times_odor3,num_components*2:num_components*3]+np.ones(3)*activation+np.random.randn(3)*5
    
    #X = X[:,np.random.permutation(10)]
    
    Y[:,0] = Y[:,0]+X[:,0:num_components].dot(v[:,0])
    Y[:,1] = Y[:,1]+X[:,0:num_components].dot(v[:,0])
    Y[:,2] = Y[:,2]+X[:,0:num_components].dot(v[:,0])
    
    Y[:,3] = Y[:,3]+X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,4] = Y[:,4]+X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,5] = Y[:,5]+X[:,num_components:num_components*2].dot(v[:,1])
    
    Y[:,6] = Y[:,6]+X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,7] = Y[:,7]+X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,8] = Y[:,8]+X[:,num_components*2:num_components*3].dot(v[:,2])
    

    cca = CCA(n_components=3,tol = 1e-12, max_iter=1000000)
    # cca = CCA(n_components=3)
    
    cca.fit(X, Y)
    
    X_proj, Y_proj = cca.transform(X, Y)
    
    params = cca.get_params()
    
    X_loadings = cca.x_weights_
    Y_loadings = cca.y_weights_
    
    
    load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])
    
    f_load_x.append(X_loadings[:,load_order[0]])
    s_load_x.append(X_loadings[:,load_order[1]])
    t_load_x.append(X_loadings[:,load_order[2]])
    
    f_load_y.append(Y_loadings[:,load_order[0]])
    s_load_y.append(Y_loadings[:,load_order[1]])
    t_load_y.append(Y_loadings[:,load_order[2]])
    
    load_order_example = load_order
    
    X_proj = X_proj[:,load_order]
    Y_proj = Y_proj[:,load_order]
    
    
    x_cs_activation_odor1.append(np.mean(X_proj[times_odor1,:],axis = 0))
    y_cs_activation_odor1.append(np.mean(Y_proj[times_odor1,:],axis = 0))
    
    x_cs_activation_odor2.append(np.mean(X_proj[times_odor2,:],axis = 0))
    y_cs_activation_odor2.append(np.mean(Y_proj[times_odor2,:],axis = 0))
    
    x_cs_activation_odor3.append(np.mean(X_proj[times_odor3,:],axis = 0))
    y_cs_activation_odor3.append(np.mean(Y_proj[times_odor3,:],axis = 0))
    
    x_r1,x_p1 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor2,:],axis = 0))
    x_r2,x_p2 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))
    x_r3,x_p3 = stats.pearsonr(np.mean(X_proj[times_odor2,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))

    y_r1,y_p1 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor2,:],axis = 0))
    y_r2,y_p2 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))
    y_r3,y_p3 = stats.pearsonr(np.mean(Y_proj[times_odor2,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))

    x_r1_simuls.append(x_r1)
    x_r2_simuls.append(x_r2)
    x_r3_simuls.append(x_r3)
    
    x_p1_simuls.append(x_p1)
    x_p2_simuls.append(x_p2)
    x_p3_simuls.append(x_p3)
    
    y_r1_simuls.append(y_r1)
    y_r2_simuls.append(y_r2)
    y_r3_simuls.append(y_r3)
    
    y_p1_simuls.append(y_p1)
    y_p2_simuls.append(y_p2)
    y_p3_simuls.append(y_p3)
    
    
X_loadings = np.vstack([np.mean(f_load_x,axis = 0),np.mean(s_load_x,axis = 0),np.mean(t_load_x,axis = 0)]).T
Y_loadings = np.vstack([np.mean(f_load_y,axis = 0),np.mean(s_load_y,axis = 0),np.mean(t_load_y,axis = 0)]).T
    

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (8,3))


gs = gridspec.GridSpec(1, 2, width_ratios=(2,1))

plt.subplot(gs[0])

sns.histplot(np.concatenate([y_r1_simuls,y_r2_simuls,y_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'Y')
sns.histplot(np.concatenate([x_r1_simuls,x_r2_simuls,x_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'X',color = 'tab:orange')
plt.xlabel('CS Activity Correlation (acorss odors)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend()

plt.xlim([-1,1])

plt.tight_layout()

plt.savefig('corr_toy_model.pdf')
#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,2))

plt.subplot(231)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])
plt.xlim([0.5,3.5])

plt.subplot(232)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xlim([0.5,3.5])

plt.ylim([-2,5])

plt.subplot(233)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])
plt.xlim([0.5,3.5])
#


plt.subplot(234)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])


plt.subplot(235)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])
plt.subplot(236)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])

plt.tight_layout()

plt.savefig('odors_stem_order.pdf')
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,4))

plt.subplot(221)
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,70])


plt.subplot(223)
plt.ylabel('Y Neurons')
plt.imshow(Y.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,70])



plt.subplot(222)

plt.plot(X_proj[:,load_order[0]]+10)
plt.plot(X_proj[:,load_order[1]]+5)
plt.plot(X_proj[:,load_order[2]]+0)
plt.xlim([0,70])

plt.subplot(224)

plt.plot(Y_proj[:,load_order[0]]+16)
plt.plot(Y_proj[:,load_order[1]]+8)
plt.plot(Y_proj[:,load_order[2]]+0)
plt.xlim([0,70])

plt.savefig('Linear_combo_odor_trace.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,6))

ax2 = plt.subplot(211, projection = '3d')
n = 8
import cycler
color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

    
ax2.scatter(X_proj[times_odor1,0],X_proj[times_odor1,2],X_proj[times_odor1,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor1,0],X_proj[times_odor1,2],X_proj[times_odor1,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.scatter(X_proj[times_odor2,0],X_proj[times_odor2,2],X_proj[times_odor2,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor2,0],X_proj[times_odor2,2],X_proj[times_odor2,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.scatter(X_proj[times_odor3,0],X_proj[times_odor3,2],X_proj[times_odor3,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor3,0],X_proj[times_odor3,2],X_proj[times_odor3,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 40)
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_zlim([-4,4])

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('X')

ax2 = plt.subplot(212, projection = '3d')
n = 8
import cycler
color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))
    
ax2.scatter(Y_proj[times_odor1,0],Y_proj[times_odor1,2],Y_proj[times_odor1,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor1,0],Y_proj[times_odor1,2],Y_proj[times_odor1,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.scatter(Y_proj[times_odor2,0],Y_proj[times_odor2,2],Y_proj[times_odor2,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor2,0],Y_proj[times_odor2,2],Y_proj[times_odor2,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.scatter(Y_proj[times_odor3,0],Y_proj[times_odor3,2],Y_proj[times_odor3,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor3,0],Y_proj[times_odor3,2],Y_proj[times_odor3,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.view_init(elev= 15., azim = 40)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_zlim([-4,4])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('Y')

plt.savefig('clusters_odors.pdf')
#%% make uncorrelated example


num_components = 3

f_load_x = []
s_load_x = []
t_load_x = []

f_load_y = []
s_load_y = []
t_load_y = []

xarr = np.random.randn(num_components,1000)
R = np.corrcoef(xarr)
w,v = np.linalg.eig(R)

v = np.array([[1,1,1],[1,1,1],[1,1,1]])

times_odor1 = np.arange(0,200,10)
times_odor2 = np.arange(0,200,10)+4
times_odor3 = np.arange(0,200,10)+8

x_cs_activation_odor1 = []
y_cs_activation_odor1 = []

x_cs_activation_odor2 = []
y_cs_activation_odor2 = []

x_cs_activation_odor3 = []
y_cs_activation_odor3 = []


x_r1_simuls = []
x_r2_simuls = []
x_r3_simuls = []

x_p1_simuls = []
x_p2_simuls = []
x_p3_simuls = []

y_r1_simuls = []
y_r2_simuls = []
y_r3_simuls = []

y_p1_simuls = []
y_p2_simuls = []
y_p3_simuls = []

for x in range(100):
    
    print(x)
    
    X = np.random.randint(0,30,size = [200,10])
    Y = np.random.randint(0,30,size = [200,10])
    
    comb1 = [1,5,8]
    comb2 = [2,3,9]
    comb3 = [4,7,6]
    
    activation = 30
    X[times_odor1,0:num_components] = X[times_odor1,0:num_components]+np.ones(3)*activation+np.random.randn(3)*5
    X[times_odor2,num_components:num_components*2] = X[times_odor2,num_components:num_components*2]+np.ones(3)*activation+np.random.randn(3)*5
    X[times_odor3,num_components*2:num_components*3] = X[times_odor3,num_components*2:num_components*3]+np.ones(3)*activation+np.random.randn(3)*5
    
    X = X[:,np.random.permutation(10)]
    #% Make 3 neurons in Y depend on 5 neurons in X
    
    Y[:,0] = Y[:,0]+X[:,0:num_components].dot(v[:,0])
    Y[:,1] = Y[:,1]+X[:,0:num_components].dot(v[:,0])
    Y[:,2] = Y[:,2]+X[:,0:num_components].dot(v[:,0])
    
    Y[:,3] = Y[:,3]+X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,4] = Y[:,4]+X[:,num_components:num_components*2].dot(v[:,1])
    Y[:,5] = Y[:,5]+X[:,num_components:num_components*2].dot(v[:,1])
    
    Y[:,6] = Y[:,6]+X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,7] = Y[:,7]+X[:,num_components*2:num_components*3].dot(v[:,2])
    Y[:,8] = Y[:,8]+X[:,num_components*2:num_components*3].dot(v[:,2])
    

    cca = CCA(n_components=3,tol = 1e-12, max_iter=1000000)
    # cca = CCA(n_components=3)
    
    cca.fit(X, Y)
    
    X_proj, Y_proj = cca.transform(X, Y)
    
    params = cca.get_params()
    
    X_loadings = cca.x_weights_
    Y_loadings = cca.y_weights_
    
    
    load_order = np.argsort(np.argmax(np.abs(X_loadings),axis = 0)[0:3])
    
    f_load_x.append(X_loadings[:,load_order[0]])
    s_load_x.append(X_loadings[:,load_order[1]])
    t_load_x.append(X_loadings[:,load_order[2]])
    
    f_load_y.append(Y_loadings[:,load_order[0]])
    s_load_y.append(Y_loadings[:,load_order[1]])
    t_load_y.append(Y_loadings[:,load_order[2]])
    
    load_order_example = load_order
    
    X_proj = X_proj[:,load_order]
    Y_proj = Y_proj[:,load_order]
    
    
    x_cs_activation_odor1.append(np.mean(X_proj[times_odor1,:],axis = 0))
    y_cs_activation_odor1.append(np.mean(Y_proj[times_odor1,:],axis = 0))
    
    x_cs_activation_odor2.append(np.mean(X_proj[times_odor2,:],axis = 0))
    y_cs_activation_odor2.append(np.mean(Y_proj[times_odor2,:],axis = 0))
    
    x_cs_activation_odor3.append(np.mean(X_proj[times_odor3,:],axis = 0))
    y_cs_activation_odor3.append(np.mean(Y_proj[times_odor3,:],axis = 0))
    
    x_r1,x_p1 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor2,:],axis = 0))
    x_r2,x_p2 = stats.pearsonr(np.mean(X_proj[times_odor1,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))
    x_r3,x_p3 = stats.pearsonr(np.mean(X_proj[times_odor2,:],axis = 0),np.mean(X_proj[times_odor3,:],axis = 0))

    y_r1,y_p1 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor2,:],axis = 0))
    y_r2,y_p2 = stats.pearsonr(np.mean(Y_proj[times_odor1,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))
    y_r3,y_p3 = stats.pearsonr(np.mean(Y_proj[times_odor2,:],axis = 0),np.mean(Y_proj[times_odor3,:],axis = 0))

    x_r1_simuls.append(x_r1)
    x_r2_simuls.append(x_r2)
    x_r3_simuls.append(x_r3)
    
    x_p1_simuls.append(x_p1)
    x_p2_simuls.append(x_p2)
    x_p3_simuls.append(x_p3)
    
    y_r1_simuls.append(y_r1)
    y_r2_simuls.append(y_r2)
    y_r3_simuls.append(y_r3)
    
    y_p1_simuls.append(y_p1)
    y_p2_simuls.append(y_p2)
    y_p3_simuls.append(y_p3)
    
    
X_loadings = np.vstack([np.mean(f_load_x,axis = 0),np.mean(s_load_x,axis = 0),np.mean(t_load_x,axis = 0)]).T
Y_loadings = np.vstack([np.mean(f_load_y,axis = 0),np.mean(s_load_y,axis = 0),np.mean(t_load_y,axis = 0)]).T
    
#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,2))

plt.subplot(231)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-1,2])
plt.xlim([0.5,3.5])

plt.subplot(232)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xlim([0.5,3.5])

plt.ylim([-1,2])

plt.subplot(233)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-1,2])
plt.xlim([0.5,3.5])
#


plt.subplot(234)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


plt.ylim([-1,2])
plt.xlim([0.5,3.5])


plt.subplot(235)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-1,2])
plt.xlim([0.5,3.5])
plt.subplot(236)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


plt.ylim([-1,2])
plt.xlim([0.5,3.5])

plt.tight_layout()

#plt.savefig('odors_stem_mixed.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,3))


gs = gridspec.GridSpec(1, 2, width_ratios=(2,1))

plt.subplot(gs[0])

sns.histplot(np.concatenate([y_r1_simuls,y_r2_simuls,y_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'Y')
sns.histplot(np.concatenate([x_r1_simuls,x_r2_simuls,x_r3_simuls]),bins = 100, element="step", cumulative = True, label = 'X',color = 'tab:orange')
plt.xlabel('CS Activity Correlation (acorss odors)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.legend()

plt.xlim([-1,1])

plt.tight_layout()

plt.savefig('corr_toy_model_disorder.pdf')
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (2,4))

plt.subplot(321)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])
plt.xlim([0.5,3.5])

plt.subplot(323)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xlim([0.5,3.5])

plt.ylim([-2,5])

plt.subplot(325)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,5])
plt.xlim([0.5,3.5])
#


plt.subplot(322)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])
plt.subplot(324)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])
plt.subplot(326)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-2,6])
plt.xlim([0.5,3.5])

plt.tight_layout()


#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,4))

plt.subplot(221)
plt.ylabel('X Neurons')
plt.imshow(X.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,70])


plt.subplot(223)
plt.ylabel('Y Neurons')
plt.imshow(Y.T,aspect = 'auto', interpolation = None)
plt.xticks(ticks = np.arange(0,101,20), labels = [])
plt.xlim([0,70])



plt.subplot(222)

plt.plot(X_proj[:,load_order[0]]+10)
plt.plot(X_proj[:,load_order[1]]+5)
plt.plot(X_proj[:,load_order[2]]+0)
plt.xlim([0,70])

plt.subplot(224)

plt.plot(Y_proj[:,load_order[0]]+16)
plt.plot(Y_proj[:,load_order[1]]+8)
plt.plot(Y_proj[:,load_order[2]]+0)
plt.xlim([0,70])

plt.savefig('Linear_combo_odor_trace_mixed.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,6))


ax2 = plt.subplot(211, projection = '3d')
n = 8
import cycler
color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

    
ax2.scatter(X_proj[times_odor1,0],X_proj[times_odor1,2],X_proj[times_odor1,1], alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor1,0],X_proj[times_odor1,2],X_proj[times_odor1,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.scatter(X_proj[times_odor2,0],X_proj[times_odor2,2],X_proj[times_odor2,1], alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor2,0],X_proj[times_odor2,2],X_proj[times_odor2,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.scatter(X_proj[times_odor3,0],X_proj[times_odor3,2],X_proj[times_odor3,1], alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(X_proj[times_odor3,0],X_proj[times_odor3,2],X_proj[times_odor3,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.view_init(elev= 15., azim = 40)
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_zlim([-4,4])

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('X')

ax2 = plt.subplot(212, projection = '3d')
n = 8
import cycler
color1 = plt.cm.Dark2(np.linspace(0, 1,n))
plt.gca().set_prop_cycle(cycler.cycler('color', color1[:,:]))

    
ax2.scatter(Y_proj[times_odor1,0],Y_proj[times_odor1,2],Y_proj[times_odor1,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor1,0],Y_proj[times_odor1,2],Y_proj[times_odor1,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.scatter(Y_proj[times_odor2,0],Y_proj[times_odor2,2],Y_proj[times_odor2,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor2,0],Y_proj[times_odor2,2],Y_proj[times_odor2,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)


ax2.scatter(Y_proj[times_odor3,0],Y_proj[times_odor3,2],Y_proj[times_odor3,1],rasterized = True, alpha = 1,s = 10)
x_ellipse, y_ellipse, z_ellipse = plot_3d_confidence_ellipse(Y_proj[times_odor3,0],Y_proj[times_odor3,2],Y_proj[times_odor3,1], num_std_dev=1.95)
ax2.plot_surface(x_ellipse, y_ellipse, z_ellipse, alpha=0.1, facecolor = None, shade = False)

ax2.view_init(elev= 15., azim = 40)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_zlim([-4,4])

ax2.set_xlabel('CS 1')
ax2.set_ylabel('CS 5')
ax2.set_zlabel('CS 3')
ax2.set_title('Y')

plt.savefig('odors_cluster_mixed.pdf')
#%%


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,4))

plt.subplot(321)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

plt.subplot(323)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

plt.subplot(325)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

#


plt.subplot(322)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])

plt.subplot(324)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])

plt.subplot(326)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])


plt.tight_layout()

#plt.savefig('cs_odor_activation.pdf')


#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,4))

plt.subplot(321)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

plt.subplot(323)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

plt.subplot(325)

plt.plot(np.arange(1,4),np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(x_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-5,5])

#


plt.subplot(322)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor1,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])

plt.subplot(324)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor2,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])

plt.subplot(326)

plt.plot(np.arange(1,4),np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.vlines(np.arange(1,4), np.mean(y_cs_activation_odor3,axis = 0),'o',color = 'black')
plt.hlines(0,1,3, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.ylim([-6,6])


plt.tight_layout()

#plt.savefig('cs_odor_activation.pdf')

