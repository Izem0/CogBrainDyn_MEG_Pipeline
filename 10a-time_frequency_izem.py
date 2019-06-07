#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:31:08 2018

@author: sh252228, ie258305
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os.path as op

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, read_tfrs , write_tfrs, AverageTFR
from mne.baseline import rescale
from mne.stats import _bootstrap_ci

from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from mne.channels import read_ch_connectivity

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from functools import partial
#%% import from config file
import config


## get nips
subjects_list = ['hm_070076','cc_150418','sf_180213','eb_180237','og_170145',
                'ga_180461','fr_190151','tr_180110','ld_190260','cg_190026',
                'ml_180318','ml_180195','ag_170045','hf_190144','lq_180242']

#subjects_list = ['eb_180237']


nips = subjects_list # ['hm_070076','cc_150418']
print(nips)


#get parameters from config file

#conditions = ['3_interval','1_intervals',
#                'long_dur','med_dur','short_dur']

conditions = ['3_intervals','1_interval']

#conditions = ['long_dur','med_dur','short_dur']

cond_colors = ['b','g']

#LOAD ALL EPOCHS

#run_tf = 0 # run it or load saved 


ch_type = 'grad' # mag/grad/eeg


tmin = -.5
tmax = 4.2

    
# load TFS  
for pp, nip in enumerate(nips):
    print(nip) 
    
    for c,condition in enumerate(conditions):
        cond_name = condition.replace("/","_")
         
        meg_subject_dir = op.join(config.meg_dir, nip)
        
        power = mne.time_frequency.read_tfrs(op.join(meg_subject_dir, '%s_%s_power_%s-tfr.h5'
                                                     % (config.study_name, nip,
                                                        condition.replace(op.sep, ''))))

        power = power.pop()
#        power.apply_baseline(mode='percent',baseline=None)
            
        power = power.crop(tmin=tmin,tmax=tmax)    
       

        if ch_type == 'mag':
            power.pick_types(meg='mag')
        elif ch_type == 'grad':
            power.pick_types(meg='grad')
        elif ch_type == 'eeg':
            power.pick_types(eeg=True,meg = False)
        
        
        # plot single subject
#        power.plot_topo(baseline=None) # , vmin = -50,vmax = 50
     
        # initiate matrices on first need
        if pp == 0 and c == 0:
            POW = np.zeros((len(nips), len(conditions), len(power.ch_names), len(power.freqs),len(power.times)))         
            ITC = np.zeros((len(nips), len(conditions), len(power.ch_names), len(power.freqs),len(power.times)))         
           
        POW[pp,c]=power.data

        
        itc =  mne.time_frequency.read_tfrs(op.join(meg_subject_dir, '%s_%s_itc_%s-tfr.h5'
                                                    % (config.study_name, nip,
                                                       condition.replace(op.sep, ''))))
        itc = itc.pop()
        itc = itc.crop(tmin=tmin,tmax=tmax)
        
        if ch_type == 'mag':
           itc.pick_types(meg='mag')
        elif ch_type == 'grad':
            itc.pick_types(meg='grad')
        elif ch_type == 'eeg':
            itc.pick_types(eeg=True,meg = False)
        
        ITC[pp,c]=itc.data

    if pp == 0:
        pow_dummy = power
        itc_dummy = itc
            
    

#%% normalize data for cluster t-test
        
POW_keep = POW.copy() 
#POW = POW_keep

for pp, nip in enumerate(nips):
    print(nip) 
    temp = POW[pp, :, :, :, :]
    baseline = np.mean(temp,0)
    baseline = np.mean(baseline, -1)
    POW[pp] = temp / baseline[np.newaxis,:,:,np.newaxis]
       

# test np.newaxis

#a = np.array([2,4,6,8,10,12])
#b = np.array([2,1,2,np.newaxis,2,np.newaxis])
#a/b

#%% compute condition differences and visualize    
         
plot_tmin = -.5
plot_tmax = 4.2
#scale_min = -1
#scale_max =  1
#scale_min = -.1*10**-20
#scale_max =  .1*10**-20

for c, condition in enumerate(conditions):          
 
    P = np.mean(POW[:,c], 0) # avg over sbs
    AVGPOW = mne.time_frequency.AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
      
    AVGPOW.plot_topo(baseline=None,mode='percent', tmin = plot_tmin, tmax = plot_tmax,
                title=('POW ' + condition ),)
#                vmin = scale_min ,vmax = scale_max)
  
    
# compute condition difference
    
# 3-1 items
P = np.mean( (POW[:,0] - POW[:,1]) , 0)

# 3.6s - 1.6s
P = np.mean( (POW[:,0] - POW[:,2]) , 0)
# 3.6s - 2.4s
P = np.mean( (POW[:,0] - POW[:,1]) , 0)
# 2.4s - 1.6s
P = np.mean( (POW[:,1] - POW[:,2]) , 0)
# relative increase
#P = np.mean( (POW[:,0] - POW[:,2]) / POW[:,2] + (POW[:,0] - POW[:,1]) / POW[:,1]) + (POW[:,1] - POW[:,2]) / POW[:,2], 0)


AVGPOW = mne.time_frequency.AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
      

# topo map
AVGPOW.plot_topo(baseline=None, mode='percent', tmin = plot_tmin, tmax = plot_tmax,
                title=('POW ' + 'difference' ),)
#                vmin = scale_min ,vmax = scale_max) # , axes=axis[c]

# joint plot
AVGPOW.plot_joint(baseline=None, mode='percent', tmin=tmin, tmax=tmax,
                 title=('3-1 intervals - 15 subj' ),
                 timefreqs=[(.7, 11.8), (1.3, 10.9), (1.9, 11), (2.2, 10.6)]
                  )

AVGPOW.plot_joint(baseline=None, mode='percent', tmin=tmin, tmax=tmax,
                 title=('3.6s - 1.6s - 15 subj' ),
                 timefreqs=[(.3, 7.7), (1.1, 10.9), (1.6, 9.6), (2.3, 8), (2.8, 9.4), (3.7, 10.8)]
                  )
                 
AVGPOW.plot_joint(baseline=None, mode='percent', tmin=tmin, tmax=tmax,
                 title=('3.6s - 2.4s - 15 subj' ),
                 timefreqs=[(.2, 9.4), (1.1, 8.8), (2.3,8.8), (2.8,9.8), (3.5,10.8), (4.1,11)]
                  )

AVGPOW.plot_joint(baseline=None, mode='percent', tmin=tmin, tmax=tmax,
                 title=('2.4s - 1.6s - 15 subj' ),
                 timefreqs=[(.3,10.7),(1.1,10),(1.7,9.3),(3,8.6),(4.1,9.)]
                  )

AVGPOW.plot_joint(baseline=None, mode='percent', tmin=tmin, tmax=tmax,
                 title=('3-1 intervals - 15 subj' ),
                 timefreqs=[(.7, 11.8), (1.3, 10.9), (1.9, 11), (2.2, 10.6), (2.8, 9.9)]
                  )




#%%    visualize condition differences at selected frequency bands

plot_tmin = tmin
plot_tmax = tmax

vmax_pow = .1
vmin_pow = -vmax_pow
vmax_itc = .02
vmin_itc = -vmax_itc

fmin = 8
fmax = 12

P = np.mean(POW[:,0] - (POW[:,1]), 0) # avg over sbs
AVGPOW = mne.time_frequency.AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
AVGPOW.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('POW difference:' + str(fmin) + '-'+ str(fmax) + ' Hz '
                      + str(plot_tmin) + '-' + str(plot_tmax) + ' s'),  
               show=False, )
               # vmin = vmin_pow, vmax = vmax_pow


#%% plot per COND and FP, at 1 selected channel
    
mne.viz.plot_sensors(pow_dummy.info,ch_type='grad')
plotchan = pow_dummy.ch_names.index('MEG2233','MEG2013')      # MEG1632    EEG012      
plot_tmin = tmin
plot_tmax = tmax   

for c, condition in enumerate(conditions):          
        
    P = np.mean(POW[:,c], 0) # avg over sbs
    AVGPOW = mne.time_frequency.AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
      
    AVGPOW.plot([plotchan], baseline=None, tmin = plot_tmin, tmax = plot_tmax,
                title=('POW ' + condition + AVGPOW.ch_names[plotchan])) # , axes=axis[c]
    
     
P = np.mean(POW[:,0] - (POW[:,1]), 0) # avg over sbs
AVGPOW = mne.time_frequency.AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
       
AVGPOW.plot([plotchan], baseline=None, tmin = plot_tmin, tmax = plot_tmax,
                title=('POW ' + 'difference' + AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#%% CLUSTER T-TESTS between single conditions
            
# POW or ITC ? 
testwhat = 'pow'
do_tfce = 0
# conds are pitch_np, pitch_p,time_np,time_p

tmin_test = 0.
tmax_test = tmax

# need to avg over freqs
# test_freqs = [15,20]
# test_freqs = [20,25]
test_freqs = [8,12]

diffname = 'diff'
test_conds = [0, 1]


tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

n_conditions = len(conditions)
n_replications = len(nips)

if do_tfce == 1: 
    threshold = dict(start=0., step = 0.2)
else:   
    p_threshold = 0.05 # should at least be 0.05
    n_samples = len(nips) # *len(conditions)
    from scipy import stats
    threshold = - stats.distributions.t.ppf(p_threshold / 2., n_samples - 1)

def find_nearest(a,a0):
    "Element in nd array 'a' closest to the scalar value a0"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

tp1 = find_nearest(itc_dummy.times,tmin_test)
tp1 = np.asscalar(np.where(itc_dummy.times == tp1)[0])

tp2 = find_nearest(itc_dummy.times,tmax_test)
tp2 = np.asscalar(np.where(itc_dummy.times == tp2)[0])

n_times = len(np.arange(tp1,tp2,1))

f1 = find_nearest(itc_dummy.freqs,test_freqs[0])
f1 = np.asscalar(np.where(itc_dummy.freqs == f1)[0])

f2 = find_nearest(itc_dummy.freqs,test_freqs[1])
f2 = np.asscalar(np.where(itc_dummy.freqs == f2)[0])

n_chans = len(pow_dummy.ch_names)

# prepare data
allconds_power = list()
allconds_itc = list()
for c, condition in enumerate(conditions):          
#    del Ptmp
    # avg over FP
    Ptmp = POW[:,c]
    Ptmp = Ptmp[:,:,f1:f2,tp1:tp2]
    Ptmp = np.mean(Ptmp, 2) # avg over FREQS
    Ptmp = Ptmp.transpose(0,2,1) 
    allconds_power.append(Ptmp)
    


times = itc_dummy.times[np.arange(tp1,tp2,1)]

# set up for the clustering
connectivity, _ = read_ch_connectivity('neuromag306planar') # neuromag306mag neuromag306planar
#from mne.channels import find_ch_connectivity
#connectivity, _ = find_ch_connectivity(pow_dummy.info, ch_type='eeg')

Ttestdata = list()
if testwhat == 'pow':
    Ttestdata.append(allconds_power[test_conds[0]])
    Ttestdata.append(allconds_power[test_conds[1]])
elif testwhat == 'itc':
    Ttestdata.append(allconds_itc[test_conds[0]])
    Ttestdata.append(allconds_itc[test_conds[1]])

print('Clustering..')  
T_obs, clusters, cluster_p_values, h0 = mne.stats.spatio_temporal_cluster_test(
        Ttestdata, connectivity=connectivity,stat_fun=None, threshold=threshold, tail=tail, 
        n_permutations=n_permutations, buffer_size=None, n_jobs=6) # out_type = 'mask'
# stat_fun=f_mway_rm

#sigma = 1e-3
#stat_fun_hat = partial(mne.stats.ttest_1samp_no_p, sigma = sigma)
#t_tfce_hat, clusters, p_tfce_hat, h0 = mne.stats.spatio_temporal_cluster_test(
#        Ttestdata, connectivity=connectivity,stat_fun=stat_fun_hat, threshold=threshold, tail=tail, 
#        n_permutations=n_permutations, buffer_size=None)

good_cluster_inds = np.where(cluster_p_values < .05)[0]
print('min (p): ' + str(np.min(cluster_p_values)))

#%% create an evoked dummy 

for subject in config.subjects_list:
    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                   config.base_fname.format(**locals()))
    epochs = mne.read_epochs(fname_in, preload=True)
    epochs.pick_types(meg='grad')


dummy_evoked = epochs.average()
del epochs

# plot cluster
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_topomap


# configure variables for visualization
plotcolors = [[243,184,157],[184,56,68],[0,123,194],[178,212,241]]
plotcolors = np.true_divide(plotcolors,255)

# get sensor positions via layout
pos = mne.find_layout(pow_dummy.info).pos

# organize data for plotting
evokeds = {conditions[cond]: mne.EvokedArray(np.mean(Ttestdata[cc],0).transpose(1,0), 
                                 dummy_evoked.info, dummy_evoked.times[0]) for cc, cond in enumerate(test_conds)}

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    # pow_dummy.crop(tmin,tmax)
    sig_times = dummy_evoked.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(f_map, pos, mask=mask, axes=ax_topo, cmap='Reds',
                            vmin=np.min, vmax=np.max, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
                         colors=plotcolors,  show=False,
                         split_legend=True, truncate_yaxis='max_ticks',
                         vlines = []) # , ylim=dict(new_ch_type=[-.5, 1])

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='gray', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()