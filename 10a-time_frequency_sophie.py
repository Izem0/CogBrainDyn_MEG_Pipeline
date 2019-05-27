#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:31:08 2018

@author: sh252228
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

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

#%% get directories
root_directory = returnRootDirectory()
preproc_directory  = returnPreprocDirectory()
results_directory  = returnResultsDirectory()
TF_directory  = returnTFDirectory()

#%% get nips
nips = returnNips()
print(nips)

#oneP = [6] #range(11,len(nips)) # [3]
#nips = [nips[pp] for pp in oneP]
#print(nips) 
# nips = nips[14:19]
#%% get parameters from config file
EpochInfo = returnEpochInfo()
tasks = EpochInfo['task_names']
conditions = ['pitch/NP','pitch/P','time/NP','time/P']

cond_colors = ['b--','g--','b','g']

#%% TF parameters

freqs = np.linspace(.5,40.,num = 40)

## either use tfr_multitaper:
# n_cycles  = freqs/2. # time smoothing 
# time_bandwidth  = 3. # frequency smoothing 

## or morelet wavelets:
n_cycles  = freqs/2. # time smoothing 

#%% LOAD ALL EPOCHS

run_tf = 0 # run it or load saved 

LOCK = 'tar' # or cue
ch_type = 'all' # mag/grad/eeg

if LOCK=='cue':
    tmin = -0.5
    tmax = 3
else:
    # tar_locked
    tmin = -3.5
    tmax = 1.

if run_tf==1: 
    
#    oneP = [0]#  range(15) #len(nips)
#    nips = [nips[pp] for pp in oneP]
#    print(nips) 
    
    for pp, nip in enumerate(nips):
    #pp = 4
    #nip = nips[pp]
        print(nip)
        
        ### Load the preprocessed DATA ########################   
        if LOCK=='cue':
            # cue_locked
            filename = preproc_directory + '/' + nip + '/' + nip + '_preprocessed.fif'
        else:
            # target_locked
            filename = preproc_directory + '/' + nip + '/' + nip + '_TARGET_preprocessed.fif'
        
       
        epochs =  mne.read_epochs(filename, preload=True)  
        epochs.set_eeg_reference(ref_channels='average', projection=True) 
        
        
        if LOCK=='cue':
            epochs.interpolate_bads()
            epochs.apply_baseline(baseline=(None,None))
            # has been qpplied for target
        
        
        # picks MEG gradiometers
        if ch_type=='eeg': 
            picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False, stim=False,exclude=[])
        elif ch_type=='mag': 
            picks = mne.pick_types(epochs.info, meg='mag', eeg=False, eog=False, stim=False,exclude=[])
        elif ch_type=='grad': 
            picks = mne.pick_types(epochs.info, meg='grad', eeg=False, eog=False, stim=False,exclude=[])
        elif ch_type=='all': 
            picks = mne.pick_types(epochs.info, meg=True, eeg=True, eog=False, stim=False,exclude=[])
        
        
        # loop over conditions 
        for c,condition in enumerate(conditions):
            
            # loop over FP
            foreperiods = np.linspace(.5,3.,num = 6) # np.unique(epochs.metadata.FP.round(decimals=1))
            
            # round FPs to 1 dec; jitter, comes from resampling
            epochs.metadata.FP = epochs.metadata.FP.round(decimals=1)
            
            for i_fp,this_fp in enumerate(foreperiods):
                
                q = 'FP=='+str(this_fp)
                epochs_fp = epochs[q] 
                
#                # wavelet
#                power = tfr_morlet(epochs_fp[condition],freqs=freqs,n_cycles=n_cycles,use_fft=True,
#                                    return_itc=False, decim=1, picks=picks,
#                                    average = False)
#                
                avg_power, itc = tfr_morlet(epochs_fp[condition],freqs=freqs,n_cycles=n_cycles,use_fft=True,
                                    return_itc=True, decim=1, picks=picks,
                                    average = True)
                
                # multitaper
#                power, itc = tfr_multitaper(epochs_fp[condition],freqs=freqs,n_cycles=n_cycles,use_fft=True,
#                                            time_bandwidth = time_bandwidth,
#                                            return_itc=True,decim=1,n_jobs=1, picks=picks)  
##                # plot to check
#                 plotchan = avg_power.ch_names.index('MEG1641')  
#                 avg_power.plot([plotchan], baseline=(tmin,tmax),mode = 'zscore', tmin = tmin, tmax = tmax)
#                 itc.plot([plotchan], baseline=(tmin,tmax),mode = 'zscore', tmin = tmin, tmax = tmax)
#                
                avg_power.crop(tmin=tmin,tmax=tmax)
                itc.crop(tmin=tmin,tmax=tmax)
            
                cond_name = condition.replace("/","_")
            
#                power.save(TF_directory + '/' + nip + '/'+ nip + '_epochsPOW_' 
#                           + LOCK + '_' +  cond_name + '_' +  str(this_fp) + 's_' 
#                           + ch_type + '.fif', overwrite=True)
                
                avg_power.save(TF_directory + '/' + nip + '/'+ nip + '_POW_' 
                           + LOCK + '_' +  cond_name + '_' +  str(this_fp) + 's_' 
                           + ch_type + '.fif', overwrite=True)
                
                itc.save(TF_directory + '/' + nip + '/'+ nip + '_ITC_' 
                          + LOCK + '_' +  cond_name + '_' +  str(this_fp) + 's_' 
                           + ch_type + '.fif', overwrite=True)

#           power.save(TF_directory + '/' + nip + '/'+ nip + '_POW_' + LOCK + '_' +  cond_name +  '_' + ch_type + 'tfr.h5',overwrite=True)
#           itc.save(TF_directory + '/' + nip + '/'+ nip + '_ITC_' + LOCK + '_'+ cond_name +  '_' + ch_type + 'tfr.h5',overwrite=True )

# load DATA
elif run_tf==0:
    
    new_ch_type = 'mag' 
    
    # leaving out the shortest FP (0.5)
    # all: np.linspace(0.5,3.,num = 6)
    if LOCK=='cue':
       foreperiods =  np.linspace(.5,3.,num = 6)
       # np.linspace(2,3.,num = 3) 
    elif  LOCK=='tar':
       # foreperiods =  np.linspace(.5,3.,num = 6)
       foreperiods =  np.linspace(1.,3.,num = 5)
    
    # load TFS  
    for pp, nip in enumerate(nips):
        print(nip) 
        
        for c,condition in enumerate(conditions):
            cond_name = condition.replace("/","_")
            
            # loop over FP
            for f,this_fp in enumerate(foreperiods):
                
                power = read_tfrs(TF_directory + '/' + nip + '/'+ nip + '_POW_' 
                           + LOCK + '_' +  cond_name + '_' +  str(this_fp) + 's_' 
                           + ch_type + '.fif')
                
                power = power.pop()
                power = power.crop(tmin=tmin,tmax=tmax)
                
                if new_ch_type == 'mag':
                    power.pick_types(meg='mag')
                elif new_ch_type == 'grad':
                    power.pick_types(meg='grad')
                elif new_ch_type == 'eeg':
                    power.pick_types(eeg=True,meg = False)
                
                # initiate matrices on first need
                if pp == 0 and c == 0 and f == 0:
                    POW = np.zeros((len(nips), len(conditions), len(foreperiods), len(power.ch_names), len(power.freqs),len(power.times)))         
                    ITC = np.zeros((len(nips), len(conditions), len(foreperiods), len(power.ch_names), len(power.freqs),len(power.times)))         
                   
                # bsl correct power: always take pre-cue window
                if LOCK=='cue':
                    power.apply_baseline(mode='zscore',baseline=(-0.5,0))
                
                elif LOCK=='tar':
                   #  power.apply_baseline(mode='zscore',baseline=(-(this_fp+0.49),-(this_fp)))
                    power.apply_baseline(mode='zscore', baseline=(tmin,tmax))
                    
                POW[pp,c,f]=power.data

                
                itc =  read_tfrs(TF_directory + '/' + nip + '/'+ nip + '_ITC_' 
                          + LOCK + '_' +  cond_name + '_' +  str(this_fp) + 's_' 
                           + ch_type + '.fif')
                
                itc = itc.pop()
                itc = itc.crop(tmin=tmin,tmax=tmax)
                
                if new_ch_type == 'mag':
                   itc.pick_types(meg='mag')
                elif new_ch_type == 'grad':
                    itc.pick_types(meg='grad')
                elif new_ch_type == 'eeg':
                    itc.pick_types(eeg=True,meg = False)
                
                ITC[pp,c,f]=itc.data
        
            if pp == 0:
                pow_dummy = power
                itc_dummy = itc
                
    ch_type = new_ch_type  
    
#%% plot per COND and FP, at 1 selected channel
    
    mne.viz.plot_sensors(pow_dummy.info,ch_type='mag')
    plotchan = pow_dummy.ch_names.index('MEG1641')      # MEG1632    EEG012      
    plot_tmin = -2.
    plot_tmax = 1.     
    
    for c, condition in enumerate(conditions):          
        # loop over FP
       
#        c = 3
#        condition = conditions[c]
#        
        for f,this_fp in enumerate(foreperiods):
            
            P = np.mean(POW[:,c,f], 0) # avg over sbs
            AVGPOW = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
              
            AVGPOW.plot([plotchan], baseline=None, tmin = plot_tmin, tmax = plot_tmax,
                        title=('POW ' + condition + str(this_fp) + 's_' + AVGPOW.ch_names[plotchan])) # , axes=axis[c]
             
##              
#            I = np.mean(ITC[:,c,f], 0) # avg over sbs
#            AVGITC = AverageTFR(itc_dummy.info,I,itc_dummy.times,itc_dummy.freqs,nave=len(nips))
#              
#            AVGITC.plot([plotchan], baseline=None, tmin = tmin, tmax = 3, vmin = 0,
#                        title=('ITC ' + condition + str(this_fp) + 's_' + AVGITC.ch_names[plotchan])) # , axes=axis[c]
            

#%% compute condition differences per FP and visualize    
         
plot_tmin = -.5
plot_tmax = 0.           
            
for f,this_fp in enumerate(foreperiods):        
    
#    # PITCH P - NP
#    P = np.mean(POW[:,1,f], 0) - np.mean(POW[:,0,f], 0) 
#    AVGPOW_Pitch_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#    # AVGPOW_Pitch_P_NP.plot([plotchan], baseline=None, title=('POW Pitch_P_NP'  + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#    AVGPOW_Pitch_P_NP.plot_topo(baseline=None, tmin = plot_tmin, tmax = plot_tmax,vmin = -1.5,vmax = 1.5,
#                                title=('POW Pitch_P_NP'  + ' ' + str(this_fp) + 's_'))
###
#    # TIME P - NP
#    P = np.mean(POW[:,3,f], 0) - np.mean(POW[:,2,f], 0) 
#    AVGPOW_Time_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
##     AVGPOW_Time_P_NP.plot([plotchan], baseline=None, title=('POW Time_P_NP'  + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#    AVGPOW_Time_P_NP.plot_topo(baseline=None, tmin = tmin, tmax = tmax,
#                               title=('POW Time_P_NP'  + ' ' + str(this_fp) + 's_'))
#    
#    
#    # TIME NP - Pitch P
#    P = np.mean(POW[:,2,f], 0) - np.mean(POW[:,1,f], 0) 
#    AVGPOW_Time_NP_Pitch_P = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#    #AVGPOW_Time_NP_Pitch_P.plot([plotchan], baseline=None, title=('POW Time_NP_Pitch_P'  + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#    AVGPOW_Time_NP_Pitch_P.plot_topo(baseline=None, tmin = tmin, tmax = tmax,
#                                     title=('Time_NP - Pitch_P'  + ' ' + str(this_fp) + 's_'))
#         
          
    # PITCH P - NP
    P = np.mean(ITC[:,1,f], 0) - np.mean(ITC[:,0,f], 0) 
    AVGITC_Pitch_P_NP = AverageTFR(itc_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
    #AVGITC_Pitch_P_NP.plot([plotchan], baseline=None, title=('ITC Pitch_P_NP'  + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
    AVGITC_Pitch_P_NP.plot_topo(baseline=None, tmin = plot_tmin, tmax = plot_tmax,vmin = -.05,vmax = .05,
                                title=('ITC Pitch_P_NP'  + ' ' + str(this_fp) + 's_'))
        
#          
#    # TIME P - NP
#    P = np.mean(ITC[:,3,f], 0) - np.mean(ITC[:,2,f], 0) 
#    AVGITC_Time_P_NP = AverageTFR(itc_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#    #AVGITC_Time_P_NP.plot([plotchan], baseline=None, title=('ITC Time_P_NP' + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#    AVGITC_Time_P_NP.plot_topo(baseline=None,tmin = tmin, tmax = tmax,
#                               title=('ITC Time_P_NP'  + ' ' + str(this_fp) + 's_'))
#       
#          
#    # TIME NP - Pitch P
#    P = np.mean(ITC[:,2,f], 0) - np.mean(ITC[:,1,f], 0) 
#    AVGITC_Time_NP_Pitch_P = AverageTFR(itc_dummy.info,P,itc_dummy.times,itc_dummy.freqs,nave=len(nips))
#    # AVGITC_Time_NP_Pitch_P.plot([plotchan], baseline=None, title=('ITC Time_NP_Pitch_P'  + ' '+ AVGPOW.ch_names[plotchan])) # , axes=axis[c]
#    AVGITC_Time_NP_Pitch_P.plot_topo(baseline=None, tmin = tmin, tmax = tmax,
#                                     title=('Time_NP_Pitch_P'  + ' ' + str(this_fp) + 's_'))
         

#%% compute condition differences for avg FP and visualize    

# POW_save = POW
# ITC_save = ITC

#POW = POW_save[:,:,np.arange(2,5,1)]  
#ITC = ITC_save[:,:,np.arange(2,5,1)]      
       
plot_tmin = -.5
plot_tmax = 0.  
    
## POW

# [Something is wrong with POWER for FP = 3 >> all NaN/Inf]

P = np.mean(POW[:,1,0:5], 0) - np.mean(POW[:,0,0:5], 0) 
P = np.mean(P, 0) # avg over FP
AVGPOW_allFP_Pitch_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
AVGPOW_allFP_Pitch_P_NP.plot_topo(baseline=None, title=('POW Pitch_P_NP'  + ' ' + 'avgFP'),
                                  vmin = -.5,vmax = .5,tmin = plot_tmin,tmax = plot_tmax)


P = np.mean(POW[:,3], 0) - np.mean(POW[:,2], 0) 
P = np.mean(P, 0) # avg over FP
AVGPOW_allFP_Time_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGPOW_allFP_Time_P_NP.plot_topo(baseline=None, title=('POW Time_P_NP'  + ' ' + 'avgFP'),
#                                 vmin = -1.,vmax = 1.,tmin = plot_tmin,tmax = plot_tmax)

#
P =(np.mean(POW[:,3], 0) - np.mean(POW[:,2], 0)) - (np.mean(POW[:,1], 0) - np.mean(POW[:,0], 0))
P = np.mean(P, 0) # avg over FP
AVGPOW_allFP_Time_Pitch = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGPOW_allFP_Time_Pitch.plot_topo(baseline=None, title=('POW _Time - Pitch'  + ' ' + 'avgFP'),
#                                        vmin = -1.,vmax = 1.,tmin = plot_tmin,tmax = plot_tmax)
#
#
P =(np.mean(POW[:,2], 0) - np.mean(POW[:,0], 0)) 
P = np.mean(P, 0) # avg over FP
AVGPOW_allFP_Time_NP_Pitch_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGPOW_allFP_Time_Pitch.plot_topo(baseline=None, title=('POW TimeNP - PitchNP'  + ' ' + 'avgFP'),
#                                vmin = -1.,vmax = 1.,tmin = plot_tmin,tmax = plot_tmax)
#          



## ITC
P = np.mean(ITC[:,1], 0) - np.mean(ITC[:,0], 0) 
P = np.mean(P, 0) # avg over FP
AVGITC_allFP_Pitch_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
AVGITC_allFP_Pitch_P_NP.plot_topo(baseline=None, title=('ITC Pitch_P_NP'  + ' '  + 'avgFP'),
                                  vmin = -.04,vmax = .04,tmin = plot_tmin,tmax = plot_tmax)


P = np.mean(ITC[:,3], 0) - np.mean(ITC[:,2], 0) 
P = np.mean(P, 0) # avg over FP
AVGITC_allFP_Time_P_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGITC_allFP_Time_P_NP.plot_topo(baseline=None, title=('ITC Time_P_NP'  + ' ' + 'avgFP'),
#                                vmin = -.05,vmax = .05,tmin = plot_tmin,tmax = plot_tmax)
#
#
P = (np.mean(ITC[:,3], 0) - np.mean(ITC[:,2], 0)) - (np.mean(ITC[:,1], 0) - np.mean(ITC[:,0], 0)) 
P = np.mean(P, 0) # avg over FP
AVGITC_allFP_Time_Pitch = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGITC_allFP_Time_Pitch.plot_topo(baseline=None, title=('ITC Time - Pitch'  + ' ' + 'avgFP'),
#                                vmin = -.05,vmax = .05,tmin = plot_tmin,tmax = plot_tmax)
#
#
P = np.mean(ITC[:,2], 0) - np.mean(ITC[:,0], 0)
P = np.mean(P, 0) # avg over FP
AVGITC_allFP_Time_NP_Pitch_NP = AverageTFR(pow_dummy.info,P,pow_dummy.times,pow_dummy.freqs,nave=len(nips))
#AVGITC_allFP_Time_Pitch.plot_topo(baseline=None, title=('ITC TimeNP - PitchNP'  + ' ' + 'avgFP'),
#                                vmin = -.05,vmax = .05,tmin = plot_tmin,tmax = plot_tmax)

#%%    visualize condition differences at selected frequency bands

# grad doenst seem to work with the same scale; only positive but data are diff

plot_tmin = -.3
plot_tmax = -.1  

vmax_pow = .3
vmin_pow = -vmax_pow
vmax_itc = .02
vmin_itc = -vmax_itc

fmin = 15
fmax = 30
   
AVGPOW_allFP_Pitch_P_NP.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('POW Pitch_P_NP:' + str(fmin) + '-'+ str(fmax) + ' Hz'),  
               show=False, vmin = vmin_pow, vmax = vmax_pow)


AVGPOW_allFP_Time_Pitch.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('POW Time_NP - Pitch_NP:' + str(fmin) + '-'+ str(fmax) + ' Hz'), 
               show=False, vmin = vmin_pow, vmax = vmax_pow)

AVGPOW_allFP_Time_P_NP.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('POW Time_P - Time_NP:' + str(fmin) + '-'+ str(fmax) + ' Hz'), 
               show=False, vmin = vmin_pow, vmax = vmax_pow)


AVGITC_allFP_Pitch_P_NP.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,  
               title=('ITC Pitch_P_NP:' + str(fmin) + '-'+ str(fmax) + ' Hz'),  show=False, vmin = vmin_itc, vmax = vmax_itc)
        
   
AVGITC_allFP_Time_P_NP.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('ITC Time_P_NP:' + str(fmin) + '-' +str(fmax) + ' Hz'),  show=False, vmin = vmin_itc, vmax = vmax_itc)

AVGITC_allFP_Time_NP_Pitch_NP.plot_topomap(ch_type=ch_type, tmin=plot_tmin, tmax=plot_tmax, fmin=fmin, fmax=fmax,
               baseline=None,
               title=('ITC Time_NP - Pitch_NP:' + str(fmin) + '-' +str(fmax) + ' Hz'),  show=False, vmin = vmin_itc, vmax = vmax_itc)

#%% Cond Diff avg FP, 1 channel
plot_tmin = -0.5
plot_tmax = 3.0

mne.viz.plot_sensors(pow_dummy.info)
plotchan = itc_dummy.ch_names.index('MEG0431')      # MEG1632    EEG012      
    
AVGITC_allFP_Pitch_P_NP.plot([plotchan], baseline=None, title=('ITC Pitch_P_NP'  + ' '+ AVGPOW.ch_names[plotchan]),tmin=tmin, tmax=plot_tmax) # , axes=axis[c]
   

#%% CLUSTER T-TESTS between single conditions
            
# POW or ITC ? 
testwhat = 'pow'
do_tfce = 0
# conds are pitch_np, pitch_p,time_np,time_p

tmin_test = -.5
tmax_test = 0.

# need to avg over freqs
# test_freqs = [15,20]
# test_freqs = [20,25]
test_freqs = [4,7]

diffname = 'pitch_p-np'
test_conds = [1, 0]

#diffname = 'pitch_p-pitch_np'
#test_conds = [1, 0]

#diffname = 'pitch_np-pitch_p'
#test_conds = [0, 1]

#diffname = 'time_np-pitch_np'
#test_conds = [2, 0]

#diffname = 'time_p-time_np'
#test_conds = [3, 2]

#diffname = 'time_np-pitch_np'
#test_conds = [3, 0]

#diffname = 'time_np-pitch_p'
#test_conds = [2, 1]

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

n_conditions = len(conditions)
n_replications = len(nips)

if do_tfce == 1: 
    threshold = dict(start=0., step = 0.2)
else:   
    p_threshold = 0.05
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
    
    # avg over FP
    Ptmp = POW[:,c]
    Ptmp = Ptmp[:,:,:,f1:f2,tp1:tp2]
    P = np.mean(Ptmp, 1) # avg over FP
    P = np.mean(P, 2) # avg over FREQS
    P = P.reshape(n_replications, n_times, n_chans) 
    allconds_power.append(P)
    
    Itmp = ITC[:,c]
    Itmp = Itmp[:,:,:,f1:f2,tp1:tp2]
    I = np.mean(Itmp, 1) # avg over FP
    I = np.mean(I, 2) # avg over FREQS
    I = I.reshape(n_replications, n_times, n_chans)   
    allconds_itc.append(I)


freqs = pow_dummy.freqs
times = itc_dummy.times[np.arange(tp1,tp2,1)]

# set up for the clustering
connectivity, _ = read_ch_connectivity('neuromag306mag') # neuromag306mag neuromag306planar
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
        n_permutations=n_permutations, buffer_size=None, out_type = 'mask')
# stat_fun=f_mway_rm

#sigma = 1e-3
#stat_fun_hat = partial(mne.stats.ttest_1samp_no_p, sigma = sigma)
#t_tfce_hat, clusters, p_tfce_hat, h0 = mne.stats.spatio_temporal_cluster_test(
#        Ttestdata, connectivity=connectivity,stat_fun=stat_fun_hat, threshold=threshold, tail=tail, 
#        n_permutations=n_permutations, buffer_size=None)


#%%
good_clusters = np.where(cluster_p_values < .08)[0]
print('min (p): ' + str(np.min(cluster_p_values)))

TestDat = Ttestdata[0] - Ttestdata[1]
TestDat_avg =  np.mean(TestDat,0)
# visualise tfce result
if do_tfce == 0:
    for i_cluster, good_cluster in enumerate(good_clusters):
          
                masky = np.invert(clusters[np.squeeze(good_cluster)])
    #                plt.imshow(np.swapaxes(clusters[np.squeeze(good_cluster)]*1,1,0),vmin=0, vmax=1)
    #                plt.colorbar()
    #                # Initialize figure
    
                T_masked = np.ma.masked_array(T_obs, masky) # masks the TRUE values (but has been inverted before)
                
                fig, ax = plt.subplots(1)    
                ax1 = plt.subplot2grid((1, 2), (0, 0))
                
                # plot topo 
                topodat = np.mean(T_masked,0)               
                topodat = topodat.reshape(102,1,1)
                TOPO = AverageTFR(pow_dummy.info,topodat,[0],[1],nave=len(nips))
                TOPO.plot_topomap(ch_type=ch_type, tmin=0, tmax=0, fmin=1, fmax=1,
                       baseline=None,
                       title=(diffname +  "\n" +
                          " cluster-level corrected (p <= " + str(p_threshold)),  
                          axes = ax1, show=True)
                
                
                # collapse mask over timepoints 
                # to get all channels that are significant at any time
                mask_chan = clusters[np.squeeze(good_cluster)].any(axis = 1)
                mask_chan_mat = np.tile(mask_chan, (1,n_chans))
                
                time_inds = np.ma.masked_array(times,np.invert(clusters[np.squeeze(good_cluster)].any(axis = 1)))
                
                TestDat_diff = Ttestdata[0] - Ttestdata[1]
                
                TestDat0_masked = np.ma.masked_array(np.mean(Ttestdata[0],0), np.invert(mask_chan_mat))
                TestDat1_masked = np.ma.masked_array(np.mean(Ttestdata[1],0), np.invert(mask_chan_mat))
                
                TestDat_diff_masked = np.ma.masked_array(np.mean(TestDat_diff,0), mask_chan_mat)
                
                plotdat0 = TestDat0_masked.mean(axis = 1)
                plotdat1 = TestDat1_masked.mean(axis = 1)
                
                ax1 = plt.subplot2grid((1, 2), (0, 1))
                plt.plot(times, plotdat0)
                plt.plot(times, plotdat1)            
                plt.title(diffname)
             
                plt.fill_betweenx((0.15, .2), time_inds.min(), time_inds.max(),
                                 color='grey', alpha=0.3)
                     
else:
    
    # visualise tfce result
    significant_points = cluster_p_values.reshape(T_obs.shape).T < .15
    print('found ' + str(significant_points.sum()) + " points selected by TFCE ...")
       
    
    AVGITC_allFP_Pitch_P_NP.plot_image(mask=significant_points, time_unit='s',
                                    )