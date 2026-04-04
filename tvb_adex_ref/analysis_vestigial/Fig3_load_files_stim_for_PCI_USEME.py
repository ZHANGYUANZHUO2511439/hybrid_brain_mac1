#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:50 2019

@author: gpi
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import Lempel_Ziv as lz

def binarise_signals(signal_m, t_stim, nshuffles = 10, 
                     percentile = 100):
    '''
    Following method from: Casali et al, A Theoretically Based Index of 
    Consciousness Independent of Sensory Processing and Behavior
    Suppl Mat, section 2.
    '''
    #%%
    ntrials, nsources, nbins = signal_m.shape
#%% centralise and normalise sources to baseline level
    
    means_prestim = np.mean(signal_m[:,:,:t_stim], axis = 2)
    
    # prestim mean to 0
    signal_centre =\
        signal_m / means_prestim[:,:, np.newaxis] - 1

    std_prestim = np.std(signal_centre[:,:,:t_stim], axis = 2)
    
    # prestim std to 1
    signal_centre_norm = signal_centre / std_prestim[:,:, np.newaxis] 
    
#%% bootstrapping: shuffle prestim signal in time, intra-trial
    signalcn_tuple = tuple(signal_centre_norm)# not affected by shuffling    
    signal_prestim_shuffle = signal_centre_norm[:,:,:t_stim]
    
    
    max_absval_surrogates = np.zeros(nshuffles)
    
    for i_shuffle in range(nshuffles):
        for i_source in range(nsources):
            for i_trial in range(ntrials):
                signal_curr = signal_prestim_shuffle[i_trial, i_source]
                np.random.shuffle(signal_curr)
                signal_prestim_shuffle[i_trial, i_source] = signal_curr
                                
                #average over trials
                shuffle_avg = np.mean(signal_prestim_shuffle, axis = 0)
                
                max_absval_surrogates[i_shuffle] = np.max(np.abs(shuffle_avg))

#%% estimate significance threshold
    max_sorted = np.sort(max_absval_surrogates)
    signalThresh = max_sorted[-int(nshuffles/percentile)] # correction?
    
#%% binarise 
    signalcn = np.array(signalcn_tuple)
    signal_binary = signalcn > signalThresh
    
    return signal_binary


plt.rcParams.update({'font.size': 15})
bE_all = [0, 10, 20, 30, 40, 50] #np.linspace(0,20,11)
ampstim_all = [1e-2, 1e-3, 1e-4] 

for ampstim in ampstim_all:
    for bE in bE_all:#bE_all:
        print('bE=', bE, 'amptstim=', ampstim) #100.0# adaptation strength in nS

#%% file reading
        n_steps = 4 #number of time steps per simulation; Currently 10000ms per step
        
        folder_path = '/mnt/usb-WD_My_Passport_260D_575834314435384B56325230-0:0-part1/TripleLoop_3b_3stimAT5_14p5sec_100seeds_15sec_andsavetime5s_Lionel_new1_connected_afterholidays/'
        # New one with lionel params '/mnt/usb-WD_My_Passport_260D_575834314435384B56325230-0:0-part1/Lionel_July2020_Parameters_TVB_ADEX/TripleLoop_3b_3stimAT5_14p5sec_100seeds_15sec_andsavetime5s_Lionel_new1/'
        # Data with oscillatory up states  bE_all = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        #'/mnt/usb-WD_My_Passport_260D_575834314435384B56325230-0:0-part1/TripleLoop_10b_3stimAT5_14p5sec_100seeds_15sec_andsavetime5se_mallory_ou_v01/'
        #/media/gpi/Elements/TVB_NOstimulus_loop/'
        #'/home/gpi/Desktop/TVB_Linux_1.5.4/Lionel_updates_June/YAdEX_TVB-master/tvb_model/'
        
        ##state_name = 'sleep75'#folder_state[-7:-1]
        n_seeds = 100 # number of independent random seeds and simulations
        n_trials = 5  # number of simulations/realisations to analyse for one PCI
            
        
        entropy_trials = []
        LZ_trials = []
        PCI_trials = []
        
        sim_names = np.arange(0,n_seeds,n_trials) #np.arange(0,20,n_trials)[::-1]
        for sim_curr in sim_names:   
            sig_all_trials = []
            for i_trials in range(sim_curr, sim_curr + n_trials):
                print(sim_curr)
                folder_state = 'b_'+str(bE)+'_stim'+str(ampstim)+'_seed'+str(i_trials)+'/'#'76connectome_secondorder_wake_1/' # '76connectome_secondorder_sleep3/' #
                print('folder is', folder_state)
                times_l = []
                rateE_m = []
            #    rateI_m = []
            #    corrEE_m = []
            #    corrEI_m = []
            #    corrII_m = []
                
                for  i_step in range(n_steps):
                    raw_curr = np.load(folder_path + folder_state +'step_'+str(i_step)+'.npy',
                                           encoding = 'latin1',
                                       allow_pickle=True)
                    print('step is', i_step)
                    for i_time in range(len(raw_curr[0])): 
                        # looks like raw_curr[0] contains the raw unsmoothed data
                        times_l.append(raw_curr[0][i_time][0])
                        rateE_m.append(np.concatenate(raw_curr[0][i_time][1][0]))
            #            rateI_m.append(np.concatenate(raw_curr[0][i_time][1][1]))
            #            corrEE_m.append(np.concatenate(raw_curr[0][i_time][1][2]))
            #            corrEI_m.append(np.concatenate(raw_curr[0][i_time][1][3]))
            #            corrII_m.append(np.concatenate(raw_curr[0][i_time][1][4]))
                        
                times_l = np.array(times_l) # in ms
                rateE_m = np.array(rateE_m) # matrix of size nbins*nregions
            #    rateI_m = np.array(rateI_m) # matrix of size nbins*nregions
            #    corrEE_m = np.array(corrEE_m) # matrix of size nbins*nregions
            #    corrEI_m = np.array(corrEI_m) # matrix of size nbins*nregions
            #    corrII_m = np.array(corrII_m) # matrix of size nbins*nregions
                
                
                t_total = times_l[-1] # last recorded time
                nbins, nregions = rateE_m.shape
                
                #%% choosing variable of interest
                var_of_interest = rateE_m
                varname = 'rateE'
                #%% discard transient
                
                nbins_transient = int(10000/times_l[0]) # to discard in analysis, looks more than enough
                
                sig_region_all = var_of_interest[nbins_transient:,:] 
                sig_region_all = np.transpose(sig_region_all) # now formatted as regions*times
                times = times_l[nbins_transient:]
                sig_all_trials.append(sig_region_all)    
                
            #%%
            sig_all_trials = np.array(sig_all_trials)
            print(range(sim_curr, sim_curr + n_trials), sig_all_trials.shape)
            t_stim_bins = int(14500/times_l[0]) - nbins_transient #remove transient
            
            
            #%% analyse only 500 ms before and after stimulus
            
            t_analysis = 500 #ms
            nbins_analysis =  int(t_analysis/times_l[0])
            
            sig_cut_analysis = sig_all_trials[:,:,
                            t_stim_bins - nbins_analysis:t_stim_bins + nbins_analysis]
            #%% binarisation
            
            sig_all_binary = binarise_signals(sig_cut_analysis, t_analysis, 
                                              nshuffles = 10, 
                                 percentile = 100)
        #    sig_all_binary = binarise_signals(sig_all_trials, t_stim_bins, 
        #                                      nshuffles = 10, 
        #                         percentile = 100)
            
            #%% return entropy
            all_entropy = lz.source_entropy(sig_all_binary.astype(int)[:,:,
                                            t_analysis:]) # signal only after the stimulus
        #    all_entropy = lz.source_entropy(sig_all_binary.astype(int)[:,:,t_stim_bins:]) # signal only after the stimulus
            print('Entropy', all_entropy[0])
            
            #%% return Lempel-Ziv
            Lempel_Ziv = lz.lz_complexity(sig_all_binary.astype(int)[:,:,t_analysis:])
        #    Lempel_Ziv = lz.lz_complexity(sig_all_binary.astype(int)[:,:,t_stim_bins:])
            print('Lempel-Ziv', Lempel_Ziv)
            
            #%% computing perturbqtionql complexity index
            pci = Lempel_Ziv/all_entropy[1]#lz.PCI(sig_all_binary.astype(int))
            print('PCI', pci)
            
            entropy_trials.append(all_entropy)
            LZ_trials.append(Lempel_Ziv)
            PCI_trials.append(pci)
    
        #%% file saving
        amp_title= np.array(ampstim)
        save_file_name = folder_path + 'LionelJune2020_Params_PCI_bE' + str(bE) + '_stim'+ str(ampstim) + '.npy'
        savefile = {}
        savefile['entropy'] = np.array(entropy_trials)
        savefile['Lempel-Ziv'] = np.array(LZ_trials)
        savefile['PCI'] = np.array(PCI_trials)
        
        np.save(save_file_name, savefile)
    ##%% test plot raw data
    #
    #plt.figure()
    #
    #for i_reg in range(nregions):
    #    # only E rate here
    #    plt.plot(times,sig_region_all[i_reg])
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'_signals.pdf')
    #
    ##%% power spectra with fft
    #
    ##setting up vector of freauencies 
    #f_sampling = 1.*len(times)/t_total
    #frq = np.fft.fftfreq(len(times), 1/f_sampling)
    #
    ## computing power spectrum using fft
    #pwr_region_all = []
    #plt.figure()
    #
    #for i_reg in range(nregions):
    #    pwr_region_all.append(np.abs(np.fft.fft(sig_region_all[i_reg]))**2)
    #    # only E rate here
    #    plt.loglog(frq[frq > 0],pwr_region_all[i_reg][frq > 0], alpha=0.4)
    #    plt.title('Power spectra')
    #    plt.xlabel('$f[Hz$]')
    #    plt.ylabel('power [$Hz^2s^2$]')
    #    plt.ylim(10e-11,10e3)
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'_frq_pwr_fft.pdf')
    #    
    #
    ##%% power spectra with welch
    #pwr_region_all_welch = []
    #plt.figure()
    #
    #for i_reg in range(nregions):
    #    f_w, p_w = signal.welch(sig_region_all[i_reg], fs = f_sampling)
    #    pwr_region_all_welch.append(p_w)
    #    # only E rate here
    #    plt.loglog(f_w,pwr_region_all_welch[i_reg])
    #    # slope (ie scaling exponent) and intercept of power spectrum
    #    print(np.polyfit(np.log(f_w[1:]), np.log(pwr_region_all_welch[i_reg][1:]), 1))
    #
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'_frq_pwr_welch.pdf')
    ##%% autocorrelograms... I commented this both because it's hard to interpret and long to compute
    #
    ### split signal into windows
    ##t_win = 2 # s
    ##nbins_win = int(np.floor(t_win*(nbins - nbins_transient)/t_total))
    ##n_win = int(np.floor(1.*(nbins - nbins_transient)/nbins_win))# number of windows
    ##
    ##sig_sum = np.sum(sig_region_all, axis = 0)
    ##sig_sum_windows = np.reshape(sig_sum[:n_win*nbins_win],
    ##                             (n_win, nbins_win))
    ##
    ### lags for x axis
    ##lags = 1.*np.arange(-len(sig_sum_windows[0]),
    ##                     len(sig_sum_windows[0]))/f_sampling
    ##
    ##corr_all = []
    ##
    ##plt.figure()
    ##for i_win in range(n_win):
    ##    corr_all.append(np.correlate(sig_sum_windows[i_win], 
    ##                                 sig_sum_windows[i_win], 'full'))
    ##    plt.plot(lags[:-1], corr_all[-1])
    ##    
    ##plt.xlabel('Lag [s]')
    ##plt.ylabel('Correlation [s**-2]') 
    ### I think the unit is Hz**2 if the signals themselves are rates in Hz
    #
    ##%%histogram of firing rates; bimodality denotes up/down state
    #plt.figure()
    #plt.hist(np.concatenate(sig_region_all), bins = 10**np.linspace(-5,-1), color = 'k', alpha=0.4, edgecolor='black', linewidth=1.2)
    #plt.xscale('log')  
    #plt.xlabel(r'$\nu_E [Hz]$')
    #plt.ylabel('Count')  
    #plt.title('Histogram')
    #
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'_hist.pdf')
    #
    #
    #
    #
    ###%% Import region names 
    ##Load_Region_key = np.load('/home/gpi/Desktop/TVB_Linux_1.5.4/TVB_Distribution/tvb_data/lib/python2.7/site-packages/tvb_data/connectivity/connectivity_76/centres.txt', allow_pickle=True)
    ##Region_key = Load_Region_key[:,1]
    #
    #
    #
    #
    #
    ##%% Pearson correlation between different mean fields
    #pearson_m = np.corrcoef(sig_region_all)
    #
    #plt.figure()
    #plt.imshow(pearson_m, cmap = 'RdBu_r', vmin = 0, vmax = 1, 
    #           interpolation = 'nearest')
    #plt.colorbar()
    #plt.xlabel('# region')
    #plt.ylabel('# region') 
    #plt.title('Pearson correlation')
    ##cmap = 'hot'
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'pearson_heatmap.pdf')
    #
    #
    ##%% Kuramoto order parameter between different mean fields
    #hilb_amplitude_region_all = np.zeros_like(sig_region_all)
    #hilb_phase_region_all = np.zeros_like(sig_region_all)
    #
    ## Taking Hilbert transform, extracting phase and amplitude
    ## be careful, this is much more meaningful when 
    ## the signal is dominated by one frequency range, eg slow waves
    #for i_reg in range(nregions):
    #    hilb = signal.hilbert(sig_region_all[i_reg])
    #    hilb_amplitude_region_all[i_reg] = np.abs(hilb)
    #    hilb_phase_region_all[i_reg] = np.angle(hilb)
    #
    ## order parameter
    #Kura_order_param = np.abs(np.sum(np.exp(1j*hilb_phase_region_all), 
    #                                 axis = 0))/nregions
    #
    ## plotting
    #plt.figure()
    #times = times/5/100
    #plt.plot(times, Kura_order_param, 'k', alpha=0.4)
    #plt.ylim(0.1,1.2)
    #plt.xlabel('Time [s]')
    #plt.ylabel('syncrhony (R)')
    #plt.title('Kuramoto order parameter')
    #
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'Kura_order.pdf')
    #
    ##%% Phase lag index between different mean fields
    #PLI_m = np.zeros((nregions, nregions))
    #
    #for i_reg in range(nregions):
    #    for j_reg in range(i_reg, nregions):
    #        phase_lags = hilb_phase_region_all[i_reg] \
    #        - hilb_phase_region_all[j_reg]
    #        PLI_m[i_reg][j_reg] = np.abs(np.mean(np.sign(phase_lags)))
    #        PLI_m[j_reg][i_reg] = PLI_m[i_reg][j_reg]
    #
    #plt.figure()
    #plt.imshow(PLI_m, cmap = 'hot', vmin = 0, vmax = 0.15, 
    #           interpolation = 'nearest')
    #plt.colorbar()
    #plt.xlabel('# region')
    #plt.ylabel('# region') 
    #plt.title('Phase lag index')
    #
    #plt.savefig(folder_path + folder_state+state_name+'_'+varname+'PLI_heatmap.pdf')
