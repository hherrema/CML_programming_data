# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
import json
import re
from glob import glob
from sklearn.metrics import mean_squared_error

# class for analyzing sync pulse alignments in System 1 data
class sync_pulse_aligner:
    STARTING_ALIGNMENT_WINDOW = 100       # try to align this many sync pulses to start with
    ALIGNMENT_WINDOW_STEP = 10            # reduces window by this amount if it cannot align
    MIN_ALIGNMENT_WINDOW = 5              # do not drop below this many aligned pulses
    ALIGNMENT_THRESHOLD = 10              # this many ms may differ between sync pulse times
    
    def __init__(self, subject, experiment, session, sync_txt):    # initial attributes from sync_pulse_sessions.csv
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.sync_txt = sync_txt
        
    # load in sample rate
    def _sample_rate(self):
        # reading from protocols (only if not a re_implant)
        if '_' not in self.subject:
            fpath = f'/protocols/r1/subjects/{self.subject}/experiments/{self.experiment}/sessions/{self.session}/ephys/current_processed/sources.json'
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as f:
                        eeg_sources = json.load(f)
                    sfreq = list(eeg_sources.values())[0]['sample_rate']
                    return sfreq
                except BaseException as e:
                    raise RuntimeError('Error reading from current_processed folder in protocols.')
        
        # look for params.txt in eeg.noreref
        params_txt = self.sync_txt.split('.')[0] + '.' + self.sync_txt.split('.')[1] + '.params.txt'
        if os.path.exists(params_txt):
            try:
                d = {}
                with open(params_txt, 'r') as f:
                    for line in f:
                        key, val = line.split()
                        d[key] = val
                sfreq = float(d['samplerate'])
                return np.round(sfreq, 0)            # round to a whole number
            except BaseException as e:
                print(e)
                raise RuntimeError('Error reading from params txt folder in eeg noreref.')
        
        # neither worked
        raise FileNotFoundError('Unable to find samplerate for session.')

    # load in sync pulses recorded in event log
    # in mstimes
    def _load_beh_syncs(self):
        behdir = f'/data/eeg/{self.subject}*/behavioral/{self.experiment}/session_{self.session}/'
        log_files = glob(behdir+'session.json*') + glob(behdir+'eeg.eeglog') + glob(behdir+'eeg.eeglog.up')
        if len(log_files) > 0:
            log_file = log_files[0]
        else:                             # search in protocols (PS) --> onlf if not a re-implant
            if '_' not in self.subject:
                prodir = f'/protocols/r1/subjects/{self.subject}/experiments/{self.experiment}/sessions/{self.session}/behavioral/current_source/logs/'
                log_files = glob(prodir+'session.json*') + glob(prodir+'eeg.eeglog') + glob(prodir+'eeg.eeglog.up')
                log_file = log_files[0]
            else:
                raise RuntimeError('Re-implant so do not search protocols.')
        
        if log_file.count('json') > 0:
            events = pd.read_json(log_file, lines=True)
            sync_events = events.query("type=='syncPulse'")
            syncs = np.array(sync_events['time'])
        else:                                                                              # older eeg logs (pre-unity)
            split_lines = [line.split() for line in open(log_file).readlines()]
            syncs = [float(line[0]) for line in split_lines if line[2] in ('CHANNEL_0_UP', 'ON')]
            
        return np.array(syncs)
    
    # load in sync pulses recorded in sync.txt
    # convert to samples
    def _load_eeg_syncs(self):
        syncs = np.loadtxt(self.sync_txt)
        diff = np.diff(syncs)
        syncs = syncs[:-1][diff > 100]           # exclude annotations from the same pulse (buggy pulse extraction)
        return syncs * 1000. / self.sfreq
    
    def coefficients(self):
        raise NotImplementedError
        
    def _matching_pulses(self):
        beh_diff = np.diff(self.beh_syncs)
        eeg_diff = np.diff(self.eeg_syncs)
        
        # match beginning and end separately, then draw line between
        beh_start_range, eeg_start_range = self._matching_window(beh_diff, eeg_diff, True, self.STARTING_ALIGNMENT_WINDOW)
        beh_end_range, eeg_end_range = self._matching_window(beh_diff, eeg_diff, False, self.STARTING_ALIGNMENT_WINDOW)
        
        # join beginning and end
        beh_range = np.union1d(np.arange(beh_start_range[0], beh_start_range[1]), 
                               np.arange(beh_end_range[0], beh_end_range[1]))
        eeg_range = np.union1d(np.arange(eeg_start_range[0], eeg_start_range[1]), 
                               np.arange(eeg_end_range[0], eeg_end_range[1]))

        # select times used
        beh_syncs_match = self.beh_syncs[beh_range]
        eeg_syncs_match = self.eeg_syncs[eeg_range]
        
        return beh_syncs_match, eeg_syncs_match

    def _matching_window(self, beh_diff, eeg_diff, from_start, alignment_window):
        beh_start_idx = None
        eeg_start_idx = None
        
        if from_start:
            start_idx = 0
            end_idx = len(eeg_diff) - alignment_window
            step = 1
        else:
            start_idx = len(eeg_diff) - alignment_window
            end_idx = 0
            step = -1
            
        # slide window, looking for one that fits differences
        for i in range(start_idx, end_idx, step):
            beh_start_idx = self._best_offset(beh_diff, eeg_diff[i:i+alignment_window], self.ALIGNMENT_THRESHOLD)
            if beh_start_idx:
                eeg_start_idx = i
                break
                
        # don't find offset, reduce window and try again
        if not beh_start_idx:
            alignment_window = alignment_window - self.ALIGNMENT_WINDOW_STEP
            if alignment_window < self.MIN_ALIGNMENT_WINDOW:
                raise ValueError('Alignment window too small.')
            else:
                #print(f'Reducing alignment window to {alignment_window} with from_start = {from_start}.')
                return self._matching_window(beh_diff, eeg_diff, from_start, alignment_window)
            
        return (beh_start_idx, beh_start_idx+alignment_window), (eeg_start_idx, eeg_start_idx+alignment_window)
    
    def _best_offset(self, beh_diff, eeg_diff, delta):
        # find any differences below threshold
        idx = np.where(np.abs(beh_diff - eeg_diff[0]) < delta)
        
        # for each difference, find the indices that still match
        for i, this_eeg_diff in enumerate(eeg_diff):
            idx = np.intersect1d(idx, np.where(np.abs(beh_diff-this_eeg_diff) < delta)[0] - i)
            # if no indices left, return None
            if len(idx) == 0:
                return None
            
        # raise exception if found more than one
        if len(idx) > 1:
            raise ValueError("Multiple matching windows so lower threshold or increase window.")
            
        return idx[0]
        
    def _remove_gaps(self):
        bsd = np.diff(self.matching_beh_syncs)
        esd = np.diff(self.matching_eeg_syncs)
        
        no_gap_idx = np.intersect1d(np.where(bsd < 5E3), np.where(esd < 5E3))     # remove large outliers
        gap_idx = np.intersect1d(np.where(bsd >= 5E3), np.where(esd >= 5E3))
        
        if len(gap_idx) >1:
            raise ValueError("Multiple gaps detected.")
        
        # no gaps, not modification necesseary
        if len(gap_idx) == 0:
            beh_syncs_diff_clean = bsd
            eeg_syncs_diff_clean = esd
            matching_beh_syncs_clean = self.matching_beh_syncs
            matching_eeg_syncs_clean = self.matching_eeg_syncs
        else:
            # remove indices where there is a big gap in the syncs (messes up interesect of fit)
            beh_syncs_diff_clean = bsd[no_gap_idx]
            eeg_syncs_diff_clean = esd[no_gap_idx]

            # modify raw matching times
            # for all times after gap, subtract first time after gap, add final time before gap, add mean difference times before gap
            before_gap_beh_syncs = self.matching_beh_syncs[:gap_idx[0]+1]
            after_gap_beh_syncs = self.matching_eeg_syncs[gap_idx[0]+1:]
            after_gap_beh_syncs = after_gap_beh_syncs - after_gap_beh_syncs[0] + before_gap_beh_syncs[-1] + np.round(np.mean(np.diff(before_gap_beh_syncs)), 0)
            matching_beh_syncs_clean = np.concatenate((before_gap_beh_syncs, after_gap_beh_syncs))

            before_gap_eeg_syncs = self.matching_eeg_syncs[:gap_idx[0]+1]
            after_gap_eeg_syncs = self.matching_eeg_syncs[gap_idx[0]+1:]
            after_gap_eeg_syncs = after_gap_eeg_syncs - after_gap_eeg_syncs[0] + before_gap_eeg_syncs[-1] + np.round(np.mean(np.diff(before_gap_eeg_syncs)), 0)
            matching_eeg_syncs_clean = np.concatenate((before_gap_eeg_syncs, after_gap_eeg_syncs))
        
        return beh_syncs_diff_clean, eeg_syncs_diff_clean, matching_beh_syncs_clean, matching_eeg_syncs_clean
    
    def _regression(self):
        # do we want regression and linear fit of differences or raw times --> Joey used differences
        slope, intercept, r, p, err = scipy.stats.linregress(self.beh_syncs_diff_clean, self.eeg_syncs_diff_clean)
        pred = self.beh_syncs_diff_clean * slope + intercept
        residuals = self.eeg_syncs_diff_clean - pred
        rmse = mean_squared_error(self.eeg_syncs_diff_clean, pred, squared=False)
        
        return slope, intercept, residuals, rmse
    
    # return mean, standard devation, standard error of the mean of residual distribution
    def _residuals(self):
        mu = np.mean(self.residuals)
        sig = np.std(self.residuals)
        se = sig / np.sqrt(len(self.residuals))
        
        return mu, sig, se
    
    # ---------- Outward Facing Methods ----------
    
    # run alignment as done by system 1 event creation
    def run_align(self):
        self.sfreq = self._sample_rate()
        self.beh_syncs = self._load_beh_syncs()
        self.eeg_syncs = self._load_eeg_syncs()
        self.matching_beh_syncs, self.matching_eeg_syncs = self._matching_pulses()
        
    # run checks for sync pulse erroneous jitter
    # remove large gaps, run linear regression, make predictions, calculate residuals
    # plot quality of fit, distribution of residuals
    def run_QC(self):
        self.beh_syncs_diff_clean, self.eeg_syncs_diff_clean, self.matching_beh_syncs_clean, self.matching_eeg_syncs_clean = self._remove_gaps()
        self.slope, self.intercept, self.residuals, self.rmse = self._regression()
        self.mu, self.sig, self.se = self._residuals()
        
    # plot relationship between behavioral and eeg sync pulses
    def plot_syncs(self, ax):
        #ax_syncs = sns.jointplot(x=self.beh_syncs_diff_clean, y=self.eeg_syncs_diff_clean, kind='reg')    # can't point joint plot on subplots
        ax_syncs = sns.regplot(x=self.beh_syncs_diff_clean, y=self.eeg_syncs_diff_clean, ax=ax[0])
        ax_syncs.set_xlabel('Inter-Pulse-Time sent', fontsize=14)
        ax_syncs.set_ylabel('Inter-Pulse-Time received', fontsize=14)
        ax_syncs.annotate(f"slope: {self.slope:.3f}\nintercept: {self.intercept:.3f}\nRMSE: {self.rmse:.3f}", 
                             xy=(.1, .7), xycoords="axes fraction", fontsize=14)
        ax_syncs.spines[['right', 'top']].set_visible(False)
        #fig.suptitle(f"Subject {self.subject}, {self.experiment}, Session {self.session}")
        #plt.tight_layout()
        #plt.show()
        #return ax
        
    # plot distribution of residuals
    def plot_residuals(self, ax):
        ax_residuals = sns.histplot(x=self.residuals, kde=True, ax=ax[1])
        ax_residuals.set_xlabel('EEG Sync Pulse Time Residual', fontsize=14)
        ax_residuals.set_ylabel('Count', fontsize=14)
        ax_residuals.annotate(fr"$\mu: {self.mu:.2e}$" + "\n" +  fr"$\sigma: {self.sig:.3f}$" + "\n" + f"SEM: {self.se:.3f}", 
                    xy=(.05, .7), xycoords="axes fraction", fontsize=14)
        ax_residuals.spines[['right', 'top']].set_visible(False)
        #ax.set_title(f"Subject {self.subject}, {self.experiment}, Session {self.session}")
        #plt.show()
        #return ax
    
    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        self.plot_syncs(ax)
        self.plot_residuals(ax)
        
        fig.suptitle(f"Subject {self.subject}, {self.experiment}, Session {self.session}")
        plt.tight_layout()
        plt.show()
        
    # return dictionary of important attributes
    def get_attrs(self):
        return {
            'subject': self.subject,
            'experiment': self.experiment,
            'session': self.session,
            'slope': self.slope,
            'intercept': self.intercept,
            'rmse': self.rmse,
            'mu': self.mu,
            'sig': self.sig,
            'se': self.se,
            'n_syncs': len(self.matching_eeg_syncs)
        }