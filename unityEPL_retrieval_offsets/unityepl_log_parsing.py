### UnityEPL-FR (session.json) log parsing for UnityEPL Retrieval Offsets

# imports
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

# replace default values with NaN in results
# select UnityEPL-FR sessions
def select_results(final_res):
    final_res['original_experiment'] = final_res['original_experiment'].replace('X', np.nan)
    final_res['original_session'] = final_res['original_session'].replace(-999, np.nan)
    unityepl_fr = final_res[final_res['session_json'] == True]
    return unityepl_fr

# deal with experiment and session changes
def deal_with_changes(row, exp_change, sess_change):
    if exp_change:
        exp_match = row.original_experiment
    else:
        exp_match = row.experiment

    if sess_change:
        sess_match = int(row.original_session)   # always cast to int
    else:
        sess_match = row.session

    return exp_match, round(sess_match)

# find session.json file
def locate_session_log(row):
    # determine if subject is re-implant or not
    re_implant = False
    # infer from the subject_alias
    if row.subject != row.subject_alias:
        re_implant = True

    # determine if original_experiment is different from experiment
    exp_change = False
    if type(row.original_experiment) == str and row.original_experiment != row.experiment:
        exp_change = True

    # determine if original_session is different from session
    sess_change = False
    # change str to int always, change value if non-matching
    if type(row.original_session) == str or (type(row.original_session)==int and row.original_session != row.session):
        sess_change = True

    exp_match, sess_match = deal_with_changes(row, exp_change, sess_change)    # deal with experiment and session changes
    if row.original_experiment == 'CatFR1':       # handle edge case
        exp_match = 'CatFR1'

    # if not a re-implants, use subject
    if not re_implant:
        behdir = f'/data10/RAM/subjects/{row.subject}/behavioral/{exp_match}/session_{sess_match}/'
    else:     # re-implant, use subject_alias
        behdir = f'/data10/RAM/subjects/{row.subject_alias}/behavioral/{exp_match}/session_{sess_match}/'

    if os.path.exists(os.path.join(behdir, 'session.json')):
        return os.path.join(behdir, 'session.json')
    else:
        raise RuntimeError('Could not locate log file.')
    
    return None
       
# extract experiment version for first line of log file
def load_exp_version(log_file):
    with open(log_file, 'r') as f:
        l1 = f.readline()
        header = json.loads(l1)
        exp_v = header['data']['Experiment version']
        
    return exp_v

# add experiment version info to dataframe
def experiment_versions(unityepl_fr):
    versions = []
    for _, row in tqdm(unityepl_fr.iterrows()):
        try:
            log_file = locate_session_log(row)
            exp_v = load_exp_version(log_file)
            versions.append((row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, 
                             row.median_beep_time, exp_v))
        except BaseException as e:
            versions.append((row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, 
                             row.median_beep_time, np.nan))

    versions = pd.DataFrame(versions, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 
                                               'median_beep_time', 'experiment_version'])
    return versions
    
# parse log file calculate offsets, expected beep times, fixation lengths, and beep durations
def parse_log_file(log_file):
    # read lines of log file
    lines = []
    with open(log_file, 'r') as f:
        for x in f:
            try:
                lines.append(json.loads(x))
            except BaseException as e:
                continue

    #lines = np.array(lines)
    
    # find retrieval, fixation, beep events
    evs = []
    for i, l in enumerate(lines):
        try:
            if l['type'] == 'display recall text' or l['type'] == 'text display cleared':
                evs.append(l)
            elif 'sound duration' in l['data'].keys():
                evs.append(l)
            elif l['data']['message']['data']['name'] == 'RETRIEVAL':
                evs.append(l)
        except:
            continue
            
    # calculate time between fixation clearing (recording start) and retrieval logged
    # calculate time between fixation displayed (recording start for pre-bug sessions) and retrival logged
    # calculate time between fixation clearing (recording start) and end of high beep (expected beep time)
    # calcualte time fixation is on screen
    offsets = []
    offsets_pre = []
    expected_bt = []
    fixation = []
    for i in range(len(evs) - 4):
        # retrieval --> high beep --> display fixation --> clear fixation
        if (
            evs[i]['type'] == 'network' and 
            (evs[i+1]['type'] == 'Sound played' and evs[i+1]['data']['sound name'] == 'high beep') and 
            evs[i+2]['type'] == 'display recall text' and 
            evs[i+3]['type'] == 'text display cleared'
        ):
            offsets.append(evs[i+3]['time'] - evs[i]['time'])
            offsets_pre.append(evs[i+2]['time'] - evs[i]['time'])
            expected_bt.append((evs[i+1]['time'] + 1000 * evs[i+1]['data']['sound duration']) - evs[i+3]['time'])
            fixation.append(evs[i+3]['time'] - evs[i+2]['time'])
            
    # sessions after fix have different order
    if len(offsets) == 0 and len(expected_bt) == 0 and len(fixation) == 0:
        # high beep --> display fixation --> retrieval --> clear fixation
        for i in range(len(evs) - 4):
            if ((evs[i]['type'] == 'Sound played' and evs[i]['data']['sound name'] == 'high beep') and
                evs[i+1]['type'] == 'display recall text' and 
                evs[i+2]['type'] == 'network' and 
                evs[i+3]['type'] == 'text display cleared'
               ):
                offsets.append(evs[i+3]['time'] - evs[i+2]['time'])
                offsets_pre.append(np.nan)                             # retrieval logged after fixation displayed
                expected_bt.append((evs[i]['time'] + 1000 * evs[i]['data']['sound duration']) - evs[i+3]['time'])
                fixation.append(evs[i+3]['time'] - evs[i+1]['time'])
                
    # calculate beep durations
    beeps = []
    for i in range(len(evs)):
        if evs[i]['type'] == 'Sound played' and evs[i]['data']['sound name'] == 'high beep':
            beeps.append(evs[i]['data']['sound duration'])
    
    return np.mean(offsets), np.nanmean(offsets_pre), np.mean(expected_bt), np.mean(fixation), np.mean(beeps)

# add durations info to dataframe
def offsets_durations(versions):
    durations = []
    for _, row in tqdm(versions.iterrows()):
        try:
            log_file = locate_session_log(row)
            off, off_pre, e_bt, fix, bd = parse_log_file(log_file)
            durations.append((row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, 
                              row.median_beep_time, row.experiment_version, off, off_pre, e_bt, fix, bd))
        except BaseException as e:
            durations.append((row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, 
                              row.median_beep_time, row.experiment_version, np.nan, np.nan, np.nan, np.nan, np.nan))

    durations = pd.DataFrame(durations, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 
                                                 'median_beep_time', 'experiment_version', 'expected_offset', 'expected_offset_pre', 'expected_beep_time', 
                                                 'fixation', 'beep_duration'])
    durations['beep_duration'] = np.round(durations['beep_duration'], 2)    # round for plotting purposes
    return durations
    
# ---------- Plotting Functions ----------

# plots to determine which sessions require offset correction
def plot_offset_sessions(durations):
    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # median beep time as a function of experiment version
    sns.stripplot(data=durations, x='experiment_version', y='median_beep_time', hue='median_beep_time', ax=ax1, legend=None)
    ax1.set(ylabel='median beep time (s)')
    ax1.spines[['right', 'top']].set_visible(False)
    
    # expected offset as a function of experiment version
    sns.stripplot(data=durations, x='experiment_version', y='expected_offset', palette='copper_r', hue='expected_offset', ax=ax2, legend=None)
    ax2.set(yticks=np.linspace(0, 1000, 5), ylabel='expected offset (ms)')
    ax2.spines[['right', 'top']].set_visible(False)
    
    # median beep time as a function of subject (chronology) and beep duration
    sns.scatterplot(data=durations, x='subject', y='median_beep_time', palette='cool', hue='beep_duration', ax=ax3)
    ax3.tick_params(axis='x', labelrotation=90)
    ax3.set(ylabel='median beep time (s)', xticks=np.arange(0, len(durations.subject.unique()), 3))
    ax3.spines[['right', 'top']].set_visible(False)
    
    plt.suptitle('Which Sessions Require Offset Correction?')
    plt.tight_layout()
    plt.show()
    
# plots to determine whether some sessions require partial offset correction
def plot_partial_offsets(durations):
    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # fixation as a function of subject (chronology) and experiment version
    sns.scatterplot(data=durations, x='subject', y='fixation', hue='experiment_version', palette='tab10', ax=ax1, alpha=0.6)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set(ylabel='fixation (ms)', xticks=np.arange(0, len(durations.subject.unique()), 3))
    ax1.spines[['right', 'top']].set_visible(False)

    # select out sessions with non-zero median beep time
    partial = durations[(durations['median_beep_time'] > 0.15)]

    # median beep time as a function of fixation and beep duration for sessions with non-zero median beep time
    sns.scatterplot(data=partial, x='fixation', y='median_beep_time', hue='beep_duration', palette='RdPu_r', ax=ax2)
    ax2.set(xlabel='fixation (ms)', ylabel='median beep time (s)')
    ax2.spines[['right', 'top']].set_visible(False)

    # expected offset as a function of median beep time
    sns.scatterplot(data=durations, x='median_beep_time', color='teal', y='expected_offset_pre', hue='median_beep_time', palette='winter_r', ax=ax3, legend=None)
    ax3.set(xlabel='median beep time (s)', ylabel='expected offset (ms)')
    ax3.spines[['right', 'top']].set_visible(False)

    plt.suptitle('Do Some Sessions Require a Partial Offset Correction?')
    plt.tight_layout()
    plt.show()

# ---------- Deprecated Plotted Functions ----------
    
# plot median beep time as a function of experiment version
def plot_beep_time_expv(durations):
    ax = sns.catplot(data=durations, x='experiment_version', y='median_beep_time', hue='median_beep_time')
    ax.set(ylabel='median beep time (s)')
    plt.show()
    
# plot offset as a function of experiment versions
def plot_offset_expv(durations):
    ax = sns.catplot(data=durations, x='experiment_version', y='expected_offset', hue='expected_offset', height=4, aspect=1.3)
    ax.set(yticks=np.linspace(0, 1000, 5), ylabel='expected_offset (ms)')
    plt.show()
    
# plot (expected_offset, expected_beep_time, fixation, beep_duration) as a function of experiment version
def plot_fxn_expv(durations):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    alp = 0.5
    sns.stripplot(data=durations, x='experiment_version', y='expected_offset', color='darkmagenta', ax=ax[0, 0], alpha=alp)
    sns.stripplot(data=durations, x='experiment_version', y='expected_beep_time', color='skyblue', ax=ax[0, 1], alpha=alp)
    sns.stripplot(data=durations, x='experiment_version', y='fixation', color='steelblue', ax=ax[1, 0], alpha=alp)
    sns.stripplot(data=durations, x='experiment_version', y='beep_duration', color='royalblue', ax=ax[1, 1], alpha=alp)
    ax[0, 0].set(ylabel='expected offset (ms)')
    ax[0, 1].set(ylabel='expected beep time (ms)')
    ax[1, 0].set(ylabel='fixation (ms)')
    ax[1, 1].set(ylabel='beep duration (s)')
    ax[0, 0].spines[['top', 'right']].set_visible(False)
    ax[0, 1].spines[['top', 'right']].set_visible(False)
    ax[1, 0].spines[['top', 'right']].set_visible(False)
    ax[1, 1].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Functions of Experiment Version', fontsize=14)
    plt.tight_layout()
    plt.show()
    
# plot interactions
# (x = median_beep_time, y = expected_offset, hue = beep_duration)
# (x = median_beep_time, y = expected_beep_time, hue = beep_duration)
# (x = median_beep_time, y = fixation, hue = beep_duration)
def plot_interactions(durations):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    alp = 0.5
    sns.scatterplot(data=durations, x='median_beep_time', y='expected_offset', palette='cool', hue='beep_duration', ax=ax[0], alpha=alp)
    sns.scatterplot(data=durations, x='median_beep_time', y='expected_beep_time', palette='PiYG', hue='beep_duration', ax=ax[1], alpha=alp)
    sns.scatterplot(data=durations, x='median_beep_time', y='fixation', palette='PiYG', hue='beep_duration', ax=ax[2], alpha=alp)
    ax[0].set(xlabel='median beep time (s)', ylabel='expected offset (ms)')
    ax[1].set(xlabel='median beep time (s)', ylabel='expected beep time (ms)')
    ax[2].set(xlabel='median beep time (s)', ylabel='fixation (ms)')
    ax[0].spines[['right', 'top']].set_visible(False); ax[1].spines[['right', 'top']].set_visible(False); ax[2].spines[['right', 'top']].set_visible(False)
    ax[0].legend(title='beep_duration', loc='lower right'); ax[1].legend(title='beep_duration', loc='lower right'); ax[2].legend(title='beep_duration', loc='lower right')
    fig.suptitle('Interaction Plots', fontsize=16)
    plt.tight_layout()
    plt.show()
    
# plot expected offset as a function of experiment version and median beep time
def plot_offset_expv_bt(durations):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    expvs = [1.0, 1.1, 6.0, 6.1, 4.1]
    colors = ['darkcyan', 'maroon', 'darkcyan', 'maroon', 'darkcyan']
    for i, expv in enumerate(expvs):
        dat = durations[durations['experiment_version'] == expv]
        sns.scatterplot(data=dat, x='median_beep_time', y='expected_offset', color=colors[i], ax=ax[i%2, i//2])
        ax[i%2, i//2].set(xlabel = 'median beep time (s)', ylabel='expected offset (ms)', title=f'Experiment Version = {expv}')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)
        if i%2 == 0:
            ax[i%2, i//2].set(xticks=np.arange(-0.1, 0.8, 0.1), yticks=np.arange(1000, 1021, 4))
        else:
            ax[i%2, i//2].set(xticks=np.arange(-0.1, 0.8, 0.1), yticks=np.arange(-0.05, 0.06, 0.02))

    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    plt.show()