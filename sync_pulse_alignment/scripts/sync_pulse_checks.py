# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
import json
import re
import math
from glob import glob
from sklearn.metrics import mean_squared_error
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from tqdm import tqdm
from find_bad_sync_pulse_alignment import AlignmentCheck
from sync_pulse_align import sync_pulse_aligner
from sync_pulse_cml import sync_pulse_aligner_cml
import sys

def run_HH_checks(sps):
    n_sess = 0
    df = pd.DataFrame(columns=['subject', 'experiment', 'session', 'slope', 'intercept', 'rmse', 
                              'mu', 'sig', 'se', 'n_syncs'])
    for _, row in tqdm(sps.iterrows()):
        # pass if multiple '_' or if lowercase letters
        if row['subject'].count('_') > 1 or any(c for c in row['subject'] if c.islower()):
            continue
        else:
            try:
                spa = sync_pulse_aligner(row.subject, row.experiment, row.session, row.sync_txt)
                spa.run_align()
                spa.run_QC()
                attrs = spa.get_attrs()
                
                # save out
                #with open(f'sync_pulse_HH/{row.subject}_{row.experiment}_{row.session}_spa.json', 'w') as f:
                #    json.dump(fp=f, obj=attrs)
                    
                n_sess += 1
                df = pd.concat([df, pd.DataFrame(attrs, index=[len(df)])])
            except BaseException as e:
                with open('sync_pulse_HH/errors.txt', 'a') as f:
                    f.write(f'{row.subject}, {row.experiment}, {row.session}: Error = {e}\n')
                continue
    
    # write out dataframe as csv
    df.to_csv('sync_pulse_HH/sync_pulse_checks_HH_results.csv', index=False)
    
    return n_sess


# I think Joey's code may fail for re_implants with '_' in subject name
def run_JR_checks(sps):
    n_sess = 0
    df = pd.DataFrame(columns=['subject', 'experiment', 'session', 'slope', 'intercept', 'rmse', 
                              'mu', 'sig', 'se', 'n_syncs', 'corr1', 'corr2'])
    for _, row in tqdm(sps.iterrows()):
        # pass if multiple '_' or if lowercase letters
        if row['subject'].count('_') > 1 or any(c for c in row['subject'] if c.islower()):
            continue
        else:
            try:
                ac = AlignmentCheck(row.subject, row.experiment, row.session)
                ac.check(plot=False)
                slope, intercept = ac.__dict__['fit']
                rmse = ac.__dict__['rmse']
                residuals = ac.__dict__['resid']
                mu = np.mean(residuals)
                sig = np.std(residuals)
                se = sig / np.sqrt(len(residuals))
                n_syncs = len(residuals)
                correlations = np.sort(ac.__dict__['correlations'])[::-1]
                corr1, corr2 = correlations[:2]
                
                attrs = {
                    'subject': row.subject,
                    'experiment': row.experiment,
                    'session': row.session,
                    'slope': slope,
                    'intercept': intercept,
                    'rmse': rmse,
                    'mu': mu,
                    'sig': sig,
                    'se': se,
                    'n_syncs': n_syncs,
                    'corr1': corr1,
                    'corr2': corr2
                }
                
                # save out
                #with open(f'sync_pulse_JR/{row.subject}_{row.experiment}_{row.session}_ac.json', 'w') as f:
                #    json.dump(fp=f, obj=attrs)
                    
                n_sess += 1
                df = pd.concat([df, pd.DataFrame(attrs, index=[len(df)])])    
            except BaseException as e:
                with open('sync_pulse_JR/errors.txt', 'a') as f:
                    f.write(f'{row.subject}, {row.experiment}, {row.session}: Error = {e}\n')
                continue
    
    # write out dataframe as csv
    df.to_csv('sync_pulse_JR/sync_pulse_checks_JR_results.csv', index=False)
    
    return n_sess

def run_CML_checks(sys1):
    n_sess = 0
    df = pd.DataFrame(columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session',
                               'slope', 'intercept', 'rmse', 'mu', 'sig', 'se', 'n_syncs'])
    for _, row in tqdm(sys1.iterrows()):
        try:
            spa = sync_pulse_aligner_cml(row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, row.sync_txt)
            spa.run_align()
            spa.run_QC()
            attrs = spa.get_attrs()

            n_sess += 1
            df = pd.concat([df, pd.DataFrame(attrs, index=[len(df)])])
        except BaseException as e:
            with open('sync_pulse_CML/errors.txt', 'a') as f:
                f.write(f'{row.subject} | {row.subject_alias} | {row.experiment} | {row.original_experiment} | {row.session} | {row.original_session} | Error = {e}\n')
            continue
            
    # write out dataframe as csv
    df.to_csv('sync_pulse_CML/sync_pulse_checks_CML_results.csv', index=False)
    
    return n_sess
        

sync_pulse_sessions = pd.read_csv('sync_pulse_sessions.csv')
sys1 = pd.read_csv('sync_pulse_CML/valid_sync_files.csv')
toggle = int(sys.argv[1])       # 0 = run my code, 1 = run Joey's code, 2 = run over cmlreaders data index

if toggle == 0:
    n_sess_success = run_HH_checks(sync_pulse_sessions)
elif toggle == 1:
    n_sess_success = run_JR_checks(sync_pulse_sessions)
elif toggle == 2:
    n_sess_success = run_CML_checks(sys1)
else:
    raise RuntimeError("toggle 0 for my code, 1 for Joey's code, 2 to run over cmlreaders data index.")
    
print(f'Succesfully ran sync pulse alignment checks on {n_sess_success} sessions.')