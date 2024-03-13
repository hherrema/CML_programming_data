### System Finder

# script attempts to determine system number for all sessions with NaN system version in data index
# Won't catch PSX sessions that have original experiement PS

# imports
import cmlreaders as cml
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import string
import math

# return sessions with NaN system version
def load_data_index_nan():
    df = cml.get_data_index('r1')
    return df[df.system_version.isna()]

def find_all_system_versions(df_nan):
    df_sysv = pd.DataFrame(columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'system_version'])
    for _, row in tqdm(df_nan.iterrows()):
        sysv = _determine_system_version(row)
        df_sysv = pd.concat([df_sysv, pd.DataFrame({'subject': row.subject,
                                                    'subject_alias': row.subject_alias,
                                                    'experiment': row.experiment, 
                                                    'original_experiment': row.original_experiment, 
                                                    'session': row.session, 
                                                    'original_session': row.original_session,
                                                    'system_version': sysv},
                                                    index = [len(df_sysv.index)])])
        
    # write out csv
    df_sysv.to_csv('/home1/hherrema/programming_data/BIDS_convert/df_sysv.csv', index=False)
    return df_sysv


def _determine_system_version(row):
    if row.subject_alias != row.subject:
        sub_root = f'/data10/RAM/subjects/{row.subject_alias}/'
    else:
        sub_root = f'/data10/RAM/subjects/{row.subject}/'
        
    if _system_4(row, sub_root):
        return 4.0
    elif _system_3(row, sub_root):
        return 3.0
    elif _system_2(row, sub_root):
        return 2.0
    else:
        return 1.0
    
    
def _system_4(row, sub_root):
    if '_' not in sub_root:
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.session}/'
    else:
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.original_session}/'
        
    if os.path.exists(sess_dir):
        return 'elemem' in os.listdir(sess_dir)
    else:
        return False

def _system_3(row, sub_root):
    if '_' not in sub_root:
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.session}/'
    else:
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.original_session}/'
        
    if not os.path.exists(sess_dir) and 'PS' in row.experiment:
        sess_dir = sub_root + f'behavioral/{row.original_experiment}/session_{row.original_session}/'     # try, but session number may be wrong
    elif not os.path.exists(sess_dir):
        sess_dir = sub_root + f'behavioral/{row.experiment}/session_{row.original_session}/'

    timestamped_directories = glob(sess_dir + 'host_pc/*')
    if len(timestamped_directories) == 0:
        return False
    
    # remove all invalid names (valid names = only contains numbers and _)
    timestamped_directories = [
        d for d in timestamped_directories 
        if os.path.isdir(d) and all([c in string.digits for c in os.path.basename(d).replace('_', '')])
    ]
    # check each timestamped directory for a .h5 file, stop if find one
    for d in timestamped_directories:
        if 'eeg_timeseries.h5' in os.listdir(d):
            return True

    return False

def _system_2(row, sub_root):
    if '_' not in sub_root:
        raw_dir = sub_root + f'raw/{row.experiment}_{row.session}/'
    else:
        raw_dir = sub_root + f'raw/{row.experiment}_{row.original_session}/'
        
    if not os.path.exists(raw_dir) and row.experiment == 'PS2.1':                              # remove '.' from PS2.1
        if math.isnan(float(row.original_session)) or row.session == row.original_session:
            raw_dir = sub_root + f'raw/PS21_{row.session}/'
        else:
            raw_dir = sub_root + f'raw/PS21_{row.original_session}/'
    elif not os.path.exists(raw_dir) and 'PS' in row.experiment:
        raw_dir = sub_root + f'raw/{row.original_experiment}_{row.original_session}/'          # try, but session number may be wrong
    elif not os.path.exists(raw_dir):
        raw_dir = sub_root + f'raw/{row.experiment}_{row.original_session}/'
        
    timestamped_directories = glob(raw_dir + '1*') + glob(raw_dir + '2*')    # some cases with 20 in year, others without
    if len(timestamped_directories) == 0:
        return False
    
    # remove all invalid names (valid names = only contains numbers and -)
    timestamped_directories = [
        d for d in timestamped_directories 
        if os.path.isdir(d) and all([c in string.digits for c in os.path.basename(d).replace('-', '')])
    ]
    # check each timestamped directory for a .ns2, stop if find one
    for d in timestamped_directories:
        if len(glob(d + '/*.ns2')) > 0:
            return True

    return False

df_nan = load_data_index_nan()
df_sysv = find_all_system_versions(df_nan)