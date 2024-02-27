### Beep Detection Algorithm for UnityEPL Retrieval Offsets

# imports
import scipy.signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import math
from glob import glob
from tqdm import tqdm
import cmlreaders as cml

# class to determine beep times from audio files
class beep_detector:
    
    # initialize
    def __init__(self, subject, subject_alias, experiment, original_experiment, session, original_session):
        self.subject = subject
        self.subject_alias = subject_alias
        self.experiment = experiment
        self.original_experiment = original_experiment
        self.session = session
        self.original_session = original_session
        
        
    # deal with experiment and session changes
    def deal_with_changes(self, exp_change, sess_change):
        if exp_change:
            exp_match = self.original_experiment
        else:
            exp_match = self.experiment

        if sess_change:
            sess_match = int(self.original_session)   # always cast to int
        else:
            sess_match = self.session

        return exp_match, round(sess_match)

    # find .wav files and .ann files
    def locate_wav_ann_files(self):
        # determine if subject is re-implant or not
        re_implant = False
        # infer from the subject_alias
        if self.subject != self.subject_alias:
            re_implant = True

        # determine if original_experiment is different from experiment
        exp_change = False
        if type(self.original_experiment) == str and self.original_experiment != self.experiment:
            exp_change = True

        # determine if original_session is different from session
        sess_change = False
        # change str to int always, change value if non-matching
        if type(self.original_session) == str or (type(self.original_session)==int and self.original_session != self.session):
            sess_change = True

        exp_match, sess_match = self.deal_with_changes(exp_change, sess_change)    # deal with experiment and session changes
        if self.original_experiment == 'CatFR1':       # handle edge case
            exp_match = 'CatFR1'

        # if not a re-implants, use subject
        if not re_implant:
            behdir = f'/data10/RAM/subjects/{self.subject}/behavioral/{exp_match}/session_{sess_match}/'
        else:     # re-implant, use subject_alias
            behdir = f'/data10/RAM/subjects/{self.subject_alias}/behavioral/{exp_match}/session_{sess_match}/'

        all_wavs = glob(behdir + '*.wav')
        all_anns = glob(behdir + '*.ann')
        
        # only use first 6 trials
        wavs = []
        for w in all_wavs:
            trialw = w.split('/')[-1].split('.')[0]
            if len(trialw) <= 2 and int(trialw) in np.arange(6):
                wavs.append(w)
            
        anns = []
        for a in all_anns:
            triala = a.split('/')[-1].split('.')[0]
            if len(triala) <= 2 and int(triala) in np.arange(6):
                anns.append(a)
                
        # sort in ascending trial order (0-5)
        wavs.sort()
        anns.sort()
        
        if len(wavs) == 0 or len(anns) == 0:
            with open('errors.txt', 'a') as f:
                f.write(f'{self.subject}, {self.experiment}, {self.session}: Unable to locate wav or ann files.\n')
            raise RuntimeError("Unable to locate wav or ann files.")
        elif len(wavs) != len(anns):
            with open('errors.txt', 'a') as f:
                f.write(f'{self.subject}, {self.experiment}, {self.session}: Different number of wav and ann files.\n')
            
        return wavs, anns
    
    # load .wav file
    # returns samplerate and arry of audio values
    def load_wav_file(self, wav_file):
        samplerate, data = wavfile.read(wav_file)
        return samplerate, data
    
    # parse .ann file
    # returns ms of first vocalization/recall, wordpool index, and recalled word
    def parse_ann_file(self, ann_file):
        with open(ann_file, 'r') as f:
            toggle = False
            for x in f:
                split_line = x.split()
                if toggle and len(split_line) == 3:     # read line of first annotation
                    ms, idx, recall = split_line
                    return float(ms), int(idx), recall
                        
                
                if len(split_line) == 0:                # empty line below header and above annotations
                    toggle = True
                    
        raise RuntimeError(f"Could not parse .ann file {ann_file}.")
        
    # filtering
    def butter_bandpass_filter(self, data_norm, samplerate, low=775.0, high=825.0, order=1):
        b, a = scipy.signal.butter(order, [low, high], fs=samplerate, btype='bandpass')
        y = scipy.signal.lfilter(b, a, data_norm)
        return y
    
    # rolling rms
    def rolling_rms(self, signal, n=30):
        ret = np.cumsum(signal**2, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.sqrt(ret[n:] / n)
    
    # find contigious True regions of boolean array
    # returns 2D array, 1st column is start index of region, 2nd column is end index
    def contiguous_regions(self, condition):

        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero() 

        # We need to start things after the change in "condition". Therefore, 
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size] # Edit

        # Reshape the result into two columns
        idx.shape = (-1,2)
        return idx
    
    
    # calculate approximate duration of beep in seconds into audio file
    def calculate_beep_time(self, wav_file, ann_file, trial, start=0.1, vr_toggle=False):
        samplerate, data = self.load_wav_file(wav_file)
        ms, idx, recall = self.parse_ann_file(ann_file)
        if ms < 1000:                                          # first recall within 1 second
            vr_toggle = True
            ms = 1000   # use first second
            
        
        s = ms / 1000                                          # convert to second of first vocalization/recall
        vr_idx = math.floor(samplerate * s)                    # sample of first vocalization/recall, floor to int
        data_no_vr = data[:vr_idx]                             # select out audio data from before first vocalization/recall index
        data_norm = data_no_vr / np.max(np.abs(data_no_vr))    # normalize to [-1, 1]
        
        signal = self.butter_bandpass_filter(data_norm, samplerate)
        rms = self.rolling_rms(signal)
        
        thresh = max(0.25*np.max(rms), 0.05)
        beeps = self.contiguous_regions(rms > thresh)
        durations = [b[1] - b[0] for b in beeps if b[0] < start * samplerate]     # keep contiguous regions beginning within 100 ms of start of audio file
        
        if len(durations):
            return max(durations) / samplerate, vr_toggle
        else:
            return 0, vr_toggle
        
    # run over trials 1 to 5
    def run_trials(self, wavs, anns):
        tups = []
        # only run trials that have .wav and .ann file
        # loop over .wavs and run if corresponding .ann exists
        for wav_file in wavs:
            behdir = '/'.join(wav_file.split('/')[:-1]) + '/'
            trial = wav_file.split('/')[-1].split('.')[0]
            if behdir + trial + '.ann' in anns:
                ann_file = behdir + trial + '.ann'
                try:
                    bt, vr_toggle = self.calculate_beep_time(wav_file, ann_file, trial)
                    tups.append((self.subject, self.subject_alias, self.experiment, self.original_experiment, self.session, self.original_session, trial, bt, vr_toggle))
                except BaseException as e:
                    with open('errors.txt', 'a') as f:
                        f.write(f'{self.subject}, {self.experiment}, {self.session}, {trial}: {e}\n')
                    continue
                
            
        res = pd.DataFrame(tups, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'trial', 'beep_time', 'vr_s1'])
        return res
    

# ---------- Finding UnityEPL Sessions ----------

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


# get all sessions between subjects s1 and s2 (inclusive)
def select_subject_range(s1, s2):
    
    df = cml.get_data_index('r1')
    mask = []
    for idx, row in tqdm(df.iterrows()):
        if row.subject.startswith('R1'):
            # increase to R1528E
            #if (row.subject >= 'R1373T' and row.subject < 'R1525J') or row.subject == 'R1347D' or row.subject == 'R1367D' or (row.subject == 'R1366J' and row.experiment == 'catFR1'):
            if (row.subject >= s1 and row.subject <= s2) or row.subject == 'R1347D' or row.subject == 'R1367D' or (row.subject == 'R1366J' and row.experiment == 'catFR1'):
                mask.append(idx)

    check_df = df.iloc[mask]
    return check_df

# determine UnityEPL v. pyEPL sessions
def unity_or_py(check_df):
    info = []
    #for _, row in tqdm(check_df.query("experiment == 'catFR1' | experiment == 'FR1'").iterrows()):
    for _, row in tqdm(check_df.iterrows()):
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

        slog = glob(behdir + 'session.log')
        sjson = glob(behdir + 'session.json*')

        info.append((row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, len(slog), len(sjson)))

    info = pd.DataFrame(info, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'session_log', 'session_json'])
    return info
    
    
# ---------- Running Algorithm ----------
    
# run beep detection algorithm over all unityEPL sessions
def run_beep_detector(py_unity_info):
    all_results = pd.DataFrame(columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'trial', 'beep_time'])
    for _, row in tqdm(py_unity_info.iterrows()):
        if row.session_json > 0:
            try:
                bd = beep_detector(row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session)
                wavs, anns = bd.locate_wav_ann_files()
                res = bd.run_trials(wavs, anns)
                all_results = pd.concat([all_results, res])
            except BaseException as e:
                continue
                
    all_results = all_results.fillna('X')      # NaN values cause problems for groupby
    return all_results

# collapse across trials and calculate median beep time
def avg_trials(all_results):
    final_res = []
    for (sub, sub_al, exp, orig_exp, sess, orig_sess), dat in all_results.groupby(by=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session']):
        ##use_rows = dat[dat['vr_s1'] == False]
        final_res.append((sub, sub_al, exp, orig_exp, sess, orig_sess, np.median(dat['beep_time'])))

    final_res = pd.DataFrame(final_res, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'median_beep_time'])
    return final_res


# save out data index of dubious subjects' sessions
check_df = select_subject_range('R1373T', 'R1525J')
check_df.to_csv('subj_range_check.csv', index=False)

# save out data frame with session log, session json information
info = unity_or_py(check_df)
info.to_csv('py_unity_info.csv', index=False)


# load in dataframes from `eegoffset_workspace.ipynb`
subj_range_check = pd.read_csv('subj_range_check.csv')
py_unity_info = pd.read_csv('py_unity_info.csv')

# results for every trial --> also writes to errors.txt
all_results = run_beep_detector(py_unity_info)
all_results.to_csv('all_results.csv', index=False)

# median beep time for each session
final_res = avg_trials(all_results)
final_res.to_csv('final_res.csv', index=False)