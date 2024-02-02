### functions used for system 1 sync pulse alignment work

# ---------- sync.txt ----------
### code for finding sync.txt files for system 1 sessions

# easily access needed data from the data index
def get_data_index_metadata(row):
    return {'subject': row.subject, 'subject_alias': row.subject_alias, 
            'experiment': row.experiment, 'original_experiment': row.original_experiment, 
            'session': row.session, 'original_session': row.original_session}

# deal with experiment and session changes
def deal_with_changes(exp_change, sess_change, **kwargs):
    if exp_change:
        exp_match = kwargs['original_experiment']
    else:
        exp_match = kwargs['experiment']
        
    if sess_change:
        sess_match = kwargs['original_session']
    else:
        sess_match = kwargs['session']
        
    return exp_match, round(sess_match)

# standard nomenclature is exp_sess.sync.txt
def standard_exp_sess(st, exp_change, sess_change, **kwargs):
    exp_match, sess_match = deal_with_changes(exp_change, sess_change, **kwargs)     # deal with experiment and session changes
    
    # validate sync.txt
    exp_sess = st.split('.')[-3]
    if '_' in exp_sess and len(exp_sess.split('_')) == 2:
        exp, sess = exp_sess.split('_')
        if exp == exp_match and sess == str(sess_match):
            return True, exp, sess
                
    return False, None, None

# experiment and session follow 'subject_'
def subject_exp_sess(st, re_implant, exp_change, sess_change, **kwargs):
    exp_match, sess_match = deal_with_changes(exp_change, sess_change, **kwargs)     # deal with experiment and session changes
    
    # filename without directories
    fname = st.split('/')[-1]
    # validate sync.txt
    if not re_implant:
        if len(fname.split('_')) > 2:
            exp, sess = fname.split('_')[1:3]
            if exp == exp_match and sess == str(sess_match):
                return True, exp, sess
    else:
        if len(fname.split('_')) > 3:
            exp, sess = fname.split('_')[2:4]
            if exp == exp_match and sess == str(sess_match):
                return True, exp, sess
    return False, None, None

def way3(st, exp_change, sess_change, **kwargs):
    exp_match, sess_match = deal_with_changes(exp_change, sess_change, **kwargs)     # deal with experiment and session changes
    
    fname = st.split('/')[-1]
    exp, sess = fname.split('.')[:2]
    #print(exp, sess)
    if exp == exp_match and sess == str(sess_match):
        return True, exp, sess
    
    return False, None, None

def dboy(st, exp_change, sess_change, **kwargs):
    exp_match, sess_match = deal_with_changes(exp_change, sess_change, **kwargs)     # deal with experiment and session changes
    
    # filename without directories
    fname = st.split('/')[-1]
    exp_sess = fname.split('.')[0]
    if '_' in exp_sess and len(exp_sess.split('_')) == 2:
        exp, sess = exp_sess.split('_')
        if exp == exp_match and sess == str(sess_match):
            return True, exp, sess
               
    return False, None, None

def pyFR(st, **kwargs):
    pyfr = st.split('.')[-3]
    if pyfr == 'pyFR':
        return True

def find_sync_txt(**kwargs):
    #print(kwargs['subject_alias'])
    # determine if subject is re-implant or not
    re_implant = False
    # infer from the subject_alias --> make sure this is not NaN
    if type(kwargs['subject_alias']) == str and kwargs['subject'] != kwargs['subject_alias'] and '_' in kwargs['subject_alias']:
        re_implant = True
        
    # determine if original_experiment is different from experiment
    exp_change = False
    if type(kwargs['original_experiment']) == str and kwargs['original_experiment'] != kwargs['experiment']:
        exp_change = True
    
    # determine if orignal_session is different from session
    sess_change = False
    if not math.isnan(kwargs['original_session']) and kwargs['original_session'] != kwargs['session']:
        sess_change = True
    
    # not a re-implant, use subject
    if not re_implant:
        sync_files = glob(f"/data/eeg/{kwargs['subject']}/eeg.noreref/*sync.txt")
    else:  # re-implant, use subject_alias
        sync_files = glob(f"/data/eeg/{kwargs['subject_alias']}/eeg.noreref/*sync.txt")
        
    for st in sync_files:
        valid, exp, sess = standard_exp_sess(st, exp_change, sess_change, **kwargs)
        if valid:
            return True, st
        valid, exp, sess = subject_exp_sess(st, re_implant, exp_change, sess_change, **kwargs)
        if valid:
            return True, st
        valid, exp, sess = dboy(st, exp_change, sess_change, **kwargs)
        if valid:
            return True, st
    
    return False, None


# ---------- Check Individual Sessions ----------
### wrapper functions to run and visualize my checks and Joey's checks

# my code over sessions on data index
# vsf = valid_sync_files.csv
def run_CML_sess(sub, exp, sess, vsf):
    # find sync pulse text file
    row = vsf[(vsf.subject==sub) & (vsf.experiment==exp) & (vsf.session==sess)].iloc[0]
    spa = sync_pulse_aligner_cml(row.subject, row.subject_alias, row.experiment, row.original_experiment, row.session, row.original_session, row.sync_txt)
    spa.run_align()
    spa.run_QC()
    spa.plot_results()
    
# my code over found sync.txt files on rhino
# sps = sync_pulse_sessions.csv
def run_HH_sess(sub, exp, sess, sps):
    # find sync pulse text file
    sync_txt = sps[(sps.subject==sub) & (sps.experiment==exp) & (sps.session==str(sess))].iloc[0].sync_txt
    spa = sync_pulse_aligner(sub, exp, sess, sync_txt)
    spa.run_align()
    spa.run_QC()
    spa.plot_results()
    
# Joey's code
def run_JR_sess(sub, exp, sess):
    ac = AlignmentCheck(sub, exp, sess)
    ac.check(plot=True)
    
# ---------- Load Errors from Alignment Checks ----------
### parse the errors.txt file written during alignment checks

# for my code run over sessions on data index
def load_errors():
    err_tups = []
    unreadable = []
    err_path_cml = 'sync_pulse_CML/errors.txt'
    with open(err_path_cml, 'r') as f:
        for line in f:
            split_line = line.split('|')
            sub = split_line[0].rstrip()
            sub_al = split_line[1].strip()
            exp = split_line[2].strip()
            orig_exp = split_line[3].strip()
            sess = split_line[4].strip()
            orig_sess = split_line[5].strip()
            err = split_line[6][9:-1].replace('.', '')
            err_tups.append((sub, sub_al, exp, orig_exp, sess, orig_sess, err))
            
    err_df = pd.DataFrame(err_tups, columns=['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session' ,'error'])
    return err_df
    
# ---------- YellowCab Multiple Gaps ----------

# generate dataframe of YC1, YC2 sessions with sync pulse time gaps at index 99
def yellow_cab_gaps(valid_sync_files):
    gap_diffs = []

    for _, row in tqdm(valid_sync_files.query("experiment in ['YC1', 'YC2']").iterrows()):
        try:
            vsf = valid_sync_files[(valid_sync_files.subject==row.subject) &
                                        (valid_sync_files.experiment==row.experiment) &
                                        (valid_sync_files.session==int(row.session))].iloc[0]
            spa = sync_pulse_aligner_cml(vsf.subject, vsf.subject_alias, vsf.experiment, vsf.original_experiment, 
                                         vsf.session, vsf.original_session, vsf.sync_txt)
            spa.run_align()
            bsd = np.diff(spa.matching_beh_syncs)
            esd = np.diff(spa.matching_eeg_syncs)
            no_gap_idx = np.intersect1d(np.where(bsd < 5E3), np.where(esd < 5E3))
            gap_idx = np.intersect1d(np.where(bsd >= 5E3), np.where(esd >= 5E3))

            if 99 in gap_idx:
                gap_diffs.append((True, esd[99]))
            else:
                gap_diffs.append((False, esd[99]))
        except BaseException as e:
            continue

    return pd.DataFrame(gap_diffs, columns=['idx_99', 'ispt'])

# ---------- Original Attempt (parse rhino for sync.txt files) ----------

# generate lists of sync.txt files and subjects
def find_sync_txts(root='/data/eeg/'):
    tot_sync = []
    sync_subs = []
    for d in os.listdir(root):
        if os.path.isdir(root+d) and d[0].isupper() and re.search(r'\d', d):     # only subject directories
            eeg_path = os.path.join(root, d, 'eeg.noreref')
            if os.path.isdir(eeg_path):
                sync_files = glob(eeg_path + '/*.sync.txt')
                if len(sync_files) > 0:
                    tot_sync.extend(sync_files)
                    if d not in sync_subs:
                        sync_subs.append(d)
                        
    return np.array(sync_subs), np.array(tot_sync)

# deduce if sync.txt is from an experimental session

# experiment and session preceed ".sync.txt"
def way1(fname):
    exp_sess = fname.split('.')[-3]
    if '_' in exp_sess:              # experimental file
        if len(exp_sess.split('_')) == 2:
            exp, sess = exp_sess.split('_')
            # should be letters in the exp, numbers in the sess
            if re.search('[a-zA-Z]', exp) and re.search(r'\d', sess):
                return True, exp, sess
    
    return False, None, None

# experiment and session follow "subject_"
def way2(fname, cml_exps):
    if len(fname.split('_')) > 2:
        exp, sess = fname.split('_')[1:3]
        if re.search('[a-zA-Z]', exp) and re.search(r'\d', sess) and exp in cml_exps + ['train', 'trackball']:
                return True, exp, sess
        
    return False, None, None

# a few sessions
def way3(fname, cml_exps):
    exp, sess = fname.split('.')[:2]
    if re.search('[a-zA-Z]', exp) and re.search(r'\d', sess) and exp in cml_exps:
        return True, exp, sess
    
    return False, None, None

def exp_sync_txts(tot_sync):
    cml_exps = list(cml.get_data_index().experiment.unique())    # list of experiments on cmlreaders
    exp_syncs = []; non_exp_syncs = []
    for stxt in tot_sync:
        sub = stxt.split('/')[3]             # subject at start
        fname = stxt.split('/')[-1]          # sync_txt filename at end
        toggle = False                       # toggle to True if found experimental sync.txt
        
        # try way 1
        toggle, exp, sess = way1(fname)
        if toggle:
            exp_syncs.append((sub, exp, sess, stxt))
            continue
            
        # try way 2
        toggle, exp, sess = way2(fname, cml_exps)
        if toggle:
            exp_syncs.append((sub, exp, sess, stxt))
            continue
            
        # try way 3
        toggle, exp, sess = way3(fname, cml_exps)
        if toggle:
            exp_syncs.append((sub, exp, sess, stxt))
            continue
            
        if not toggle:    # should always be False if reached
            non_exp_syncs.append(stxt)
            
    return pd.DataFrame(exp_syncs, columns=['subject', 'experiment', 'session', 'sync_txt']), np.array(non_exp_syncs)