### functions used to validate sources.json data
### used in validate_sources_json.ipynb

# ---------- Data Validation ----------

# handle sessions with multiple EEG files
def handle_multiple_eegfiles(sub, exp, sess, sources_emr):
    noreref = f'/protocols/r1/subjects/{sub}/experiments/{exp}/sessions/{sess}/ephys/current_processed/noreref/'
    eeg_samples = []
    eeg_files = []
    
    for i, fname in enumerate(sources_emr['name']):
        eeg_files.append(fname)
        
        # handles files with no samples
        if isinstance(sources_emr['n_samples'], np.ndarray) and sources_emr['n_samples'][i] == 0:
            eeg_samples.append(0)
            continue
        
        # h5 file
        if 'h5' in fname:
            fpath = os.path.join(noreref, fname)
            with h5py.File(fpath, 'r') as hfile:
                ts = hfile['/timeseries']
                if 'orient' in ts.attrs.keys() and ts.attrs['orient'] == b'row':
                    eeg_samples.append(ts.shape[0])
                else:
                    eeg_samples.append(ts.shape[1])
        
        # split EEG file
        else:
            pattern = os.path.join(noreref, fname)
            split_files = glob(f'{pattern}.*')
            split_file = split_files[0]     # only need 1 file
            memmap = np.memmap(split_file, mode='r')
            eeg_samples.append(len(memmap))
               
    return eeg_samples, eeg_files


# load non-epoched EEG to extract n_samples, samplerate
def load_eeg_samples(reader):
    eeg = reader.load_eeg()
    return eeg.data.shape[-1], eeg.samplerate


# run validation checks for single sessions
def validate_sources_1_sess(sub, exp, sess, loc, mont, sysv):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    pf = cml.PathFinder(sub, exp, sess, loc, mont)
    sources_emr = cml.EEGMetaReader.fromfile(pf.find('sources'), subject=sub)
    
    multiple_eegfiles = False
    idx = 1
    if isinstance(sources_emr['n_samples'], np.ndarray) or isinstance(sources_emr['sample_rate'], np.ndarray) or isinstance(sources_emr['name'], np.ndarray):
        multiple_eegfiles = True
        idx = len(sources_emr['name'])
        
    if not multiple_eegfiles:
        eeg_samples, eeg_samplerate = load_eeg_samples(reader)
        eeg_files = sources_emr['name']
    else:
        eeg_samples, eeg_files = handle_multiple_eegfiles(sub, exp, sess, sources_emr)
        eeg_samplerate = sources_emr['sample_rate']        # in all cases sample rate is same for multiple files
        
    print(eeg_files, eeg_samples)
        
    return pd.DataFrame({'subject': sub, 'experiment': exp, 'session': sess, 'localization': loc, 'montage': mont, 'system_version': sysv, 
                         'multiple_eegfiles': multiple_eegfiles, 'eegfile': eeg_files, 'sources_samples': sources_emr['n_samples'], 'eeg_samples': eeg_samples,
                         'sources_samplerate': sources_emr['sample_rate'], 'eeg_samplerate': eeg_samplerate}, index=[x for x in range(idx)])

# ---------- Post-Processing ----------

# add equal and close labels (allow for buffer = 1)
def compare_samples(row, buffer):
    equal = False
    close = False
    
    if row.sources_samples == row.eeg_samples:
        equal = True
        close = True
    elif math.isclose(row.sources_samples, row.eeg_samples, abs_tol=buffer):
        close = True
            
    return equal, close

def apply_samples_comparisons(res):
    equal_list = []
    close_list = []
    for _, row in tqdm(res.iterrows()):
        equal, close = compare_samples(row, 1)
        equal_list.append(equal)
        close_list.append(close)

    res['equal_samples'] = equal_list
    res['close_samples'] = close_list
    
    return res


# check for mismatch by a factor of 2
def check_factor_2(res):
    factor_2_list = []
    for _, row in res.iterrows():
        factor_2 = False
        if math.isclose(row.sources_samples/2, row.eeg_samples, rel_tol=1E-4) or math.isclose(row.sources_samples, row.eeg_samples/2, rel_tol=1E-4):
            factor_2 = True

        factor_2_list.append(factor_2)

    res['factor_2'] = factor_2_list
    
    return res