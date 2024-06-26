### script to correct erroneous n_samples values in sources.json

# imports
import os
import json
import pandas as pd
import copy
from tqdm import tqdm

class SourcesJSONWriter:
    def __init__(self, subject, experiment, session, data, rhino_root, write_toggle):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.data = data
        self.rhino_root = rhino_root
        self.write_toggle = write_toggle
      
    # find processed folder that current_processed is linked to
    def locate_folder(self):
        curr_proc = f'{self.rhino_root}protocols/r1/subjects/{self.subject}/experiments/{self.experiment}/sessions/{self.session}/ephys/current_processed'
        proc_dir = os.path.join(os.path.dirname(curr_proc), os.readlink(curr_proc))
        return proc_dir
    
    # read current sources.json
    def read_sources(self):
        sources_json = os.path.join(self.proc_dir, 'sources.json')
        with open(sources_json, 'r') as f:
            sources = json.load(f)
            
        if len(self.data) != len(sources.keys()):
            raise RuntimeError(f"Number of EEG files ({len(self.data)}) in data does not match number of EEG files ({len(sources.keys())}) in sources.json.")
            
        return sources_json, sources
    
    # update sources.json
    def update_sources(self):
        new_sources = copy.deepcopy(self.sources)
        
        for key in new_sources.keys():
            dat = self.data[self.data['eegfile'] == key].iloc[0]    # data corresponding to eegfile
            val = new_sources[key]                                  # value = dictionary
            val['n_samples'] = int(dat.eeg_samples)                 # update n_samples
            new_sources[key] = val                                  # update sources
            
        return new_sources
            
    # write sources.json
    def write_sources(self):
        # rename previous sources.json to sources.json.bak
        os.rename(self.sources_json, os.path.join(self.proc_dir, 'sources.json.bak'))
        
        # write new sources.json
        with open(self.sources_json, 'w') as f:
            json.dump(self.new_sources, f, indent=2)
            
        return True
        
    
    def run(self):
        self.proc_dir = self.locate_folder()
        self.sources_json, self.sources = self.read_sources()
        self.new_sources = self.update_sources()
        
        if self.write_toggle:
            success = self.write_sources()
        

# read in results from sources.json validation
res = pd.read_csv('/home1/hherrema/programming_data/correct_sources_json/output_dir/sources_json_df.csv')

updates = []
for (sub, exp, sess), data in tqdm(res.groupby(['subject', 'experiment', 'session'])):
    # pass for sessions that don't require corrections
    if False not in data.close_samples.unique():
        continue
        
    try:
        sources_json_writer = SourcesJSONWriter(sub, exp, sess, data, '/', True)
        sources_json_writer.run()
        updates.append((sub, exp, sess, True))
    except BaseException as e:
        updates.append((sub, exp, sess, False))
        with open('/home1/hherrema/programming_data/correct_sources_json/output_dir/errors.txt', 'a') as f:
            f.write(f"{sub} {exp} {sess}: {e}\n")
            
updates = pd.DataFrame(updates, columns=['subject', 'experiment', 'session', 'success'])
updates.to_csv('/home1/hherrema/programming_data/correct_sources_json/output_dir/updates.csv', index=False)
        
        