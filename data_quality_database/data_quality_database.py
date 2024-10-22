### Class for reporting session-level data quality issues

# imports
import cmlreaders as cml
import pandas as pd
import warnings
import json

class data_quality_database:
    # database fields
    FIELDS = ['protocol', 'subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 'localization', 'montage', 
              'system_version', 'events', 'contacts', 'pairs', 'monopolar', 'bipolar', 'step', 'category', 'notes']
    TYPES = [str, str, str, str, str, int, (int, float), int, int, (int, float), bool, bool, bool, bool, bool, str, str, str]

    # data issue steps
    STEPS = {
        'collection': 'Error during data collection at hospital or in scalp lab.',
        'processing_imaging': 'Error during image processing (localization pipeline).',
        'processing_behavior': 'Error during event processing (event creation).',
        'processing_eeg': 'Error durring eeg processing (event creation).',
        'analysis': 'Error during data loading/analyses (cmlreaders).'
    }

    """
    # data issue categories
    CATEGORIES = {
        'sync_pulse_alignment': 'Questionable alignment via sync pulses.',
        'retrieval_offset_1000ms': 'Requires 1000 ms eegoffset correction for retrieval events.',
        'retrieval_offset_500ms': 'Requires 500 ms eegoffset correction for retrieval events.',
        'multiple_eegfiles': 'Recommend sorting multiple EEG files.',
        'other': 'Any other uncategorized issue.'
    }
    """

    # initialize
    def __init__(self, categories_path = 'output_dir/data_quality_categories.json', db_path='output_dir/data_quality_records.csv'):
        self.categories_path = categories_path
        self.db_path = db_path

        # data issue categories
        with open(self.categories_path, 'r') as f:
            self.CATEGORIES = json.load(f)

    
    # ---------- Utility ----------
    
    # return possible steps
    def possible_steps(self):
        return self.STEPS
    
    # return possible categories
    def possible_categories(self):
        return self.CATEGORIES
    
    # return dictionary structure for reporting issue
    def report_structure(self):
        return dict(zip(self.FIELDS, self.TYPES))


    # ---------- Query ----------
    
    # return dataframe with all known issues
    def all_records(self):
        return pd.read_csv(self.db_path)
    
    # query single session
    def query_session(self, subject, experiment, session, exc=True):
        records = self.all_records()
        sess_records = records[(records.subject == subject) &
                                (records.experiment == experiment) &
                                (records.session == session)]
        
        if len(sess_records) == 0 and exc:
            raise LookupError(f'{subject} {experiment} {session} not in records.')
        
        return sess_records

    # query single subject
    def query_subject(self, subject, exc=True):
        records = self.all_records()
        sub_records = records[records.subject == subject]

        if len(sub_records) == 0 and exc:
            raise LookupError(f'{subject} not in records.')
        
        return sub_records
    
    # ---------- Report ----------

    # validate step
    def _validate_step(self, step):
        if step not in self.STEPS.keys():
            raise ValueError(f"step must be one of {list(self.STEPS.keys())}")
    
    # validate category
    def _validate_category(self, category):
        if category not in self.CATEGORIES.keys():
            raise ValueError(f"category must be one of {list(self.CATEGORIES.keys())}. "
                             "Add a new category using the new_category method.")
        
    # add new category
    def new_category(self, category, description):
        if category in self.CATEGORIES.keys():
            raise ValueError(f"{category} already exists in possible categories")
        else:
            warnings.warn(f"Creating new data issue category {category}")
            self.CATEGORIES[category] = description
            with open(self.CATEGORIES, 'w') as f:
                json.dump(self.CATEGORIES, self.categories_path, indent=2)
    
    # report each field individually, check for type
    def _add_field_to_report(self, rs, k, typ, v):
        if type(v) == typ or type(v) in typ:
            rs[k] = v
        else:
            raise TypeError(f"{k} must have type {typ}")
        
        return rs
    
    # build report in correct structure
    def build_report(self, protocol, subject, subject_alias, experiment, original_experiment, session, original_session, localization, montage,
                     system_version, events, contacts, pairs, monopolar, bipolar, step, category, notes):
        
        # validate arguments
        self._validate_step(step)
        self._validate_category(category)
        
        report_list = list(zip(self.FIELDS, self.TYPES, [protocol, subject, subject_alias, experiment, original_experiment, session, original_session, localization, montage,
                                                         system_version, events, contacts, pairs, monopolar, bipolar, step, category, notes]))
        
        report_dict = {}
        for (k, t, v) in report_list:
            report_dict = self._add_field_to_report(report_dict, k, t, v)

        return report_dict
    
    # report issue
    def report(self, report_dict, force=False):
        sess_records = self.query_session(report_dict['subject'], report_dict['experiment'], report_dict['session'], exc=False)

        # new submission
        if len(sess_records) == 0:
            warnings.warn(f"Reporting issue for {report_dict['subject']}, "
                          f"{report_dict['experiment']}, {report_dict['session']}")
            row = self._update_database(report_dict)
            return row
        
        # repeat submission
        else:
            if not force:
                warnings.warn(f"{report_dict['subject']}, {report_dict['experiment']}, "
                              f"{report_dict['session']} already has {len(sess_records)} "
                              "reported issues.  If you are reporting a new issue, re-submit "
                              "your report with the argument: force=True")
                return sess_records
            else:
                warnings.warn(f"Reporting issue for {report_dict['subject']}, "
                          f"{report_dict['experiment']}, {report_dict['session']}")
            row = self._update_database(report_dict)
            return row


    # update database
    def _update_database(self, report_dict):
        records = self.all_records()
        row = pd.DataFrame(report_dict, index=[len(records)])
        records = pd.concat([records, row], ignore_index=True)

        # sort by subject, experiment, session
        records = records.sort_values(by=['subject', 'experiment', 'session'])

        # save out
        records.to_csv(self.db_path, index=False)

        return row
    
    
    # ---------- Remove ----------
    
    # remove data issue entry
    def remove_report(self, subject, experiment, session, step, category):
        raise NotImplementedError
    
    
    # ---------- Loading ----------
    
    # create CMLReader object
    def cml_reader(self, subject, experiment, session, localization, montage):
        return cml.CMLReader(subject, experiment, session, localization, montage)
        
    # load events
    def load_events(self, reader):
        try:
            evs = reader.load('events')
            return True, evs
        except BaseException:
            return False, None
        
    # load contacts
    def load_contacts(self, reader):
        try:
            contacts = reader.load('contacts')
            return True, contacts
        except BaseException:
            return False, None
        
    # load pairs
    def load_pairs(self, reader):
        try:
            pairs = reader.load('pairs')
            return True, pairs
        except BaseException:
            return False, None
        
    # load monopolar EEG
    def load_monopolar(self, reader, contacts):
        try:
            eeg_m = reader.load_eeg(scheme=contacts)
            return True, eeg_m
        except BaseException:
            return False, None
        
    # load bipolar EEG
    def load_bipolar(self, reader, pairs):
        try:
            eeg_b = reader.load_eeg(scheme=pairs)
            return True, eeg_b
        except BaseException:
            return False, None
