### Class for reporting session-level data quality issues

# imports
import cmlreaders as cml
import pandas as pd
import warnings

class data_quality_database:
    # database fields
    FIELDS = ['subject', 'subject_alias', 'experiment', 'original_experiment', 'session', 'original_session', 
              'localization', 'montage', 'on_cmlreaders', 'category', 'notes']
    TYPES = [str, str, str, str, int, int, int, int, bool, str, str]

    # data issue categories
    CATEGORIES = {
        'sync_pulse_alignment': 'Questionable alignment via sync pulses.',
        'retrieval_offset_1000ms': 'Requires 1000 ms eegoffset correction for retrieval events.',
        'retrieval_offset_500ms': 'Requires 500 ms eegoffset correction for retrieval events.',
        'multiple_eegfiles': 'Recommend sorting multiple EEG files.',
        'other': 'Any other uncategorized issue.'
    }

    # initialize
    def __init__(self, path='output_dir/data_quality_records.csv'):
        self.path = path

    
    # ---------- Query ----------
    
    # return dataframe with all known issues
    def all_records(self):
        return pd.read_csv(self.path)
    
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
    def query_subject(self, subject):
        records = self.all_records()
        sub_records = records[records.subject == subject]

        if len(sub_records) == 0:
            raise LookupError(f'{subject} not in records.')
        
        return sub_records
    
    # ---------- Report ----------
    
    # return possible categories
    def possible_categories(self):
        return self.CATEGORIES
    
    # return dictionary structure for reporting issue
    def report_structure(self):
        return dict(zip(self.FIELDS, self.TYPES))
    
    # build report in correct structure
    def build_report(self, subject, subject_alias, experiment, orignal_experiment, session,
                     original_session, localization, montage, on_cmlreaders, category, notes):
        rs = self.report_structure()

        # subject
        if type(subject) == str:
            rs['subject'] = subject
        else:
            raise TypeError("'subject' must have type str")
        
        # subject_alias
        if type(subject_alias) == str:
            rs['subject_alias'] = subject_alias
        else:
            raise TypeError("'subject_alias' must have type str")
        
        # experiment
        if type(experiment) == str:
            rs['experiment'] = experiment
        else:
            raise TypeError("'experiment' must have type str")
        
        # original experiment

        # session
        if type(session) == int:
            rs['session'] = session
        else:
            raise TypeError("'session' must have type int")
        
        # original session

        # localization
        
    # type checks
    def _type_check_report_build(self, rs, key, val, typ):
        if type(val) == typ:
            rs[key] = val
            return rs
        else:
            raise TypeError(f"{key} must have type {typ}")
    
    # report issue
    def report(self, kwargs, force=False):
        valid_kwargs = self._validate_kwargs(kwargs)

        sess_records = self.query_session(valid_kwargs['subject'],
                                          valid_kwargs['experiment'],
                                          valid_kwargs['session'],
                                          exc=False)
        # new submission
        if len(sess_records) == 0:
            warnings.warn(f"Reporting problem for {valid_kwargs['subject']}, "
                          f"{valid_kwargs['experiment']}, {valid_kwargs['session']}")
            row = self._update_database(valid_kwargs)
            return row
        
        # repeat submission
        else:
            if not force:
                warnings.warn(f"{valid_kwargs['subject']}, {valid_kwargs['experiment']}, "
                              f"{valid_kwargs['session']} already has {len(sess_records)} "
                              "reported issues.  If you are reporting a new issue, re-submit "
                              "your report with the argument: force=True")
                return sess_records
            else:
                warnings.warn(f"Reporting problem for {valid_kwargs['subject']}, "
                              f"{valid_kwargs['experiment']}, {valid_kwargs['session']}")
                row = self._update_database(valid_kwargs)
                return row

    # validate structure of keyword arguments
    def _validate_kwargs(self, kwargs):
        # remove keys that are not valid fields
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k in self.FIELDS}
        removed_keys = list(set(kwargs.keys()) - set(cleaned_kwargs.keys()))
        if len(removed_keys) > 0:
            warnings.warn(f'Removing invalid arguments: {removed_keys}')

        # required fields not in keys
        missing_keys = [x for x in self.FIELDS if x not in cleaned_kwargs.keys()]
        if len(missing_keys) > 0:
            raise ValueError(f'Required fields not in input: {missing_keys}')
        
         # values with wrong types
        correct_structure = self.report_structure()
        wrong_types = [(k, type(v), correct_structure[k]) for k, v in cleaned_kwargs.items() 
                       if type(v) != correct_structure[k]]
        if len(wrong_types) > 0:
            raise TypeError(f'Following fields have wrong types (field, input type, expected type): {wrong_types}')
        
        # category in excepted categories
        if cleaned_kwargs['category'] not in self.CATEGORIES:
            raise ValueError(f"'category' field must be one of {list(self.CATEGORIES.keys())}")
        
        return cleaned_kwargs

    # update database
    def _update_database(self, valid_kwargs):
        records = self.all_records()
        row = pd.DataFrame(valid_kwargs, index=[len(records)])
        records = pd.concat([records, row])

        # sort by subject, experiment, session
        records = records.sort_values(by=['subject', 'experiment', 'session'])

        # save out
        records.to_csv(self.path, index=False)

        return row
