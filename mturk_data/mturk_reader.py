### sqlite3 MTurk database
### utility functions for querying and loading data

# imports
import sqlite3
import pandas as pd
import json

class MTurk_sqlite3_reader:
    
    # initialize
    def __init__(self, table_name=None, db_root='/home1/maint/cmlpsiturk_files/cmlpsiturk_sqlite3.db'):
        self.table_name = table_name
        self.db_root = db_root
        
    # ---------- Connecting to Database ----------
    
    # create connection and cursor
    def _create_connection_cursor(self):
        connection = sqlite3.connect(self.db_root)
        cursor = connection.cursor()
        return connection, cursor
        
    # close connection --> call at end of every method
    def _close_connection(self, connection):
        connection.close()
        
    # ---------- Query & Load ----------
    
    # read all table names
    def all_tables(self):
        connection, cursor = self._create_connection_cursor()
        tables = [t for t in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]   # list of 1-tuples
        self._close_connection(connection)
        return tables
    
    # load table metadata into pandas dataframe
    def summary(self):
        if not self.table_name:
            raise ValueError("Initialize MTurk_sqlite3_reader with a table_name.")
        connection, cursor = self._create_connection_cursor()
        summary_df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", connection)
        self._close_connection(connection)
        summary_df = summary_df[['uniqueid', 'workerid', 'cond', 'counterbalance', 'beginhit', 
                                 'beginexp', 'endhit', 'status', 'mode', 'datastring']]       # select columns to keep
        summary_df = summary_df[summary_df['mode'] != 'debug']                # remove debug data
        summary_df = summary_df[summary_df['status'].isin([3,4,5,7])]         # only completed sessions
        self.summary_df = summary_df
        return summary_df
        
    # load events data into pandas dataframe
    def events(self):
        data = []
        for _, row in self.summary_df.iterrows():
            data.append(row['datastring'])
            
        subject_data = []
        for subject_json in data:
            try:
                subject_dict = json.loads(subject_json)
                subject_data.append(subject_dict['data'])
            except:
                continue
        
        trialdata = []
        for part in subject_data:
            for record in part:
                record['trialdata']['uniqueid'] = record['uniqueid']
                trialdata.append(record['trialdata'])
                
        events_df = pd.DataFrame(trialdata)    # put all trial data into pandas dataframe
        events_df['subject'] = events_df.uniqueid.astype('category').cat.codes
        events_df['subject'] = 'MTK' + self.table_name + '_' + events_df['subject'].astype(str)
        events_df.drop('view_history', axis=1, inplace=True)
        self.events_df = events_df
        return events_df