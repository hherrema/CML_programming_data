import json
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from sklearn.metrics import mean_squared_error
class AlignmentCheck:
    """
    Class for analyzing sync pulse alignments in System 1 data.
    """
    def __init__(self, subject, experiment, session, sync_send=None, syncs=None):
        if (syncs is not None) and (sync_send is not None):
            # override loading with manual syncs
            self.syncs = syncs
            self.sync_send = sync_send
            return
        self.subject = subject
        self.experiment = experiment
        self.session = session
        if glob(f"/data/eeg/{self.subject}*/raw/*{self.experiment}_{self.session}/*.log"):
            raise ValueError("Subject was run on System 2")
        files = glob(f"/data/eeg/{self.subject}*/eeg.noreref/*{self.experiment}_{self.session}*.sync.txt")
        print(f"/data/eeg/{self.subject}/eeg.noreref/*{self.experiment}_{self.session}*.sync.txt")
        if len(files)==0:
            raise FileNotFoundError("No sync pulse file found")
        if len(files)>1:
            raise ValueError("More than one matching sync file found!")
        self.sync_file = files[0]
        sess_dir = f"/data/eeg/{self.subject}*/behavioral/{self.experiment}/session_{self.session}/"
        files = glob(sess_dir + "session.json*")
        files += glob(sess_dir + "eeg.eeglog")
        if len(files)==0:
            raise FileNotFoundError("No log file found")
        if len(files)>1:
            warnings.warn(f"More than one matching log file found!{files}")
        self.log_file = files[0]
        with open(f"/protocols/r1/subjects/{self.subject}/experiments/{self.experiment}/sessions/{self.session}/ephys/current_processed/sources.json", "r") as f:
            eeg_sources = json.load(f)
        eeg_source = list(eeg_sources.values())[0]
        self.sample_rate = eeg_source["sample_rate"]
        self.load_rhino_data()
        
    def load_rhino_data(self):
        self.syncs = self.get_sync_pulse_diffs()
        self.sync_send = self.get_logged_pulse_diffs()
        
    def get_logged_pulse_diffs(self):
        # using older eeg logs (pre-unity)
        if self.log_file.count("json")>0:
            events = pd.read_json(self.log_file, lines=True)
            sync_events = events.query("type == 'syncPulse'")
            sync_send = np.diff(sync_events["time"].values)
        else:
            split_lines = [line.split() for line in open(self.log_file).readlines()]
            sync_send = np.diff([float(line[0]) for line in split_lines if line[2] in ('CHANNEL_0_UP', 'ON')])
        sync_send = sync_send[sync_send>30]
        return sync_send
    
    def get_sync_pulse_diffs(self):
        syncs = np.loadtxt(self.sync_file)
        syncs = syncs * 1000. / self.sample_rate
        syncs = np.diff(syncs)
        # exclude annotations from the same pulse (buggy pulse extraction)
        syncs = syncs[np.abs(syncs)>50][:100]
        # shrink window to match if there is a gap in the syncs
        while not np.all(np.abs(syncs)<5e3):
            syncs = syncs[:-5]
            if len(syncs)<25:
                raise ValueError("Alignment window too small")
        if not np.all(syncs>0):
            raise ValueError(f"Extracted sync pulses from {self.sync_file} are not monotonically increasing.")
        return syncs
    
    def highest_corr(self, x, y):
        if len(x)==len(y):
            return 0, np.corrcoeff(x, y)
        elif len(x) > len(y):
            windows = sliding_window_view(x, len(y))
            test = y
        else:
            windows = sliding_window_view(y, len(x))
            test = x
        correlations = []
        for i in tqdm(range(len(windows))):
            corr = np.corrcoef(windows[i, :], test)[0, 1]
            correlations.append(corr)
        return np.argmax(correlations), np.max(correlations), correlations
    
    def plot(self, save=False):
#         ax = sns.histplot(x=self.correlations, kde=True)
#         sw_stat, sw_sig = scipy.stats.shapiro(self.correlations)
#         kurt = scipy.stats.kurtosis(self.correlations)
#         ax.annotate(f"Kurtosis: {kurt:.2f}\nShapiro-Wilk:\np={sw_sig:.1e}",
#                      xy=(.6, .7), xycoords="axes fraction", fontsize=14)
#         ax.axvline(self.correlations[self.match_idx], color='r')
        ax = sns.jointplot(x=self.sync_send[self.match_idx:self.match_idx+len(self.syncs)], y=self.syncs, kind='reg')
        ax.ax_joint.set_xlabel('Inter-Pulse-Time sent', fontsize=14)
        ax.ax_joint.set_ylabel('Inter-Pulse-Time received', fontsize=14)
        ax.ax_joint.annotate(f"slope: {self.fit[0]:.3f}\nintercept: {self.fit[1]:.3f}\n$R^2$: {self.max_corr**2:.3f}" +
                             f"\nRMSE: {self.rmse:.3f}",
                             xy=(.1, .7), xycoords="axes fraction", fontsize=14)
        plt.suptitle(f"Subject {self.subject}, {self.experiment}, Session {self.session}")
        plt.tight_layout()
        return ax
        
    def check(self, plot=False):
        # compute correlations
        self.match_idx, self.max_corr, self.correlations = self.highest_corr(self.sync_send, self.syncs)
        # fit 
        self.fit = np.polyfit(self.sync_send[self.match_idx:self.match_idx+len(self.syncs)], self.syncs, 1)
        preds = self.fit[0]*self.sync_send[self.match_idx:self.match_idx+len(self.syncs)] + self.fit[1]
        self.resid = preds - self.syncs
        self.rmse = np.sqrt(mean_squared_error(self.syncs, preds))
        if plot:
            ax = self.plot()
        small_resid = np.median(np.abs(self.resid)) <= 5
        small_intercept = np.isclose(0., self.fit[1], atol=20)
        slope_is_one = np.isclose(1., self.fit[0], atol=.1)
        high_corr = np.isclose(1., self.max_corr, atol=.01)
        return small_resid & small_intercept & slope_is_one & high_corr