from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored as ci_skurv

from gnpc.default_modules import *


class PatientStratification:
    def __init__(
        self,
        df,
        cut_off=None,
        high_is_high=True,
    ):
        """
        high_is_high = higher score related to higher risk
        """
        # self.threshold = threshold
        self.df = df

        event_data = np.array(
            self.df.groupby("patient")["event"].mean().values.tolist(), dtype=int
        )
        time_data = np.array(self.df.groupby("patient")["time"].mean().values.tolist())
        survival_score = np.array(
            self.df.groupby("patient")["score"].mean().values.tolist()
        )
        self.survival_score = survival_score
        self.time_data = time_data
        self.event_data = event_data

        self.cut_off = cut_off

        if high_is_high:
            self.high_group = self.survival_score > self.cut_off
            self.low_group = self.survival_score <= self.cut_off
            self.df["group"] = self.df["score"].apply(
                lambda x: "high" if x > self.cut_off else "low"
            )
        else:
            self.high_group = self.survival_score <= self.cut_off
            self.low_group = self.survival_score > self.cut_off
            self.df["group"] = self.df["score"].apply(
                lambda x: "high" if x <= self.cut_off else "low"
            )

        self.high_time = self.time_data[self.high_group]
        self.high_event = self.event_data[self.high_group]
        self.low_time = self.time_data[self.low_group]
        self.low_event = self.event_data[self.low_group]

        result = logrank_test(
            self.high_time, self.low_time, self.high_event, self.low_event
        )
        self.p_value = result.p_value

        self.event_data = self.event_data.astype(bool)
        self.c_index = ci_skurv(self.event_data, self.time_data, self.survival_score)[0]

    @staticmethod
    def _get_sig_level(pvalue):
        out = f"p : {pvalue:.3e}", False
        if pvalue < 0.05:
            out = "p < 0.05", True
        if pvalue < 0.01:
            out = "p < 0.01", True
        if pvalue < 0.001:
            out = "p < 0.001", True
        return out

    def plot(self, plot_table=True, fpath_save=None, **kwargs):

        kmf_h = KaplanMeierFitter(label="High Risk")
        kmf_l = KaplanMeierFitter(label="Low Risk")

        kmf_h.fit(self.high_time, self.high_event)
        kmf_l.fit(self.low_time, self.low_event)
        pvalue_text, is_significant = self._get_sig_level(self.p_value)
        print(
            f"median-high:{kmf_h.median_survival_time_}, median-low:{kmf_l.median_survival_time_}"
        )
        _, ax = plt.subplots(**kwargs)
        kmf_h.plot_survival_function(
            ax=ax,
            ci_show=False,
            c="#dc3978",
        )  # red

        kmf_l.plot_survival_function(
            ax=ax,
            ci_show=False,
            c="#029099",
        )
        # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), frameon=False)
        ax.legend().set_visible(False)
        axislim = ax.get_ylim()
        ax.text(
            axislim[1] - ((1 + axislim[1]) * 1.5),
            axislim[0] + ((1 - axislim[0]) * 0.1),
            f"{pvalue_text}\nC-index: {self.c_index:.3f}",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if plot_table:
            add_at_risk_counts(kmf_h, kmf_l, ax=ax, rows_to_show=["At risk"])
        # plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.tight_layout()
        if fpath_save is not None:
            if os.path.exists(os.path.dirname(fpath_save)) is False:
                os.makedirs(os.path.dirname(fpath_save))
            plt.savefig(fpath_save, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.show()

    def plot_vector(
        self, plot_table=True, fpath_save=None, unit_mm_to_pixel=25.4, **kwargs
    ):
        plt.rcParams.update({"font.size": kwargs["fontsize"]})
        if fpath_save != None:
            if not os.path.exists(os.path.dirname(fpath_save)):
                os.makedirs(os.path.dirname(fpath_save))

        kmf_h = KaplanMeierFitter(label="High Risk")
        kmf_l = KaplanMeierFitter(label="Low Risk")

        kmf_h.fit(self.high_time, self.high_event)
        kmf_l.fit(self.low_time, self.low_event)
        pvalue_text, is_significant = self._get_sig_level(self.p_value)
        print(
            f"median-high:{kmf_h.median_survival_time_}, median-low:{kmf_l.median_survival_time_}"
        )

        _, ax = plt.subplots(
            figsize=([x / unit_mm_to_pixel for x in kwargs["figsize"]]),
            dpi=kwargs["dpi"],
        )
        kmf_h.plot_survival_function(
            ax=ax,
            ci_show=False,
            c="#dc3978",
        )  # red

        kmf_l.plot_survival_function(
            ax=ax,
            ci_show=False,
            c="#029099",
        )
        # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), frameon=False)
        ax.legend().set_visible(False)
        axislim = ax.get_ylim()
        ax.text(
            axislim[1] - ((1 + axislim[1]) * 1.5),
            axislim[0] + ((1 - axislim[0]) * 0.1),
            f"{pvalue_text}\nC-index: {self.c_index:.3f}",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if plot_table:
            add_at_risk_counts(kmf_h, kmf_l, ax=ax, rows_to_show=["At risk"])

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        if fpath_save is not None:
            if os.path.exists(os.path.dirname(fpath_save)) is False:
                os.makedirs(os.path.dirname(fpath_save))
            plt.savefig(fpath_save, dpi=300)
        plt.show()
