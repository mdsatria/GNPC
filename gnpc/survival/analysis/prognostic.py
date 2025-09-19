import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored as ci_skurv


def censor_df(df, col_duration, col_event, threshold):
    loc = df[col_duration] > threshold
    df.loc[loc, col_duration] = threshold
    df.loc[loc, col_event] = 0
    df[col_event] = df[col_event].astype(bool)


class SurvivalAnalysis:
    def __init__(self, censored_dataframe, col_event, col_duration):
        self.censored_dataframe = censored_dataframe
        self.col_event = col_event
        self.col_duration = col_duration

        self.censored_dataframe[self.col_event] = self.censored_dataframe[
            self.col_event
        ].astype(bool)
        self.censored_dataframe[self.col_duration] = self.censored_dataframe[
            self.col_duration
        ].astype(int)


class KMFAnalysis(SurvivalAnalysis):
    def __init__(self, censored_dataframe, col_event, col_duration):
        super().__init__(censored_dataframe, col_event, col_duration)
        self.event = self.censored_dataframe[self.col_event].values
        self.duration = self.censored_dataframe[self.col_duration].values

        self.dct_group = None
        self.p_value = None
        self.c_index = None

    def evaluate(
        self,
        covariate,
        col_estimate=None,
        is_numeric=False,
        numeric_threshold=None,
        categorical_threshold={},
    ):
        dct_group = {}
        if is_numeric:
            groups = ["high", "low"]
            for group in groups:
                if group == "high":
                    loc = np.where(
                        self.censored_dataframe[covariate].values > numeric_threshold
                    )
                else:
                    loc = np.where(
                        self.censored_dataframe[covariate].values <= numeric_threshold
                    )
                dct_group[group] = {
                    "duration": self.duration[loc],
                    "event": self.event[loc],
                }
        else:
            groups = list(categorical_threshold.keys())
            for k, v in categorical_threshold.items():
                loc = np.where(self.censored_dataframe[covariate].values == v)
                dct_group[k] = {
                    "duration": self.duration[loc],
                    "event": self.event[loc],
                }
        self.dct_group = dct_group

        results = logrank_test(
            self.dct_group[groups[0]]["duration"],
            self.dct_group[groups[1]]["duration"],
            self.dct_group[groups[0]]["event"],
            self.dct_group[groups[1]]["event"],
        )
        self.p_value = results.p_value
        if col_estimate is not None:
            self.c_index = ci_skurv(
                event_indicator=self.censored_dataframe[self.col_event].values,
                event_time=self.censored_dataframe[self.col_duration].values,
                estimate=self.censored_dataframe[col_estimate].values,
            )

    def plot(self, colors=["#EE7D7F", "#367BA6"]):
        assert self.dct_group is not None
        group = list(self.dct_group.keys())
        dct_kmfs = {}

        for g in group:
            # dct_kmfs[g] =
            kmf = KaplanMeierFitter(label=g)
            kmf.fit(self.dct_group[g]["duration"], self.dct_group[g]["event"])
            dct_kmfs[g] = kmf

        _, ax = plt.subplots(figsize=(8, 4))
        for i, (k, kmf) in enumerate(dct_kmfs.items()):
            kmf.plot_survival_function(ax=ax, ci_show=False, c=colors[i])
        ax.legend(loc="upper right")
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        axislim = ax.get_ylim()
        plt.text(
            axislim[1] - ((1 + axislim[1]) * 1.5),
            axislim[0] + ((1 - axislim[0]) * 0.1),
            f"p-value: {self.p_value:.3e}",
        )
        plt.tight_layout()
        plt.show()


class CovarAnalysis(SurvivalAnalysis):
    def __init__(
        self,
        censored_dataframe,
        col_event,
        col_duration,
        covariates: list,
        categorical_covars: list,
        penalizer=1e-2,
    ):
        super().__init__(censored_dataframe, col_event, col_duration)
        self.covariates = covariates
        self.categorical_covars = categorical_covars
        self.penalizer = penalizer
        assert all(
            x in self.covariates for x in self.categorical_covars
        ), "check covars!"
        # self.formula_multivariate = None

    def _create_mulivariate_formula(self, covariates):
        formula = ""
        for i, col in enumerate(covariates):
            if col in self.categorical_covars:
                uni_formula = f"C({col})"
            else:
                uni_formula = col

            if i > 0:
                formula = formula + f" + {uni_formula}"
            else:
                formula = uni_formula
        return formula

    # Function to transform strings
    @staticmethod
    def transform_string(s):
        pattern = re.compile(r"C\(([^)]+)\)\[T\.(\d+)\.\d+\]")
        match = pattern.search(s)
        if match:
            return f"{match.group(1)}: {match.group(2)}"
        return s  # Return the original string if no match is found

    def _clean_result(self, cph_model):
        df_result = cph_model.summary.reset_index()[
            [
                "covariate",
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        ]
        df_result.rename(
            columns={
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "lower HR 95%",
                "exp(coef) upper 95%": "upper HR 95%",
                "p": "p-value",
            },
            inplace=True,
        )
        df_result["p-val"] = df_result["p-value"]
        df_result["p-value"] = df_result["p-value"].apply(
            lambda x: (
                "<0.001***"
                if x < 0.001
                else (
                    "<0.005**"
                    if x < 0.005
                    else (f"<0.05*" if x < 0.05 else round(x, 4))
                )
            )
        )
        df_result[["HR", "lower HR 95%", "upper HR 95%"]] = df_result[
            ["HR", "lower HR 95%", "upper HR 95%"]
        ].round(2)
        # df_result["group"] = col_name
        df_result["covariate"] = df_result["covariate"].map(self.transform_string)
        return df_result

    def univariate_analysis(self, covariates=None, return_dict=False):
        dct_univariate = {}
        if covariates == None:
            covariates = self.covariates.copy()
        for col in covariates:
            unicovar = [col, self.col_duration, self.col_event]

            cph_model = CoxPHFitter(penalizer=self.penalizer)
            if col in self.categorical_covars:
                formula = f"C({col})"
            else:
                formula = col

            cph_model.fit(
                df=self.censored_dataframe[unicovar],
                duration_col=self.col_duration,
                event_col=self.col_event,
                formula=formula,
            )
            clean_result = self._clean_result(cph_model)
            dct_univariate[col] = clean_result

        df_result = pd.concat(x for _, x in dct_univariate.items())
        if return_dict:
            return dct_univariate
        else:
            return df_result

    def multivariate_incremental_analysis(self, main_covariate, return_dict=False):
        dct_univariate = {}
        covariates = self.covariates.copy()
        covariates = [x for x in covariates if x != main_covariate]
        for col in covariates:
            covar_incremental = [main_covariate, col, self.col_duration, self.col_event]
            formula = self._create_mulivariate_formula([main_covariate, col])
            cph_model = CoxPHFitter(penalizer=self.penalizer)
            cph_model.fit(
                df=self.censored_dataframe[covar_incremental],
                duration_col=self.col_duration,
                event_col=self.col_event,
                formula=formula,
            )
            clean_result = self._clean_result(cph_model)
            dct_univariate[col] = clean_result

        df_result = pd.concat(x for _, x in dct_univariate.items())
        df_result = df_result[df_result["covariate"] != "score"]
        if return_dict:
            return dct_univariate
        else:
            return df_result

    def multivariate_analysis(self, covariates=None, return_dict=False):
        if covariates == None:
            covariates = self.covariates.copy()
        all_col = covariates + [self.col_duration] + [self.col_event]
        cph_model = CoxPHFitter(penalizer=self.penalizer)
        formula = self._create_mulivariate_formula(covariates)
        cph_model.fit(
            df=self.censored_dataframe[all_col],
            duration_col=self.col_duration,
            event_col=self.col_event,
            formula=formula,
        )
        clean_result = self._clean_result(cph_model)
        if return_dict:
            return clean_result.to_csv()
        else:
            return clean_result
