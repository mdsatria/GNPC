from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold, train_test_split

from gnpc.default_modules import *


def censor_data(df, time_col, event_col, threshold):
    df[time_col] = df[time_col].apply(lambda x: round(x))
    idx_censor = df[df[time_col] > threshold].index
    df.loc[idx_censor, time_col] = threshold
    df.loc[idx_censor, event_col] = 0
    return df


class SurvivalDataHandling:
    def __init__(self, col_time, col_event):
        self.col_time = col_time
        self.col_event = col_event
        self.bins_values = None
        self.sub_bins = None
        self.is_censored = False

    def _create_sublists(self):
        self.sub_bins = [[0, self.bins_values[0]]]  # Add the [0, first bin] interval
        self.sub_bins.extend(
            [
                [self.bins_values[i], self.bins_values[i + 1]]
                for i in range(len(self.bins_values) - 1)
            ]
        )
        self.sub_bins.append([self.bins_values[-1]])  # Add the

    def _categorise(self, row):
        for i, time_interval in enumerate(self.sub_bins):
            if i == 0:
                if row[self.col_time] <= time_interval[1]:
                    return i
            elif i == len(self.sub_bins) - 1:
                if row[self.col_time] > time_interval[0]:
                    return i
            else:
                if (row[self.col_time] > time_interval[0]) & (
                    row[self.col_time] <= time_interval[1]
                ):
                    return i

    def discretise(self, df, bins=3):
        if self.is_censored:
            percentile_list = [i / bins for i in range(bins)][1:]
            self.bins_values = (
                df[df[self.col_event] == 1][self.col_time]
                .quantile(percentile_list)
                .values.tolist()
            )
            self.bins_values = [round(x) for x in self.bins_values]
            self._create_sublists()
            df["time_label"] = df.apply(self._categorise, axis=1)
            return df
        else:
            return "CENSOR THE DATA FIRST! .censor()"

    def apply_discretise(self, df):
        if self.bins_values is None or self.sub_bins is None:
            return "QUANTILE VALUES NOT SET! Run .discretise() on main DataFrame first."
        else:
            df["time_label"] = df.apply(self._categorise, axis=1)
            return df

    def censor(self, df, threshold):
        df[self.col_time] = df[self.col_time].apply(lambda x: round(x))
        idx_censor = df[df[self.col_time] > threshold].index
        df.loc[idx_censor, self.col_time] = threshold
        df.loc[idx_censor, self.col_event] = 0
        self.is_censored = True
        return df


def splits_test_plot(df, col_time, col_event, label, threshold):
    x_time = df[col_time].values
    x_event = df[col_event].values
    idx = x_time > threshold
    x_time[idx] = threshold
    x_event[idx] = 0
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(event_observed=x_event, durations=x_time)
    kmf.plot_survival_function()
    plt.ylim(0.7, 1.01)
    plt.show()


class SurvivalDataSplit:
    def __init__(self, df, event_column, time_column):
        self.df = df
        self.event_column = event_column
        self.time_column = time_column

    def split_train_test(
        self,
        test_size: float,
        random_state: int = 42,
        iteration: int = 1000,
        adjusted: bool = True,
    ):
        df_split_1, df_split_2 = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df[self.event_column],
        )
        if adjusted:
            df_split_1, df_split_2 = self._adjust_survival_dist_train_test(
                df_split_1, df_split_2, iteration
            )

        return df_split_1, df_split_2

    def split_n_folds(
        self,
        n_splits,
        random_state: int = 42,
        iteration: int = 1000,
        adjusted: bool = True,
    ):
        # Initialize the StratifiedKFold to split by event
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = []

        for _, test_index in skf.split(self.df, self.df[self.event_column]):
            df_split = self.df.iloc[test_index].copy()
            splits.append(df_split)
        if adjusted:
            # Adjust the time distribution for each split
            adjusted_splits = self._adjust_survival_dist_n_folds(splits, iteration)

            return adjusted_splits
        else:
            return splits

    def _adjust_survival_dist_train_test(self, df_split_1, df_split_2, iteration):
        # Calculate mean and median for both splits
        mean_1, median_1 = (
            df_split_1[self.time_column].mean(),
            df_split_1[self.time_column].median(),
        )
        mean_2, median_2 = (
            df_split_2[self.time_column].mean(),
            df_split_2[self.time_column].median(),
        )

        # Swap rows between the two splits until the distributions are similar
        for _ in range(iteration):  # Limiting the number of iterations
            if np.isclose(mean_1, mean_2, atol=1) and np.isclose(
                median_1, median_2, atol=1
            ):
                break  # Stop if the distributions are close enough
            self._optimise(df_split_1, df_split_2, median_1, median_2)

            mean_1, median_1 = (
                df_split_1[self.time_column].mean(),
                df_split_1[self.time_column].median(),
            )
            mean_2, median_2 = (
                df_split_2[self.time_column].mean(),
                df_split_2[self.time_column].median(),
            )

        return df_split_1, df_split_2

    def _adjust_survival_dist_n_folds(self, splits, iteration):
        # Calculate target mean and median for the time column
        overall_mean = self.df[self.time_column].mean()
        overall_median = self.df[self.time_column].median()

        # Adjust the splits
        for i in range(len(splits)):
            current_mean = splits[i][self.time_column].mean()
            current_median = splits[i][self.time_column].median()

            # Check and adjust splits to get closer to the overall mean and median
            for _ in range(iteration):  # Limit iterations to avoid infinite loops
                if np.isclose(current_mean, overall_mean, atol=1) and np.isclose(
                    current_median, overall_median, atol=1
                ):
                    break

                for j in range(len(splits)):
                    if i != j:
                        self._optimise(
                            splits[i], splits[j], overall_median, overall_median
                        )

                        current_mean = splits[i][self.time_column].mean()
                        current_median = splits[i][self.time_column].median()

        return splits

    def _optimise(self, df1, df2, median_1, median_2):
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        idx_split_1 = (df1[self.time_column] - median_2).abs().idxmax()
        idx_split_2 = (df2[self.time_column] - median_1).abs().idxmax()

        swap_1 = df1.loc[idx_split_1].copy()
        swap_2 = df2.loc[idx_split_2].copy()

        df1.loc[idx_split_1] = swap_1
        df2.loc[idx_split_1] = swap_2
