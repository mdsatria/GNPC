from matplotlib.patches import Patch
from scipy import stats

from gnpc.default_modules import *


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


def plot_features(stat_high, stat_low, feature_name, show_fliers=False, show_mean=True):
    # print(f"Median High: {np.median(stat_high)}  Median Low: {np.median(stat_low)}")
    # print(f"Mean High: {np.mean(stat_high)}  Mean Low: {np.mean(stat_low)}")

    # t_statistic, p_value = stats.ttest_ind(stat_high, stat_low, equal_var=False)
    t_statistic, p_value = stats.mannwhitneyu(stat_high, stat_low)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    combined_data = [stat_high, stat_low]
    axs.boxplot(combined_data, showfliers=show_fliers, showmeans=show_mean)
    axs.set_title(f"{feature_name.upper()}: {'<0.001**' if p_value<0.001 else p_value}")
    axs.yaxis.grid(True)
    axs.set_xticks([y + 1 for y in range(2)], labels=["High Risk", "Low Risk"])
    axs.set_xlabel("Group")
    axs.set_ylabel("Value")
    plt.show()


def plot_genes(df, clinical_fpath="dataset/clinical/gene.csv", fpath_save=None):
    cols_genes = ["TP53", "TRAF3", "NFKBIA", "AEBP1", "NLRC5"]

    df_gene_raw = pd.read_csv(clinical_fpath)
    df_gene_altered = df_gene_raw.copy(deep=True)
    unique_values = [-1, 0, 1]
    dct_alter = {-2: -1, -1: -1, 0: 0, 1: 1, 2: 1}

    for col in cols_genes:
        for old, new in dct_alter.items():
            df_gene_altered.loc[df_gene_raw[col] == old, col] = new

    df_gene = df_gene_altered.copy(deep=True)
    group_risk = []
    for c in df_gene["Cases"]:
        try:
            val = df[df["patient"] == c]["group"].values.tolist()[0]
        except:
            val = np.nan
        group_risk.append(val)
    df_gene["group"] = group_risk
    df_gene = df_gene.dropna()
    df_gene.reset_index(drop=True, inplace=True)
    df_gene = df_gene.sort_values("group")
    df_gene["group_numeric"] = np.where(df_gene["group"] == "high", 1, 0)

    ytick_labels_pval = []
    dct_pval = {}
    for c in cols_genes:
        group_lst = []
        for group in ["high", "low"]:
            contingency_table = pd.crosstab(
                index=df_gene[df_gene["group"] == group][c], columns="count"
            )
            contingency_table = contingency_table.reindex(unique_values, fill_value=0)
            group_lst.append(contingency_table)
        chi2, p_value, _, _ = stats.chi2_contingency(group_lst)
        if p_value < 0.001:
            ytick_labels_pval.append(f"{c} **")
        elif p_value < 0.05:
            ytick_labels_pval.append(f"{c} *")
        else:
            ytick_labels_pval.append(c)
        dct_pval[c] = p_value

    fig, ax = plt.subplots(
        2, 1, figsize=(20, 3), gridspec_kw={"height_ratios": [5, 1]}, sharex=False
    )
    sns.heatmap(
        df_gene[cols_genes].reset_index(drop=True).T,
        annot=False,
        cmap="viridis",
        ax=ax[0],
        cbar=False,
        yticklabels=ytick_labels_pval,
    )  # Change the colormap as needed
    sns.heatmap(
        df_gene[["group_numeric"]].reset_index(drop=True).T,
        annot=False,
        # cmap="coolwarm",
        cmap=["#EE7D7F", "#367BA6"],
        ax=ax[1],
        cbar=False,
    )

    gene_legend = [
        Patch(facecolor=(0.992, 0.906, 0.145), label=" 1"),
        Patch(facecolor=(0.129, 0.569, 0.549), label=" 0"),
        Patch(facecolor=(0.267, 0.004, 0.329), label="-1"),
    ]
    ax[0].legend(
        handles=gene_legend,
        loc="center",
        bbox_to_anchor=(1.05, 0.5),
        title="Gene Status",
    )
    ax[0].tick_params(bottom=False, left=False)
    ax[0].set_xticklabels("")

    # group_legend = [
    #     Patch(facecolor=(0.705, 0.015, 0.149), label="High"),
    #     Patch(facecolor=(0.231, 0.298, 0.752), label="Low"),
    # ]
    group_legend = [
        Patch(facecolor="#EE7D7F", label="High"),
        Patch(facecolor="#367BA6", label="Low"),
    ]
    ax[1].legend(
        handles=group_legend,
        loc="center",
        bbox_to_anchor=(1.05, 0.5),
        title="Risk Group",
    )
    ax[1].set_xticks(range(len(df_gene["Cases"])))
    ax[1].set_xticklabels(df_gene["Cases"].values.tolist(), rotation=90)
    ax[1].set_ylabel("")
    ax[1].set_yticklabels("")
    ax[1].tick_params(bottom=False, left=False)
    plt.tight_layout()
    if fpath_save is not None:
        if os.path.exists(os.path.dirname(fpath_save)) is False:
            os.makedirs(os.path.dirname(fpath_save))
        plt.savefig(fpath_save)
    plt.show()
    return dct_pval


def plot_boxplot(
    stat_high,
    stat_low,
    title_name,
    log_scale=False,
    num_space=3.0,
    width=0.4,
    high_risk_color="#EE7D7F",
    low_risk_color="#367BA6",
    fpath_save=None,
):
    # stat_high = df[df["group"] == "high"]["lmp1_status"].values
    # stat_low = df[df["group"] == "low"]["lmp1_status"].values

    _, p_value = stats.mannwhitneyu(stat_high, stat_low)
    if p_value < 0.001:
        p_value_text = "<0.001**"
    elif p_value < 0.05:
        p_value_text = "<0.05"
    else:
        p_value_text = f" p-value:{p_value:.2e}"

    combined_data = [stat_high, stat_low]
    plt.figure(figsize=(8, 4))
    plt.title(f"{title_name}: {p_value_text}")

    plt.grid(visible=False, axis="x")
    plt.grid(visible=True, axis="y", alpha=0.5, linestyle="--")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    bp_high = plt.boxplot(
        [stat_high],
        positions=np.array(range(len([stat_high]))) * num_space - width,
        sym="",
        widths=0.6,
        showmeans=True,
    )
    bp_low = plt.boxplot(
        [stat_low],
        positions=np.array(range(len([stat_low]))) * num_space + width,
        sym="",
        widths=0.6,
        showmeans=True,
    )
    set_box_color(bp_high, high_risk_color)
    set_box_color(bp_low, low_risk_color)
    plt.plot([], c=high_risk_color, linewidth=4, label="High Risk")
    plt.plot([], c=low_risk_color, linewidth=4, label="Low Risk")
    plt.legend(loc="center", bbox_to_anchor=(1.15, 0.5), frameon=False)
    # range_group = range(0, len(group_label) * len(group_label), int(num_space))
    plt.xticks([])
    if log_scale:
        plt.yscale("log")
    # plt.yticks(np.arange(min(min_y), max(max_y) + 0.2, 0.2))
    # plt.yticks(y_ticks)
    plt.tight_layout()
    if fpath_save is not None:
        if os.path.exists(os.path.dirname(fpath_save)) is False:
            os.makedirs(os.path.dirname(fpath_save))
        plt.savefig(fpath_save)
    plt.show()


def plot_morphology(
    group_high_risk,
    group_low_risk,
    group_label,
    feature_name,
    num_space=3.0,
    width=0.4,
    high_risk_color="#EE7D7F",
    low_risk_color="#367BA6",
    fpath_save=None,
):

    dct_name = {
        "area_convex": "Convex Area",
        "area_contour": "Contour Area",
        "eccentricity": "Eccentricity",
        "equiv_diameter": "Diameter",
        "major_axis_len": "Major Axis Length",
        "minor_axis_len": "Minor Axis Length",
        "perimeter": "Perimeter",
        "solidity": "Solidity",
        "orientation": "Orientation",
        "radius": "Radius",
        "area_bbox": "Bounding Box Area",
        "roundness": "Roundness",
        "spindle": "Spindleness",
    }
    plt.figure(figsize=(8, 4))
    plt.title(f"{dct_name[feature_name]}")

    plt.grid(visible=False, axis="x")
    plt.grid(visible=True, axis="y", alpha=0.5, linestyle="--")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    bp_high = plt.boxplot(
        group_high_risk,
        positions=np.array(range(len(group_high_risk))) * num_space - width,
        sym="",
        widths=0.6,
        showmeans=True,
    )
    bp_low = plt.boxplot(
        group_low_risk,
        positions=np.array(range(len(group_low_risk))) * num_space + width,
        sym="",
        widths=0.6,
        showmeans=True,
    )

    top_coor_high = []
    for ix, item in enumerate(bp_high["whiskers"]):
        if ix % 2 != 0:
            top_coor_high.append(item.get_ydata()[1])

    top_coor_low = []
    for ix, item in enumerate(bp_low["whiskers"]):
        if ix % 2 != 0:
            top_coor_low.append(item.get_ydata()[1])

    set_box_color(bp_high, high_risk_color)
    set_box_color(bp_low, low_risk_color)

    range_group = range(0, len(group_label) * len(group_label), int(num_space))
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=high_risk_color, linewidth=4, label="High Risk")
    plt.plot([], c=low_risk_color, linewidth=4, label="Low Risk")
    plt.legend(loc="center", bbox_to_anchor=(1.15, 0.5), frameon=False)

    max_y = []
    min_y = []
    for i in range(len(group_label)):
        _, p_value = stats.mannwhitneyu(group_high_risk[i], group_low_risk[i])
        print(p_value)
        if p_value < 0.05:
            if p_value < 1e-4:
                p_value_text = "***"  # <1e-4
            elif p_value < 1e-3:
                p_value_text = "**"  # <1e-3
            elif p_value < 5e-2:
                p_value_text = "*"  # <5e-2

            x1, x2 = (range_group[i] - width), (range_group[i] + width)
            temp_y_max = max(top_coor_high[i], top_coor_low[i])
            if temp_y_max < 1.0:
                h = 0.008
            else:
                h = temp_y_max / 50
            temp_y_max += h
            max_y.append(temp_y_max)
            y = temp_y_max

            col = "k"
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c=col)
            plt.text(
                (x1 + x2) * 0.5,
                y + h,
                p_value_text,
                ha="center",
                va="bottom",
                color=col,
            )

    plt.xticks(range_group, group_label)
    # plt.yticks(np.arange(min(min_y), max(max_y) + 0.2, 0.2))
    # plt.yticks(y_ticks)
    plt.tight_layout()
    if fpath_save is None:
        pass
    else:
        if os.path.exists(os.path.dirname(fpath_save)) is False:
            os.makedirs(os.path.dirname(fpath_save))
        plt.savefig(fpath_save)
        # plt.close()
    plt.show()
