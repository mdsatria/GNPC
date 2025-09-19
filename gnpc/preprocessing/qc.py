import csv

import numpy as np
import pandas as pd


def histo_tsv_to_df(
    fpath,
    subset=None,
    index_start=5,
):
    tsv_line = []
    with open(fpath) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            tsv_line.append(line)
    dct_tsv = {x: [] for x in tsv_line[index_start]}

    for i in range(index_start + 1, len(tsv_line)):
        for k, j in zip(dct_tsv.keys(), range(len(dct_tsv.keys()))):
            dct_tsv[k].append(tsv_line[i][j])

    df = pd.DataFrame(dct_tsv)
    if subset != None:
        df["subset"] = subset
    df = df.rename(columns={"#dataset:filename": "wsi"})
    return df


def qc_assesment(df, is_ihc=False):

    df[
        [
            "pen_markings",
            "coverslip_edge",
            "bright",
            "dark",
            "flat_areas",
            "fatlike_tissue_removed_num_regions",
            "fatlike_tissue_removed_mean_area",
            "fatlike_tissue_removed_max_area",
            "fatlike_tissue_removed_percent",
            "small_tissue_filled_num_regions",
            "small_tissue_filled_mean_area",
            "small_tissue_filled_max_area",
            "small_tissue_filled_percent",
            "small_tissue_removed_num_regions",
            "small_tissue_removed_mean_area",
            "small_tissue_removed_max_area",
            "small_tissue_removed_percent",
            "blurry_removed_num_regions",
            "blurry_removed_mean_area",
            "blurry_removed_max_area",
            "blurry_removed_percent",
            "spur_pixels",
            "areaThresh",
            "template1_MSE_hist",
            "template2_MSE_hist",
            "template3_MSE_hist",
            "template4_MSE_hist",
            "tenenGrad_contrast",
            "michelson_contrast",
            "rms_contrast",
            "grayscale_brightness",
            "grayscale_brightness_std",
            "chan1_brightness",
            "chan1_brightness_std",
            "chan2_brightness",
            "chan2_brightness_std",
            "chan3_brightness",
            "chan3_brightness_std",
            "chan1_brightness_YUV",
            "chan1_brightness_std_YUV",
            "chan2_brightness_YUV",
            "chan2_brightness_std_YUV",
            "chan3_brightness_YUV",
            "chan3_brightness_std_YUV",
            "deconv_c0_mean",
            "deconv_c0_std",
            "deconv_c1_mean",
            "deconv_c1_std",
            "deconv_c2_mean",
            "deconv_c2_std",
            "pixels_to_use",
        ]
    ] = df[
        [
            "pen_markings",
            "coverslip_edge",
            "bright",
            "dark",
            "flat_areas",
            "fatlike_tissue_removed_num_regions",
            "fatlike_tissue_removed_mean_area",
            "fatlike_tissue_removed_max_area",
            "fatlike_tissue_removed_percent",
            "small_tissue_filled_num_regions",
            "small_tissue_filled_mean_area",
            "small_tissue_filled_max_area",
            "small_tissue_filled_percent",
            "small_tissue_removed_num_regions",
            "small_tissue_removed_mean_area",
            "small_tissue_removed_max_area",
            "small_tissue_removed_percent",
            "blurry_removed_num_regions",
            "blurry_removed_mean_area",
            "blurry_removed_max_area",
            "blurry_removed_percent",
            "spur_pixels",
            "areaThresh",
            "template1_MSE_hist",
            "template2_MSE_hist",
            "template3_MSE_hist",
            "template4_MSE_hist",
            "tenenGrad_contrast",
            "michelson_contrast",
            "rms_contrast",
            "grayscale_brightness",
            "grayscale_brightness_std",
            "chan1_brightness",
            "chan1_brightness_std",
            "chan2_brightness",
            "chan2_brightness_std",
            "chan3_brightness",
            "chan3_brightness_std",
            "chan1_brightness_YUV",
            "chan1_brightness_std_YUV",
            "chan2_brightness_YUV",
            "chan2_brightness_std_YUV",
            "chan3_brightness_YUV",
            "chan3_brightness_std_YUV",
            "deconv_c0_mean",
            "deconv_c0_std",
            "deconv_c1_mean",
            "deconv_c1_std",
            "deconv_c2_mean",
            "deconv_c2_std",
            "pixels_to_use",
        ]
    ].astype(
        float
    )
    important_cols = [
        "wsi",
        "subset",
        "spur_pixels",  # <0
        "areaThresh",  # <0
        "blurry_removed_percent",  # ==1
    ]
    if is_ihc:
        condition = (df["blurry_removed_percent"] > 0.99) & (df["spur_pixels"] < 0)
    else:
        condition = (
            (df["blurry_removed_percent"] > 0.99)
            & (df["spur_pixels"] < 0)
            & (df["areaThresh"] < 0)
        )

    # Create the new column based on the condition
    df["bad_wsi"] = np.where(condition, True, False)
    df.insert(1, "bad_wsi", df.pop("bad_wsi"))
    return df
