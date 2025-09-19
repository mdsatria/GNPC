import os
import pathlib
import random
import string
from time import time

import numpy as np
import torch


def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_significance_sign(pvalue):
    out = (f"p = {pvalue:.2e}", False)
    if pvalue < 0.05:
        out = ("*", True)
    if pvalue < 0.01:
        out = ("*", True)
    if pvalue < 0.001:
        out = ("*", True)
    return out


def check_significance_val(pvalue):
    out = (f"p = {pvalue:.2e}", False)
    if pvalue < 0.05:
        out = ("p < 0.05", True)
    if pvalue < 0.01:
        out = ("p < 0.01", True)
    if pvalue < 0.001:
        out = ("p < 0.001", True)
    return out


def get_feature_len(config):
    # if config["graph_config"]["feat_reduction"] is not None:
    #     num_feature = 256
    # else:
    num_feature = 0
    if config["graph_config"]["embedding"] == "dino":
        num_feature = 384
    elif config["graph_config"]["embedding"] == "uni":
        num_feature = 1024
    elif config["graph_config"]["embedding"] == "resnet50":
        num_feature = 2048
    elif config["graph_config"]["embedding"] == "virchow":
        num_feature = 2560
    elif config["graph_config"]["embedding"] == "conch":
        num_feature = 512
    if config["graph_config"]["is_morph"]:
        # 72 len per each nuclei type, mean, std, range, median ,kurtosis, skew
        nuc_morph_len = 144

        num_feature += nuc_morph_len
    return num_feature


def get_graph_type(deep_feature, nuclei_feature, is_weighted):
    """
    embedding
    morph
    weighted
    """
    # deep_feature = config["graph_config"]["embedding"]
    # nuclei_feature = config["graph_config"]["morph"]
    # is_weighted = config["graph_config"]["weighted"]

    return f"emb_{deep_feature}-mrp_{nuclei_feature}-wgt_{is_weighted}"


def get_graph_type_from_json(config):
    """
    embedding
    morph
    weighted
    """
    deep_feature = config["graph_config"]["embedding"]
    nuclei_feature = config["graph_config"]["is_morph"]
    is_weighted = config["graph_config"]["is_weighted"]
    # deep_feature = config["graph_config"]["embedding"]
    # nuclei_feature = config["graph_config"]["morph"]
    # is_weighted = config["graph_config"]["weighted"]

    return f"emb_{deep_feature}-mrp_{nuclei_feature}-wgt_{is_weighted}"


class StopWatch:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time()

    def stop_and_count(self):
        assert self.start_time is not None, "call method start first!"
        time_elapsed = time() - self.start_time
        self.start_time = None
        return time_elapsed


# def generate_unique_strings(num_strings, prefix=None, string_length=8):
#     unique_strings = set()

#     while len(unique_strings) < num_strings:
#         # Generate a random string of random length up to max_length
#         # length = string_length
#         random_string = "".join(random.choices(string.ascii_lowercase, k=string_length))

#         # Add the string to the set if it's unique and its length is less than or equal to max_length
#         # if len(random_string) <= string_length:
#         if prefix is None:
#             unique_strings.add(random_string)
#         else:
#             unique_strings.add(f"{prefix}-{random_string}")

#     unique_strings = list(unique_strings)
#     unique_strings = [f"{i}_{x}" for i, x in enumerate(unique_strings)]
#     if num_strings == 1:
#         return unique_strings[0]
#     else:
#         return unique_strings


# def get_folder_list(path):
#     directories = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
#     return directories


class DirUtils:
    def __init__(self, rootdir: str):
        self.rootdir = rootdir

    def get_folder_list(self):
        directories = [
            x
            for x in os.listdir(self.rootdir)
            if os.path.isdir(os.path.join(self.rootdir, x))
        ]
        return directories

    def get_file_list(self):
        return sorted(os.listdir(self.rootdir))

    def get_file_path(self):
        fpath = [os.path.join(self.rootdir, x) for x in self.get_file_list()]
        return fpath

    def get_filename(self):
        filename = [pathlib.Path(x).stem for x in self.get_file_list()]
        return filename
