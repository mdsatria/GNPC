import torch
import numpy as np
import pickle
import json, ujson
import joblib
from torch_geometric.data import Data


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(fpath, fast=True):
    with open(fpath, "r") as f:
        if fast:
            data = ujson.load(f)
        else:
            data = json.load(f)
    return data


def load_new_hovernet_data(fpath):
    with open(fpath, "rb") as f:
        temp = joblib.load(fpath)
    source_resolution = temp["source-image-resolution"]["mpp"][0]
    element_resolution = temp["element-resolution"]["resolution"]

    factor_resize = np.ceil(element_resolution / source_resolution)
    # print(element_resolution, source_resolution, factor_resize)
    data = {"nuc": temp["elements"]}
    return data, factor_resize


def save_pickle(data, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(data, f)


def save_json(data, fpath):
    with open(fpath, "w") as f:
        json.dump(data, f, indent=5)


def save_graph(graph_dict, fpath):
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
    graph_dict["x"] = graph_dict["x"].type(torch.float32)
    graph_dict["edge_index"] = graph_dict["edge_index"].type(torch.int64)
    if "edge_weight" in graph_dict.keys():
        graph_dict["edge_weight"] = graph_dict["edge_weight"].type(torch.float32)
    datagraph = Data(**graph_dict)
    torch.save(datagraph, fpath)
