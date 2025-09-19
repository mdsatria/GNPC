from functools import partial

import path_project
from torch_geometric.data import Data
from tqdm.contrib.concurrent import process_map

from gnpc.default_modules import *
from gnpc.feature.graph import create_weighted_graph_delaunay
from gnpc.utils.io import *
from gnpc.utils.logger import LogMsg
from gnpc.utils.misc import StopWatch, get_graph_type

# def get_nuc_feature_indices(
#     len_morph_per_type: int = 91,
#     total_nuc_type: int = 5,
#     include_nuc_type: list = [1, 2],
# ):
#     sub_indices_morph = np.array_split(
#         np.arange(len_morph_per_type * total_nuc_type), total_nuc_type
#     )
#     include_indices = []
#     for ix, indices in enumerate(sub_indices_morph):
#         if ix + 1 in include_nuc_type:
#             include_indices.extend(indices)
#     return include_indices


def make_node_feature(
    wsi_id: str,
    dir_patch: str,
    dir_emb: str,
    dir_nuc: str,
):
    # use this if all nuclei type is in feature
    # include_indices = get_nuc_feature_indices()

    fpath_patch = os.path.join(dir_patch, f"{wsi_id}.pkl")
    patch_info = load_pickle(fpath_patch)["patch"]
    patch_ids = list(patch_info.keys())

    coords = np.vstack([patch_info[x]["top_left"] for x in patch_ids])
    centroids = np.vstack([patch_info[x]["center"] for x in patch_ids])

    if dir_emb != None:
        fpath_emb = os.path.join(dir_emb, f"{wsi_id}.pkl")
        data_emb = load_pickle(fpath_emb)

    features = []
    for i in patch_ids:
        features_ = []
        if dir_emb != None:
            feat_emb = data_emb["embeddings"][i]
            features_.append(feat_emb)
        if dir_nuc != None:
            fpath_nuc = os.path.join(
                dir_nuc, wsi_id, "stat", f"{i}_{coords[i][0]}_{coords[i][1]}.pkl"
            )
            feat_nuc = load_pickle(fpath_nuc)  # [include_indices]
            features_.append(feat_nuc)

        features_ = np.concatenate(features_)
        features.append(features_)

    features = np.vstack(features)

    return coords, centroids, features, wsi_id


def combine_patient_level_graph(graph_list: list):
    if len(graph_list) > 1:
        graph_dict = {
            "x": None,
            "edge_index": None,
            "coordinates": None,
            "topleft": None,
            "threshold_distance": None,
            "wsi_id": None,
        }

        graph_dict["threshold_distance"] = graph_list[0].threshold_distance
        # combine x/features
        graph_dict["x"] = torch.concatenate([data.x for data in graph_list])
        # combine coordinates
        graph_dict["coordinates"] = torch.concatenate(
            [data.coordinates for data in graph_list]
        )
        # combine topleft
        graph_dict["topleft"] = torch.concatenate([data.topleft for data in graph_list])
        # combine edge_weight
        if "edge_weight" in graph_list[0].keys():
            graph_dict["edge_weight"] = torch.concatenate(
                [data.edge_weight for data in graph_list]
            )

        # combine edge index
        # index in 2nd graph + max index from 1st graph
        edge_index = graph_list[0]["edge_index"]
        for ix in range(1, len(graph_list)):
            max_index = edge_index.max()
            temp_edge_index = graph_list[ix].edge_index + max_index
            edge_index = torch.concatenate([edge_index, temp_edge_index], axis=1)

        graph_dict["edge_index"] = edge_index

    else:
        graph_dict = graph_list[0]

    graph_dict["x"] = graph_dict["x"].type(torch.float32)
    graph_dict["edge_index"] = graph_dict["edge_index"].type(torch.int64)
    if "edge_weight" in graph_dict.keys():
        graph_dict["edge_weight"] = graph_dict["edge_weight"].type(torch.float32)

    graph = Data(**graph_dict)
    return graph


def get_patient_dct(csv_file, col_patient="patient", col_slide="wsi"):
    df = pd.read_csv(csv_file)
    dct_patient = []
    for patient in df[col_patient].unique().tolist():
        wsi_list = df[df[col_patient] == patient][col_slide].values.tolist()
        dct_patient.append({str(patient): wsi_list})
    return dct_patient


def construct_graph(
    dct_patient: dict,
    dir_patch: str,
    dir_emb: str,
    dir_nuc: str,
    dir_save: str,
    use_weight: bool,
    graph_type: str,
    max_node_distance: int = 2000,
):
    for patient_id, wsi_list in dct_patient.items():
        dir_save_patient_graph = os.path.join(dir_save, patient_id)
        fpath_graph_patient = os.path.join(dir_save_patient_graph, f"{patient_id}.pt")
        dir_save_slide_graph = os.path.join(dir_save, patient_id, "slide")
        if os.path.exists(fpath_graph_patient):
            pass
        else:
            if not os.path.exists(dir_save_patient_graph):
                os.makedirs(dir_save_patient_graph)
                os.makedirs(dir_save_slide_graph)

            graph_list = []
            for wsi_id in wsi_list:
                wsi_id = pathlib.Path(wsi_id).stem

                coords, centroids, features, _ = make_node_feature(
                    wsi_id=wsi_id,
                    dir_patch=dir_patch,
                    dir_emb=dir_emb,
                    dir_nuc=dir_nuc,
                )
                if graph_type == "delaunay":
                    graph_dict = create_weighted_graph_delaunay(
                        centroids=centroids,
                        features=features,
                        topleft=coords,
                        is_weighted=use_weight,
                        threshold_distance=max_node_distance,
                    )
                else:
                    raise NotImplementedError
                graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
                graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
                graph_dict["x"] = graph_dict["x"].type(torch.float32)
                graph_dict["edge_index"] = graph_dict["edge_index"].type(torch.int64)
                if "edge_weight" in graph_dict.keys():
                    graph_dict["edge_weight"] = graph_dict["edge_weight"].type(
                        torch.float32
                    )

                graph_tensor = Data(**graph_dict)
                graph_list.append(graph_tensor)

                fpath_graph_slide = os.path.join(dir_save_slide_graph, f"{wsi_id}.pt")
                torch.save(graph_tensor, fpath_graph_slide)

            graph_patient = combine_patient_level_graph(graph_list=graph_list)
            torch.save(graph_patient, fpath_graph_patient)


def main(
    csv_patient: str,
    root_dir_emb: str,
    root_dir_patch: str,
    root_dir_nuc: str,
    dir_save: str,
    cohort: str,
    use_weight: bool,
    use_nuc: bool,
    emb_name: str,
    max_node_distance: int = 2000,
    graph_type: str = "delaunay",
):
    dct_patient = get_patient_dct(csv_file=csv_patient)

    graph_config = get_graph_type(
        deep_feature=emb_name,
        nuclei_feature=use_nuc,
        is_weighted=use_weight,
    )

    dir_patch = os.path.join(root_dir_patch, cohort, "coords")
    dir_final_save = os.path.join(dir_save, cohort, graph_config)
    if not os.path.exists(dir_final_save):
        os.makedirs(dir_final_save)

    dir_emb = None
    dir_nuc = None
    if use_nuc != False:
        dir_nuc = os.path.join(root_dir_nuc, cohort)
    if emb_name != False:
        dir_emb = os.path.join(root_dir_emb, cohort, emb_name)

    partial_func = partial(
        construct_graph,
        dir_patch=dir_patch,
        dir_emb=dir_emb,
        dir_nuc=dir_nuc,
        dir_save=dir_final_save,
        use_weight=use_weight,
        graph_type=graph_type,
        max_node_distance=max_node_distance,
    )
    process_map(partial_func, dct_patient, max_workers=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--dir_patch", type=str)
    parser.add_argument("--dir_emb", type=str)
    parser.add_argument("--dir_nuc", type=str)
    parser.add_argument("--dir_save", type=str)
    parser.add_argument("--emb", type=str, help="Feature encoder/type for node")
    parser.add_argument("--max_dist", type=int)
    parser.add_argument("--gtype", default="delaunay", type=str, help="Graph type")
    parser.add_argument(
        "--weighted",
        action="store_true",
        default=False,
        help="Use weighted graph or not",
    )
    parser.add_argument(
        "--use_nuc",
        action="store_true",
        default=False,
        help="Use morphology of nuclei in graph node",
    )
    parser.add_argument("--cohort", type=str)

    args = parser.parse_args()

    main(
        csv_patient=args.csv,
        root_dir_emb=args.dir_emb,
        root_dir_patch=args.dir_patch,
        root_dir_nuc=args.dir_nuc,
        dir_save=args.dir_save,
        cohort=args.cohort,
        use_weight=args.weighted,
        use_nuc=args.use_nuc,
        emb_name=args.emb,
        max_node_distance=args.max_dist,
        graph_type=args.gtype,
    )
