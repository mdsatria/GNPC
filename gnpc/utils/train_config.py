# import string
# import itertools
# from copy import deepcopy


# def get_graph_type(is_weighted, is_morph, feat_reduction):
#     prefix = "delaunay"
#     if is_weighted:
#         prefix = f"{prefix}_weighted"
#     else:
#         prefix = f"{prefix}_nonweighted"
#     if is_morph:
#         prefix = f"{prefix}_usemorph"
#     else:
#         prefix = f"{prefix}_nomorph"
#     if feat_reduction is not None:
#         prefix = f"{prefix}_umap"
#     return prefix


# def make_exp_ids(n_exp, string_len=4, prefix=None):
#     alphabet = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
#     result = []
#     for i in range(n_exp):
#         s = []
#         current = i
#         # Convert the number `i` into a base-26-like string
#         for _ in range(string_len):
#             s.append(alphabet[current % 26])  # Find corresponding letter
#             current //= 26  # Move to the next "digit"
#         result.append("".join(reversed(s)))  # Reversing to get correct order
#     if prefix != None:
#         result = [f"{prefix}{x}" for x in result]
#     return result


# def get_base_train_config(is_cv: bool = False):
#     train_config = {
#         "endpoint": None,
#         "exp_id": None,
#         "dir_save": None,
#         "batch_size": None,
#         "epochs": None,
#         "lr": None,
#         "scheduler": None,
#         "num_worker": 4,
#         "optimiser": "sgd",
#         "weight_decay": 0,
#         "l2_nll": 1e-4,
#         "device": "cuda",
#     }

#     model_config = {
#         "m_conv": "GCN",
#         "m_act_conv": "LeakyReLU",
#         "m_act_fcn": "SELU",
#         "m_use_last": False,
#         "m_use_norm_fcn": True,
#     }
#     if is_cv:
#         graph_config = {
#             "embedding": None,
#             "morph": None,
#             "weighted": None,
#             "dir_folds": None,
#             "dir_graph": None,
#         }
#     else:
#         graph_config = {
#             "embedding": None,
#             "morph": None,
#             "weighted": None,
#             "csv_train": None,
#             "csv_test_internal": None,
#             "csv_test_external": None,
#             "dir_graph": None,
#         }

#     config = {
#         "train_config": train_config,
#         "graph_config": graph_config,
#         "model_config": model_config,
#     }
#     return config


# def get_config(
#     is_cv: bool,
#     node_morph: list[bool],
#     node_embd: list[str],
#     is_weighted: list[bool],
#     lr: list[float],
#     epochs: list[int],
#     batch_size: list[int],
#     use_scheduler: list[bool],
#     endpoint: str,
#     dir_output_train: str,
#     dir_graph: str,
#     root_csv: str = None,
# ):
#     base_config = get_base_train_config(is_cv=is_cv)
#     base_config["train_config"]["dir_save"] = dir_output_train
#     base_config["train_config"]["endpoint"] = endpoint
#     base_config["graph_config"]["dir_graph"] = dir_graph
#     if not is_cv:
#         base_config["graph_config"][
#             "csv_train"
#         ] = f"{root_csv}/{endpoint}/validation/train.csv"
#         base_config["graph_config"][
#             "csv_test_internal"
#         ] = f"{root_csv}/{endpoint}/validation/test_internal.csv"
#         base_config["graph_config"][
#             "csv_test_external"
#         ] = f"{root_csv}/{endpoint}/validation/test_external.csv"

#     exp_ids = make_exp_ids(
#         n_exp=len(node_morph)
#         * len(node_embd)
#         * len(is_weighted)
#         * len(lr)
#         * len(epochs)
#         * len(batch_size)
#         * len(use_scheduler)
#     )
#     combination_list = list(
#         itertools.product(
#             node_morph, node_embd, is_weighted, lr, epochs, batch_size, use_scheduler
#         )
#     )

#     all_configs = []
#     for ix, c in enumerate(combination_list):
#         if (
#             c[0] == False and c[1] == None
#         ):  # must use one of type node, either morph or embd
#             pass
#         else:
#             current_config = deepcopy(base_config)
#             current_config["graph_config"]["morph"] = c[0]
#             current_config["graph_config"]["embedding"] = c[1]
#             current_config["graph_config"]["weighted"] = c[2]
#             current_config["train_config"]["lr"] = c[3]
#             current_config["train_config"]["epochs"] = c[4]
#             current_config["train_config"]["batch_size"] = c[5]
#             current_config["train_config"]["scheduler"] = c[6]
#             current_config["train_config"]["exp_id"] = exp_ids[ix]
#             all_configs.append(current_config)

#     return all_configs


# if __name__ == "__main__":
#     is_cv = True
#     node_morph = [True, False]
#     node_embd = ["dino", False]
#     is_weighted = [True]
#     lr = [1e-2]
#     epochs = [100]
#     batch_size = [32]
#     use_scheduler = [True]
#     endpoint = "os"
#     run_name = "baseline"
#     dir_output_train = "dir_out"
#     dir_graph = "dir_graph"
#     root_csv = "root_csv"
#     configs = get_config(
#         is_cv=is_cv,
#         node_morph=node_morph,
#         node_embd=node_embd,
#         is_weighted=is_weighted,
#         lr=lr,
#         epochs=epochs,
#         batch_size=batch_size,
#         use_scheduler=use_scheduler,
#         endpoint=endpoint,
#         run_name=run_name,
#         dir_output_train=dir_output_train,
#         dir_graph=dir_graph,
#         root_csv=root_csv,
#     )
