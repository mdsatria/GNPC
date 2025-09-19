from functools import partial
from multiprocessing.pool import Pool

import networkx as nx
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity

from gnpc.default_modules import *


def cosine_similarity_sklearn(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def process_point(centroids, feature, mat_connect, is_weighted, ix):
    local_edge_index = []
    local_edge_weight = []
    point = centroids[ix]
    for dxy in mat_connect:
        neighbour = point + dxy
        neighbour_index = np.where((centroids == neighbour).all(axis=1))[0]
        if len(neighbour_index) > 0:
            coord_neighbour = neighbour_index[0]
            local_edge_index.append([ix, coord_neighbour])
            if is_weighted:
                first_node = feature[ix]
                second_node = feature[coord_neighbour]
                similarity = cosine_similarity_sklearn(first_node, second_node)
                local_edge_weight.append(similarity)

    return local_edge_index, local_edge_weight


def create_weighted_graph_neighbour(
    centroids, features, topleft, distance=1024, n_neighbour=4, is_weighted=True
):
    if n_neighbour == 4:
        mat_connect = np.array(
            [
                (0, distance),
                (distance, 0),
                (0, -distance),
                (-distance, 0),
            ]
        )
    elif n_neighbour == 8:
        mat_connect = np.array(
            [
                (0, distance),
                (distance, 0),
                (0, -distance),
                (-distance, 0),
                (distance, distance),
                (distance, -distance),
                (-distance, distance),
                (-distance, -distance),
            ]
        )

    part_func = partial(process_point, centroids, features, mat_connect, is_weighted)
    with Pool() as pool:
        temp_result = pool.map(part_func, range(len(centroids)))
    edge_index = []
    edge_weight = []
    for res in temp_result:
        edge_index.extend(res[0])
        edge_weight.extend(res[1])

    edge_index = np.array(edge_index)
    edge_weight = np.array(edge_weight)

    dct = {
        "x": features,
        "edge_index": edge_index.T,
        "coordinates": centroids,
        "topleft": topleft,
    }

    if is_weighted:
        dct["edge_weight"] = edge_weight.T

    return dct


def create_weighted_graph_delaunay(
    centroids, features, topleft, threshold_distance=2_000, is_weighted=True
):
    G = nx.Graph()
    for i, point in enumerate(centroids):
        G.add_node(i, pos=point)

    tri = Delaunay(centroids)
    edge_index = []
    edge_weight = []
    for simplex in tri.simplices:
        combinations = list(itertools.combinations(simplex, 2))
        for c1, c2 in combinations:
            point1 = centroids[c1]
            point2 = centroids[c2]
            dist = np.linalg.norm(point1 - point2)
            if dist <= threshold_distance:
                if not G.has_edge(c1, c2):
                    weight = cosine_similarity_sklearn(features[c1], features[c2])
                    edge_index.append([c1, c2])
                    edge_weight.append(weight)

    edge_index = np.array(edge_index)
    edge_weight = np.array(edge_weight)

    dct = {
        "x": features,
        "edge_index": edge_index.T,
        "coordinates": centroids,
        "topleft": topleft,
        "threshold_distance": threshold_distance,
    }
    if is_weighted:
        dct["edge_weight"] = edge_weight.T

    return dct


# def get_cohort_stat(graph_files, output_file=None):
#     N_total = 0
#     mu_global = 0.0
#     var_global = 0.0

#     for i, g_in in enumerate(graph_files):
#         # Load tensor from disk

#         gr = torch.load(g_in)
#         X = gr.x
#         # Calculate local statistics
#         N_i = X.shape[0]
#         mu_i = torch.mean(X, dim=0)
#         var_i = torch.var(X, dim=0, unbiased=True)

#         # Update global statistics
#         delta = mu_i - mu_global
#         mu_global = mu_global + (N_i / (N_total + N_i)) * delta
#         var_global = (
#             N_total * var_global + N_i * var_i + N_i * delta * (mu_i - mu_global)
#         ) / (N_total + N_i)

#         N_total += N_i

#     if output_file is not None:
#         torch.save({"mean": mu_global, "std": var_global}, output_file)
#     return mu_global, var_global


# def standardised_graphs(graph_files_input, dir_out, mu_global=None, var_global=None):
#     if (mu_global == None) or (var_global == None):
#         mu_global, var_global = get_cohort_stat(graph_files_input)

#     pbar = tqdm(graph_files_input)
#     for i, g_in in enumerate(pbar):
#         pbar.set_description(
#             f"normalising graph {i+1}/{len(graph_files_input)}: {g_in}"
#         )
#         fpath_out = os.path.join(dir_out, os.path.basename(g_in))
#         # print(fpath_out)

#         gr = torch.load(g_in)
#         X = gr.x

#         # Normalize tensor
#         X = (X - mu_global) / torch.sqrt(var_global)

#         gr.x = X.to(torch.float32)
#         torch.save(gr, fpath_out)
#         # print(f"graph i")
