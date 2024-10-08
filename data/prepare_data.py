import enum
import os
import pickle

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.constants import *

def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_data(training_config, device):
    dataset_name = training_config["dataset_name"].lower()
    should_visualise = training_config["should_visualise"]

    if dataset_name == DatasetType.CORA.name.lower():
        # shape = (N, F_IN) where N is the number of nodes and F_IN is the number of input features
        node_features_csr = pickle_read(os.path.join(CORA_PATH, "node_features.csr"))
        # shape = (N, 1)
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, "node_labels.npy"))
        # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix
        adjacency_list_dict = pickle_read(
            os.path.join(CORA_PATH, "adjacency_list.dict")
        )

        # normalise the features (helps with training)
        node_features_csr = normalise_features_sparse(node_features_csr)
        num_of_nodes = len(node_labels_npy)

        # shape = (2, E), where E is the number of edges, and 2 for source and target nodes
        topology = build_edge_index(
            adjacency_list_dict, num_of_nodes, add_self_edges=True
        )

        if should_visualise:
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)
            visualise_graph(topology, node_labels_npy, dataset_name)

        # convert data to tensors
        topology = torch.tensor(topology, dtype=torch.long, device=device)
        node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)
        node_features = torch.tensor(node_features_csr.todense(), device=device)

        # indices for dataset splits
        train_indices = torch.arange(
            CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device
        )
        val_indices = torch.arange(
            CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device
        )
        test_indices = torch.arange(
            CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device
        )

        return (
            node_features,
            node_labels,
            topology,
            train_indices,
            val_indices,
            test_indices,
        )
    else:
        raise Exception(f"{dataset_name} not yet supported")


def normalise_features_sparse(node_features_sparse):
    if not sp.issparse(node_features_sparse):
        raise TypeError(f"Expected a sparse matrix, got {type(node_features_sparse)}")

    # calculate the sum of features of every node (sum across all columns)
    node_features_sum = np.array(node_features_sparse.sum(-1))

    # invert the sum of features
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # handling infinite values
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.0

    # create a diagonal matrix of the inverse of the sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we ad it to the edge index
            if (src_node, trg_node) not in seen_edges:
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E) where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # if not isinstance(edge_index.cpu().numpy()):
    #     raise TypeError(f'Expected a NumPy array got {type(edge_index)}.')

    # store input and output degree
    in_degrees = np.zeros(num_of_nodes, dtype=int)
    out_degrees = np.zeros(num_of_nodes, dtype=int)

    num_of_edges = edge_index.shape[1]
    for node in range(num_of_edges):
        source_node_id = edge_index[0, node]
        target_node_id = edge_index[1, node]

        in_degrees[target_node_id] += 1
        out_degrees[source_node_id] += 1

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12, 8), dpi=300)
    fig.subplots_adjust(hspace=0.6)
    plt.subplot(311)
    plt.plot(in_degrees, color="red")
    plt.xlabel("node id")
    plt.ylabel("in-degree count")
    plt.title("Input degree for different node ids")

    plt.subplot(312)
    plt.plot(out_degrees, color="green")
    plt.xlabel("node id")
    plt.ylabel("out-degree count")
    plt.title("Out degree for different node ids")

    plt.subplot(313)
    plt.plot(hist, color="blue")
    plt.xlabel("node degree")
    plt.ylabel("# nodes for a given out-degree")
    plt.title(f"Node out-degree distribution for {dataset_name} dataset")
    plt.xticks(np.arange(0, len(hist), 5.0))

    plt.grid(True)
    # plt.show()


def visualise_graph(
    edge_index,
    node_labels,
    dataset_name,
    visualisation_tool=GraphVisualisationTool.IGRAPH,
):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    if isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.cpu().numpy()

    num_of_nodes = len(node_labels)
    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

    if visualisation_tool == GraphVisualisationTool.NETWORKX:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edge_index_tuples)
        nx.draw_networkx(nx_graph)
        plt.show()

    elif visualisation_tool == GraphVisualisationTool.IGRAPH:
        # create graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        # graph style details
        visual_style = {}
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 35

        # edge thickness dependant on the the number of shortest paths that go through an edge
        edge_weights_raw = np.clip(
            np.log(np.asarray(ig_graph.edge_betweenness()) + 1e16), a_min=0, a_max=None
        )
        edge_weights_raw_normalised = edge_weights_raw / np.max(edge_weights_raw)
        edge_weights = [w**6 for w in edge_weights_raw_normalised]
        visual_style["edge_width"] = edge_weights

        # simple heuristic for vertex size.
        visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]
        if dataset_name.lower() == DatasetType.CORA.name.lower():
            visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]
        else:
            print("Feel free to add custom color scheme.")

        visual_style["layout"] = ig_graph.layout_kamada_kawai()
        print("Plotting results ... (it may take couple of seconds)")
        ig.plot(ig_graph, "output.png", **visual_style)
        print("done")

    else:
        raise Exception(f"Visualisation tool {visualisation_tool.name} not supported.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {"dataset_name": DatasetType.CORA.name, "should_visualise": False}


node_features, node_labels, edge_index, train_indices, val_indices, test_indices = (
    load_graph_data(config, device)
)

print(node_features.shape, node_features.dtype)
print(node_labels.shape, node_labels.dtype)
print(edge_index.shape, edge_index.dtype)
print(train_indices.shape, train_indices.dtype)

num_of_nodes = len(node_labels)
plot_in_out_degree_distributions(edge_index, num_of_nodes, config["dataset_name"])
visualise_graph(edge_index, node_labels, config["dataset_name"])
