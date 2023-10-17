import ast

import networkx as nx
import numpy as np

_HEAD_TO_COLUMN_MAP = {
    "biota": "CATAMI Biota",
    "substrate": "CATAMI Substrate",
    "relief": "CATAMI Relief",
    "bedforms": "CATAMI Bedforms",
}

_HEAD_TO_MASK_MAP = {
    "biota": "Biota Mask",
    "substrate": "Substrate Mask",
    "relief": "Relief Mask",
    "bedforms": "Bedforms Mask",
}


# Utilities for graph functions
def check_elements(list_a, list_b):
    return any(element in set(list_b) for element in set(list_a))


# General graph related functions
def create_subgraph(G, radius):
    G = G.copy()

    G.remove_edges_from(nx.selfloop_edges(G))

    root_nodes = [node for node in nx.topological_sort(G) if G.in_degree(node) == 0]

    pruned_G = nx.DiGraph()

    nodes_within_radius = []
    nodes_exact_radius = []
    for start_node in root_nodes:
        sub_G = nx.ego_graph(G=G, n=start_node, radius=radius)

        nodes_and_distances = nx.single_source_shortest_path_length(G, start_node)

        # nodes within radius
        nodes_within_radius_sub = [
            node for node, distance in nodes_and_distances.items() if distance <= radius
        ]

        # nodes with exact radius (include start_node)
        nodes_exact_radius_sub = [
            node for node, distance in nodes_and_distances.items() if distance == radius
        ]

        # updates
        nodes_within_radius += nodes_within_radius_sub
        nodes_exact_radius += nodes_exact_radius_sub
        pruned_G.update(sub_G)

    return pruned_G, nodes_within_radius, nodes_exact_radius


def filter_data(df, head, depth, leaf_nodes, column):
    df = df.copy()
    df = df.dropna(subset=[column])

    df.loc[:, column] = df.loc[:, column].fillna("")
    df.loc[:, f"_{head}"] = df.loc[:, column].apply(ast.literal_eval)
    df.loc[:, f"{head}_depth_{depth}_applicable"] = df.loc[:, f"_{head}"].apply(
        check_elements, args=(leaf_nodes,)
    )
    df.drop([f"_{head}"], axis=1, inplace=True)

    return df


def filter_data_with_hops(df, hops_dict, root_graphs, column):
    for head, hop in hops_dict.items():
        root_graph = root_graphs[head]
        _, nodes_within_radius, nodes_exact_radius = create_subgraph(root_graph, hop)
        df = filter_data(df, head, hop, nodes_exact_radius, column)
    return df, nodes_within_radius


def find_max_depth(graph):
    max_depth = 0

    graph_nodes = set(graph.nodes())
    nodes_within_max_depth = set()

    while not graph_nodes.issubset(nodes_within_max_depth):
        _, nodes_within_radius, _ = create_subgraph(graph, max_depth)
        nodes_within_max_depth = set(nodes_within_radius)
        max_depth += 1

    return max_depth - 1
