import networkx as nx
import numpy as np


def get_category_center(subgraph, clusters, nodes_list):
    """
    Get the original cluster centroids.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param clusters:
    :param nodes_list: list of node ids in the subgraph
    :return: dictionary of cluster centroids.
    """
    cluster_centroids = {}
    for cluster in np.unique(clusters):
        centroid = [0, 0]
        nodes_in_cluster = np.where(clusters == cluster)[0]
        for node_index in nodes_in_cluster:
            node_id = nodes_list[node_index]
            location = nx.get_node_attributes(subgraph, "location")[node_id]
            centroid[0] += location[0]
            centroid[1] += location[1]
        centroid[0] /= len(nodes_in_cluster)
        centroid[1] /= len(nodes_in_cluster)

        cluster_centroids[cluster] = centroid
    return cluster_centroids


def get_adjusted_category_center(clusters, sp_lengths, start_node, subgraph, nodes_list):
    """
    Get the distorted cluster centroids based on category adjustments.
    :param clusters: node cluster assignment.
    :param sp_lengths: all-pairs shortest paths.
    :param start_node: origin node.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param nodes_list: list of node ids in the subgraph.
    :return: adjusted cluster centroids.
    """
    cluster_centroids = {}
    for cluster in np.unique(clusters):
        nodes_in_cluster = np.where(clusters == cluster)[0]
        cluster_nodes = [nodes_list[node_index] for node_index in nodes_in_cluster]
        distances = [(sp_lengths[start_node][node] + sp_lengths[node][start_node]) / 2. for node in cluster_nodes]
        order = np.argsort(distances)
        weights = 1 / (order.argsort() + 1)
        locations_x = [nx.get_node_attributes(subgraph, "location")[node][0] for node in cluster_nodes]
        locations_y = [nx.get_node_attributes(subgraph, "location")[node][1] for node in cluster_nodes]

        c_x = np.sum(locations_x * weights) / np.sum(weights)
        c_y = np.sum(locations_y * weights) / np.sum(weights)

        cluster_centroids[cluster] = (c_x, c_y)
    return cluster_centroids


def category_adjustment_model(node_loc, cluster_center):
    """
    Calculate the recalled node location based on distorted cluster centroids.
    :param node_loc: not distorted node location.
    :param cluster_center: distorted cluster centroids based on category adjustments.
    :return: adjusted node location coordinates.
    """
    # (λ) is the weight of the cluster center in the adjustment model.
    lambda_weight = 0.9
    recalled_loc = (lambda_weight * node_loc) + ((1 - lambda_weight) * cluster_center)

    return recalled_loc


def set_shifted_node_location(clusters, subgraph, nodes_list, shifted_cluster_centroids, cluster_name):
    """
    Set the adjusted node location in the subgraph.
    :param clusters: node cluster assignment.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param nodes_list: list of node ids in the subgraph.
    :param shifted_cluster_centroids: adjusted cluster centroids.
    :param cluster_name: cluster names to differentiate between clusters.
    """
    node_attr_loc_cluster = {}

    for n in subgraph.nodes():
        node_index = nodes_list.index(n)
        cluster = clusters[node_index]
        location = np.array([nx.get_node_attributes(subgraph, "location")[n][0],
                             nx.get_node_attributes(subgraph, "location")[n][1]])
        node_attr_loc_cluster[n] = category_adjustment_model(location, np.array(shifted_cluster_centroids[cluster]))

        nx.set_node_attributes(subgraph, node_attr_loc_cluster, cluster_name)
