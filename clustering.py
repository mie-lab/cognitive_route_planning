import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster


def create_distance_matrix(nodes_list, dist_weight, subgraph, attribute_list):
    """
    Create a distance matrix for the nodes in the subgraph.
    :param nodes_list: list of node ids in the subgraph.
    :param dist_weight: weight of distance in the clustering.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param attribute_list:  attributes which values are used to calculate the distance.
    :return: distance_matrix
    """

    distance_matrix = np.zeros((len(nodes_list), len(nodes_list)))

    for i, node_i in enumerate(nodes_list):
        for j, node_j in enumerate(nodes_list):
            if i != j:

                euclidean_distance = np.sqrt(np.sum((np.array(subgraph.nodes[node_i]['location']) - np.array(
                    subgraph.nodes[node_j]['location'])) ** 2)) / 1000
                ij_diff = 0

                for attribute in attribute_list:
                    node_i_attribute = subgraph.nodes[node_i][attribute]
                    node_j_attribute = subgraph.nodes[node_j][attribute]
                    if pd.isnull(node_i_attribute) or pd.isnull(node_j_attribute):
                        ij_diff += 0.5
                    else:
                        ij_diff += (node_i_attribute - node_j_attribute) ** 2
                ij_diff = dist_weight * euclidean_distance + (1 - dist_weight) * ij_diff
            else:
                ij_diff = 0

            distance_matrix[i, j] = ij_diff

    return distance_matrix


def get_level_clusters(linkage, cluster_counts):
    """
    Get node cluster assignment and the height of the cluster level.
    :param linkage: clustering results.
    :param cluster_counts: number of clusters.
    :return: node cluster assignment and the height of the cluster level
    """

    cluster_height = linkage[-int(cluster_counts), 2]
    clusters_at_level = fcluster(linkage, cluster_height, criterion='distance')

    return clusters_at_level, cluster_height
