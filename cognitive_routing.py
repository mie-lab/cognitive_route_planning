import networkx as nx
import numpy as np

L1_NODE_BASE = 100000
L2_NODE_BASE = 2 * L1_NODE_BASE
ALPHA = 0.5


def get_level_and_cluster(node):
    """
    Get node level and cluster.
    :param node: node id.
    :return: cluster level and cluster id.
    """
    return node // L1_NODE_BASE, node % L1_NODE_BASE


def compute_abstract_edge_weight(source, target, subgraph, cluster_membership, all_pairs_sp, max_sp):
    """
    Compute the weight of an edge in the abstract graph.
    :param source: origin node.
    :param target: destination node.
    :param subgraph: generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param cluster_membership: node cluster assignment.
    :param all_pairs_sp: all-pairs shortest paths.
    :param max_sp: longest shortest path.
    :return: an average of the weights in both directions of the abstract graph nodes.
    """
    dist1 = get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, source, target)
    dist2 = get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, target, source)
    return (dist1 + dist2) / 2


def get_mixed_node_distance(node_id_2, node_id_1, cluster_membership, all_pairs_sp, max_sp):
    """
    Calculate the normalized mean distance between node and a cluster.
    :param node_id_2: node id.
    :param node_id_1: node id.
    :param cluster_membership: node cluster assignment.
    :param all_pairs_sp: all-pairs shortest paths.
    :param max_sp: maximum length the shortest path.
    :return: normalized mean distance between node and a cluster.
    """
    level, cluster = get_level_and_cluster(node_id_2)
    cluster_nodes = [node for node in cluster_membership.keys() if cluster_membership[node][level] == cluster]
    distances = [all_pairs_sp[node_id_1][node] for node in cluster_nodes]

    return np.mean(distances) / max_sp


def get_cluster_node_distance(node_id_1, node_id_2, cluster_membership, all_pairs_sp, max_sp):
    """
    Calculate the normalized mean distance between two clusters.
    :param node_id_1: cluster node id.
    :param node_id_2: cluster node id.
    :param cluster_membership: node cluster assignment.
    :param all_pairs_sp: all-pairs shortest paths.
    :param max_sp: maximum length the shortest path.
    :return: normalized mean distance between two clusters.
    """

    level1, cluster1 = get_level_and_cluster(node_id_1)
    cluster_1_nodes = [node for node in cluster_membership.keys() if cluster_membership[node][level1] == cluster1]
    level2, cluster2 = get_level_and_cluster(node_id_2)
    cluster_2_nodes = [node for node in cluster_membership.keys() if cluster_membership[node][level2] == cluster2]

    distances = []
    for cluster_1_node in cluster_1_nodes:
        for cluster_2_node in cluster_2_nodes:
            distances.append(all_pairs_sp[cluster_1_node][cluster_2_node])

    return np.mean(distances) / max_sp


def get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, node_id_1, node_id_2):
    """
    Calculate distance between relevant nodes which can be either the cluster or network nodes.
    :param subgraph: generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param cluster_membership: node cluster assignment.
    :param all_pairs_sp: all-pairs shortest paths.
    :param max_sp: maximum length the shortest path.
    :param node_id_1: node id.
    :param node_id_2: node id.
    :return: weighted distance between relevant nodes.
    """

    if node_id_2 in cluster_membership:
        # get saliency for network node
        saliency = subgraph.nodes[node_id_2]['saliency']
    else:
        # compute saliency for cluster node
        level, cluster = get_level_and_cluster(node_id_2)
        saliency = min([subgraph.nodes[node]['saliency'] for node in cluster_membership if
                        cluster_membership[node][level] == cluster])

    # scenario 1: two nodes
    if (node_id_1 in cluster_membership) and (node_id_2 in cluster_membership):
        distance = subgraph.edges[(node_id_1, node_id_2)]['length']
    # scenario 2: node and cluster
    elif node_id_1 in cluster_membership:
        distance = get_mixed_node_distance(node_id_2, node_id_1, cluster_membership, all_pairs_sp, max_sp)
    # scenario 3: cluster and node
    elif node_id_2 in cluster_membership:
        distance = get_mixed_node_distance(node_id_1, node_id_2, cluster_membership, all_pairs_sp, max_sp)
    # scenario 4: cluster and cluster
    else:
        distance = get_cluster_node_distance(node_id_1, node_id_2, cluster_membership, all_pairs_sp, max_sp)

    return ALPHA * distance + (1 - ALPHA) * (1 - saliency)


def create_abstract_graph(subgraph, cluster_membership, all_pairs_sp, max_sp, start_node, visited_nodes=[]):
    """
    Create an abstract subgraph for the cognitive routing algorithm at every step.
    The 'length' attribute depending on different scenarios is the mean sp distance.
    :param subgraph: generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param cluster_membership: node cluster assignment.
    :param all_pairs_sp: all-pairs shortest paths.
    :param max_sp: longest shortest path.
    :param start_node: origin node.
    :param visited_nodes: list of visited nodes.
    :return: an abstract subgraph containing nodes at different levels of the clustering.
    """

    edges_out = []
    edge_weights = {}

    # edges that have nodes in two different clusters
    edges_inter = list(set([e for e in subgraph.edges() if cluster_membership[e[0]][1] != cluster_membership[e[1]][1]]))

    current_lev_1_cluster = cluster_membership[start_node][1]
    current_lev_2_cluster = cluster_membership[start_node][2]

    for e in subgraph.edges():
        if (cluster_membership[e[0]][1] == current_lev_1_cluster) and (
                cluster_membership[e[1]][1] == current_lev_1_cluster):
            edges_out.append(e)
            edge_weights[e] = get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, e[0], e[1])
    for e in edges_inter:
        if (cluster_membership[e[0]][2] == current_lev_2_cluster) and (
                cluster_membership[e[1]][2] == current_lev_2_cluster):
            if cluster_membership[e[0]][1] == current_lev_1_cluster:
                # need to check if there is an edge between physical points(i.e., they are neighbors)
                e_new = (e[0], L1_NODE_BASE + cluster_membership[e[1]][1])
            elif cluster_membership[e[1]][1] == current_lev_1_cluster:
                # need to check if there is an edge between physical points(i.e., they are neighbors)
                e_new = (L1_NODE_BASE + cluster_membership[e[0]][1], e[1])
            else:
                e_new = (L1_NODE_BASE + cluster_membership[e[0]][1], L1_NODE_BASE + cluster_membership[e[1]][1])
            edges_out.append(e_new)
            edge_weights[e_new] = get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, e_new[0], e_new[1])

        if cluster_membership[e[0]][2] != cluster_membership[e[1]][2]:
            if cluster_membership[e[0]][2] == current_lev_2_cluster:
                if cluster_membership[e[0]][1] == current_lev_1_cluster:
                    # need to check if there is an edge between physical points(i.e., they are neighbors)
                    e_new = (e[0], L2_NODE_BASE + cluster_membership[e[1]][2])
                else:
                    e_new = (
                        L1_NODE_BASE + cluster_membership[e[0]][1], L2_NODE_BASE + cluster_membership[e[1]][2])
            elif cluster_membership[e[1]][2] == current_lev_2_cluster:
                if cluster_membership[e[1]][1] == current_lev_1_cluster:
                    # need to check if there is an edge between physical points(i.e., they are neighbors)
                    e_new = (L2_NODE_BASE + cluster_membership[e[0]][2], e[1])
                else:
                    e_new = (L2_NODE_BASE + cluster_membership[e[0]][2],
                             L1_NODE_BASE + cluster_membership[e[1]][1])
            else:
                e_new = (L2_NODE_BASE + cluster_membership[e[0]][2],
                         L2_NODE_BASE + cluster_membership[e[1]][2])
            edges_out.append(e_new)
            edge_weights[e_new] = get_cluster_distances(subgraph, cluster_membership, all_pairs_sp, max_sp, e_new[0], e_new[1])
    keep_edges = []
    for edge in edges_out:
        if (edge[0] not in visited_nodes) and (edge[1] not in visited_nodes):
            keep_edges.append(edge)

    abstract_graph = nx.Graph(keep_edges)
    nx.set_edge_attributes(abstract_graph, {edge: edge_weights[edge] for edge in keep_edges}, name='length')

    return abstract_graph


def get_next_step(abstract_graph, start_node, end_node, all_pairs_sp, cluster_membership, subgraph):
    """
    Get the next node in the cognitive routing algorithm.
    :param abstract_graph: abstract subgraph containing nodes at different levels of the clustering.
    :param start_node: origin node.
    :param end_node: destination node.
    :param all_pairs_sp: all-pairs shortest paths.
    :param cluster_membership: node cluster assignment.
    :param subgraph: generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :return: next node in the cognitive routing algorithm.
    """
    start_lev_1_cluster = cluster_membership[start_node][1]
    start_lev_2_cluster = cluster_membership[start_node][2]
    end_lev_1_cluster = cluster_membership[end_node][1]
    end_lev_2_cluster = cluster_membership[end_node][2]

    if start_lev_2_cluster != end_lev_2_cluster:
        destination = L2_NODE_BASE + end_lev_2_cluster
    elif start_lev_1_cluster != end_lev_1_cluster:
        destination = L1_NODE_BASE + end_lev_1_cluster
    else:
        destination = end_node
    path = nx.shortest_path(abstract_graph, source=start_node, target=destination, weight='length', method='dijkstra')
    next_node = path[1]
    if next_node > L1_NODE_BASE:
        level, cluster = get_level_and_cluster(next_node)
        cluster_nodes = [node for node in cluster_membership if cluster_membership[node][level] == cluster]
        distances = [all_pairs_sp[start_node][node] for node in cluster_nodes]
        destination = cluster_nodes[np.argsort(distances)[0]]
        path = nx.shortest_path(subgraph, source=start_node, target=destination, weight='length', method='dijkstra')

        return path[1], next_node
    else:
        return next_node, None