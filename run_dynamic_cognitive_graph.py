import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import os
import shapely
import configparser

from network_processing import get_enriched_node_df, get_enriched_edge_df, create_network_graph, \
    set_network_node_attributes, set_cycling_saliency
from trajectory_processing import get_trajectories, filter_trips, get_trajectory_edges, create_graph, \
    create_base_subgraph, remove_duplicate_traj_segments
from clustering import create_distance_matrix, get_level_clusters
from plotting import plot_base_subgraph, plot_clustering, plot_route_task, plot_metric_values, plot_final_route
from memory_distortions import get_category_center, get_adjusted_category_center, set_shifted_node_location
from cognitive_routing import create_abstract_graph, get_next_step, L1_NODE_BASE, L2_NODE_BASE
from similarity_comparison import get_distances, get_metric_values


def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'paths.ini'))

    node_path = os.path.abspath(config['paths']['nodes'])
    edge_path = os.path.abspath(config['paths']['edges'])
    significance_path = os.path.abspath(config['paths']['significance'])
    od_path = os.path.abspath(config['paths']['od'])
    out_dir = os.path.abspath(config['paths']['out_dir'])
    credentials = os.path.abspath(config['paths']['credentials'])

    # dataframe with enriched nodes and edges
    nodes = get_enriched_node_df(node_path, significance_path, od_path)
    edges = get_enriched_edge_df(edge_path, 'ln_desc_after')

    # the base with saliency values
    G = create_network_graph(edges, 'source', 'target')
    G, nodes = set_network_node_attributes(G, edges, nodes)

    saliency_dict = {'avg_speed': {'factor': 0.2, 'high': 0},
                     'lanes_updated': {'factor': 0.2, 'high': 0},
                     'bike_lanes': {'factor': 0.2, 'high': 1},
                     'degree': {'factor': 0.1, 'high': 0},
                     'bedeutung': {'factor': 0.1, 'high': 1},
                     'destinations': {'factor': 0.2, 'high': 1}}
    G = set_cycling_saliency(G, saliency_dict)

    # relevant for later
    minx, miny, maxx, maxy = nodes.total_bounds
    polygon_from_bounds = shapely.box(minx, miny, maxx, maxy)

    trips = get_trajectories(credentials)
    trips = filter_trips(trips[trips.within(polygon_from_bounds)], 1000, 5000, 500)

    # only for testing
    trip_ids = [213663, 218658, 265956, 279873, 281980, 287527, 296171, 303317, 322378, 352336, 418952, 432747, 458765]
    trip_id = 296171
    test_traj = trips[trips['trip_id'] == trip_id]

    # map-match trajectory to the network.
    traj_path = remove_duplicate_traj_segments(get_trajectory_edges(test_traj, nodes, G))
    traj_G = create_graph(G, traj_path)
    start_node, end_node = traj_path[0], traj_path[-1]

    # create the shortest path graph for further use.
    sp = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
    sp_G = create_graph(G, sp)

    # construct base subgraph based on routes that is not 15% longer than the shortest path between the OD pair
    subgraph, hull = create_base_subgraph(test_traj, 500, nodes, G, sp, start_node, end_node, 1.15)
    all_pairs_sp = dict(nx.shortest_path_length(subgraph, weight='length'))
    max_sp = max(value for inner_dict in all_pairs_sp.values() for value in inner_dict.values())
    nodes_list = list(subgraph.nodes())

    plot_base_subgraph(subgraph, traj_G, sp_G, hull, out_dir)

    # distance matrix and clustering
    attributes = ['degree', "bedeutung", "avg_speed", 'destinations', 'lanes_updated', 'bike_lanes', 'elevation']
    distance_matrix = create_distance_matrix(nodes_list, 0., subgraph, attributes)
    Z = linkage(squareform(distance_matrix), method='complete', optimal_ordering=False)

    # getting the num
    level_2_cluster_count = np.cbrt(len(nodes_list))
    level_1_cluster_count = min(np.square(level_2_cluster_count), 15)

    # individual clusters and the heights for plotting
    L2_clusters, cluster_level_2_height = get_level_clusters(Z, level_2_cluster_count)
    L1_clusters, cluster_level_1_height = get_level_clusters(Z, level_1_cluster_count)

    # assign node to clusters at different levels.
    cluster_member = {node: (node, L1_clusters[nodes_list.index(node)], L2_clusters[nodes_list.index(node)]) for node in nodes_list}
    plot_clustering(Z, cluster_level_1_height, cluster_level_2_height, out_dir)

    # get not distorted centroid coordinates
    cluster_centroids_1 = get_category_center(subgraph, L1_clusters, nodes_list)
    cluster_centroids_2 = get_category_center(subgraph, L2_clusters, nodes_list)

    folder_name = f"route_{trip_id}"
    os.makedirs(folder_name, exist_ok=True)

    path = [start_node]
    next_node = start_node
    visited_nodes = []
    counter = 0

    # route planning
    while next_node != end_node:
        cognitive_G = create_abstract_graph(subgraph, cluster_member, all_pairs_sp, max_sp, next_node, visited_nodes=visited_nodes)
        abstract_graph_v = create_abstract_graph(subgraph, cluster_member, all_pairs_sp, max_sp, next_node)
        next_node, level_node = get_next_step(cognitive_G, next_node, end_node, all_pairs_sp, cluster_member, subgraph)
        visited_nodes.append(next_node)

        if level_node is not None:
            visited_nodes.append(L1_NODE_BASE + cluster_member[next_node][1])
            visited_nodes.append(L2_NODE_BASE + cluster_member[next_node][2])
        path.append(next_node)
        counter += 1

        plot_route_task(abstract_graph_v, subgraph, G, cluster_centroids_1, cluster_centroids_2, level_node, sp_G, path,
                        next_node, traj_G, start_node, end_node, counter, folder_name)

    planned_path_G = create_graph(G, path)
    plot_final_route(subgraph, sp_G, planned_path_G, traj_G, start_node, end_node, out_dir)

    param = 'saliency'
    sp_saliency = get_metric_values(sp, sp_G, param)
    traj_saliency = get_metric_values(traj_path, traj_G, param)
    cog_path_saliency = get_metric_values(path, planned_path_G, param)

    plot_metric_values(sp_saliency, traj_saliency, cog_path_saliency, param, trip_id, out_dir)
    distances = get_distances(traj_path, sp, path)

if __name__ == "__main__":
    main()
