import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import os
from IPython import display
from cognitive_routing import L1_NODE_BASE, L2_NODE_BASE
from trajectory_processing import create_graph


def plot_base_subgraph(subgraph, traj_G, sp_G, hull, out_dir):
    """
    Plot the constructed base subgraph with the shortest path, the original trajectory and resulting hull.
    :param out_dir: output directory.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param traj_G: original trajectory graph.
    :param sp_G: shortest path graph.
    :param hull: a buffered convex hull of the trajectory.
    :return:
    """
    fig, ax = plt.subplots(1, figsize=(6, 6))
    pos = {node: (data['location'][0], data['location'][1]) for node, data in subgraph.nodes(data=True)}
    pos_t = {node: (data['location'][0], data['location'][1]) for node, data in traj_G.nodes(data=True)}
    sp_pos = {node: (data['location'][0], data['location'][1]) for node, data in sp_G.nodes(data=True)}

    # draw subgraph, original trajectory, and shortest path
    nx.draw(subgraph, pos, ax=ax, edge_color='black', node_size=5, node_color='black', width=0.5, arrows=False)
    nx.draw_networkx_edges(traj_G, pos_t, ax=ax, style='--', edge_color='blue', width=1.5, arrows=False)
    nx.draw_networkx_edges(sp_G, sp_pos, ax=ax, style='--', edge_color='red', width=1.5, arrows=False)

    hull.plot(ax=ax, color='None', edgecolor='orange', lw=0.5)
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldGrayCanvas, zoom='auto')
    plt.savefig(os.path.join(out_dir, 'base_subgraph.png'))


def plot_clustering(linkage, level_1_height, level_2_height, out_dir):
    """
    Plot the dendrogram of the hierarchical clustering.
    :param out_dir: output directory.
    :param linkage: generated linkage matrix.
    :param level_1_height: height of the first level of clustering.
    :param level_2_height: height of the second level of clustering.
    """
    plt.figure(figsize=(8, 5))
    dendrogram(linkage)

    plt.axhline(y=level_1_height, color='black', linestyle='--')
    plt.text(x=(plt.gca().get_xlim()[1]), y=level_1_height + .05, s='Level 1', va='center', ha='right', color='black')
    plt.axhline(y=level_2_height, color='black', linestyle='--')
    plt.text(x=(plt.gca().get_xlim()[1]), y=level_2_height + .05, s='Level 2', va='center', ha='right', color='black')
    plt.title('Agglomerate Hierarchical Clustering Dendrogram')
    plt.xlabel('Node ID')
    plt.ylabel('Distance')
    plt.savefig(os.path.join(out_dir, 'dendrogram.png'))


def plot_route_task(abstract_graph, subgraph, G, cluster_centroids_1, cluster_centroids_2, level_node, sp_G, path,
                    next_node, traj_G, start_node, end_node, counter, out_dir, crs=3857):
    """
    Dynamically plot the cognitive graph with the shortest path and the original trajectory.
    :param abstract_graph: abstract graph without the removed elements.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param G: original graph.
    :param cluster_centroids_1: level 1 cluster centroids.
    :param cluster_centroids_2: level 2 cluster centroids.
    :param level_node:
    :param sp_G: shortest path graph.
    :param path: list of node ids in the planned path.
    :param next_node: next node to move.
    :param traj_G: original trajectory graph.
    :param start_node: origin node.
    :param end_node: destination node.
    :param counter: step counter.
    :param out_dir: output directory.
    :param crs: Coordinate Reference System.
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))

    pos = {node: (data['location'][0], data['location'][1]) for node, data in subgraph.nodes(data=True)}
    l1_nodes = [node for node in abstract_graph.nodes() if (L1_NODE_BASE < node < L2_NODE_BASE)]
    pos_l1 = {node: cluster_centroids_1[node - L1_NODE_BASE] for node in l1_nodes}
    l2_nodes = [node for node in abstract_graph.nodes() if node > L2_NODE_BASE]
    pos_l2 = {node: cluster_centroids_2[node - L2_NODE_BASE] for node in l2_nodes}

    # abstract graph flattened (level 0 nodes)
    original_nodes = [node for node in abstract_graph.nodes() if node < L1_NODE_BASE]
    all_pos = {node: pos[node] for node in original_nodes}
    all_pos.update(pos_l1)
    all_pos.update(pos_l2)

    # draw the abstract graph.
    nx.draw(abstract_graph, all_pos, ax=ax, with_labels=False, node_size=10, node_color='black',
            width=0.5, edge_color='black', node_shape='o')
    nx.draw_networkx_nodes(abstract_graph, pos_l1, ax=ax, nodelist=l1_nodes, node_size=30, node_color='black',
                           node_shape='v')
    nx.draw_networkx_nodes(abstract_graph, pos_l2, ax=ax, nodelist=l2_nodes, node_size=30, node_color='black',
                           node_shape='s')

    # draw sp and original trajectory.
    pos_sp = {node: (data['location'][0], data['location'][1]) for node, data in sp_G.nodes(data=True)}
    pos_t = {node: (data['location'][0], data['location'][1]) for node, data in traj_G.nodes(data=True)}
    nx.draw_networkx_edges(sp_G, pos_sp, style='--', edge_color='red', width=1, arrows=False)
    nx.draw_networkx_edges(traj_G, pos_t, ax=ax, style='--', edge_color='green', width=1, arrows=False)
    nx.draw_networkx_labels(subgraph, pos, labels={start_node: 'O', end_node: 'D'}, font_size=20,
                            font_weight='bold', verticalalignment='top', font_color='black')

    planned_path_G = create_graph(G, path)
    pos_route = {node: (subgraph.nodes[node]['location'][0], subgraph.nodes[node]['location'][1]) for node in planned_path_G.nodes()}
    nx.draw(planned_path_G, pos_route, with_labels=False, node_size=30, node_color='blue', edge_color='blue')

    # draw the next node to move
    if level_node is not None:
        if level_node < L2_NODE_BASE:
            nx.draw_networkx_nodes(abstract_graph, pos_l1, ax=ax, nodelist=[level_node], node_size=30, node_color='red', node_shape='v')
        else:
            nx.draw_networkx_nodes(abstract_graph, pos_l2, ax=ax, nodelist=[level_node], node_size=30, node_color='red', node_shape='s')
    else:
        nx.draw_networkx_nodes(abstract_graph, all_pos, ax=ax, nodelist=[next_node], node_size=10, node_color='red', node_shape='o')

    # figure miscellaneous.
    legend = route_legend_helper()
    for item, value in {'L2 Cluster Node': 's', 'L1 Cluster Node': 'v', 'L0 Network Node': 'o'}.items():
        legend.append(Line2D([], [], color='black', marker=value, linestyle='None', markersize=10, label=item))
    plt.legend(handles=legend, loc='upper right')
    ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldGrayCanvas, zoom='auto')
    ax.set_title('Dynamic Cognitive Graph')
    plt.savefig(os.path.join(out_dir, f'{counter}_step.png'))
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close(plt.gcf())


def route_legend_helper():
    """
    Helper function to create the legend for the route plot.
    :return: list of legend patches.
    """
    blue_patch = mpatches.Patch(color='blue', label='Cognitive Route')
    red_patch = mpatches.Patch(color='red', label='Shortest Path')
    green_patch = mpatches.Patch(color='green', label='Original Trajectory')
    return [blue_patch, red_patch, green_patch]


def plot_final_route(subgraph, sp_G, planned_path_G, traj_G, start_node, end_node, out_dir):
    """
    Plot the final cognitive route with the shortest path and the original trajectory.
    :param subgraph: a generated subgraph from all paths not longer than 15% of the shortest between OD pair.
    :param sp_G: shortest path graph.
    :param planned_path_G: planned path graph.
    :param traj_G: original trajectory graph.
    :param start_node: origin node.
    :param end_node: end node.
    :param out_dir:  directory to save the plot.
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))

    saliency_values = list(nx.get_node_attributes(subgraph, 'saliency').values())
    normalized_saliency = [(value - min(saliency_values)) / (max(saliency_values) - min(saliency_values)) for value in saliency_values]
    pos = {node: (data['location'][0], data['location'][1]) for node, data in subgraph.nodes(data=True)}
    nx.draw(subgraph, pos, ax=ax, edge_color='black', node_size=5, node_color=normalized_saliency, cmap=plt.cm.winter,  width=0.5, arrows=False)

    # shortest path
    pos_sp = {node: (data['location'][0], data['location'][1]) for node, data in sp_G.nodes(data=True)}
    nx.draw_networkx_edges(sp_G, pos_sp, ax=ax, style='--', edge_color='red', width=2, arrows=False)

    # original trajectory
    pos_t = {node: (data['location'][0], data['location'][1]) for node, data in traj_G.nodes(data=True)}
    nx.draw_networkx_edges(traj_G, pos_t, ax=ax, style='--', edge_color='green', width=2, arrows=False)

    # final route
    pos_route = {node: (data['location'][0], data['location'][1]) for node, data in planned_path_G.nodes(data=True)}
    nx.draw_networkx_edges(planned_path_G, pos_route, ax=ax, style='--', width=2, edge_color='blue')

    # origin-destination labels
    labels = {start_node: 'O', end_node: 'D'}
    nx.draw_networkx_labels(subgraph, pos, ax=ax, labels=labels, font_size=20, font_weight='bold', verticalalignment='top', font_color='black')

    plt.legend(handles=route_legend_helper(), loc='upper left', title='Route Type')
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldGrayCanvas, zoom='auto')
    ax.set_title('Cognitive route planning and trajectory comparison')
    plt.savefig(os.path.join(out_dir, 'final_path.png'))


def plot_metric_values(sp, tr, cr, param, trip_id, out_dir):
    """
    Plots selected metric values along the shortest path, trajectory, and cognitive route
    where number of segments normalized between 0-1.
    :param out_dir: output directory.
    :param sp: metric values along the shortest path.
    :param tr: metric values along the trajectory.
    :param cr: metric values along the cognitive route.
    :param param: metric name.
    :param trip_id: trip identifier.
    """
    plt.plot(np.linspace(0, 1, len(sp)), sp, label='shortest path', color='red', lw=1)
    plt.plot(np.linspace(0, 1, len(tr)), tr, label='trajectory', color='blue', lw=1)
    plt.plot(np.linspace(0, 1, len(cr)), cr, label='cognitive route', color='green', lw=1)
    plt.legend()
    plt.title('{} values along each route: {}'.format(param, trip_id))
    plt.xlabel('Mapped Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'metric_values.png'))
