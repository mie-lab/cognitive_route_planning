import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
import re


def get_enriched_node_df(node_path, significance_path=None, od_path=None, crs=3857):
    """
    Load and enrich node dataset with significance and trip destination counts.
    :param clip_ods: ods dataset is hihgly scewed and needs to be clipped for meaningful normalization
    :param node_path: path to the node dataset.
    :param significance_path: path to the significance dataset, openly available here:
    https://www.stadt-zuerich.ch/geodaten/download/Bedeutungsplan?format=10007
    :param od_path: path to the origin-destination dataset.
    :param crs: Coordinate Reference System, by default set to 3857 for Zurich.
    :return: pre-processed DataFrame with enriched network nodes.
    """

    nodes = gpd.read_file(node_path)
    nodes = nodes[~nodes['geometry'].isna()]
    nodes = nodes.to_crs(epsg=crs).reset_index()

    # merge nodes and significance data
    if significance_path is not None:
        significance = gpd.read_file(significance_path).to_crs(epsg=crs)
        nodes = (gpd.sjoin(nodes, significance, how='left', predicate='within')
                 .drop_duplicates(subset=['osmid', 'x', 'y'])
                 .drop(columns=['x', 'y', 'index_right', 'index', 'zukunfig_f', 'name', 'id1', 'objectid'])).fillna(0.0)

    # merge nodes and trip destination counts
    if od_path is not None:
        ods = pd.read_csv(od_path)
        summed_dest = pd.DataFrame(ods.groupby('t')['trips'].sum())
        nodes = nodes.merge(summed_dest, how='left', left_on='osmid', right_on='t').fillna(0.0).reset_index()

    return nodes


def get_enriched_edge_df(edge_path, lane_column=None, surface_mapping=None, crs=3857):
    """
    Load and enrich edge dataset with lane and bike lane counts.
    :param surface_mapping: mapping of surface types to numerical values.
    :param edge_path: path to the edge dataset.
    :param lane_column:
    :param crs: Coordinate Reference System, by default set to 3857 for Zurich.
    :return: pre-processed DataFrame with enriched network edges.
    """
    edges = gpd.read_file(edge_path)
    edges = edges.to_crs(epsg=crs).rename(columns={'u': 'source', 'v': 'target'})

    # filling in missing maxspeed values with 50 km/h as a reasonable speed limit in urban areas.
    edges['maxspeed'] = edges['maxspeed'].fillna(50.0)

    # normalize edge length attribute for later use.
    edges['length_norm'] = normalize_values(edges['length'])

    # bug in the network data - there is a missing node in the node dataset that exist in the rebuilt graph.
    edges = edges[edges['target'] != 8374]

    # update motorised vehicle and bike lanes counts.
    if lane_column is not None:
        edges[lane_column] = edges[lane_column].apply(lambda x: re.findall(r'[A-Za-z]', x))
        edges['motorized_lanes'] = edges[lane_column].apply(lambda x: x.count('H') + x.count('M') + x.count('T'))
        edges['bike_lanes'] = edges[lane_column].apply(
            lambda x: 1 if (x.count('P') + x.count('L') + x.count('S') + x.count('X')) > 0 else 0)

    return edges


def normalize_values(series):
    min_val, max_val = series.min(), series.max()
    return (series - min_val) / (max_val - min_val)


def set_node_attributes(edges, nodes, maxspeed_bins, clip_ods=None, surface_mapping=None):
    """
    Set node attributes from the node and edge datasets by normalizing node attributes and aggregating edge attributes
        on nodes.
    :param maxspeed_bins: bins for normalizing maxspeed attribute.
    :param clip_ods: upper boundary to clip the ods dataset.
    :param surface_mapping: surface mapping for numerical values.
    :param edges: network edge DataFrame.
    :param nodes: network node DataFrame.
    :return: network graph and enriched node DataFrame.
    """

    nodes.set_index("osmid", inplace=True, drop=False)
    nodes['location'] = [(row.geometry.x, row.geometry.y) for node, row in nodes.iterrows()]
    nodes['elevation_norm'] = normalize_values(nodes['elevation'])
    nodes['trips'] = nodes['trips'].clip(lower=0, upper=clip_ods)
    nodes['trips_norm'] = normalize_values(nodes['trips'])
    nodes['bedeutung_norm'] = normalize_values(nodes['bedeutung'])
    nodes = aggregate_edge_attributes(edges, nodes, ['motorized_lanes', 'maxspeed', 'bike_lanes', 'surface'],
                                      maxspeed_bins=maxspeed_bins, surface_mapping=surface_mapping)

    return nodes


def aggregate_edge_attributes(edges, nodes, node_attr, maxspeed_bins=None, surface_mapping=None):
    """
    Aggregate edge attributes on nodes.
    :param surface_mapping: surface mapping for numerical values.
    :param maxspeed_bins: bins for normalizing maxspeed attribute.
    :param edges: network edge DataFrame.
    :param nodes: network node DataFrame.
    :param node_attr: node attribute to aggregate.
    :return: aggregated node attribute DataFrame.
    """

    edges_copy = edges.copy()

    for attr in node_attr:

        if attr == 'surface' and surface_mapping is not None:
            edges_copy['surface_norm'] = edges_copy[attr].map(surface_mapping)
            surf_agg = edges_copy.groupby('source')['surface_norm'].mean()
            nodes = nodes.merge(surf_agg.rename(f'{attr}_norm'), how='left', left_index=True, right_index=True)

        elif attr == 'motorized_lanes':
            mlanes_sum = edges_copy.groupby('source')[attr].sum()
            mlanes_agg = normalize_values(mlanes_sum)
            nodes = nodes.merge(mlanes_agg.rename(f'{attr}_norm'), how='left', left_index=True, right_index=True)

        elif attr == 'maxspeed' and maxspeed_bins is not None:
            edges_copy['maxspeed'] = pd.cut(edges_copy['maxspeed'], bins=maxspeed_bins,
                                            labels=[1, 0.5, 0], right=True).astype(float)
            maxspeed_agg = edges_copy.groupby('source')[attr].mean()
            nodes = nodes.merge(maxspeed_agg.rename(f'{attr}_norm'), how='left', left_index=True, right_index=True)

        elif attr == 'bike_lanes':
            blanes_agg = edges_copy.groupby('source')[attr].mean()
            nodes = nodes.merge(blanes_agg.rename(f'{attr}_norm'), how='left', left_index=True, right_index=True)

    return nodes

def set_node_attributes_from_df(G, df, attr_columns):
    for attr in attr_columns:
        nx.set_node_attributes(G, df[attr].to_dict(), attr)


def create_network_graph(edges, nodes, source, target):
    """
    Create a network graph from the edge dataset.
    :param edges: edge dataset.
    :param source: column name for edge starts.
    :param target: column name for edge ends.
    :return: a network graph.
    """

    G = nx.from_pandas_edgelist(
        edges,
        source=source,
        target=target,
        edge_attr=list(edges.columns),
        create_using=nx.Graph,
    )

    node_attrs = ['location', 'elevation_norm', 'trips_norm', 'bedeutung_norm', 'motorized_lanes_norm',
                  'maxspeed_norm', 'bike_lanes_norm', 'surface_norm']

    set_node_attributes_from_df(G, nodes, node_attrs)

    return G


def set_cycling_saliency(G, saliency_dict):
    """
    Set cycling saliency on the network graph based on a given attribute weights.
    :param G: network graph.
    :param saliency_dict: weights for different node properties contributing to the saliency.
    :return: network graph with saliency attribute.
    """
    saliency = {}
    for n in G.nodes():
        cur_saliency = 0
        for feature, params in saliency_dict.items():
            if not pd.isnull(G.nodes[n][feature]):
                if not params['high']:
                    cur_saliency += (1 - G.nodes[n][feature]) * params['factor']
                else:
                    cur_saliency += G.nodes[n][feature] * params['factor']
        saliency[n] = cur_saliency
    nx.set_node_attributes(G, saliency, 'saliency')

    return G
