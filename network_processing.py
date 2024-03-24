import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import re


def get_enriched_node_df(node_path, significance_path, od_path, crs=3857):
    """
    Load and enrich node dataset with significance and trip destination counts.
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

    # significance data
    significance = gpd.read_file(significance_path).to_crs(epsg=crs)
    nodes = (gpd.sjoin(nodes, significance, how='left', predicate='within')
             .drop_duplicates(subset=['osmid', 'x', 'y'])
             .drop(columns=['index_right', 'index'])).fillna(0.0)

    # trip destination counts
    ods = pd.read_csv(od_path)
    summed_dest = pd.DataFrame(ods.groupby('t')['trips'].sum())
    nodes = nodes.merge(summed_dest, how='left', left_on='osmid', right_on='t').fillna(0.0).reset_index()

    return nodes


def get_enriched_edge_df(edge_path, lane_column, crs=3857):
    """
    Load and enrich edge dataset with lane and bike lane counts.
    :param edge_path:
    :param lane_column:
    :param crs: Coordinate Reference System, by default set to 3857 for Zurich.
    :return: pre-processed DataFrame with enriched network edges.
    """
    edges = gpd.read_file(edge_path)
    edges = edges.to_crs(epsg=crs).rename(columns={'u': 'source', 'v': 'target'}).fillna(0.0)

    # normalize edge distance.
    scaler = MinMaxScaler()
    edges['length'] = scaler.fit_transform(edges['length'].values.reshape(-1, 1))

    # bug in the network data - there is a missing node in the node dataset that exist in the rebuilt graph.
    edges = edges[edges['target'] != 8374]

    # update motorised vehicle and bike lanes counts.
    edges[lane_column] = edges[lane_column].apply(lambda x: re.findall(r'[A-Za-z]', x))
    edges['lanes_updated'] = edges[lane_column].apply(lambda x: x.count('H') + x.count('M') + x.count('T'))
    edges['bike_lanes'] = edges[lane_column].apply(lambda x: 1 if (x.count('P') + x.count('L') + x.count('S') + x.count('X')) > 0 else 0)

    return edges


def create_network_graph(edges, source, target):
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

    return G


def set_network_node_attributes(G, edges, nodes):
    """
    Set node attributes from the node and edge datasets by normalizing node attributes and aggregating edge attributes
        on nodes.
    :param G:  network graph.
    :param edges: network edge DataFrame.
    :param nodes: network node DataFrame.
    :return: network graph and enriched node DataFrame.
    """

    nodes.set_index("osmid", inplace=True, drop=False)

    # helper functions.
    def normalize_values(series):
        min_val, max_val = series.min(), series.max()
        return (series - min_val) / (max_val - min_val)

    def set_node_attributes_from_df(G, df, attr_columns):
        for attr in attr_columns:
            nx.set_node_attributes(G, df[attr].to_dict(), attr)

    nodes['location'] = [(row.geometry.x, row.geometry.y) for node, row in nodes.iterrows()]
    nodes['degree'] = normalize_values(pd.DataFrame.from_dict(dict(G.degree()), orient='index')[0])
    nodes['elevation'] = normalize_values(nodes['elevation'])
    nodes['destinations'] = normalize_values(nodes['trips'])
    nodes['bedeutung'] = normalize_values(nodes['bedeutung'])
    edges['lanes_updated'] = normalize_values(edges['lanes_updated'])
    edges['avg_speed'] = normalize_values(edges['maxspeed'])
    edge_attrs_agg = edges.groupby('source')[['lanes_updated', 'avg_speed', 'bike_lanes']].mean()
    nodes = nodes.merge(edge_attrs_agg, how='left', left_index=True, right_index=True)

    set_node_attributes_from_df(G, nodes,
                                ['location', 'degree', 'elevation', 'destinations', 'bedeutung', 'lanes_updated',
                                 'avg_speed', 'bike_lanes'])
    return G, nodes


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
            if not params['high']:
                cur_saliency += (1 - G.nodes[n][feature]) * params['factor']
            else:
                cur_saliency += G.nodes[n][feature] * params['factor']
        saliency[n] = cur_saliency
    nx.set_node_attributes(G, saliency, 'saliency')

    return G
