from sqlalchemy import create_engine
import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry import Point
import requests


def get_trajectories(credentials, crs=3857):
    """
    Get cycling trajectories from the database.
    :param credentials:  credentials to the database.
    :param crs: Coordinate Reference System, by default set to 3857 for Zurich.
    :return: DataFrame with cycling trajectories.
    """
    with open(credentials, 'r') as f:
        password = f.readline().strip()
    engine_bike = create_engine(
        f"postgresql://ayda:{password}@mie-lab-db.ethz.ch:5432/case_study_cache"
    )

    CRS_WGS84 = "epsg:4326"
    trips = gpd.GeoDataFrame.from_postgis(
        sql=f"""SELECT * FROM public.triplegs WHERE study = 'Green Class 1' 
        AND mode IN ('Mode::Bicycle') AND ST_isValid(geom)""",
        con=engine_bike,
        crs=CRS_WGS84,
        geom_col="geom",
        index_col="id"
    )
    trips = trips.set_geometry('geom').to_crs(crs).reset_index()

    return trips


def filter_trips(trip_df, min_trip_ln=0, max_trip_ln=float('inf'), od_dist=0):
    """
    Filter trips based on length and origin-destination distance.
    :param trip_df: DataFrame with trajectories.
    :param min_trip_ln: minimum trip length.
    :param max_trip_ln: maximum trip length.
    :param od_dist: Euclidean distance between origin and destination.
    :return: DataFrame with trips that satisfy the conditions.
    """
    proj_trip_df = trip_df.copy().to_crs(3857)
    proj_trip_df = proj_trip_df[proj_trip_df['geom'].length > min_trip_ln]
    proj_trip_df = proj_trip_df[proj_trip_df['geom'].length < max_trip_ln]
    proj_trip_df = proj_trip_df[
        [abs(i.coords[-1][0] - i.coords[0][0]) > od_dist and abs(i.coords[-1][1] - i.coords[0][1]) > od_dist for i in
         proj_trip_df.geom]]
    proj_trip_df = proj_trip_df.sort_values(by='user_id')

    return proj_trip_df.to_crs(4326)


def get_osrm_matched_nodes(trip):
    """
    Get nodes from the OSRM map-matching service.
    :param trip: cycling trajectory.
    :param crs: Coordinate Reference System, by default set to 3857 for Zurich.
    :return: list of nodes map-matched to the input trajectory.
    """

    api_url = "http://router.project-osrm.org/match/v1"
    profile = "bike"
    coordinates = ';'.join([f"{lon},{lat}" for lon, lat in shapely.get_coordinates(trip.geom[trip.index])])
    steps = "true"
    geometries = "geojson"
    overview = "simplified"
    annotations = "nodes"

    # Constructing the full URL
    full_url = f"{api_url}/{profile}/{coordinates}?steps={steps}&geometries={geometries}&overview={overview}&annotations={annotations}"

    # Making the GET request
    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
    else:
        print("Failed to fetch data from the API. Status code:", response.status_code)
    points = [Point(x['steps'][0]['maneuver']['location'][0], x['steps'][0]['maneuver']['location'][1]) for x in
              data['matchings'][0]['legs']]

    return points

def get_trajectory_edges(osm_matched_nodes, nodes, edges, graph):
    """
    Map-match trajectory to network.
    :param edges: network edges df.
    :param osm_matched_nodes: cycling trajectory
    :param nodes: network nodes df.
    :param graph: network graph.
    :return: list of node ids corresponding to input trajectory
    """

    point_ids = []
    for point in osm_matched_nodes:
        cur_edge_matching = edges.loc[shapely.distance(point, edges['geometry']).sort_values().index[0], :]
        edge_nodes = (cur_edge_matching['source'], cur_edge_matching['target'])
        cur_point_id = nodes.loc[
            shapely.distance(point, nodes.loc[nodes['osmid'].isin(edge_nodes), 'geometry']).sort_values().index[
                0], 'osmid']
        if cur_point_id not in point_ids:
            point_ids.append(cur_point_id)

    # get edges matching with traj points - can't use points directly because they might not cover all nodes along traj.
    complete_path = []
    for i in range(len(point_ids) - 1):
        path = nx.shortest_path(graph, source=point_ids[i], target=point_ids[i + 1], weight='length')
        for j in range(len(path) - 1):
            complete_path.append((path[j], path[j + 1]))
    traj_path = [complete_path[0][0]] + [edge[1] for edge in complete_path]

    return traj_path


def create_graph(G, path):
    """
    Create a graph from a list of node ids.
    :param G: network graph.
    :param path: list of node ids that exist in the network graph.
    :return: a subgraph.
    """

    graph = nx.Graph()
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if G.has_edge(edge[0], edge[1]):
            graph.add_node(edge[0], **G.nodes[edge[0]])
            graph.add_node(edge[1], **G.nodes[edge[1]])
            edge_attrs = G[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], **edge_attrs)

    return graph


def create_base_subgraph(traj, buffer, nodes, G, sp, start_node, end_node, detour, crs=3857):
    """
    Create a subgraph based on subset of shortest paths between the trajectory OD pair and the buffered convex hull of
    the trajectory.
    :param traj: trajectory LineString.
    :param buffer: buffer distance.
    :param nodes: Graph nodes.
    :param G: network graph.
    :param sp: shortest path for the trajectory OD pair.
    :param start_node: origin node.
    :param end_node: destination node.
    :param detour: detour factor to sample paths that are not longer than 15% than the shortest path.
    :param crs: Coordinate Reference System.
    :return: subgraph for the particular trajectory and the corresponding buffered convex hull.
    """
    traj_metric = traj.copy().to_crs(3857)
    hull = gpd.GeoDataFrame(geometry=traj_metric.convex_hull.buffer(buffer), crs=crs).reset_index().to_crs(4326)
    subgraph = G.subgraph(list(nodes[nodes.within(hull.geometry[0])]['osmid']))
    subgraph_connected = G.subgraph(max(nx.connected_components(subgraph), key=len))

    sp_length = nx.path_weight(subgraph_connected, sp, 'length') * detour
    unique_nodes_in_routes = set()

    for path in nx.shortest_simple_paths(subgraph_connected, start_node, end_node, weight='length'):
        path_length = nx.path_weight(subgraph_connected, path, 'length')
        if path_length > sp_length:
            break
        else:
            unique_nodes_in_routes.update(path)
    final_subgraph = G.subgraph(unique_nodes_in_routes)

    return final_subgraph, hull
