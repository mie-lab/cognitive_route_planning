from shapely import LineString
from sqlalchemy import create_engine
import geopandas as gpd
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import Point
import requests


def get_trajectories(credentials, user_data=None, simplify=0, crs=4326):
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
    trips['geom'] = shapely.simplify(trips['geom'], tolerance=simplify)

    if user_data is not None:
        trips = trips.merge(user_data, how='left', left_on='user_id', right_on='userid').dropna()
        trips = preprocess_trajectory_data(trips, user_data)

    return trips


def get_user_data(id_path, survey_data_path, survey_columns):
    """
    Load and merge user data with survey data.
    :param id_path: path to the user ID mapping.
    :param survey_data_path: path to the survey data.
    :param survey_columns: columns to keep from the survey data.
    :return: merged DataFrame with user data.
    """
    id_mapping = pd.read_csv(id_path, sep=';', dtype=object).drop(columns=['vid'])
    id_mapping['lfdn'] = pd.to_numeric(id_mapping['lfdn'])
    id_mapping = id_mapping.drop_duplicates(keep='first').dropna()

    assert id_mapping.shape[0] == id_mapping[
        'lfdn'].nunique(), 'There are users with missing lfdn or multiple lfdn with multiple user ID mappings'
    assert id_mapping.shape[0] == id_mapping[
        'userid'].nunique(), 'There are users with missing userid or multiple userid with multiple lfdn mappings'

    survey_data = pd.read_csv(survey_data_path, sep=';')
    assert survey_data.shape[0] == survey_data['lfdn'].nunique(), 'There are users in survey with the same lfdn'

    # merge personal data.
    user_data = id_mapping.merge(survey_data.loc[:, survey_columns], how='right', left_on='lfdn', right_on='lfdn')

    for col in ['work_lon', 'work_lat', 'home_lon', 'home_lat']:
        user_data[col] = pd.to_numeric(user_data[col].str.replace(',', '.'))

    return user_data


def get_points(lon_col, lat_col, user_data, crs=4326):
    """
    Get home and work point coordinates from user data.
    :param lon_col: longitute column to be used.
    :param lat_col: latitude column to be used.
    :param user_data: dataframe with user information.
    :param crs: coordinate reference system.
    :return: a GeoDataFrame with home and work points.
    """
    user_data_geo = user_data.copy()
    user_data_geo['geometry'] = [Point(lon, lat) for lon, lat in zip(user_data[lon_col], user_data[lat_col])]
    return gpd.GeoDataFrame(user_data_geo, geometry='geometry').set_crs('4326').to_crs(crs)


def preprocess_trajectory_data(trips, user_data):
    """
    Preprocess trajectory data.
    :param user_data: dataframe with user information.
    :param trips: DataFrame with cycling trajectories.
    :return: DataFrame with preprocessed trajectories.
    """
    # duration in minutes
    trips['duration'] = (trips.loc[:, 'finished_at'] - trips.loc[:,'started_at']).dt.total_seconds() / 60
    # length in meters
    trips['length'] = trips['geom'].length

    work_points = get_points('work_lon', 'work_lat', user_data, crs=3857)
    home_points = get_points('home_lon', 'home_lat', user_data, crs=3857)

    # home distances
    home_distance = []
    work_distance = []
    for trip in trips.index:
        user_id = trips.loc[trip, 'user_id']
        work_point = work_points.loc[work_points['userid'] == user_id, 'geometry'].iloc[0]
        home_point = home_points.loc[home_points['userid'] == user_id, 'geometry'].iloc[0]
        start_point = Point(trips.loc[trip, 'geom'].coords[0])
        end_point = Point(trips.loc[trip, 'geom'].coords[-1])
        home_distance.append(min([home_point.distance(point) for point in [start_point, end_point]]))
        work_distance.append(min([work_point.distance(point) for point in [start_point, end_point]]))
    trips['home_distance'] = home_distance
    trips['work_distance'] = work_distance

    return trips


def filter_trips(trip_df, polygon_from_bounds, min_trip_ln=0, max_trip_ln=float('inf'), od_dist=0):
    """
    Filter trips based on length and origin-destination distance.
    :param polygon_from_bounds: bounding box polygon for the area of interest.
    :param trip_df: DataFrame with trajectories.
    :param min_trip_ln: minimum trip length.
    :param max_trip_ln: maximum trip length.
    :param od_dist: Euclidean distance between origin and destination.
    :return: DataFrame with trips that satisfy the conditions.
    """

    trips_in_area = trip_df[trip_df.within(polygon_from_bounds)]

    trips_in_area = trips_in_area[trips_in_area['geom'].length > min_trip_ln]
    trips_in_area = trips_in_area[trips_in_area['geom'].length < max_trip_ln]
    trips_in_area = trips_in_area[[abs(i.coords[-1][0] - i.coords[0][0]) > od_dist and
                                   abs(i.coords[-1][1] - i.coords[0][1]) > od_dist for i in trips_in_area.geom]]
    trips_in_area = trips_in_area.sort_values(by='user_id')

    return trips_in_area


def get_osrm_matched_trips(trips):
    """
    Get nodes from the OSRM map-matching service.
    :param trips: trip df.
    :param crs: Coordinate Reference System.
    :return: list of nodes map-matched to the input trajectory.
    """

    api_url = "http://router.project-osrm.org/match/v1"
    profile = "car"
    steps = "true"
    geometries = "geojson"
    overview = "full"
    annotations = "nodes"

    original_crs = trips.crs
    trips = trips.to_crs(4326)
    errorneous_trajectories = []

    for i in trips.index:
        coordinates = ';'.join([f"{lon},{lat}" for lon, lat in shapely.get_coordinates(trips.geom[i])])
        full_url = f"{api_url}/{profile}/{coordinates}?steps={steps}&geometries={geometries}&overview={overview}&annotations={annotations}"
        response = requests.get(full_url)
        if response.status_code == 200:
            data = response.json()
            points = [Point([x['steps'][0]['maneuver']['location'][0], x['steps'][0]['maneuver']['location'][1]]) for x in
                      data['matchings'][0]['legs']]
            if len(points) > 1:
                trips.loc[i, 'geom'] = LineString(points)
            else:
                errorneous_trajectories.append(i)
                print("Failed to map to a proper geometry. Status code:", response.status_code, 'At index: ', i)
        else:
            errorneous_trajectories.append(i)
            print("Failed to fetch data from the API. Status code:", response.status_code, 'At index: ', i)

    print(len(errorneous_trajectories), ' trajectories were removed due to mapmatching related errors.')
    #trips = trips.drop(index=errorneous_trajectories)
    trips = trips.to_crs(original_crs)

    return trips, errorneous_trajectories


def get_trajectory_node_ids(trips, nodes, edges):
    """
    Get the node ids for each trajectory and write it as a new attribute in the column 'node_ids'.
    :param trips: trip df.
    :param nodes: simpified network nodes.
    :param edges: simplified network edges.
    """
    trip_node_ids = []
    for trip in trips.index:
        points = trips.loc[trip, 'geom'].coords
        node_ids = []
        for point in points:
            cur_edge_matching = edges.loc[shapely.distance(Point(point), edges['geometry']).sort_values().index[0], :]
            source, target = cur_edge_matching['source'], cur_edge_matching['target']
            #cur_point_id = nodes.loc[shapely.distance(Point(point), nodes.loc[nodes['osmid'].isin(edge_nodes), 'geometry']).sort_values().index[0], 'osmid']
            #if cur_point_id not in node_ids:
            #    node_ids.append(cur_point_id)
            if source not in node_ids:
                node_ids.append(source)
            if target not in node_ids:
                node_ids.append(target)

        # get edges matching with traj points - can't use points directly because they might not cover all nodes along traj.
        #complete_path = []
        #for i in range(len(node_ids) - 1):
        #    path = nx.shortest_path(G, source=node_ids[i], target=node_ids[i + 1], weight='length')
        #    for j in range(len(path) - 1):
        #        complete_path.append((path[j], path[j + 1]))
        #traj_path = [complete_path[0][0]] + [edge[1] for edge in complete_path]


        trip_node_ids.append(node_ids)
    trips['node_ids'] = trip_node_ids

    return trips


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


'''-----------Trip stats functions-----------'''


def get_peak_hour_trajectories(df):
    """
    Get the number of peak hour trajectories and unique users during peak hours.
    :param df:
    :return:
    """
    # Define peak hours
    peak_morning_start = pd.to_datetime('07:00:00').time()
    peak_morning_end = pd.to_datetime('09:00:00').time()
    peak_evening_start = pd.to_datetime('16:00:00').time()
    peak_evening_end = pd.to_datetime('19:00:00').time()

    # Function to check if a time is within peak hours
    def is_peak_hour(time):
        return (peak_morning_start <= time <= peak_morning_end) or (peak_evening_start <= time <= peak_evening_end)
    # Apply function to filter trajectories during peak hours
    df['is_peak_hour'] = df['started_at'].dt.time.apply(is_peak_hour)

    # Filter the DataFrame for peak hour trajectories
    peak_hour_trajectories = df[df['is_peak_hour']]
    return len(peak_hour_trajectories), len(peak_hour_trajectories['user_id'].unique())


def get_trip_location(df, location):
    return len(df[df[location] < 1000]), len(df[df[location] < 1000]['user_id'].unique())


def calculate_statistics(df):
    """
    Calculate descriptive statistics for a given trajectory data.
    :param df: trajectory df.
    :return: dictionery with stats.
    """

    peak_hour_trip_count, peak_hour_users = get_peak_hour_trajectories(df)
    trips_from_home, home_commuters = get_trip_location(df, 'home_distance')
    trips_from_work, work_commuters = get_trip_location(df, 'work_distance')

    return {
        'user_count': len(df['user_id'].unique()),
        'trip_count': len(df),
        'min_duration': min(df['duration']),
        'max_duration': max(df['duration']),
        'avg_duration': df['duration'].mean(),
        'std_duration': df['duration'].std(),
        'min_length': min(df['length']),
        'max_length': max(df['length']),
        'avg_length': df['length'].mean(),
        'std_length': df['length'].std(),
        'min_age': min(df['age']),
        'max_age': max(df['age']),
        'avg_age': df['age'].mean(),
        'std_age': df['age'].std(),
        'od_dist_home': df['home_distance'].mean(),
        'std_home_dist': df['home_distance'].std(),
        'od_dist_work': df['work_distance'].mean(),
        'std_work_dist': df['work_distance'].std(),
        'peak_hour_trips': peak_hour_trip_count,
        'peak_hour_users': peak_hour_users,
        'trips_from_home': trips_from_home,
        'home_commuters': home_commuters,
        'trips_from_work': trips_from_work,
        'work_commuters': work_commuters,

    }


def get_trajectory_stats(trips, stats_per_gender=False):
    """
    Generates descritive statistics for the trajectory data.
    :param trips: trajectory data
    :return: stats overview for the data sample.
    """

    all_users_stats = calculate_statistics(trips)
    print("All Users Statistics:")
    for key, value in all_users_stats.items():
        print(f"{key}: {value}")

    if stats_per_gender == True:
        women_stats = calculate_statistics(trips[trips['sex'] == 0.0])
        print("\nWomen Statistics:")
        for key, value in women_stats.items():
            print(f"{key}: {value}")

        men_stats = calculate_statistics(trips[trips['sex'] == 1.0])
        print("\nMen Statistics:")
        for key, value in men_stats.items():
            print(f"{key}: {value}")

