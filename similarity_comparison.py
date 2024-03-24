import textdistance


def get_lcs_length(path_1, path_2):
    """
    Calculate the length of the longest common subsequence of two paths.
    :param path_1: list with node ids in the path.
    :param path_2: list with node ids in the path.
    :return: the length of the longest common subsequence.
    """
    table = {}
    lcs_length = 0
    for i, n1 in enumerate(path_1, 1):
        for j, n2 in enumerate(path_2, 1):
            if n1 == n2:
                length = table[i, j] = table.get((i - 1, j - 1), 0) + 1
                lcs_length = max(lcs_length, length)
    return lcs_length


def measure_path_distance(path_1, path_2, dist_type):
    """
    Measure a particular distance between two paths.
    :param path_1: list with node ids in the path.
    :param path_2: list with node ids in the path.
    :param dist_type: the type of distance to be measured.
    Can be 'edit_distance', 'long_seq_distance', or 'jaccard_distance'.
    :return: distance between the two paths.
    """
    if dist_type == 'edit_distance':
        dist = textdistance.levenshtein.distance(path_1, path_2)
        dist /= max(len(path_1), len(path_2))
    elif dist_type == 'long_seq_distance':
        dist = 1 - (get_lcs_length(path_1, path_2) / min(len(path_1), len(path_2)))
    elif dist_type == 'jaccard_distance':
        dist = 1 - len(set(path_1).intersection(path_2)) / len(set(path_1).union(path_2))
    else:
        dist = 1
    return dist


def get_distances(tr, sp, cr):
    """
    Get multiple types of distances between the trajectory, shortest path, and cognitive path.
    :param tr: list with node ids in the trajectory path.
    :param sp: list with node ids in the shortest path.
    :param cr: list with node ids in the cognitive path.
    :return: a dictionary of distances between the paths.
    """
    dist = {}
    for i in ['edit_distance', 'jaccard_distance', 'long_seq_distance']:
        tr_sp = measure_path_distance(tr, sp, i)
        cr_tr = measure_path_distance(tr, cr, i)
        cr_sp = measure_path_distance(sp, cr, i)
        dist[i] = {'tr_sp': tr_sp, 'cr_tr': cr_tr, 'cr_sp': cr_sp}
    return dist


def get_metric_values(path, G, metric):
    """
    Get a specific metric values for the nodes in the path.
    :param path: list with node ids in the path.
    :param G: network graph.
    :param metric: metric name.
    :return: list of the specific metric values for the nodes in the path.
    """
    metric_list = []
    for i in path:
        cur_node = G.nodes.data(True)[i]
        metric_list.append(cur_node[metric])

    return metric_list
