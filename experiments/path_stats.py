import numpy as np

def check_positive_probability(model, paths):
    final_points = np.empty((len(paths), paths[0].shape[1]))
    for i, path in enumerate(paths):
        final_points[i] = path.to_numpy()[-1]
    predictions = model.predict_proba(final_points)
    return predictions[:,1]


def check_path_count(paths):
    return np.array([len(path) for path in paths])


def check_final_point_distance(paths):
    distances = np.empty(len(paths))
    for i, path in enumerate(paths):
        path = path.to_numpy()
        start = path[0]
        end = path[-1]
        diff = start - end
        dist = np.sqrt(diff@diff)
        distances[i] = dist
    return distances


def check_path_length(paths):
    lengths = np.empty(len(paths))
    for path_idx, path in enumerate(paths):
        path = path.to_numpy()
        path_dist = 0
        i = 0
        j = 1
        while j < path.shape[0]:
            diff = path[i] - path[j]
            path_dist += np.sqrt(diff@diff)
            i += 1
            j += 1
        lengths[path_idx] = path_dist
    return lengths


"""
For each path,
    calculates the average euclidean distance between each point
    and its nearest valid counterpoint
"""
def check_validity_distance(preprocessor, paths):
    path_validity = np.empty(len(paths))
    for path_idx, path in enumerate(paths):
        valid_path = make_valid(path, preprocessor).to_numpy()
        diff = valid_path - path.to_numpy()
        dist = np.sqrt((diff**2).sum(axis=1)).mean()
        path_validity[path_idx] = dist
    return path_validity


def make_valid(path, preprocessor):
    p = preprocessor.inverse_transform(path)
    return preprocessor.transform(p)


def check_cluster_size(cluster_assignments, n_clusters):
    cluster_sizes = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_sizes[i] = cluster_assignments[cluster_assignments == i].shape[0]
    return cluster_sizes


"""
For each path,
    calculates the average euclidean distance between each point
    and its nearest immutable-enforced point
"""
def check_immutability(immutable_columns, paths):
    path_immutability = np.empty(len(paths))
    for path_idx, path in enumerate(paths):
        for i in range(path.shape[0] - 1):
            diff = path.iloc[i][immutable_columns].to_numpy() - path.iloc[i+1][immutable_columns].to_numpy()
            dist = np.sqrt(diff@diff)
            path_immutability[path_idx] = dist
    return path_immutability

"""
For each path,
    calculates the average number of altered features for each point
"""
def check_sparsity(paths):
    path_sparsity = np.empty(len(paths))
    for path_idx, path in enumerate(paths):
        for i in range(path.shape[0] - 1):
            diff = path.iloc[i].to_numpy() - path.iloc[i+1].to_numpy()
            nonzero = (diff != 0).sum()
            path_sparsity[path_idx] = nonzero
    return path_sparsity
