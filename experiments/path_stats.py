import numpy as np

EQ_EPSILON = 1e-10


def check_path_count(paths):
    return np.array([len(path) for path in paths])


def check_path_length(paths):
    lengths = np.zeros(len(paths))
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


def density_metric():
    pass


"""
For each path,
    calculates the average euclidean distance between each point
    and its nearest valid counterpoint
"""
def check_validity_distance(preprocessor, paths):
    path_validity = np.zeros(len(paths))
    for path_idx, path in enumerate(paths):
        valid_path = make_valid(path, preprocessor).to_numpy()
        diff = valid_path - path.to_numpy()
        dist = np.sqrt((diff**2).sum(axis=1)).mean()
        path_validity[path_idx] = dist
    return path_validity


def check_validity(preprocessor, column_names_per_feature, paths):
    """The average number of features per-point which are not valid.
    """
    path_validity = np.zeros(len(paths))
    for path_idx, path in enumerate(paths):
        valid_path = make_valid(path, preprocessor)
        diff = np.abs(valid_path - path) > EQ_EPSILON
        path_totals = np.zeros(path.shape[0])
        for column_names in column_names_per_feature:
            path_totals += diff[column_names].sum(axis=1).to_numpy()
        path_validity[path_idx] = path_totals.sum() / path_totals.shape[0]
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
def check_immutability(preprocessor, immutable_features, paths):
    path_immutability = np.zeros(len(paths))
    for path_idx, path in enumerate(paths):
        path = preprocessor.inverse_transform(path)
        for i in range(path.shape[0] - 1):
            p1 = path.iloc[[i]][immutable_features]
            p2 = path.iloc[[i+1]][immutable_features]

            numeric_columns = p1.select_dtypes(include=np.number).columns
            other_columns = p2.columns.difference(numeric_columns)

            numeric_diff = (np.abs(p1[numeric_columns].to_numpy() - p2[numeric_columns].to_numpy()) >= EQ_EPSILON).sum(axis=1)
            other_diff = (p1[other_columns].to_numpy() != p2[other_columns].to_numpy()).sum(axis=1)


            diff = numeric_diff + other_diff
            path_immutability[path_idx] += diff.sum()
        path_immutability[path_idx] = path_immutability[path_idx]/(path.shape[0] - 1)
    return path_immutability


"""
For each path,
    calculates the average number of altered features for each point
"""
def check_sparsity(preprocessor, paths):
    path_sparsity = np.zeros(len(paths))
    for path_idx, path in enumerate(paths):
        path = preprocessor.inverse_transform(path)
        for i in range(path.shape[0] - 1):
            nonzero = (path.iloc[i] != path.iloc[i+1]).sum()
            path_sparsity[path_idx] += nonzero
        path_sparsity[path_idx] = path_sparsity[path_idx]/(path.shape[0] - 1)
    return path_sparsity


def check_diversity(paths):
    points = np.empty((len(paths), paths[0].shape[1]))
    for i in range(len(paths)):
        points[i] = paths[i].iloc[-1].to_numpy()

    dist = lambda p1, p2: np.sqrt((p1-p2)@(p1-p2))
    K = np.empty((points.shape[0], points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(i, points.shape[0]):
            d = dist(points[i], points[j])
            K[i,j] = 1/(1 + d)
            K[j,i] = 1/(1 + d)
            if i == j:
                K[i,j] += np.random.normal(0,0.0001)
    diversity = np.linalg.det(K)
    return np.full(len(paths), diversity)
