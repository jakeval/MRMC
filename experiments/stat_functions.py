import numpy as np
from core import utils

EQ_EPSILON = 1e-10


def count_path_successes(paths, preprocessor, model, cutoff=0.7):
    successes = 0
    for path in paths:
        point = preprocessor.transform(path.iloc[[-1]])
        is_positive = model.predict_proba(point.to_numpy())[0,1] >= cutoff
        if is_positive:
            successes += 1
    return successes


def distance_comparison(baseline_paths, paths, preprocessor):
    avg_distance = 0
    for baseline_path, path in zip(baseline_paths, paths):
        point = preprocessor.transform(path.iloc[[-1]])
        baseline_point = preprocessor.transform(baseline_path.iloc[[-1]])
        diff = point.to_numpy() - baseline_point.to_numpy()
        avg_distance += np.linalg.norm(diff)
    return avg_distance / len(baseline_paths)


def cosine_comparison(baseline_paths, paths, preprocessor):
    avg_distance = 0
    for baseline_path, path in zip(baseline_paths, paths):
        poi = preprocessor.transform(path.iloc[[0]])
        point = preprocessor.transform(path.iloc[[-1]])
        baseline_point = preprocessor.transform(baseline_path.iloc[[-1]])
        diff = point.to_numpy() - poi.to_numpy()
        baseline_diff = baseline_point.to_numpy() - poi.to_numpy()
        avg_distance += utils.cosine_similarity(diff, baseline_diff)
    return avg_distance / len(baseline_paths)


def check_path_count(paths):
    return np.array([path.shape[0] for path in paths]).mean()


def check_path_length(preprocessor, paths):
    transformed_paths = []
    for path in paths:
        transformed_paths.append(preprocessor.transform(path))
    paths = transformed_paths

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
    return lengths.mean()


def check_positive_probability(preprocessor, model, paths, cutoff):
    return count_path_successes(paths, preprocessor, model, cutoff=cutoff) / len(paths)


def check_final_point_distance(preprocessor, paths):
    dist = 0
    for path in paths:
        poi = preprocessor.transform(path.iloc[[0]]).to_numpy()
        cf_point = preprocessor.transform(path.iloc[[-1]]).to_numpy()
        dist += np.linalg.norm(poi - cf_point)
    return dist / len(paths)


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
def check_immutability(preprocessor, immutable_features, poi, cf_points):
    cf_points = preprocessor.inverse_transform(cf_points)
    poi = preprocessor.inverse_transform(poi)

    cf_points = cf_points[immutable_features]
    poi = poi[immutable_features]

    numeric_columns = poi.select_dtypes(include=np.number).columns
    other_columns = poi.columns.difference(numeric_columns)

    numeric_diff = (np.abs(cf_points[numeric_columns].to_numpy() - poi[numeric_columns].to_numpy()) >= EQ_EPSILON).sum(axis=1)
    other_diff = (cf_points[other_columns].to_numpy() != poi[other_columns].to_numpy()).sum(axis=1)
    return numeric_diff + other_diff



def check_sparsity(paths, num_columns, cat_columns):
    avg_sparsity = 0
    for path in paths:
        poi = path.iloc[[0]]
        cf = path.iloc[[-1]]
        num_count = (np.abs(cf[num_columns].to_numpy() - poi[num_columns].to_numpy()) >= EQ_EPSILON).sum()
        cat_count = (cf[cat_columns].to_numpy() != poi[cat_columns].to_numpy()).sum()
        avg_sparsity += num_count + cat_count
    return avg_sparsity / len(paths)


def check_diversity(preprocessor, paths):
    transformed_paths = []
    for path in paths:
        transformed_paths.append(preprocessor.transform(path))
    paths = transformed_paths
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
    return diversity
