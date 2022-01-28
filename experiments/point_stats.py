import numpy as np
import math

EQ_EPSILON = 1e-10

def check_positive_probability(model, poi, cf_points, cutoff):
    predictions = model.predict_proba(cf_points.to_numpy())[:,1]
    return (predictions >= cutoff).astype(np.int32)


def check_model_certainty(model, poi, cf_points):
    return model.predict_proba(cf_points.to_numpy())[:,1]


def check_final_point_distance(poi, cf_points):
    diff = cf_points.to_numpy() - poi.to_numpy()
    return np.sqrt((diff**2).sum(axis=1))


def check_validity(preprocessor, column_names_per_feature, poi, cf_points):
    """The number of features per-point which are not valid.
    """
    valid_points = preprocessor.inverse_transform(cf_points)
    valid_points = preprocessor.transform(valid_points)
    diff = (cf_points - valid_points) > EQ_EPSILON
    total = 0
    for column_names in column_names_per_feature:
        total += diff[column_names].any(axis=1)
    return total


"""
For each path,
    calculates the average euclidean distance between each point
    and its nearest immutable-enforced point
"""
def check_immutability(preprocessor, immutable_features, poi, cf_points):
    cf_points = preprocessor.inverse_transform(cf_points)
    poi = preprocessor.inverse_transform(poi)
    return (cf_points[immutable_features].to_numpy() != poi[immutable_features].to_numpy()).sum(axis=1)


"""
For each path,
    calculates the number of altered features for each path
"""
def check_sparsity(preprocessor, poi, cf_points):
    poi = preprocessor.inverse_transform(poi)
    cf_points = preprocessor.inverse_transform(cf_points)
    return (cf_points.to_numpy() != poi.to_numpy()).sum(axis=1)


def check_diversity(poi, cf_points):
    d = cf_points.shape[1]
    total = 0
    for i in range(cf_points.shape[0] - 1):
        for j in range(i+1, cf_points.shape[0] - 1):
            ci = cf_points.iloc[i]
            cj = cf_points.iloc[j]
            total += (np.abs(ci - cj) > EQ_EPSILON).to_numpy().sum()
    diversity = total * 1/(math.comb(d, 2) * d)
    return np.full(cf_points.shape[0], diversity)

"""
def check_diversity(preprocessor, poi, cf_points):
    points = cf_points.to_numpy()
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
    return np.full(cf_points.shape[0], diversity)

"""