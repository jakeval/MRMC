import numpy as np


def check_positive_probability(model, poi, cf_points):
    predictions = model.predict_proba(cf_points)
    return predictions[:,1]


def check_final_point_distance(preprocessor, poi, cf_points):
    cf_points = preprocessor.transform(cf_points)
    poi = preprocessor.transform(poi)
    diff = cf_points.to_numpy() - poi.to_numpy()
    return np.sqrt((diff**2).sum(axis=1))


"""
The returned points are already valid
"""
def check_validity_distance(poi, cf_points):
    return np.zeros(cf_points.shape[0])


"""
For each path,
    calculates the average euclidean distance between each point
    and its nearest immutable-enforced point
"""
def check_immutability(preprocessor, immutable_features, poi, cf_points):
    cf_points = preprocessor.transform(cf_points)
    poi = preprocessor.transform(poi)
    diff = (cf_points[immutable_features].to_numpy() - poi[immutable_features].to_numpy())
    return np.sqrt((diff**2).sum(axis=1))


"""
For each path,
    calculates the average number of altered features for each point
"""
def check_sparsity(preprocessor, poi, cf_points):
    cf_points = preprocessor.transform(cf_points)
    poi = preprocessor.transform(poi)
    return ((cf_points.to_numpy() - poi.to_numpy()) != 0).sum(axis=1)


def check_diversity(preprocessor, poi, cf_points):
    points = preprocessor.transform(cf_points).to_numpy()
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
