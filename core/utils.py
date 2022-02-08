import numpy as np

MIN_DIRECTION = 1e-32

def size_normalization(dir, poi, X):
    return dir / X.shape[0]

def cosine_similarity(x1, x2):
    return (x1@x2) / (np.sqrt(x1@x1) * np.sqrt(x2@x2))

"""
Normalizes direction based on the distance to the data centroid
"""
def centroid_normalization(dir, poi, X, alpha=0.7):
    if dir@dir == 0: # if the direction is zero or very near it, return the original direction
        return dir
    centroid = X.mean(axis=0)
    diff = centroid - poi
    centroid_dist = np.sqrt(diff@diff)
    dir = (alpha * dir * centroid_dist) / np.sqrt(dir@dir)
    return dir

def privacy_perturb_dir(dir, epsilon=0.1, delta=0.01, C=1):
    beta = np.sqrt(2*np.log(1.25/delta))
    stdev = (beta*C**2)/epsilon
    return dir + np.random.normal(0, stdev, size=dir.shape)

def random_perturb_dir(scale, dir, immutable_column_indices=None):
    # generate random noise
    r = np.random.normal(0, scale, dir.shape)
    # zero-out immutable columns
    if immutable_column_indices is not None:
        r[:,immutable_column_indices] = 0
    
    # rescale random noise to a percentage of the original direction's magnitude
    original_norm = np.linalg.norm(dir)
    r = (r * (scale * original_norm)) / np.linalg.norm(r)

    # rescale the perturbed direction to the original magnitude
    new_dir = dir + r
    new_dir = (new_dir * np.linalg.norm(dir)) / np.linalg.norm(new_dir)
    return new_dir

def perturb_point(scale, x):
    perturbation = np.random.normal(loc=(0,0), scale=scale)
    return x + perturbation

def constant_priority_dir(dir, k=1, step_size=1):
    return constant_step_size(priority_dir(dir, k), step_size)

def priority_dir(dir, k=5):
    #dir_arry = dir.to_numpy()
    sorted_idx = np.argsort(-np.abs(dir[0,:]))
    dir_new = np.zeros_like(dir)
    dir_new[:,sorted_idx[:k]] = dir[:,sorted_idx[:k]]
    #sparse_dir = dir.copy()
    #sparse_dir.loc[:] = dir_new
    return dir_new

def constant_step_size(dir, step_size=1):
    return step_size*dir / np.sqrt(dir@dir)

def preference_dir(preferences, epsilon, max_step_size, dir):
    for dimension in preferences:
        if np.abs(dir[dimension]) > epsilon:
            perturbed_dir = np.zeros_like(dir)
            perturbed_dir[dimension] = min(dir[dimension], max_step_size)
            return perturbed_dir
    return np.zeros_like(dir)

def normal_alpha(dist, width=1):
    return np.exp(-0.5 * (dist/width)**2)

def volcano_alpha(dist, cutoff=0.5, degree=2):
    return np.where(dist <= cutoff, 1/cutoff**degree, 1/dist**degree)

def private_alpha(dist, cutoff=0.5, degree=2):
    return 1/dist * np.where(dist <= cutoff, 1/cutoff**degree, 1/dist**degree)

def model_early_stopping(model, point, cutoff=0.7):
    _, pos_proba = model.predict_proba(point.to_numpy())[0]
    return pos_proba >= cutoff
