import numpy as np

def size_normalization(dir, poi, X):
    return dir / X.shape[0]

"""
Normalizes direction based on the distance to the data centroid
"""
def centroid_normalization(dir, poi, X, alpha=0.7):
    centroid = X.mean(axis=0)
    diff = centroid - poi
    centroid_dist = np.sqrt(diff@diff)
    dir = alpha * dir * (centroid_dist / np.sqrt(dir@dir))
    return dir

def perturb_point(scale, x):
    perturbation = np.random.normal(loc=(0,0), scale=scale)
    return x + perturbation

def constant_priority_dir(dir, k=1, step_size=1):
    return constant_step_size(priority_dir(dir, k), step_size)

def priority_dir(dir, k=5):
    dir_arry = dir.to_numpy()
    sorted_idx = np.argsort(-np.abs(dir_arry))  
    dir_new = np.zeros_like(dir_arry)
    dir_new[sorted_idx[:k]] = dir_arry[sorted_idx[:k]]
    sparse_dir = dir.copy()
    sparse_dir.loc[:] = dir_new
    return sparse_dir

def constant_step_size(dir, step_size=1):
    return step_size*dir / np.sqrt(dir@dir)

def preference_dir(preferences, epsilon, max_step_size, dir):
    for dimension in preferences:
        if np.abs(dir[dimension]) > epsilon:
            perturbed_dir = np.zeros_like(dir)
            perturbed_dir[dimension] = min(dir[dimension], max_step_size)
            return perturbed_dir
    return np.zeros_like(dir)

def volcano_alpha(u, sigma_close, sigma_far, scale, dist):
    close_val = np.exp(-0.5 * ((dist - u)/sigma_close)**2)
    far_val = np.exp(-0.5 * ((dist - u)/sigma_far)**2)
    return scale * np.where(dist < u, close_val, far_val)

def cliff_alpha(dist, cutoff=0.5, degree=2):
    return np.where(dist <= cutoff, 1/cutoff**degree, 1/dist**degree)

def model_early_stopping(model, point, cutoff=0.7):
    _, pos_proba = model.predict_proba(point.to_numpy())[0]
    return pos_proba >= cutoff
