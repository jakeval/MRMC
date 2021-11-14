import numpy as np

def size_normalization(dir, X):
  return dir / X.shape[0]

def perturb_point(scale, x):
  perturbation = np.random.normal(loc=(0,0), scale=scale)
  return x + perturbation

def constant_priority_dir(dir, k=1, step_size=1):
  return constant_step_size(priority_dir(dir, k), step_size)

def priority_dir(dir, k=5):
  sorted_idx = np.argsort(-np.abs(dir))  
  dir_new = np.zeros_like(dir)
  dir_new[sorted_idx[:k]] = dir[sorted_idx[:k]]
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

def volcano_alpha(u, sigma_close, sigma_far, scale, dist):
  close_val = np.exp(-0.5 * ((dist - u)/sigma_close)**2)
  far_val = np.exp(-0.5 * ((dist - u)/sigma_far)**2)
  return scale * np.where(dist < u, close_val, far_val)

def cliff_alpha(dist, cutoff=0.5, degree=2):
  return np.where(dist <= cutoff, 1/cutoff**degree, 1/dist**degree)