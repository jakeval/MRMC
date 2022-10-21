import numpy as np
import pandas as pd
from data import recourse_adapter
from typing import Any, Mapping, Sequence


# TODO(@jakeval): Most functions here are no longer used and should be removed.
#                 The remaining functions should be retouched.


MIN_DIRECTION = 1e-32
EQ_EPSILON = 1e-10


def recategorize_feature(column: pd.Series, inverse_category_dict: Mapping[str, Sequence[str]]) -> pd.Series:
    """Returns a Series where some values are remapped to others.

    Given a dictionary like {'new_val': [val1, val2]}, val1 and val2 are relabeled as new_val in the new Series.
    
    Args:
        column: The series of data to remap.
        inverse_category_dict: The remapping dict formatted as described above.
        
    Returns:
        A new series with remapped values."""
    new_column = column.copy()
    for key, val_list in inverse_category_dict.items():
        for val in val_list:
            new_column = np.where(new_column == val, key, new_column)
    return new_column


def size_normalization(dir, poi, X):
    return dir / X.shape[0]

def cosine_similarity(x1, x2):
    return (x1@x2) / (np.sqrt(x1@x1) * np.sqrt(x2@x2))

def centroid_normalization(dir, poi, X, alpha=0.7):
    """Normalizes direction based on the distance to the data centroid."""
    if dir@dir == 0: # if the direction is zero or very near it, return the original direction
        return dir
    centroid = X.mean(axis=0)
    diff = centroid - poi
    centroid_dist = np.sqrt(diff@diff)
    dir = (alpha * dir * centroid_dist) / np.sqrt(dir@dir)
    return dir

def rescale_dir(dir: recourse_adapter.EmbeddedSeries, rescale_factor: float):
    new_dir = dir * rescale_factor
    return new_dir

def privacy_perturb_dir(dir, epsilon=0.1, delta=0.01, C=1):
    beta = np.sqrt(2*np.log(1.25/delta))
    stdev = (beta*C**2)/epsilon
    return dir + np.random.normal(0, stdev, size=dir.shape)

def randomly_perturb_dir(dir: recourse_adapter.EmbeddedSeries, ratio: float):
    norm = np.linalg.norm(dir)
    noise = np.random.normal(0, 1, len(dir))
    noise = (noise / np.linalg.norm(noise)) * ratio * norm
    new_dir = dir + noise
    new_dir = (new_dir / np.linalg.norm(new_dir)) * norm
    return new_dir

def random_perturb_dir(adapter, scale, categorical_prob, p1, dir, num_features, cat_features, immutable_features=None, immutable_column_indices=None):
    """Randomly perturb a direction.

    Continuous variables are perturbed with random noise.
    Categorical variables are randomly changed with probability p
    """

    if immutable_features is not None:
        num_features = num_features.difference(immutable_features)
        cat_features = cat_features.difference(immutable_features)

    new_dir = dir.copy()

    for feature in cat_features:
        # set the original category to -1 and a new category to +1
        if np.random.random() <= categorical_prob:
            ohe_columns = adapter.get_feature_names_out([feature])
            p1_cat = ohe_columns[np.argmax(p1[ohe_columns].to_numpy())]
            unselected_categories = ohe_columns
            if (new_dir[ohe_columns].to_numpy() == 0).all():
                #unselected_categories = list(filter(lambda cat: cat != p1_cat, unselected_categories))
                new_dir.loc[:,p1_cat] = -1
            else:
                dir_cat = ohe_columns[np.argmax(new_dir[ohe_columns].to_numpy())]
                new_dir.loc[:,dir_cat] = 0
                #unselected_categories = list(filter(lambda cat: cat != dir_cat, unselected_categories))
            new_cat = np.random.choice(unselected_categories, 1)[0]
            new_dir.loc[:,new_cat] += 1
    
    # generate random noise
    r = np.random.normal(0, 1, len(num_features))
    
    # rescale random noise to a percentage of the original direction's magnitude
    original_norm = np.linalg.norm(dir[num_features].to_numpy())
    if original_norm == 0:
        return dir
    r = (r * (scale * original_norm)) / np.linalg.norm(r)

    # rescale the perturbed direction to the original magnitude
    num_only = False
    if num_only:
        old_numeric_values = new_dir[num_features].to_numpy()
        new_numeric_values = old_numeric_values + r
        new_numeric_values = (new_numeric_values * np.linalg.norm(old_numeric_values)) / np.linalg.norm(new_numeric_values)
        new_dir.loc[:,num_features] = new_numeric_values
    else:
        new_dir = new_dir * np.linalg.norm(dir.to_numpy()) / np.linalg.norm(new_dir.to_numpy())

    return new_dir

def perturb_point(scale, x):
    perturbation = np.random.normal(loc=(0,0), scale=scale)
    return x + perturbation

def constant_priority_dir(dir, k=1, step_size=1):
    return constant_step_size(priority_dir(dir, k), step_size)

def priority_dir(dir, k=5):
    dir_arry = dir.to_numpy()
    sorted_idx = np.argsort(-np.abs(dir_arry[0,:]))
    dir_new = np.zeros_like(dir_arry)
    dir_new[:,sorted_idx[:k]] = dir_arry[:,sorted_idx[:k]]
    #sparse_dir = dir.copy()
    #sparse_dir.loc[:] = dir_new
    return pd.DataFrame(columns=dir.columns, data=dir_new)

def constant_step_size(dir, step_size=1):
    normalization = np.linalg.norm(dir.to_numpy())
    if normalization <= MIN_DIRECTION:
        return dir
    return (step_size*dir) / normalization

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

def epsilon_compare(point, df):
    numeric_columns = point.select_dtypes(include=np.number).columns
    other_columns = point.columns.difference(numeric_columns)

    numeric_diff = np.ones((df.shape[0], df.shape[1])).astype(np.bool8)
    if len(numeric_columns) > 0:
        numeric_diff = (np.abs(df[numeric_columns].to_numpy() - point[numeric_columns].to_numpy()) >= EQ_EPSILON)
    other_diff = np.ones((df.shape[0], df.shape[1])).astype(np.bool8)
    if len(other_columns) > 0:
        other_diff = (df[other_columns].to_numpy() != point[other_columns].to_numpy())

    return numeric_diff & other_diff


def random_poi(dataset: pd.DataFrame, column = 'Y', label: Any = -1, drop_label: bool = True) -> pd.Series:
    """Selects a random POI of the given label from the dataset.
    
    Args:
        dataset: The dataset to sample from.
        label: The label of the point to select.
        drop_label: Whether to drop the label from the returned POI.
    """
    poi = dataset[dataset[column] == label].sample(1)
    if drop_label:
        poi = poi.drop(column, axis=1)

    return poi.iloc[0]
