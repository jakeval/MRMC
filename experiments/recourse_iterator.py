from core.utils import cosine_similarity

def iterate_recourse(transformed_poi,
                     preprocessor,
                     max_iterations,
                     certainty_cutoff,
                     k_paths,
                     get_recourse,
                     get_positive_probability,
                     weight_function,
                     perturb_dir=None):
    path_starts = get_recourse(transformed_poi, k_paths) # dataframe of points in transformed space
    if path_starts is None:
        return None
    paths = []
    for i in range(path_starts.shape[0]):
        point = path_starts.iloc[[i]]
        full_path = generate_path(transformed_poi, point, preprocessor, max_iterations, certainty_cutoff, get_recourse, get_positive_probability, weight_function, perturb_dir=perturb_dir)
        paths.append(full_path)
    return paths

def validate_point(point, preprocessor):
    point = preprocessor.inverse_transform(point)
    point = preprocessor.transform(point)
    return point

def generate_path(p1, 
                  p2, 
                  preprocessor, 
                  max_iterations, 
                  certainty_cutoff, 
                  get_recourse, 
                  get_positive_probability, 
                  weight_function, 
                  perturb_dir=None):
    path = p1.reset_index(drop=True)
    for i in range(max_iterations):
        dir = p2.to_numpy() - p1.to_numpy()
        dir = weight_function(dir) # rescale the direction
        if perturb_dir is not None:
            dir = perturb_dir(dir)
        perturbed_point = p1 + dir
        perturbed_point = validate_point(perturbed_point, preprocessor)
        path = path.append(perturbed_point, ignore_index=True)
        p1 = perturbed_point
        if get_positive_probability(perturbed_point) >= certainty_cutoff:
            return path
        p2 = get_recourse(perturbed_point, 1)
        if p2 is None:
            return path
    
    return path
