import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
from scipy import sparse


LINEAR_APPROXIMATION = True


def weight_function(z):
    return 1/z + 1 # -np.log(z)


class Face:
    def __init__(self, k_paths, clf, distance_threshold, confidence_threshold, density_threshold, kde_bandwidth, kde_rtol=None, weight_function=weight_function):
        self.density_estimator = None # N x N x D -> N x N
        self.density_scores = None
        self.clf = clf
        self.distance_threshold = distance_threshold
        self.weight_function = weight_function # N x N -> N x N
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold
        self.graph = None
        self.candidate_mask = None
        self.original_candidate_mask = None
        self.k_paths = k_paths
        self.X = None
        self.dataset = None
        self.preprocessor = None
        self.kde_bandwidth = kde_bandwidth
        self.kde_rtol = kde_rtol
        if self.kde_rtol is not None:
            self.kde = KernelDensity(bandwidth=self.kde_bandwidth, rtol=self.kde_rtol)
        else:
            self.kde = KernelDensity(bandwidth=self.kde_bandwidth)


    def clean_number(f):
        return ''.join(map(lambda c: '-' if c == '.' else c, f'{f:.3f}'))

    def get_kde_filename(dataset, bandwidth, rtol):
        """Convert KDE parameters to a filename containing the scores."""
        bandwidth_str = Face.clean_number(bandwidth)
        density_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            density_path = f'{dataset}_density_scores_{bandwidth_str}_rtol_{rtol_str}.npy'
        else:
            density_path = f'{dataset}_density_scores_{bandwidth_str}.npy'
        return density_path

    def get_graph_filename(dataset, bandwidth, rtol, distance):
        """Convert graph parameters to a filename containing the graph."""
        bandwidth_str = Face.clean_number(bandwidth)
        distance_str = Face.clean_number(distance)
        graph_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            graph_path = f'{dataset}_density_{bandwidth_str}_rtol_{rtol_str}_distance_{distance_str}.npz'
        else:
            graph_path = f'{dataset}_density_{bandwidth_str}_distance_{distance_str}.npz'
        return graph_path

    def load_kde(dataset, bandwidth, rtol=None, dir='./face_graphs'):
        """Load the density scores from the filesystem.
        """
        density_path = os.path.join(dir, Face.get_kde_filename(dataset, bandwidth, rtol))
        density_scores = np.load(density_path)
        return density_scores

    def load_graph(dataset, bandwidth, distance_threshold, rtol=None, dir='./face_graphs'):
        """Load the graph from the filesystem.
        """
        graph_path = os.path.join(dir, Face.get_graph_filename(dataset, bandwidth, rtol, distance_threshold))
        graph = sparse.load_npz(graph_path)
        return graph

    def load(dataset, bandwidth, distance_threshold, rtol=None, dir='./face_graphs'):
        """Load the density scores and graph from the filesystem.
        """
        return Face.load_kde(dataset, bandwidth, rtol, dir), Face.load_graph(dataset, bandwidth, distance_threshold, rtol, dir)

    def generate_kde_scores(preprocessor, dataset, bandwidth, rtol=None, save_results=True, dataset_str='default', dir='./face_graphs'):
        """Performs density estimation on a dataset and saves it to the filesystem.
        """
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()
        
        kde = None
        if rtol is not None:
            kde = KernelDensity(bandwidth=bandwidth, rtol=rtol)
        else:
            kde = KernelDensity(bandwidth=bandwidth)

        kde.fit(X)
        density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        density_scores = density_estimator(X)
        if save_results:
            density_path = os.path.join(dir, Face.get_kde_filename(dataset_str, bandwidth, rtol))
            np.save(density_path, density_scores)
        return density_scores

    def generate_graph(preprocessor,
                            dataset,
                            bandwidth,
                            density_scores,
                            distance_threshold,
                            rtol=None,
                            dir='./face_graphs',
                            save_results=True,
                            dataset_str='default'):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy().astype(np.float32)

        # Get the KDE
        kde = None
        if rtol is not None:
            kde = KernelDensity(bandwidth=bandwidth, rtol=rtol)
        else:
            kde = KernelDensity(bandwidth=bandwidth)        
        kde.fit(X)
        density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        
        # calculate the pairwise distances
        distances = np.zeros((X.shape[0], X.shape[0]))
        distances += (X*X).sum(axis=1)[:,None]
        distances += (X*X).sum(axis=1)[None,:]
        distances -= 2*(X@X.T)
        distances = np.sqrt(distances)

        # calculate the neighbor mask
        neighbor_mask = ~(distances > distance_threshold) # true wherever there's an edge
        distances[~neighbor_mask] = 0.0

        # calculate the weighted densities
        density = None
        if LINEAR_APPROXIMATION:
            # Instead of using the density of the midpoint of all points, choose the minimum endpoint density
            idx = np.empty((X.shape[0], X.shape[0], 2))
            idx[:,:,0] = np.arange(X.shape[0])[None,:].repeat(X.shape[0], axis=0)
            idx[:,:,1] = np.arange(X.shape[0])[:,None].repeat(X.shape[0], axis=1)
            idx = idx.astype(np.int32)
            density = np.zeros((distances.shape[0], X.shape[0]))
            density[neighbor_mask] = density_scores[idx].min(axis=2)[neighbor_mask]
        else:
            # Beware -- This will not fit in memory
            midpoints = ((X[:,None] + X) / 2)[neighbor_mask]
            density = np.zeros((distances.shape[0], X.shape[0]))
            density[neighbor_mask] = weight_function(density_estimator(midpoints))

        graph = distances * density
        graph = sparse.coo_matrix(graph)
        if save_results:
            graph_path = os.path.join(dir, Face.get_graph_filename(dataset_str, bandwidth, rtol, distance_threshold))
            sparse.save_npz(graph_path, graph)
        
        return graph

    def fit(self, dataset, preprocessor, graph=None, density_scores=None):
        if graph is None:
            density_scores = Face.generate_kde_scores(preprocessor, dataset, self.kde_bandwidth, rtol=self.kde_rtol, save_results=False)
            graph = Face.generate_graph(preprocessor, dataset, self.kde_bandwidth, density_scores, self.distance_threshold, rtol=self.kde_rtol, save_results=False)
        self.graph = graph
        self.density_scores = density_scores

        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()
        self.dataset = dataset
        self.X = X

        if self.density_estimator is None:
            self.kde.fit(X)
            self.density_estimator = lambda Z: np.exp(self.kde.score_samples(Z))

        self.preprocessor = preprocessor
        self.candidate_mask = (self.clf(self.X) >= self.confidence_threshold) & (self.density_scores >= self.density_threshold)

    def add_immutable_condition(self, poi, immutable_features, feature_tolerances=None):
        """Culls points from the candidate set which would violate immutability constraints.
        """
        self.original_candidate_mask = self.candidate_mask
        new_candidate_mask = self.candidate_mask

        for feature in immutable_features:
            if (feature_tolerances is not None) and (feature in feature_tolerances):
                continue
            feature_val = poi.loc[poi.index[0],feature]
            new_candidate_mask = new_candidate_mask & (self.dataset[feature] == feature_val)

        if feature_tolerances is not None:
            for feature, tolerance in feature_tolerances.items():
                feature_val = poi[feature]
                feature_mask = np.abs(self.dataset[feature].to_numpy() - feature_val.to_numpy()) <= tolerance
                new_candidate_mask = new_candidate_mask & feature_mask
        
        self.candidate_mask = new_candidate_mask

    def clear_immutable_condition(self):
        if self.original_candidate_mask is not None:
            self.candidate_mask = self.original_candidate_mask

    def iterate(self, poi_loc, X=None, graph=None, candidate_mask=None):
        """Perform dijkstra's search on the graph.

        poi_loc is not the pandas index, but the numpy index (ie df.index.get_loc(index)).
        It is the index of the point to iterate in the graph.
        """

        if graph is None:
            graph = self.graph
            candidate_mask = self.candidate_mask
            X = self.X

        # perform the search
        csgraph = graph.tocsr()
        dist_matrix, predecessors = sparse.csgraph.dijkstra(csgraph, indices=poi_loc, return_predecessors=True)

        # find the nearest candidate points
        candidates = np.arange(X.shape[0])[candidate_mask]
        k = min(self.k_paths, candidates.shape[0])
        sorted_indices = np.argpartition(dist_matrix[candidate_mask], k-1)[:k]
        cf_idx = candidates[sorted_indices]

        # reconstruct and return the paths
        processed_data = None
        if 'Y' in self.dataset.columns:
            processed_data = self.preprocessor.transform(self.dataset).drop('Y', axis=1)
        else:
            processed_data = self.preprocessor.transform(self.dataset)
        columns = processed_data.columns
        paths = []
        for final_point in cf_idx:
            path = []
            point = final_point
            if predecessors[point] == -9999:
                continue
            while point != -9999:
                path.append(X[point])
                point = predecessors[point]
            pathdf = pd.DataFrame(columns=columns, data=path).loc[::-1].reset_index(drop=True)
            paths.append(pathdf)
        return paths

    def iterate_new_point(self, poi, num_cfs):
        """Perform dijkstra's search on the graph.
        """
        graph, X, candidate_mask = self.add_new_point(self.graph, poi)
        poi_loc = graph.shape[1] - 1
        paths = self.iterate(poi_loc, X=X, graph=graph, candidate_mask=candidate_mask)
        return paths

    def add_new_point(self, graph, point):
        point = self.preprocessor.transform(point)
        point_density = self.density_estimator(point.to_numpy())

        # calculate the (N x N x D) pairwise difference matrix
        differences = self.X - point.to_numpy()

        # calculate the distances matrix
        distances = np.linalg.norm(differences, axis=1) # shape N
        
        # calculate the neighbor mask
        neighbor_mask = ~(distances > self.distance_threshold) # true wherever there's an edge
        distances[~neighbor_mask] = 0.0

        # calculate the weighted densities
        density = None
        # Instead of using the density of the midpoint of all points, choose the minimum endpoint density
        density = np.minimum(self.density_scores, point_density)
        graph_update = distances * density
        col_update = sparse.coo_matrix(np.concatenate([graph_update, [0]])[None,:]).T
        row_update = sparse.coo_matrix(graph_update)

        graph = sparse.vstack([graph, row_update])
        graph = sparse.hstack([graph, col_update])

        X = np.concatenate([self.X, point.to_numpy()])
        candidate_mask = np.concatenate([self.candidate_mask, [False]])
        return graph, X, candidate_mask
