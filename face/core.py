import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import heapq
import os
from scipy import sparse


LINEAR_APPROXIMATION = True


def euclidean_distance(X):
    return np.linalg.norm(X[:,None] - X, axis=2)


def weight_function(z):
    return 1/z + 1 # -np.log(z)


def immutable_conditions(differences, immutable_column_indices, tolerances=None):
    """
    differences: N x N x D array of differences. differences[i, j, k] = X[i,k] - X[j,k]
    """
    if tolerances is None:
        return (differences[:,:,np.array(immutable_column_indices)] == 0).all(axis=2)
    else:
        strict_mask = (differences[:,:,np.array(immutable_column_indices)] == 0).all(axis=2)
        masks = [strict_mask]
        for index, tolerance in tolerances.items():
            masks.append(np.abs(differences[:,:,index]) <= tolerance)
        masks = np.array(masks)
        mask = masks.all(axis=0)
        return mask


class Face:
    def __init__(self, k_paths, clf, distance_threshold, confidence_threshold,
                 density_threshold, conditions_function=None, 
                 weight_function=weight_function, distance_function=euclidean_distance):
        self.density_estimator = None # N x N x D -> N x N
        self.density_scores = None
        self.clf = clf
        self.distance_threshold = distance_threshold
        self.weight_function = weight_function # N x N -> N x N
        self.conditions_function = conditions_function
        self.distance_function = distance_function
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold
        self.graph = None
        self.candidate_mask = None
        self.k_paths = k_paths
        self.X = None
        self.dataset = None
        self.preprocessor = None
        self.original_graph = None
        self.original_density_scores = None
        self.recover_originals = False

    def clean_number(f):
        return ''.join(map(lambda c: '-' if c == '.' else c, f'{f:.3f}'))

    def set_kde(self, preprocessor, dataset, dataset_str, bandwidth, rtol=None, dir='./face_graphs', save_results=True):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = Face.clean_number(bandwidth)
        density_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}_rtol_{rtol_str}.npy')
        else:
            density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}.npy')
        kde = None
        if kde is not None:
            kde = KernelDensity(bandwidth=bandwidth, rtol=rtol)
        else:
            kde = KernelDensity(bandwidth=bandwidth)        
        kde.fit(X)
        self.density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        if os.path.exists(density_path):
            self.density_scores = np.load(density_path)
        else:
            self.density_scores = self.density_estimator(X)
            if save_results:
                np.save(density_path, self.density_scores)

    def set_kde_subset(self, preprocessor, dataset, dataset_str, bandwidth, idx, rtol=None, dir='./face_graphs'):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = Face.clean_number(bandwidth)
        density_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}_rtol_{rtol_str}.npy')
        else:
            density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}.npy')
        kde = None
        if kde is not None:
            kde = KernelDensity(bandwidth=bandwidth, rtol=rtol)
        else:
            kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        self.density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        self.density_scores = np.load(density_path)[idx]

    def get_num_blocks(preprocessor, dataset, block_size=80000000):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        rows_per_block = int(np.floor(block_size / (X.shape[1] * X.shape[0])))
        num_blocks = int(np.ceil(X.shape[0] / rows_per_block))
        return num_blocks

    def generate_graph_block(self, preprocessor, dataset, 
                             block_index, bandwidth, 
                             block_size=80000000, dir='./face_graphs'):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        density_estimator = lambda Z: np.exp(kde.score_samples(Z))

        rows_per_block = int(np.floor(block_size / (X.shape[1] * X.shape[0])))

        block_start = block_index*rows_per_block
        block_end = (block_index+1)*rows_per_block
        
        # calculate the (N x N x D) pairwise difference matrix
        differences = X[block_start:block_end,None] - X

        # calculate the (N x N) conditions mask
        conditions_mask = np.ones((differences.shape[0], differences.shape[1])).astype(np.bool)
        if self.conditions_function is not None:
            conditions_mask = self.conditions_function(differences)

        # calculate the distances matrix
        distances = np.linalg.norm(differences, axis=2)
        
        # calculate the neighbor mask
        neighbor_mask = ~((distances > self.distance_threshold) | ~conditions_mask) # true wherever there's an edge
        distances[~neighbor_mask] = 0.0

        # calculate the weighted densities
        # num_rows = block_end - block_start
        midpoints = ((X[block_start:block_end,None] + X) / 2)[neighbor_mask]
        density = np.zeros((distances.shape[0], X.shape[0]))
        density[neighbor_mask] = self.weight_function(density_estimator(midpoints))

        graph_block = sparse.coo_matrix(distances * density)
        return graph_block

    def set_graph_from_blocks(self, graph_blocks, preprocessor, dataset, dataset_str, bandwidth, rtol=None, dir='./face_graphs'):
        graph = None
        for block in graph_blocks:
            if graph is None:
                graph = block
            else:
                graph = sparse.vstack([graph, block])

        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = Face.clean_number(bandwidth)
        distance_str = Face.clean_number(self.distance_threshold)
        graph_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_rtol_{rtol_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')
        else:
            graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')

        self.graph = graph
        sparse.save_npz(graph_path, self.graph)

    def set_graph(self, preprocessor, dataset, dataset_str, bandwidth, block_size=80000000, rtol=None, dir='./face_graphs', save_results=True):
        if self.density_scores is None:
            self.set_kde(preprocessor, dataset, dataset_str, bandwidth, rtol=rtol, dir=dir, save_results=save_results)

        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy().astype(np.float32)

        bandwidth_str = Face.clean_number(bandwidth)
        distance_str = Face.clean_number(self.distance_threshold)
        graph_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_rtol_{rtol_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')
        else:
            graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')
        if os.path.exists(graph_path):
            self.graph = sparse.load_npz(graph_path)
            return

        rows_per_block = int(np.floor(block_size / (X.shape[1] * X.shape[0])))
        num_blocks = int(np.ceil(X.shape[0] / rows_per_block))
        graph = None
        for i in range(num_blocks):
            block_start = i*rows_per_block
            block_end = (i+1)*rows_per_block
            
            # calculate the (N x N x D) pairwise difference matrix
            differences = X[block_start:block_end,None] - X

            # calculate the (N x N) conditions mask
            conditions_mask = np.ones((differences.shape[0], differences.shape[1])).astype(np.bool)
            if self.conditions_function is not None:
                conditions_mask = self.conditions_function(differences)

            # calculate the distances matrix
            distances = np.linalg.norm(differences, axis=2)
            
            # calculate the neighbor mask
            neighbor_mask = ~((distances > self.distance_threshold) | ~conditions_mask) # true wherever there's an edge
            distances[~neighbor_mask] = 0.0

            # calculate the weighted densities
            # num_rows = block_end - block_start
            density = None
            if LINEAR_APPROXIMATION:
                # Instead of using the density of the midpoint of all points, choose the minimum endpoint density
                idx = np.empty((differences.shape[0], differences.shape[1], 2))
                idx[:,:,0] = np.arange(differences.shape[1])[None,:].repeat(differences.shape[0], axis=0)
                idx[:,:,1] = np.arange(differences.shape[0])[:,None].repeat(differences.shape[1], axis=1)
                idx = idx.astype(np.int32)
                density = np.zeros((distances.shape[0], X.shape[0]))
                density[neighbor_mask] = self.density_scores[idx].min(axis=2)[neighbor_mask]
            else:
                midpoints = ((X[block_start:block_end,None] + X) / 2)[neighbor_mask]
                density = np.zeros((distances.shape[0], X.shape[0]))
                density[neighbor_mask] = self.weight_function(self.density_estimator(midpoints))

            graph_update = distances * density
            graph_update = sparse.coo_matrix(graph_update)
            if graph is None:
                graph = graph_update
            else:
                graph = sparse.vstack([graph, graph_update])

        self.graph = graph
        if save_results:
            sparse.save_npz(graph_path, self.graph)

    def load_graph(dataset, bandwidth, distance_threshold, use_conditions, rtol=None, dir='./face_graphs'):
        bandwidth_str = Face.clean_number(bandwidth)
        density_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            density_path = os.path.join(dir, f'{dataset}_density_scores_{bandwidth_str}_rtol_{rtol_str}.npy')
        else:
            density_path = os.path.join(dir, f'{dataset}_density_scores_{bandwidth_str}.npy')
        density_scores = np.load(density_path)

        distance_str = Face.clean_number(distance_threshold)
        graph_path = None
        if rtol is not None:
            rtol_str = Face.clean_number(rtol)
            graph_path = os.path.join(dir, f'{dataset}_density_{bandwidth_str}_rtol_{rtol_str}_conditions_{use_conditions}_distance_{distance_str}_.npz')
        else:
            graph_path = os.path.join(dir, f'{dataset}_density_{bandwidth_str}_conditions_{use_conditions}_distance_{distance_str}_.npz')
        graph = sparse.load_npz(graph_path)

        return density_scores, graph

    def set_graph_from_memory(self, graph, density_scores):
        self.graph = graph
        self.density_scores = density_scores
        
    def fit(self, dataset, preprocessor, verbose=False):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()
        self.dataset = dataset
        self.X = X
        self.preprocessor = preprocessor
        if self.recover_originals:
            self.density_scores = self.original_density_scores
            self.graph = self.original_graph
            self.recover_originals = False
        self.candidate_mask = (self.clf(self.X) >= self.confidence_threshold) & (self.density_scores >= self.density_threshold)
        if verbose:
            mask = (self.density_scores >= self.density_threshold)
            total = mask.shape[0]
            culled = mask.shape[0] - mask[mask].shape[0]
            mask = (self.clf(self.X) >= self.confidence_threshold)
            total = mask.shape[0]
            culled = mask.shape[0] - mask[mask].shape[0]

    def add_age_condition(self, age_tolerance, poi_index):
        self.recover_originals = True
        self.original_graph = self.graph
        self.original_density_scores = self.density_scores
        poi_age = self.dataset.loc[poi_index, 'age']
        age_mask = np.abs(self.dataset['age'] - poi_age) <= age_tolerance
        self.dataset = self.dataset[age_mask]
        self.X = self.X[age_mask]
        self.candidate_mask = self.candidate_mask[age_mask]
        self.graph = self.graph.tocsr()[age_mask][:,age_mask].tocoo()

    def iterate(self, poi_index):
        # graph = sparse.csgraph.csgraph_from_dense(self.graph)
        poi_index = self.dataset.index.get_loc(poi_index)
        csgraph = self.graph.tocsr()
        # print(csgraph)
        dist_matrix, predecessors = sparse.csgraph.dijkstra(csgraph, indices=poi_index, return_predecessors=True)

        candidates = np.arange(self.X.shape[0])[self.candidate_mask]
        k = min(self.k_paths, candidates.shape[0])
        sorted_indices = np.argpartition(dist_matrix[self.candidate_mask], k-1)[:k]
        cf_idx = candidates[sorted_indices]

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
                path.append(self.X[point])
                point = predecessors[point]
            pathdf = pd.DataFrame(columns=columns, data=path)
            paths.append(pathdf)
        return paths

    def iterate2(self, poi_index):
        indices = np.arange(0, self.X.shape[0])
        distances = np.full(self.X.shape[0], np.inf)
        distances[poi_index] = 0
        frontier = list([(distance, i) for distance, i in zip(distances, range(self.X.shape[0]))])
        frontier_set = np.ones(len(frontier)).astype(np.bool)
        heapq.heapify(frontier)
        parents = np.full(self.X.shape[0], -1)
        while len(frontier) > 0:
            curr_idx = None
            end_loop = False
            while curr_idx is None:
                if len(frontier) == 0:
                    end_loop = True
                    break
                _, curr_idx = heapq.heappop(frontier)
                if not frontier_set[curr_idx]:
                    curr_idx = None
            if end_loop:
                break
            frontier_set[curr_idx] = False
            new_distances = self.graph[curr_idx]
            curr_distance = distances[curr_idx]
            distance_mask = distances > curr_distance + new_distances
            neighbor_mask = (self.graph[curr_idx] > 0) & frontier_set
            mask = distance_mask & neighbor_mask

            for i in indices[mask]:
                distances[i] = curr_distance + new_distances[i]
                parents[i] = curr_idx
                heapq.heappush(frontier, (curr_distance + new_distances[i], i))
        
        candidates = np.arange(self.X.shape[0])[self.candidate_mask]
        k = min(self.k_paths, candidates.shape[0])
        sorted_indices = np.argpartition(distances[self.candidate_mask], k-1)[:k]
        cf_idx = candidates[sorted_indices]

        paths = []
        for final_point in cf_idx:
            path = np.array([])
            point = final_point
            if distances[point] == np.inf:
                continue
            while point != -1:
                path = np.append(point, path)
                point = parents[point]
            paths.append(path.astype(np.int32))
        
        return paths
