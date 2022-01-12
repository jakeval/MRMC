import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import heapq
import os
from scipy import sparse


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
            masks.append(np.abs(differences[:,:,index] <= tolerance))
        masks = np.array(masks)
        return masks.all(axis=0)


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

    def load_graph(self, graph_path, density_path):
        self.graph = sparse.load_npz(graph_path)
        self.density_scores = np.load(density_path)

    def clean_number(self, f):
        return ''.join(map(lambda c: '-' if c == '.' else c, f'{f:.3f}'))

    def set_kde(self, preprocessor, dataset, dataset_str, bandwidth, dir='./face_graphs'):
        print("Set the KDE")
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = self.clean_number(bandwidth)
        density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}.npy')
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        self.density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        if os.path.exists(density_path):
            print("Load density scores from .npy")
            self.density_scores = np.load(density_path)
        else:
            print("Generate new density scores and save to .npy")
            self.density_scores = self.density_estimator(X)
            np.save(density_path, self.density_scores)
        print("Finished setting the KDE")

    def set_kde_subset(self, preprocessor, dataset, dataset_str, bandwidth, idx, dir='./face_graphs'):
        print("Set the KDE")
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = self.clean_number(bandwidth)
        density_path = os.path.join(dir, f'{dataset_str}_density_scores_{bandwidth_str}.npy')
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        self.density_estimator = lambda Z: np.exp(kde.score_samples(Z))
        print("Load density scores from .npy")
        self.density_scores = np.load(density_path)[idx]

    def get_num_blocks(preprocessor, dataset, block_size=80000000):
        print("Begin generating graph in blocks.")
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        rows_per_block = int(np.floor(block_size / (X.shape[1] * X.shape[0])))
        num_blocks = int(np.ceil(X.shape[0] / rows_per_block))
        print("Rows per block: ", rows_per_block)
        print("Number of blocks: ", num_blocks)
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

    def set_graph_from_blocks(self, graph_blocks, preprocessor, dataset, dataset_str, bandwidth, dir='./face_graphs'):
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

        bandwidth_str = self.clean_number(bandwidth)
        distance_str = self.clean_number(self.distance_threshold)
        graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')

        print(f"Number of edges: \t{graph.getnnz()} / {X.shape[0] * (X.shape[0] - 1)}")
        self.graph = graph
        sparse.save_npz(graph_path, self.graph)
        print("Finished setting the graph")

    def set_graph(self, preprocessor, dataset, dataset_str, bandwidth, block_size=80000000, dir='./face_graphs', save_results=True):
        print("Set the graph")
        if self.density_scores is None:
            self.set_kde(preprocessor, dataset, dataset_str, bandwidth, dir=dir)

        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()

        bandwidth_str = self.clean_number(bandwidth)
        distance_str = self.clean_number(self.distance_threshold)
        graph_path = os.path.join(dir, f'{dataset_str}_density_{bandwidth_str}_conditions_{self.conditions_function is not None}_distance_{distance_str}_.npz')
        if os.path.exists(graph_path):
            print("Load graph from .npz")
            self.graph = sparse.load_npz(graph_path)
            print(f"Number of edges: \t{self.graph.getnnz()} / {X.shape[0] * (X.shape[0] - 1)}")
            print("Finished setting the graph")
            return

        rows_per_block = int(np.floor(block_size / (X.shape[1] * X.shape[0])))
        num_blocks = int(np.ceil(X.shape[0] / rows_per_block))
        print("Rows per block: ", rows_per_block)
        print("Number of blocks: ", num_blocks)
        graph = None
        for i in range(num_blocks):
            print(f"Process block {i}/{num_blocks}")
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
            midpoints = ((X[block_start:block_end,None] + X) / 2)[neighbor_mask]
            density = np.zeros((distances.shape[0], X.shape[0]))
            density[neighbor_mask] = self.weight_function(self.density_estimator(midpoints))

            graph_update = sparse.coo_matrix(distances * density)
            if graph is None:
                graph = graph_update
            else:
                graph = sparse.vstack([graph, graph_update])
            print(f"Number of edges: \t{graph.getnnz()} / {X.shape[0] * (X.shape[0] - 1)}")

        self.graph = graph
        sparse.save_npz(graph_path, self.graph)
        print("Finished setting the graph")
        
    def fit(self, dataset, preprocessor, verbose=False):
        X = preprocessor.transform(dataset)
        if 'Y' in X.columns:
            X = X.drop('Y', axis=1)
        X = X.to_numpy()
        self.dataset = dataset
        self.X = X
        self.preprocessor = preprocessor
        self.candidate_mask = (self.clf(self.X) >= self.confidence_threshold) & (self.density_scores >= self.density_threshold)
        if verbose:
            mask = (self.density_scores >= self.density_threshold)
            total = mask.shape[0]
            culled = mask.shape[0] - mask[mask].shape[0]
            print(f"{culled} out of {total} candidates culled for density")
            mask = (self.clf(self.X) >= self.confidence_threshold)
            total = mask.shape[0]
            culled = mask.shape[0] - mask[mask].shape[0]
            print(f"{culled} out of {total} candidates culled for model confidence")

    def iterate(self, poi_index):
        # graph = sparse.csgraph.csgraph_from_dense(self.graph)
        print("running dijkstra...")
        csgraph = self.graph.tocsr()
        # print(csgraph)
        dist_matrix, predecessors = sparse.csgraph.dijkstra(csgraph, indices=poi_index, return_predecessors=True)
        print("got the answer!")

        print("finding candidates...")
        candidates = np.arange(self.X.shape[0])[self.candidate_mask]
        k = min(self.k_paths, candidates.shape[0])
        print(f"Found {k} out of {self.k_paths} requested candidates.")
        sorted_indices = np.argpartition(dist_matrix[self.candidate_mask], k-1)[:k]
        cf_idx = candidates[sorted_indices]
        print(f"Candidates have path distances {dist_matrix[cf_idx]}")

        print("reconstructing paths...")
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
