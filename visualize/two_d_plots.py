import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Display2DPaths:
    def __init__(self, X, Y, title=None):
        self.X = X
        self.Y = Y
        self.title = title
        self.pca = None
        self.paths = None
        self.clusters = None


    def do_pca(self, data_to_fit=None):
        pipe = Pipeline(steps=[
            ('standardize', StandardScaler()),
            ('reduce_dim', PCA(2))
        ])
        X = data_to_fit if data_to_fit is not None else self.X
        pipe.fit(self.X)
        self.pca = pipe
        return self

    def heatmap(self):
        heatmap_plot = lambda ax: ax.hexbin(self.X[:,0], self.X[:,1], C=self.Y, gridsize=60, bins=None, alpha=0.6)
        fig, ax = self._plot_in_2d(heatmap_plot)

        cb = fig.colorbar(cm.ScalarMappable(), ax=ax)
        cb.set_label("Mean Label Value")

        return fig, ax

    def scatter(self):        
        def scatter_plot(ax):
            pos = self.X[self.Y == 1]
            neg = self.X[self.Y == -1]
            ax.scatter(pos[:,0], pos[:,1], alpha=0.6, label="Positive Class")
            ax.scatter(neg[:,0], neg[:,1], alpha=0.6, label="Negative Class")

        return self._plot_in_2d(scatter_plot)

    def _plot_in_2d(self, plot_function):
        if self.pca is not None:
            self._fit_pca()

        fig, ax = plt.subplots(figsize=(10,7))

        plot_function(ax)

        if self.clusters is not None:
            ax.scatter(self.clusters[:,0], self.clusters[:,1], label="Cluster Centers", marker='x', c='fuchsia')
            for i, cluster in enumerate(self.clusters):
                ax.annotate(f"Cluster {i}", cluster, c='fuchsia')

        if self.paths is not None:
            for i, poi_path in enumerate(self.paths):
                ax.plot(poi_path[:,0], poi_path[:,1])
                ax.scatter(poi_path[1:,0], poi_path[1:,1], label=f"Path {i}")
            ax.scatter([self.paths[0][0,0]], [self.paths[0][0,1]], label="POI origin")

        if self.title:
            ax.set_title(self.title)

        ax.legend()

        return fig, ax

    def set_clusters(self, clusters):
        self.clusters = clusters
        return self

    def set_paths(self, paths):
        self.paths = paths
        return self

    def _fit_pca(self):
        self.X = self.pca.transform(self.X)
        if self.clusters is not None:
            self.clusters = self.pca.transform(self.clusters)
        if self.paths is not None:
            pca_paths = []
            for path in self.paths:
                pca_paths.append(self.pca.transform(path))
            self.paths = pca_paths
