import numpy as np
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# import matplotlib.fontmanager as fm

#matplotlib.rcParams['font.family'] = 'Avenir'
#plt.rcParams['font.size'] = 18
#plt.rcParams['axes.linewidth'] = 2
plt.style.use('style.mplstyle')

class Display2DPaths:
    def __init__(self, X, Y, title=None):
        self.X = X
        self.Y = Y
        self.title = title
        self.pca = None
        self.paths = None
        self.clusters = None
        self.colors = ['lightskyblue', 'mediumseagreen']
        self.marker_cycle = ['s', 'D', '^', 'p', 'X', '+']
        self.small_legend = False

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

    def scatter(self, ax=None):
        def scatter_plot(ax):
            pos = self.X[self.Y == 1]
            neg = self.X[self.Y == -1]
            cp, cn = self.colors
            ax.scatter(pos[:,0], pos[:,1], alpha=0.6, label="Positive Class", c=cp)
            ax.scatter(neg[:,0], neg[:,1], alpha=0.6, label="Negative Class", c=cn)

        return self._plot_in_2d(scatter_plot, ax)

    def _plot_in_2d(self, plot_function, ax=None):
        if self.pca is not None:
            self._fit_pca()

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,7))

        plot_function(ax)

        if self.paths is not None:
            color = 'midnightblue'
            for i, poi_path in enumerate(self.paths):
                marker = self.marker_cycle[i % len(self.marker_cycle)]
                ax.plot(poi_path[:,0], poi_path[:,1], c=color)
                label=f"Recourse Path {i}"
                if self.small_legend:
                    label=None
                ax.scatter(poi_path[1:,0], poi_path[1:,1], label=label, marker=marker, c=color, s=20)
            ax.scatter([self.paths[0][0,0]], [self.paths[0][0,1]], label="POI origin", s=30, c=color)

        if self.clusters is not None:
            cluster_colors = 'red'
            label = "Cluster Centers"
            if self.small_legend:
                label=None
            ax.scatter(self.clusters[:,0], self.clusters[:,1], label=label, s=30, marker='x', c=cluster_colors)
            for i, cluster in enumerate(self.clusters):
                ax.annotate(f"Cluster {i}", cluster + np.array([0.025, 0]), c=color)

        if self.title:
            ax.set_title(self.title)

        return fig, ax

    def use_small_legend(self):
        self.small_legend = True
        return self

    def use_large_legend(self):
        self.small_legend = False
        return self

    def set_colors(self, pos_color, neg_color):
        self.colors = [pos_color, neg_color]
        return self

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
