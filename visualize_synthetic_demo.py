import numpy as np
from visualize.two_d_plots import Display2DPaths
from matplotlib import pyplot as plt
from core.mrmc import MRM, MRMCIterator
from sklearn.cluster import KMeans
import pandas as pd
from data.adult_data_adapter import AdultPreprocessor
from data import data_adapter as da
from core.mrmc import utils

np.random.seed(88557)

# generate three clusters
# generate a "path" between them

def generate_cluster(center, std, N):
    return np.random.normal(center, std, size=(N,2))

def generate_path(start, end, width, N, rand_dist):
    r = rand_dist(N)
    r = rescale(r)
    v = (end - start)
    normal_noise = np.random.normal(0, width, N)
    normal = np.array([-v[1]/v[0], 1])
    normal = normal / np.linalg.norm(normal)
    noiseless_points = r[:,None] * v + start
    return noiseless_points + normal * normal_noise[:,None]

def rescale(points):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    return (points - pmin) / (pmax - pmin)


def get_labels(points, cutoff):
    return (points.max(axis=1) > cutoff).astype('int') * 2 - 1


std = 0.15
mincoord = 0.3
c1 = np.array([mincoord, 1])
c2 = np.array([1, mincoord])
cr = np.array([0,0])
N = 200
N2 = 100

extra_curriculars = generate_cluster(c1, (std, std), N)
academics = generate_cluster(c2, (std, std), N)
rejected = generate_cluster(cr, (std, std), N)

# rand_dist = lambda N: np.random.random(N) # lambda N: np.random.chisquare(50, N)
rand_dist = lambda N: np.random.normal(0, 0.85, N)
split = np.array([mincoord+0.1, mincoord+0.1])
width = 0.08
pstem = generate_path(cr, split, width, N2, rand_dist)
p1 = generate_path(c1, split, width, N2, rand_dist)
p2 = generate_path(c2, split, width, N2, rand_dist)

data = np.concatenate([extra_curriculars, academics, rejected, pstem, p1, p2])
# data = np.concatenate([pstem, p1, p2])
X = rescale(data)
Y = get_labels(data, 0.6)

data = pd.DataFrame({'grades': X[:,0], 'extracurriculars': X[:,1], 'Y': Y})

n_clusters=2
km = KMeans(n_clusters=n_clusters)
km.fit(X[Y == 1])
cluster_assignments = km.predict(X[Y == 1])

preprocessor = AdultPreprocessor([], ['grades', 'extracurriculars'])
preprocessor.fit(data)

poi = da.random_poi(data)


#fig, (ax1, ax2) = plt.subplots(1,2)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
adjustment = 0.02
w = 5.5 / 2
fig1.set_size_inches(w=w*(1 + adjustment), h=2.75)
fig2.set_size_inches(w=w*(1 - adjustment), h=2.75)
#fig.set_size_inches(w=5.5, h=2.75)
# fig.suptitle("Recourse for College Admission")


# Generate low degree
alpha = lambda dist: utils.volcano_alpha(dist, cutoff=0.3, degree=2)
weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=0.7)
mrm = MRM(perturb_dir=None, alpha=alpha, weight_function=weight_function)
mrmc = MRMCIterator(n_clusters, mrm, preprocessor=preprocessor, max_iterations=3, validate=False)
mrmc.fit(data, cluster_assignments)

paths = mrmc.iterate(poi)

print("Figure One:")
print("-------------------")
paths_new = []
for path in paths:
    path = preprocessor.inverse_transform(path)
    paths_new.append(path.to_numpy())
    start = path.iloc[0,:]
    end = path.iloc[1,:]
    print("\tstarting point")
    print(start)
    print("\tending point")
    print(end)
    print("\tdifference")
    print(end - start)
    print("-------------------")

pos_color = 'lightskyblue'
neg_color = 'mediumseagreen'

display = Display2DPaths(X, Y)
display.use_small_legend().set_colors(pos_color, neg_color).set_clusters(km.cluster_centers_).set_paths(paths_new).scatter(ax1)
ax1.set_xlabel('Academics Score')
ax1.set_ylabel('Extracurriculars Score')
ax1.set_xlim((-0.05, 1.05))
ax1.set_ylim((-0.05, 1.05))


# Generate high degree
alpha = lambda dist: utils.volcano_alpha(dist, cutoff=0.3, degree=32)
weight_function = lambda dir, poi, X: utils.centroid_normalization(dir, poi, X, alpha=0.3)
mrm = MRM(perturb_dir=None, alpha=alpha, weight_function=weight_function)
mrmc = MRMCIterator(n_clusters, mrm, preprocessor=preprocessor, max_iterations=6, validate=False)
mrmc.fit(data, cluster_assignments)

paths = mrmc.iterate(poi)

print("\n")
print("Figure Two:")
print("-------------------")
paths_new = []
for path in paths:
    path = preprocessor.inverse_transform(path)
    paths_new.append(path.to_numpy())
    start = path.iloc[0,:]
    end = path.iloc[1,:]
    print("\tstarting point")
    print(start)
    print("\tending point")
    print(end)
    print("\tdifference")
    print(end - start)
    print("-------------------")

display = Display2DPaths(X, Y)
display.use_small_legend().set_colors(pos_color, neg_color).set_clusters(km.cluster_centers_).set_paths(paths_new).scatter(ax2)
ax2.set_xlabel('Academics Score')
#ax2.set_ylabel('Extracurriculars Score')
ax2.set_xlim((-0.05, 1.05))
ax2.set_ylim((-0.05, 1.05))
ax2.legend()

#ax1.label_outer()
#ax2.label_outer()
fig1.tight_layout()
fig2.tight_layout()

plt.show()
