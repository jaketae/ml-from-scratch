import numpy as np
from mlfromscratch.utils.ops import euclidean_distance

class KMeans:

	def __init__(self, k, n_iters):
		self.k = k
		self.n_iters = n_iters

	def predict(self, X):
		self.X = np.asarray(X)
		self._init_centroids(self.X)

		for _ in range(self.n_iters):
			centroids_old = self.centroids
			self.clusters = self._create_clusters(centroids_old)
			self.centroids = self._update_centroids(self.clusters, self.X)
			diff = centroids_old - self.centroids
			if not diff.any():
				break

		return self._label_clusters(self.clusters, self.X)

	def _init_centroids(self, X):
		self.n_samples, self.n_features = X.shape
		idx = np.random.choice(self.n_samples, self.k, replace=False)
		self.centroids = X[idx]

	def _closest_centroid(self, x, centroids):
		distances = [euclidean_distance(x, centroid) for centroid in centroids]
		return np.argmin(distances)

	def _create_clusters(self, centroids):
		clusters = [[] for _ in range(self.k)]
		for idx, x in enumerate(self.X):
			cluster = self._closest_centroid(x, centroids)
			clusters[cluster].append(idx)
		return clusters

	def _update_centroids(self, clusters, X):
		centroids = np.zeros((self.k, self.n_features))
		for i, cluster in enumerate(clusters):
			centroid = np.mean(X[cluster], axis=0)
			centroids[i] = centroid
		return centroids

	def _label_clusters(self, clusters, X):
		labels = np.empty(self.n_samples)
		for i, cluster in enumerate(clusters):
			labels[cluster] = i
		return labels
