#!/usr/bin/env python
# encoding: utf-8

# Author: Yiru Zhang <zyrbruce@gmail.com>

# Reference:
# M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm. 
# Pattern Recognition, Vol. 41, Issue 4, pages 1384 1397, 2008.


import numpy as np

class ECM: 
	"""Evidential c-means
	
	Parameters
	----------
	n_clusters: int, optional (default=10)
        The number of clusters to form as well as the number of
        centroids to generate
	max_iter: int, optional (default=150)
	        Hard limit on iterations within solver.
    
	alpha: float
		exponent for the cardinality (default 1) 
	beta: float
		exponent for m (defaut 2) 
	delta: float
		delta : distance to the empty set (default 10)
	
    error: float, optional (default=1e-5)
        Tolerance for stopping criterion.
    random_state: int, optional (default=42)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
	
    Attributes
    ----------
    n_samples: int
        Number of examples in the data set
    n_features: int
        Number of features in samples of the data set
    
	
    centers: array, shape = [n_class-1, n_SV]
        Final cluster centers, returned as an array with n_clusters rows
        containing the coordinates of each cluster center. The number of
        columns in centers is equal to the dimensionality of the data being
        clustered.
    
	m: credal partition = maxtrix of size nxf; the corresponding focal elements are given in F; 
	g: matrix of size Kxd of the centers of the clusters
	F: matrix (f,K) of focal elements
		F(i,j)=1 if omega_j belongs to focal element i
			0 otherwise
	pl: plausibilities of the clusters
	BetP: pignistic probabilities of the clusters
	J: objective function
	N: non specifity index (to be minimized)
	
    Methods
    -------
    fit(X)
        fit the data
    _predict(X)
        use fitted model and output cluster memberships
    predict(X)
        use fitted model and output 1 cluster for each sample
	"""
	def __init__(self, n_clusters=10, max_iter=150, error=1e-5, random_state=42):
	        # assert m > 1
	        self.m, self.centers = None, None
	        self.K = n_clusters
	        self.max_iter = max_iter
	        self.error = error
	        self.key = random.PRNGKey(random_state)
			
	def _dist(A, B):
		""" Compute the Euclidean distance between two matrices"""
		return np.sqrt(np.einsum('ijk->ij', (A[:,None,:] - B) ** 2))
	
	def fit(self, X):
		"""
		Compute evidential c-means clustering.
		
		Parameters
		----------
		X: array-like, shape = [n_samples, n_features]
			Training instances to cluster.
		"""
		self.n_samples = X.shape[0]
		
		self.m = np.
		