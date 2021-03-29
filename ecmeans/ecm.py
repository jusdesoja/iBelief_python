#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yiru Zhang <zyrbruce@gmail.com>


Reference:
M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm. 
Pattern Recognition, Vol. 41, Issue 4, pages 1384-1397, 2008.
"""

import numpy as np
from scipy.spatial import distance
from util import DST_fmt_functions as fmt


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
            distance to the empty set (default 10)

    epsilon: float, optional (default=1e-5)
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

    def __init__(self, n_clusters=10, max_iter=150, alpha = 1.0, beta = 2.0, delta=10, epsilon=1e-5, random_state=42, verbose=False):
        
        self.centers, self.S, self.Cardinal = None, None, None
        self.c = n_clusters
        self.c2 = 2 ** n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        #self.key = random.PRNGKey(random_state)
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        

    def fit(self, X):
        """
        Compute evidential c-means clustering.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
                Training instances to cluster.
        """
        self.n_samples, self.dim = X.shape
        self.centers = np.zeros((self.c, self.dim))
        n_iter = 0
        running = True
        J_old = 9999  # initialize J by a big number
        
        self._calculateCS(self)
        
        self.centers = X[np.random.choice(range(self.n_samples), self.c, replace=False)]
        
        while running:
            n_iter += 1
            self.D = self._calculateD(self,X)
            M = self._calculateM(self, X)
            self._calculateV(self, X, M)
            self._calculateD(self, X)
            # self.calculateB()
            # self.calculateV()
            J = np.sum((self.Cardinal ** self.alpha) * (M[:, 1:] ** self.beta) * self.D) + (
                self.delta ** 2) * np.sum(M[:, 1] ** self.beta)
            diff = np.abs(J_old - J)
            J_old = J
            if self.verbose:
                print("iteration: %d, difference: = %f", n_iter, diff)
            if diff < self.epsilon or n_iter > self.max_iter:
                running = False

    def __predict(self, X):
        """
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        M: array, shape = [n_samples, nbFoc-1]
            Evidential partition array, returned as an array with n_samples rows
            and nbFoc-1 columns with meta-clusters.
        """
        n_samples_temp, dim_temp = X.shape
        assert dim_temp == self.dim

        D_temp = ECM._dist(X, self.centers)
        M_temp = np.zeros((n_samples_temp, self.c2))
        CMulD = 1.0/(((self.Cardinal ** self.alpha) * D_temp) ** (self.beta-1))
        denominator = (np.sum(CMulD[:, 1:], axis=1) +
                       self.delta ** (-2/(self.beta-1)))[np.newaxis]
        M_temp[:, 1:] = CMulD/denominator
        M_temp[:, 0] = 1 - np.sum(M_temp[:, 1:], axis=1)
        return M_temp

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
           X : array, shape = [n_samples, n_features]
               New data to predict.
        Returns
        -------
           labels : array, shape = [n_samples,]
              Index of the cluster each sample belongs to.

        """

    @staticmethod
    # @jit
    def _dist(A, B):
        """ Compute the Euclidean distance between two matrices"""
        return np.sqrt(np.einsum('ijk->ij', (A[:, None, :] - B) ** 2))

    @staticmethod
    # @jit
    def _calculateC(self):
        """ 
        Calcultate the matrix of cardinal of focal elements 
        """

        self.Cardinal = np.sum(self.S, 1)
        return self.Cardinal

    @staticmethod
    # @jit
    def _calculateS(self):
        """
        Calculate the matrix of association s_kj
        Equation (17) 
        """
        #self.S = np.zeros((self.c, self.c2))
        self.S = (np.arange(self.c2)[:, None] & (
            1 << np.arange(self.c)) > 0)[1:, :].astype(int)
        return self.S
        
    @staticmethod
    # @jit
    def _calculateCS(self):
        self._calculateS(self)
        self._calculateC(self)
        
    @staticmethod
    # @jit
    def _calculateD(self, Xin):
        """
        Calculate distances from samples to each barycenter vbar
        Equation (19) in ref.
        """
        self._calculateVBar(self)
        #for j in range(self.c2):
            # for l in range(self.c2):
        #    self.VBars[j] = 1.0 / self.Cardinal[j] * np.product(self.S[j], self.centers[j])
            # if self.S[l][j] != 0:
            #   vbar=1/(double)self.Cardinal[j] * np.product()
        #f_one_sample_to_vbars = lambda x: np.apply_along_axis(np.linalg.norm(), 1, self.VBars)
        #D = np.zero((self.n_samples, self.c2))
        

        # A naive implementation by for-loop from another function.
        D = np.apply_along_axis(self._calculateSam2Vbars, 1, Xin, self.VBars)
        
        # A matrix operation version will be implemented later

        return D

    @staticmethod
    # @jit
    def _calculateSam2Vbars(sample, Vbars):
        """
        Calculate distance from one sample to each Vbars
        """
        # distance.euclidean takes time.
        #print(sample, Vbars)
        oneSamToVbars = np.apply_along_axis(
            distance.euclidean, 1, Vbars, sample)
        # To be replaced by np.eisum
        return oneSamToVbars

    @staticmethod
    # @jit
    def _calculateB(self, Xin, Mass):
        """
        Calculate matrix B (Equation (36) in ref.)
        """
        B = np.zeros((self.c, self.dim))
        for l in range(self.c):
            for q in range(self.dim):
                B[l][q] = np.dot(Xin[:, q], np.dot(
                    (Mass[:, 1:] ** self.beta) * (self.Cardinal ** (self.alpha-1)), self.S[:, l][np.newaxis].T))
        return B

    def _calculateH(self, Mass):
        """
        Calculate matrix H (Equation (37) in ref.)
        """
        H = np.zeros((self.c, self.c))
        for l in range(self.c):
            for k in range(self.c):
                H[l][k] = np.sum(np.dot(
                    (self.Cardinal ** (self.alpha - 2)) * Mass[:, 1:], (self.S[:, l] * self.S[:, k])[np.newaxis].T))
        return H

    @staticmethod
    # @jit
    def _calculateM(self, X):
        """
        Calculate matrix M 
        Equation (29) and (30) in ref.
        """
        self.D = ECM._calculateD(self, X)
        M = np.zeros((self.n_samples, self.c2))
        CMulD = 1.0/(((self.Cardinal ** self.alpha) * self.D) ** (self.beta-1))
        CMulD_copy = CMulD.copy()
        CMulD_copy[CMulD_copy == np.inf] = 1000 # replace inf values by a relative large number
        #print('shape:',CMulD.shape)
        #denominatorTail = delta ** (-2/(beta-1))
        #denominatorHead = np.sum(CMulD[:,1:], axis = 1)

        denominator = (np.sum(CMulD[:, 1:], axis=1) +
                       self.delta ** (-2/(self.beta-1)))[np.newaxis].T
        #print("denominator",denominator)
        M[:, 1:] = CMulD_copy / denominator
        M[:, 0] = 1 - np.sum(M[:, 1:], axis=1)
        return M

    @staticmethod
    # @jit
    def _next_centers(B, H):
        return np.dot(np.linalg.inv(H), B)

    @staticmethod
    # @jit
    def _calculateV(self, Xin, Mass):
        """
        Calculate V
        Equation (38) ref.
        """
        H = self._calculateH( Mass)
        B = self._calculateB(self, Xin, Mass)
        self.centers = np.dot(np.linalg.inv(H), B)
        return self.centers

    @staticmethod
    # @jit
    def _calculateVBar(self):
        """
        Calculate VBar 
        Equation (18) in ref.
        """
        self.VBars = np.zeros((self.c2 - 1, self.dim))
        
            #print('coeff', self.S[j,:].T)
            
        for j in range(self.c2 - 1):
            self.VBars[j] = np.sum(
                self.S[j][np.newaxis].T * self.centers, axis=0) / self.Cardinal[j]

    def _decisionFromM(self, M, func='bel'):
        """
        Decide the cluster assignement from mass functions

        Parameters
        ----------
        M: array, shape = [n_samples, nbFoc (self.c2)]
                mass function for all samples
        func: 'bel' or 'm' or 'pl'
                function to decide with, one of belief functions, mass functions or plausibility functions 
        Return
        ------
        labels: array, shape = [n_samples, ]
                labels for each sample
        """
        assert np.log2(M.shape[1]) % 1 == 0
        # labels = np.zeros(M.shape[0])
        if func == 'm':
            labels = np.argmax(M, axis=1).T
        if func == 'bel':
            labels = np.argmax(np.apply_along_axis(
                fmt.mtobel, M, axis=1), axis=1).T
        if func == 'pl':
            labels = np.argmax(np.apply_along_axis(
                fmt.mtopl, M, axis=1), axis=1).T
        return labels
