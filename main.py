from sklearn import datasets

#import unittest
import numpy as np
from ecmeans.ecm import ECM	
if __name__ == "__main__":
	
	iris = datasets.load_iris()
	X = iris.data
	estimator = ECM(n_clusters=3, verbose = True)
	estimator.fit(X)
