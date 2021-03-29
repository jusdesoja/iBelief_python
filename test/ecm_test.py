#!/usr/bin/env python
# encoding: utf-8
"""
Test for ecm algorithm
Author: Yiru Zhang <zyrbruce@gmail.com>
"""

from sklearn import datasets

import unittest
import numpy as np
from .ecmeans.emc import ECM

class TestMethods(unittest.TestCase):
	def test_fcm():
		iris = datasets.load_iris()
		X = iris.data
		estimator = ECM(n_clusters=3)
		fcm.fit(X)
		self.assertTrue(ecm.m is not None)
