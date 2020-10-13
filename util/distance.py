#!/usr/bin/env python
# encoding: utf-8

"""Calculate distance between two BBA on the same frame of discernment.
The distance is represented by Jousselme Distance
"""

import numpy as np
import math
#import pdb
from .Dcalculus import Dcalculus
#from exceptions import IllegalMassSizeError
def JousselmeDistance(mass1,mass2, D = "None"):
    m1 = np.array(mass1).reshape((1,mass1.size))
    m2 = np.array(mass2)
    if m1.size != m2.size:
        raise ValueError("mass vector should have the same size, given %d and %d" % (m1.size, m2.size))
    else:
        if type(D)== str:
            D = Dcalculus(m1.size)
        m_diff = m1 - m2
        #print(D)
        #print(m_diff.shape,D.shape)
        #print(np.dot(m_diff,D))


        #----JousselmeDistance modified for testing, don't forget to correct back------#

        out = math.sqrt(np.dot(np.dot(m_diff,D),m_diff.T)/2.0)
        #out = np.dot(np.dot(m_diff,D),m_diff.T)
        return out

def _calculateDistanceMat(singleton_dist_mat):
    """
    Calculate weighted distance on multiple singletons of preference.
    The simmilarity between two singletons is considered as a common part even though they are exclusive in the definition level.
    To simplify the calculation the dissimilarity between unions, we assume that the common part exist only between pairs. (i.e. no common part is shared among three or more elements)


    """
    n_singleton = singleton_dist_mat.shape[0]
    #singleton_sim_mat = (np.ones(n_singleton) - np.eye(n_singleton)) - singleton_dist_mat # To avoid erreur in singleton self similarity, one singleton's self similarity is 0
    #singleton_sim_mat = singleton_sim_mat
    #print(singleton_sim_mat)
    singleton_sim_mat = (1-np.eye(n_singleton)) - singleton_dist_mat # To avoid erreur in singleton self similarity, on singleton's self similarity is 0
    singleton_sim_mat = 2 * singleton_sim_mat/(1+singleton_sim_mat)
    print(singleton_sim_mat)
    n_element = 2 ** n_singleton
    dist_mat = np.zeros((n_element,n_element))
    for i in range(1,n_element + 1):
        for j in range(i+1,n_element):
            #print(i,j,bin(i),bin(j))
            A_vec = np.array([int(d) for d in bin(i)[2:][::-1]])
            B_vec = np.array([int(d) for d in bin(j)[2:][::-1]])
            A_vec = np.pad(A_vec,(0,n_singleton-A_vec.size),'constant',constant_values=(0))
            #pdb.set_trace()
            B_vec = np.pad(B_vec,(0,n_singleton-B_vec.size),'constant',constant_values=(0))
            #print(A_vec,B_vec)
            common_vec = np.logical_and(A_vec,B_vec)
            diff_vec = np.logical_xor(A_vec,B_vec)
            all_vec = np.logical_or(A_vec,B_vec)
            #common_vec = np.zeros((n_singleton))
            #common_element = np.array([int(d) for d in bin(i & j)[2:][::-1]]) # binary and of i and j
            #common_vec[:common_element.size] = common_element
            #common_vec = common_vec.reshape((1,common_vec.size))
            #print(common_vec)
            #diff_vec = np.zeros((n_singleton))
            #diff_element = np.array([int(d) for d in bin(i ^ j)[2:][::-1]]) # binary xor of i and j
            #print(i,j,diff_element)
            #diff_vec[:diff_element.size] = diff_element
            #diff_vec = diff_vec.reshape((1,diff_vec.size))
            #print(diff_vec)
            #all_vec = np.zeros((n_singleton))
            #all_element = np.array([int(d) for d in bin(i | j)[2:][::-1]]) # binary or of i and j
            #all_vec[:all_element.size] = all_element
            #all_vec = all_vec.reshape((1,all_vec.size))
            #print(all_vec)
            #pdb.set_trace()
            p_common = np.count_nonzero(common_vec==1) \
                - np.dot(common_vec, np.dot(singleton_sim_mat, common_vec.T)) / 2 \
                + np.dot(diff_vec, np.dot(singleton_sim_mat, diff_vec.T)) / 2 \
                - np.dot(np.logical_and(A_vec,diff_vec),np.dot(singleton_sim_mat,np.logical_and(A_vec,diff_vec)))/2\
                - np.dot(np.logical_and(B_vec,diff_vec),np.dot(singleton_sim_mat,np.logical_and(B_vec,diff_vec)))/2
            p_all =  np.count_nonzero(all_vec == 1)\
                - np.dot(all_vec, np.dot(singleton_sim_mat,all_vec.T)) / 2
            #if (i==1 and j==6):
                #pdb.set_trace()
            #npmath.sqrt(np.dot(np.dot(m_diff,D),m_diff.T)/2.0)
            #print("common,all",p_common,p_all)
            dist_mat[i][j] = p_common / p_all
    dist_mat = 1 - (dist_mat + dist_mat.T + np.eye(n_element))
    return dist_mat

def weighted3SingletonDistance(mass1, mass2, singleton_dist_mat):
    """
    Weighted distance on multiple singletons of preferences.

    Parameter
    ---------
    mass1: mass function of size 8
    mass2: mass function of size 8
    singleton_dist_mat: matrix of distances between different singletons

    Return
    ---------
    dist: distance of mass1 and mass2

    """
    n_singleton = round(math.log2(mass1.size))
    if singleton_dist_mat.shape[0] != n_singleton:
        raise ValueError("mass and singleton distance matrix are not compatible!")
    else:
        m1 = np.array(mass1).reshape((1,mass1.size))
        m2 = np.array(mass2).reshape((1,mass2.size))
        dist_mat = _calculateDistanceMat(singleton_dist_mat)
        m_diff = m1 - m2
        return math.sqrt(np.dot(np.dot(m_diff,dist_mat),m_diff.T)/2.0)
#m1 = np.array([0., 0.31,0.59,0., 0.1, 0., 0. ,0.])

#m2 = np.array([0., 0.5,0.3,0., 0.2, 0., 0. ,0.])
#m2 = np.array([0., 0.28, 0.22, 0., 0.5, 0., 0., 0.])
#m3 = np.array([0.,1.,0.,0.,0.,0.,0.,0.])
#D = Dcalculus(8)
#print(JousselmeDistance(m1, m3,D), JousselmeDistance(m2,m3,D))
#p = 2/3
#p=1
#print(_calculateDistanceMat(np.array([[0, 1, p,], [1,0,p],[p,p,0]]))[1:,1:])
