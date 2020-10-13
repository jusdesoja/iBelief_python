#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math
from .distance import JousselmeDistance

def inclusion(F1, F2, sizeDS):
    return np.sum(np.equal(np.array(list(np.binary_repr(np.bitwise_and(F1,F2),sizeDS))).astype(int), np.array(list(np.binary_repr(F1,sizeDS))).astype(int)))==sizeDS



def d_incS(m1,m2):
    if m1.size == m2.size:
        sizeDS = int(math.log(m1.size, 2))
        focElem1 = np.where(m1 != 0)[0].astype(int)
        nbFoc1 = focElem1.size
        focElem2 = np.where(m2 != 0)[0].astype(int)
        nbFoc2 = focElem2.size
        d12 = 0
        for F1 in focElem1:
            for F2 in focElem2:
                if inclusion(F1, F2, sizeDS):
                    d12 += 1
        return d12 / (nbFoc1*nbFoc2)
    else:
        raise ValueError("m1 and m2 have different sizes, m1 has size %d and m2 has size %d" % (m1.size, m2.size))

def d_incL(m1,m2):
    if m1.size == m2.size:
        sizeDS = int(math.log(m1.size, 2))
        focElem1 = np.where(m1 != 0)[0].astype(int)
        nbFoc1 = focElem1.size
        focElem2 = np.where(m2 != 0)[0].astype(int)
        d12 = 0
        for F1 in focElem1:
            F1inF = False
            for F2 in focElem2:
                #print("type%s, %s, %s" % (type(F1), type(F2), type(sizeDS)))
                if inclusion(F1, F2, sizeDS):
                    F1inF = True
            if F1inF:
                d12 += 1
        return d12 / (nbFoc1)
    else:
        raise ValueError("m1 and m2 have different sizes, m1 has size %d and m2 has size %d" % (m1.size, m2.size))

def inclusionDegree(m1,m2,type = 'S'):
    if m1.size == m2.size:
        if type == 'S':
            return max(d_incS(m1,m2), d_incS(m2,m1))
        else:
            return max(d_incL(m1,m2), d_incL(m2,m1))
    else:
        raise ValueError("m1 and m2 have different sizes, m1 has size %d and m2 has size %d" % (m1.size, m2.size))

def conflict(m1,m2, D = "N"):
    if m1.size == m2.size:
        #import pdb; pdb.set_trace()
        if not type(D) == str:
            return (1-inclusionDegree(m1,m2)) * JousselmeDistance(m1, m2,D)
        else:
            return (1-inclusionDegree(m1,m2)) * JousselmeDistance(m1, m2)
    else:
        raise ValueError("m1 and m2 have different sizes, m1 has size %d and m2 has size %d" % (m1.size, m2.size))


########################################################################
from iBelief.DST_fmt_functions import *
import scipy.spatial.distance as spdist
def func_distance(m1,m2, f):
    if f=='q':
        return spdist.euclidean(mtoq(m1),mtoq(m2))
    elif f == 'pl':
        return spdist.euclidean(mtopl(m1),mtopl(m2))
    elif f == 'b':
        return spdist.euclidean(mtob(m1),mtob(m2))
    else:
        raise ValueError("f should be a function in q, pl or b, %s is given" % f)


"""
#print(inclusion(10,2,4))
#print(inclusion(2,3,4))

m1 = np.array([0,0.5, 0, 0.5])
m2 = np.array([0, 0.4, 0.5, 0.1])
print(d_incL(m1,m2))
print(conflict(m1,m2))
"""
