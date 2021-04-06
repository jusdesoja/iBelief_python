#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math
from qpsolvers import solve_qp

import pdb

def conjQP_multi(mass, dist):
    """
    This function computes a conjunctive combination of two input mass functions by minimizing a distance.
    Distances are computed between the vaccuous mass function and all functions belonging to those more comitted than both input mass functions.
    The minimization is guaranteed to converge to a unique solution

    Parameters:
    -----------
    mass: a M x N matrix containing M mass functions. Each line of the matrix is an input mass functions. N is the size of the power set.
    dist: a string indicating which distance to use for approximation. Possible choices are 'pl', 'b' and 'q'.
    Hereby we use k=2 since we will be using quadratic programming.

    Return:
    mout: the result of the combination of m1 with m2. mout < m1 and m2 for some partial order linked with the chosen distance.

    """
    mass = mass.astype(np.float)
    M, N = mass.shape
    n = round(math.log2(N))
    if (math.pow(2,n) != N):
        raise ValueError("the size of mass functions must be a power of 2, %d is given" % N)
    elif (np.sum(np.absolute(np.sum(mass, axis=1)-np.ones((1,M))))>1e-10):
        raise ValueError("matrix mass does not contain mass functions")
    else:
        #Building the incidence matrix M
        M = np.array([[1.0, 1.0],[0, 1.0]])
        if (n>=2):
            for i in range(1, n):
                M = np.kron(np.array([[1.0, 1.0], [0, 1.0]]), M)

        #initial mass function in the conditional subspace
        m0 = np.zeros(N).astype(np.float)
        m0[0] = 1.0
        m0 = m0[np.newaxis].T
        #Vaccuous mass function
        migno = np.zeros(N).astype(np.float)
        migno[-1] = 1.0
        migno = migno[np.newaxis].T
        #Bounds for mass functions
        mass_lb = np.zeros((N, 1)).astype(np.float)
        mass_ub = np.ones((N, 1)).astype(np.float)

        #Selecting a distance
        if (dist == 'pl'):
            #Compute the optimum in the plausibility space
            #The matrix 1-J*M' maps mass functions to plausibility functions.
            J = np.fliplr(np.identity(N))
            A = 1.0 - np.dot(J, M.T)
            # There is no reverse transfer matrix for plausibility
            # instead revA will compute mass function values from a plausibility for non empty subsets
            R = np.tile(migno, (1,N)).T
            pdb.set_trace()
            revA = np.dot(np.dot(np.linalg.inv(M.T), J), (R-np.identity(N)))
            #Boundary conditions for plausibility (open world assumption)
            lower_bound = np.zeros((1,N))
            upper_bound = np.nanmin(np.dot(A, mass.T), axis=1)[np.newaxis].T
            upper_bound[0] = 0
        elif (dist == 'b'):
            A = M.T
            revA = np.linalg.inv(M.T)
            lower_bound = np.nanmax(np.dot(A,mass.T), axis=1)[np.newaxis].T
            lower_bound[-1] = 1.0
            upper_bound = np.ones((1,N))
        elif(dist == 'q'):
            A = M
            revA = np.linalg.inv(M)
            lower_bound = np.zeros((1,N))
            lower_bound[0] = 1.0
            upper_bound = np.nanmin(np.dot(A, mass.T), axis = 1)[np.newaxis].T
        else:
            raise ValueError("undefined distance")
            
        #ub2 = upper_bound/A
        #lb2 = lower_bound/A
        #ub = min(mass_ub, ub2)
        #lb = max(mass_lb, lb2)
        
        mout = solve_qp(P = np.dot(A.T, A), q = -np.dot(np.dot(A.T,A), migno).T[0], G = A, h = upper_bound.T[0],  A = mass_ub.T[0], b = np.array([1.]), lb=mass_lb.T[0], ub= mass_ub.T[0])
        mout = np.multiply(mout, mout>1e-10) 
    return mout


