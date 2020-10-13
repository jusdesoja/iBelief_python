## Combination rules for masses
##' @param MassIn The matrix containing the masses. Each column represents a
##' piece of mass.
##' @param criterion The combination criterion:
##'
##' criterion=1 Smets criterion (conjunctive combination rule)
##'
##' criterion=2 Dempster-Shafer criterion (normalized)
##'
##' criterion=3 Yager criterion
##'
##' criterion=4 Disjunctive combination criterion
##'
##' criterion=5 Dubois criterion (normalized and disjunctive combination)
##'
##' criterion=6 Dubois and Prade criterion (mixt combination), only for Bayesian masses whose focal elements are singletons
##'
##' criterion=7 Florea criterion
##'
##' criterion=8 PCR6
##'
##' criterion=9 Cautious Denoeux Min for functions non-dogmatics
##'
##' criterion=10 Cautious Denoeux Max for separable masses
##'
##' criterion=11 Hard Denoeux for functions sub-normal
##'
##' criterion=12 Mean of the bbas
##'
##' criterion=13 LNS rule, for separable masses
##'
##' criterion=131 LNSa rule, for separable masses
##' @param TypeSSF If TypeSSF = 0, it is not a SSF, the general case. If TypeSSF = 1, a SSF with a singleton as a focal element. If TypeSSF = 2, a SSF with any subset of \eqn{\Theta} as a focal element.
##' @return The combined mass vector. One column.

import numpy as np
from .DST_fmt_functions import *

def DST(massIn, criterion, TypeSSF=0):
    """
    Combination rules for multiple masses.

    Parameters:
    ----------
    massIn: ndarray
        Masses to be combined, represented by a 2D matrix
    criterion: integer
        Combination rule to be applied
        The criterion values represented respectively the following rules:
            criterion=1 Smets criterion (conjunctive combination rule)
            criterion=2 Dempster-Shafer criterion (normalized)
            criterion=3 Yager criterion
            criterion=4 Disjunctive combination criterion
            criterion=5 Dubois criterion (normalized and disjunctive combination)
            criterion=6 Dubois and Prade criterion (mixt combination), only for Bayesian masses whose focal elements are singletons
            criterion=7 Florea criterion
            criterion=8 PCR6
            criterion=9 Cautious Denoeux Min for functions non-dogmatics
            criterion=10 Cautious Denoeux Max for separable masses
            criterion=11 Hard Denoeux for functions sub-normal
            criterion=12 Mean of the bbas
            criterion=13 LNS rule, for separable masses
            criterion=131 LNSa rule, for separable masses
    TypeSSF: integer
        If TypeSSF = 0, it is not a SSF (the general case).
        If TypeSSF = 1, it is a SSF with a singleton as a focal element.
        If TypeSSF = 2, it is a SSF with any subset of \Theta as a focal element.

    Return:
    ----------
    Mass: ndarray
        a final mass vector combining all masses
    """
    n, m = massIn.shape
    if criterion in (4,5,6,7):
        b_mat = np.apply_along_axis(mtob, axis = 0, arr = massIn)
        b = np.apply_along_axis(np.prod, axis = 1, arr = b_mat )
    if criterion in (1,2,3,6,7, 14):

        q_mat = np.apply_along_axis(mtoq, axis = 0, arr = massIn) #apply on column. (2 in R)
        q = np.apply_along_axis(np.prod, axis = 1,arr = q_mat) # apply on row (1 in R)
    if criterion == 1:
        #Smets criterion
        Mass = qtom(q)
        Mass[0] = 1.0 - np.sum(Mass[1:])
    elif criterion == 2:
        #Dempster-Shafer criterion (normalized)
        Mass = qtom(q)
        Mass = Mass/(1-Mass[0])
        Mass[0] = 0
    elif criterion == 3:
        #Yager criterion
        Mass = qtom(q)
        Mass[-1] = Mass[-1]+Mass[0]
        Mass[0]=0
    elif criterion == 4:
        #disjunctive combination
        Mass = btom(b)
    elif criterion == 5:
        # Dubois criterion (normalized and disjunctive combination)
        Mass = btom(b)
        Mass = Mass/(1-Mass[0])
        Mass[0] = 0
    #elif criterion == 6:
    #elif criterion == 7:
    #elif criterion == 8:
    elif criterion == 9:
        #Cautious Denoeux min for fonctions non-dogmatic
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.min , axis = 1, arr = wtot )
        Mass = wtom(w)
    elif criterion == 10:
        #Cautious Denoeux max only for separable fonctions
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.max, axis = 1, arr = wtot)
        Mass = wtom(w)
    #elif criterion == 11:
    elif criterion == 12:
        # mean of the masses
        Mass = np.apply_along_axis(np.mean, axis = 1, arr = massIn)
    elif criterion == 13:
        if TypeSSF==0:
            Mass = LNS(massIn, mygamma = 1)
        elif TypeSSF == 1:
            Mass = LNS_SSF(massIn, mygamma =1, singleton =True)
        elif TypeSSF == 2:
            Mass = LNS_SSF(massIn, mygamma = 1, singleton= False)
    elif criterion == 14:
        Mass = qtom(np.apply_along_axis(np.mean, axis=1, arr=q_mat))
    return Mass[np.newaxis].transpose()

def LNS(massIn, mygamma,ifnormalize = False, ifdiscount = True, approximate=False, eta = 0):
    nf,n = massIn.shape
    ThetaSize = np.log2(nf)
    w_mat = np.apply_along_axis(mtow,axis = 0,arr = massIn)
    if approximate:
        # This case has not been tested
        num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6) ,1,arr = w_mat)
        id_eff = np.where(num_eff > 0)
        num_group_eff = num_eff[id_eff]
        beta_vec = np.ones(len(id_eff))
        if (eta != 0):
            myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
            beta_vec = (ThetaSize / myc[id_eff]) **eta
        alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
        w_eff = 1-alpha_vec
        w_vec = np.ones(nf)
        w_vec[id_eff] = w_eff
    else:
        if mygamma == 1:
            w_vec = np.apply_along_axis(np.prod,axis=1, arr = w_mat)
        elif mygamma == 0:
            w_vec = np.apply_along_axis(np.ndarray.min, axis = 1, arr = w_mat)
        #else:
        if ifdiscount:
            num_eff= np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6), axis=1, arr=w_mat)
            id_eff = np.where(num_eff>0)
            w_eff = w_vec[id_eff]
            num_group_eff = num_eff[id_eff]
            beta_vec = np.ones(len(id_eff))
            if eta != 0:
                myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
                beta_vec = (ThetaSize / myc[id_eff]) ** eta
            alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
            w_eff = 1 - alpha_vec + alpha_vec * w_eff
            w_vec[id_eff] = w_eff
    out = wtom(w_vec)
    if ifnormalize and mygamma:
        out[0] = 0
        out = out/np.sum(out)
    return out

def LNS_SSF(massIn, mygamma, ifnormalize = False, ifdiscount =True, approximate=False, eta = 0, singleton = False):
    m,n = massIn.shape
    if singleton:
        ThetaSize = m
        nf = 2 ** m
        w_mat = massIn[0:-1,::]
        w_mat = 1 - w_mat
        eta = 0
    else:
        nf = m
        ThetaSize = np.log2(nf)
        w_mat = massIn[0: -1, ::]
        w_mat = 1-w_mat
    if approximate:
        num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x - 1) > 1e-6), axis = 1, arr = w_mat)
        id_eff = np.argwhere(num_eff > 0)
        num_group_eff = num_eff[id_eff]
        if(eta != 0):
            beta_vec = np.ones(len(id_eff))
            myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
            beta_vec = (ThetaSize/myc[id_eff]) ** eta
            alpha_vec = beta_vec * num_group_eff /np.sum(beta_vec * num_group_eff)
        else:
            alpha_vec = num_group_eff / sum(num_group_eff)
        w_eff = 1 - alpha_vec
        if(singleton):
            w_vec = np.ones(ThetaSize)
        else:
            w_vec = np.ones(nf-1)
        w_vec[id_eff] = w_eff
    else:
        if mygamma == 1:
            w_vec = np.apply_along_axis(np.prod, 1, arr = w_mat)
        elif mygamma == 0:
            w_vec = np.apply_along_axis(np.ndarray.min, 1,arr = w_mat)
        #else: TODO to complete
        if ifdiscount:
            num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6), axis = 1, arr =w_mat)
            id_eff = np.argwhere(num_eff > 0)
            w_eff = w_vec[id_eff]
            num_group_eff = num_eff[id_eff]
            if eta != 0:
                beta_vec = np.ones(len(id_eff))
                myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
                beta_vec = (ThetaSize/myc[id_eff]) ** eta
                alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
            else:
                alpha_vec = num_group_eff / np.sum(num_group_eff)
            w_eff = 1 - alpha_vec + alpha_vec * w_eff
            w_vec[id_eff] = w_eff
    w_vec_complete = np.ones(nf)
    if singleton:
        w_vec_complete[2*np.arange(ThetaSize)] = w_vec
    else:
        w_vec_complete[0:-1] = w_vec
    if np.ndarray.min(w_vec_complete)>0:
        out = wtom(w_vec_complete)
    else:
        id = np.argwhere(w_vec_complete==0)
        out = np.zeros(nf)
        out[id] = 1
    if(ifnormalize & mygamma == 1):
        out[0] = 0
        out = out/np.sum(out)
    return out.T



"""
m = np.array([[0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])
LNS(m,1)
"""
