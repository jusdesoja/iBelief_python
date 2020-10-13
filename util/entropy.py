#!/usr/bin/env python
# encoding: utf-8

"""
Calculate different entropy for mass functions.
The calculations based on the following paper
"A new definition of entropy of belief functions in the Dempster-Shafer theory." International Journal of Approximate Reasoning 92(2018):49-65. Jirousek R, Shenoy P P.
"""

import numpy as np
try:
    from .DST_fmt_functions import *
except Exception:
    from DST_fmt_functions import *
def entropy(mass, criterion):
    """
    Calculate entropy of the mass function.

    Parameters:
    mass: ndarray
        mass function to calculate entropy
    criterion: integer
        criterion index for different entropy
        1: Höhle's entropy
        2: Smets's entropy
        3: Yager entropy
        4: Nguyen entropy
        5: Dubois and Prad entropy
        6: Pal entropy
        7: Maeda and Ichihashi entropy
        8: Harmanec and Klir entropy
    """
    if criterion == 1:
        bel = mtobel(mass)
        h = np.sum(mass[1:] * np.log2(1/bel[1:]))
    if criterion == 2:
        if mass[-1] != 0:
            q = mtoq(mass)
            h = np.sum(np.log2(1/q[1:]))
        else:
            h = float("inf")
    if criterion == 3:
        pl = mtopl(mass)
        h = np.sum(mass[1:] * np.log2(1/pl[1:]))

    if criterion == 4:
        h = np.sum(mass[1:] * np.log2(1/mass[1:]))

    if criterion == 5:
        absInd = np.array([bin(i).count('1') for i in range(mass.size)])
        h = np.sum(mass[1:] * np.log2(absInd[1:]))
    if criterion == 6:
        absInd = np.array([bin(i).count('1') for i in range(mass.size)])
        h = np.sum(mass[1:] * np.log2(absInd[1:] / mass[1:]))

    return h
"""
m = np.array([0,0.3,0.4,0.3])
m_i = np.array([0,0,0,1])
print(entropy(m_i, 6))
"""
