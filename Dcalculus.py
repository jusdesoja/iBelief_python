#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math

def Dcalculus(lm):
    """Compute the table of conflict """
    natoms = round(math.log2(lm))
    ind = [{}]*lm
    if (math.pow(2, natoms) == lm):
        ind[0] = {0} #In fact, the first element should be a None value (for empty set).
        #But in the following calculate, we'll deal with 0/0 which shoud be 1 bet in fact not calculable. So we "cheat" here to make empty = {0}
        ind[1] = {1}
        step = 2
        while (step < lm):
            ind[step] = {step}
            step = step+1
            indatom = step
            for step2 in range(1,indatom - 1):
                #print(type(ind[step2]))
                ind[step] = (ind[step2] | ind[indatom-1])
                #ind[step].sort()
                step = step+1
    out = np.zeros((lm,lm))

    for i in range(lm):
        for j in range(lm):
            out[i][j] = float(len(ind[i] & ind[j]))/float(len(ind[i] | ind[j]))
    return out
#print(Dcalculus(16))
