###------DST fmt functions--------------#####

### from R codes by Kuang Zhou refering to matlab codes by Philippe Smets. FMT = Fast Mobius Transform
### depend: numpy ####
#Author: Yiru Zhang <yiru.zhang@irisa.fr>


#TODO: exit function should be replaced by exceptions.


import numpy as np
import math
from sys import exit

def mtobetp(InputVec):
    """Computing BetP on the signal points from the m vector (InputVec) out = BetP
    vector beware: not optimize, so can be slow for >10 atoms
    """
    # the length of the power set, f
    mf = InputVec.size
    # the number of the signal point clusters
    natoms = round(math.log(mf,2))
    if math.pow(2,natoms) == mf:
        if InputVec[0] == 1:
            #bba of the empty set is 1
            exit("warning: all bba is given to the empty set, check the frame\n")
            out = np.ones(natoms)/natoms
        else:
            betp = np.zeros(natoms)
            for i in range(1, mf):
                # x , the focal sets InputVec the dec2bin form
                x = np.array(list(map(int,np.binary_repr(i, natoms)))[::-1]) # reverse the binary expression
                # m_i is assigned to all the signal points equally

                betp = betp + np.multiply(InputVec[i]/sum(x), x)
            out = np.divide(betp,(1.0 - InputVec[0]))
        return out
    else:
        raise ValueError("Error: the length of the InputVec vector should be power set of 2, given %d \n" % mf)


def mtoq(InputVec):
    """
    Computing FMT from m to q
    Argument:
    InputVec : vector m

    Return:
    out: vector q
    """
    InputVec = InputVec.copy()
    mf = InputVec.size
    natoms =round(math.log2(mf))
    if 2 ** natoms == mf:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842,order='F')
            #for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):    #not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2 ] = InputVec[:, j * 2 ] + InputVec[:, j * 2+1]
        out = InputVec.reshape(1,mf,order='F')[0]
        return out
    else:
        raise ValueError("ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)






def mtob(InputVec):
    """
    Comput InputVec from m to b function.  belief function + m(emptset)
    InputVec = m
    vector out = b vector
    """
    InputVec = InputVec.copy()
    mf = InputVec.size
    natoms =round(math.log(mf,2))
    if math.pow(2, natoms) == mf:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842,order='F')
            #for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):    #not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2 + 1 ] = InputVec[:, j * 2 + 1 ] + InputVec[:, j * 2]
        out = InputVec.reshape(1,mf,order='F')[0]
        return out
    else:
        raise ValueError("ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)

#def btopl(InputVec):
#    """Compute pl from b InputVec
#    InputVec : vector f*1
#    out = pl

#    """
#    mf = InputVec.size
#    natoms = round(math.log(mf,2))
#    if math.pow(2, natoms) == mf:
#        InputVec = InputVec[-1] -

def mtonm(InputVec):
    """
    Transform bbm into normalized bbm
    """
    if InputVec[0] < 1:
        out = InputVec/(1-InputVec[0])
        out[0] = 0
    return out

def mtobel(InputVec):
    return mtob(mtonm(InputVec))

def qtom(InputVec):
    """
    Compute FMT from q to m.
    input: vector q
    output: vector m
    """
    InputVec = InputVec.copy()
    lm = InputVec.size
    natoms =round(math.log(lm,2))
    if math.pow(2, natoms) == lm:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842, order='F')
            #for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):    #not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2 ] = InputVec[:, j * 2 ] - InputVec[:, j * 2+1]
        out = InputVec.reshape(1,lm,order='F')[0]
        return out
    else:
        exit("ACCIDENT in qtom: length of input vector not OK: should be a power of 2\n")


def btom(InputVec):
    mass_t = InputVec.copy()
    mf = mass_t.size
    natoms = round(math.log2(mf))
    if 2 ** natoms == mf:
        for i in range(natoms):
            i124 = int(2 ** i)
            i842 = int(2 ** (natoms - i))
            i421 = int(2 ** (natoms -i - 1))
            mass_t = mass_t.reshape(i124, i842, order = 'F')
            #for j in range(i421):    #not a good way for add operation coz loop matrix for i842 times
            #    InputVec[:, j * 2 ] = InputVec[:, j * 2 - 1] - InputVec[:, j * 2 - 2]
            #print(i421)
            mass_t[:, np.array(range(1, i421 + 1)) * 2 - 1] = mass_t[:, np.array(range(1, i421 + 1)) * 2 - 1] - mass_t[:, np.array(range(i421)) * 2]
        out  = mass_t.reshape(1, mf, order = 'F')
        return out
    else:
        raise ValueError("ACCIDENT in btom: length of input vector not OK: should be a power of 2, given %d\n" % mf)

def pltob(InputVec):
    mf = InputVec.size
    natoms = round(math.log2(mf))
    if 2 ** natoms == mf:
        InputVec = 1 - InputVec[::-1]
        out = InputVec
        return out
    else:
        raise ValueError("ACCIDENT in pltob: length of input vector not OK: should be a power of 2, given %d\n" % mf)
def mtopl(InputVec):
    InputVec = mtob(InputVec)
    out = btopl(InputVec)
    return out

def pltom(InputVec):
    out = btom(pltob(InputVec))
    return out
def qtow(InputVec):
    """
    Compute FMT from q to w, Use algorithm qtom on log q
    input: vector q
    output: vector w
    """
    InputVec = InputVec.astype(float)
    lm = InputVec.size
    natoms = round(math.log(lm,2))
    if math.pow(2, natoms) == lm:
        if InputVec[-1] >0: #non dogmatic
            out = np.exp(-qtom(np.log(InputVec)))
            out[-1] = 1
        else:
            #print("""Accident in qtow: algorithm works only if q(lm) > 0\n
            #add an epsilon to m(lm)\n
            #Nogarantee it is really OK\n
            #""")
            """
            mini = 1
            for i in range(lm):
                if (InputVec[i] >0):
                    mini = min(mini, InputVec[i])
            mini = mini / 10000000.0
            for i in range(lm):
                InputVec[i] = max(InputVec[i],mini)
            """
            for i in range(lm):
                if InputVec[i] == 0:
                    InputVec[i] = 1e-9
            out = np.exp(-qtom(np.log(InputVec)))
            out[-1] = 1
    else:
        raise ValueError("ACCIDENT in qtom: length of input vector not OK: should be a power of 2, given %d\n" % lm)
    return out

def btopl(InputVec):
    #compute pl from b Inputvec
    #InputVec vector f*1
    #return pl
    lm = InputVec.size
    natoms = round(math.log2(lm))
    if 2 ** natoms == lm:
        InputVec = InputVec[-1] - InputVec[::-1]
        out = InputVec
        return out
    else:
        raise ValueError("ACCIDENT in btopl: length of input vector not OK: should be a power of 2, given %d\n" % lm)
############################################
#functions below are not tested
############################################
def wtoq(InputVec):
    lm = InputVec.size
    natoms = round(math.log2(lm))
    if 2 ** natoms == lm:
        if np.ndarray.min(InputVec) > 0:
            out = np.prod(InputVec)/np.exp(mtoq(np.log(InputVec)))
            return out
        else:
            raise ValueError('ACCIDENT in wtoq: one of the weights are non positive\n')
    else:
        raise ValueError('Accident in wtoq: length of input vector illegal: should be a power of 2')

def mtow(InputVec):
    """Compute FMT from m to w.
    input: vector m
    output: vector w
    """
    out = qtow(mtoq(InputVec))
    return out

def wtom(InputVec):
    out = qtom(wtoq(InputVec))
    return out

############test##########################

#m = np.array([0,0.2,0,0.8])
#print(mtobetp(m))

"""
m = np.array([0, 0.1, 0.3, 0, 0.3, 0, 0, 0.3])
pl = np.array([0, 0.4, 0.6, 0.7, 0.6, 0.7, 0.9, 1])
b = np.array([0, 0.1, 0.3, 0.4, 0.3, 0.4, 0.6, 1])
q = np.array([1,0.4,0.6,0.3,0.6,0.3,0.3,0.3])
print("%s"%b)
#print("mtopl: ", mtopl(m))
#print("pltob: ", pltob(pl))
mfromb = btom(b)
print("btom: b:", b,"m:",mfromb)
#print("pltom: ", pltom(pl))
print("btopl: b:%s, pl:%s" % (b, btopl(b)))
print("qtow:", qtow(q))
"""
