


from exceptions import IllegalMassSizeError
import numpy as np
import math

from DST_fmt_functions import *

def decisionDST(mass, criterion, r=0.5):
    """Different rules for decision making in the framework of belief functions
    
    Parameters
    -----------
    mass: 
    
    criterion: integer
        different decision rules to apply.
            criterion=1 maximum of the plausibility
            criterion=2 maximum of the credibility
            criterion=3 maximum of the credibility with rejection
	        criterion=4 maximum of the pignistic probability
	        criterion=5 Appriou criterion (decision onto \eqn{2^\Theta})
    """
    mass = mass.copy()
    if (mass.size in mass.shape):   #if mass is a 1*N or N*1 array
        mass = mass.reshape(mass.size,1)
    nbEF, nbvec_test = mass.shape   #number of focal elements = number of mass rows 
    nbClasses = round(math.log(nbEF,2))
    class_fusion = []
    for k in range(nbvec_test):
        massTemp = mass[:,k]
        if criterion == 1:
            #TODO
            pass
        elif criterion == 2:
            #TODO
            pass
        elif criterion == 3:
            #TODO
            pass
        elif criterion == 4:
            pign = mtobetp(massTemp.T)
            indice = np.argmax(pign)
            class_fusion.append(indice)
    
    return np.array(class_fusion)
