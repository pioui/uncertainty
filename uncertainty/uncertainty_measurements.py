import logging
import numpy as np

from uncertainty.maximum_hu_utils import get_max_hu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DO_OVERALL = True

def GU(p, d = "euclidean", n = 1):
    """
    Geometry-based classification uncertainty measure

    @param p : np.array(N,C) N data points x C probabilities each for each class
    @param d : distance considered on the standard (C-1)-simplex, 
        The distance function can be "euclidean", "kullbackleibler", "fisherrao" default "euclidean"
    @param n : positive integer, default 1
    @return : uncertainty claculated with respect of the distance from center of the standard (C-1)-simplex
    """

    if not isinstance(n, int) or n<=0:
        raise ValueError('n should be a nonegative integer')
    elif not isinstance(d, str):
        raise ValueError("d must be a string identifier")

    p = np.asarray(p)

    if p.ndim==1:
        p = p[None,:]
    elif p.ndim>2:
        raise ValueError(f"p must have two dimensions but {p.ndim} dimensions were found")


    # The probabilities should sum up to 1
    p = p/np.sum(p, axis = 1, keepdims = True)

    # Get number of classes
    C = p.shape[-1]
    
    if d == "euclidean":
        return 1 - np.sqrt(np.sum((p - 1 / C) ** 2, axis=1) / (1 - 1 / C))**n
    elif d == "fisherrao":
        return 1 - np.arccos(np.sum(np.sqrt(p/ C), axis=1))** n / np.arccos(np.sqrt(1/ C))** n
    elif d == "kullbackleibler":
        p = p + 1e-10
        return 1 - (np.sum(p * np.log(C*p), axis=1) / np.log(C))**n
    else: raise ValueError(f'Unknown Distance Metric: {d}')


def HU(p, H):
    """
    Homophily-based classification uncertainty measure

    @param p : np.array(N,C) N pixels x C probabilities each for each class
    @param H : np.array(C,C) classes' pairwise distance matrix
    @return : np.array(N) the modified variance of the C-dimentional categorical distribution
    """

    p = np.asarray(p)

    if p.ndim==1:
        p = p[None,:]
    elif p.ndim>2:
        raise ValueError(f"p must have two dimensions but {p.ndim} dimensions were found")
        
    if H.ndim!=2:
        raise ValueError(f"The matrix H must have two dimensions but {p.ndim} dimensions were found")
    
    if np.any(H < 0):
        raise ValueError(f"The elements of the matrix H should be positive")
    
    print(p.shape)

    # The probabilities should sum up to 1
    p = p/np.sum(p, axis = 1, keepdims = True)

    # Scale the values of H and get the squared values
    H = H/np.amax(H)
    H2 = np.power(H, 2)

    # get the maximum value of HU
    maxHU = get_max_hu(H)
    HU = np.zeros(len(p))
    # Get number of data points
    N = len(p)

    step = 1000

    if N<=step:
        HU = np.diag(np.matmul(np.matmul(p, H2),np.transpose(p)))
    else: 
        for i in range(0,len(p),step):
            if i+step>len(p):
                HU[i:] = np.diag(np.matmul(np.matmul(p[i:,:], H2),np.transpose(p[i:,:])))
            else:
                HU[i:i+step-1] = np.diag(np.matmul(np.matmul(p[i:i+step-1,:], H2),np.transpose(p[i:i+step-1,:])))
        
    return HU/(2*maxHU)

def variance(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : np.array(N) the variance of the N-dimentional categorical distribution
    """
    
    # The probabilities should sum up to 1
    p = p/np.sum(p, axis = 1, keepdims = True)

    p_max = np.amax(p, axis=1)
    return p_max*(1-p_max)*4