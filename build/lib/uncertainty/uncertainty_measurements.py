import os
import logging
import numpy as np
import random
random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(42)

DO_OVERALL = True


def centroid(p):
    """
    @param p :  number of pixels x N probability for each class
    @return : the centroid of the N-dimentional triangle defined by one-hot encoding points.
    """
    N = p.shape[-1]
    return 1-np.sqrt(np.sum((p-1/N)**2, axis = 1))/ np.sqrt((1-1/N))


def variance(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : np.array(N) the variance of the N-dimentional categorical distribution
    """
    N = p.shape[-1]
    x = np.arange(N)+1
    var = np.sum(x**2*p, axis =1) - np.sum(x*p, axis = 1)**2
    mean = np.sum(x*p, axis =1)

    return var/mean

def variance_heterophil(p,w):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @param w : np.array(C,C) distances of classes / heterophily matrix
    @return : np.array(N) the modified variance of the C-dimentional categorical distribution
    """
    d = w[p.argmax(1)]
    return (np.sum(d**2*p, axis =1) - np.sum(d*p, axis = 1)**2)



def entropy(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : np.array(N) the entropy of the N-dimentional categorical distribution
    """
    p=p+1e-7
    N = p.shape[-1]

    return -np.sum(p*np.log(p), axis =1)/np.log(N)




