import os
import logging
import numpy as np
import random

from uncertainty.maximum_variance_utils import get_var_max_from_matrix

random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(42)

DO_OVERALL = True


def geometry_based_uncertainty(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : uncertainty claculated with respect of the distance from center of the standard (C-1)-simplex
    """
    C = p.shape[-1]
    return 1 - np.sum((p - 1 / C) ** 2, axis=1) / (1 - 1 / C)


def variance(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : np.array(N) the variance of the N-dimentional categorical distribution
    """
    C = p.shape[-1]
    x = np.arange(C) + 1
    var = np.sum(x ** 2 * p, axis=1) - np.sum(x * p, axis=1) ** 2

    return 4 * var / (C - 1) ** 2


def semantic_based_uncertainty(p, C):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @param C : np.array(C,C) compatibility / compatibilityy matrix
    @return : np.array(N) the modified variance of the C-dimentional categorical distribution
    """

    #This calculation requires enormous memory (73 Gb for trento) which I don't have 
    #Var = np.diag(np.matmul(np.matmul(p, C*C),np.transpose(p)))
    #TODO: Something parallel, I don't like this implementation but I am stuck now

    Var = np.zeros(len(p))
    for i in range(len(p)):
        pTCC = np.matmul(p[i],C*C)
        pTCCp = np.matmul(pTCC,np.transpose(p[i]))
        Var[i] = pTCCp
    
    # general maximum
    maxVar = get_var_max_from_matrix(C)
    # local maximum
    # maxVar = np.max(Var) 

    return Var/maxVar

def semantic_based_uncertainty_old(p, C):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @param w : np.array(C,C) compatibility / compatibilityy matrix
    @return : np.array(N) the modified variance of the C-dimentional categorical distribution
    """
    s_ij = C[p.argmax(1)]
    return np.sum(**2 * p, axis=1) - np.sum(s_ij * p, axis=1) ** 2


def shannon_entropy(p):
    """
    @param p : np.array(N,C) N pixels x C probability for each class
    @return : np.array(N) the entropy of the N-dimentional categorical distribution
    """
    p = p + 1e-10
    N = p.shape[-1]

    return -np.sum(p * np.log(p), axis=1) / np.log(N)
