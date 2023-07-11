
from scipy.optimize import minimize, Bounds
import numpy as np


def get_max_hu(H):
    '''
    Calculate the maximum values of homophily-based uncertainty HU given the matrix H

    @param H : np.array(C,C) classes' pairwise distance matrix
    @return : the maximum value of HU
    '''
    def rosen(x):
        return -np.transpose(x)@np.power(H,2)@x/2 #
        
    eq_cons = {'type': 'eq',
             'fun' : lambda x: np.sum(x) - 1}
    x0 = np.ones(H.shape[0])/H.shape[0]
    bounds = Bounds(np.zeros(H.shape[0]),np.ones(H.shape[0]))
    res = minimize(rosen, x0, method='SLSQP',
               constraints=eq_cons, options={'ftol': 1e-9, 'disp': False},
               bounds=bounds)
    return -res.fun
    