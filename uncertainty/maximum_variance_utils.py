'''
This script includes functions to generate the maximum of semantic uncertainty
    - "get_var_max_from_matrix" calculates the maximum using the "Compatibility matrix"
    - "get_var_max" generates the maximum in Latex form
    - "get_var_opt" finds the maximum via an optimization algorithm
    
author: Saloua Chlaily (UiT)
date: 07/2022
'''

from itertools import permutations
from sympy import parse_expr, latex
from sympy.utilities.iterables import partitions
from scipy.optimize import minimize, Bounds
import math
import numpy as np


def get_var_opt(D):
    def rosen(x):
        return -np.transpose(x)@np.power(D,2)@x/2 #
        
    eq_cons = {'type': 'eq',
             'fun' : lambda x: np.sum(x) - 1}
    x0 = np.ones(D.shape[0])/D.shape[0]
    bounds = Bounds(np.zeros(D.shape[0]),np.ones(D.shape[0]))
    res = minimize(rosen, x0, method='SLSQP',
               constraints=eq_cons, options={'ftol': 1e-9, 'disp': False},
               bounds=bounds)
    return -res.fun
    
def get_var_max_from_matrix(D):
    '''
    This function calculates the maximum using the "Compatibility matrix"
    
    Inputs:
    -------
    D: CxC matrix
        compatibility matrix where C is the number of classes
    
    Outputs:
    -------
    Vmax: float
        Maximum value of semantic uncertainty
        Vmax = alpha/(2*beta) if C!=3 else Vmax = -alpha/(2*beta)
    '''
    C = len(D)
    alpha = sum(get_AC_d(C, i, D) for i in range(1, C // 2 + 1))
    beta = sum(get_BC_d1(C, i, D) for i in range(1, C // 2 + 1))
    if C % 2 == 1:
        beta += sum(get_BC_d2(C, i, D) for i in range(1, C // 2 + 1))
    return np.max(D)**2 / 4 if beta <= 0 else -alpha / (2 * beta)


def get_s_value(i, j, D=None):
    '''
    This function returns the symbol "s_ij**2" or its value
    
    Inputs:
    -------
    i,j: integers
        the subscripts of "s_ij"
    D: CxC matrix
        Compatibility matrix
    Outputs:
    -------
    r: string if D is None else float
        
    '''
    if D is not None:
        return D[i - 1][j - 1]**2
    if i < j:
        return parse_expr(f"s_{i}{j}**2")
    elif i == j:
        return 0
    else:
        return parse_expr(f"s_{j}{i}**2")
    
def get_var_max(C):
    '''
    This function calculates the maximum of semantic uncertainty
    
    Inputs:
    -------
    C: int
        Number of classes
    
    Outputs:
    -------
    Vmax: string
        Latex form of Maximum value of semantic uncertainty
        Vmax = alpha/(2*beta) if C!=3 else Vmax = -alpha/(2*beta)
    '''
    alpha = sum(get_AC_d(C, i) for i in range(1, C // 2 + 1))
    beta = sum(get_BC_d1(C, i) for i in range(1, C // 2 + 1))
    if C % 2 == 1:
        if C == 3:
            beta += sum(get_BC_d2(C, i) for i in range(1, C // 2 + 1))
        else:
            beta += sum(get_BC_d2(C, i) for i in range(2, C // 2 + 1))
    return latex(-1 * alpha / (2 * beta)) if C != 3 else latex(alpha / (2 * beta))


def get_AC(C,D=None):
    a = 0
    for i in list(permutations(range(1, C+1), C)):
        b = 1
        for j in range(C-1):
            b = b*get_s_value(i[j],i[j+1],D)
        b = b*get_s_value(i[C-1],i[0],D)
        a+=b
    return - a/C

def get_BC_d1(C, d,D=None):
    a = 0
    for i in list(permutations(range(1, C + 1), C)):
        for p in partitions(C, m=d):
            if sum(p.values()) == d:
                for kk in p.keys():
                    b = 1
                    c = 0
                    f = 0
                    for k in p.keys():
                        w = 1
                        for l in range(p[k]):
                            c = f + l * k
                            for j in range(c, c + k - 1):
                                w = w * get_s_value(i[j], i[j + 1],D)
                            if k != kk or l != 0:
                                w = w * get_s_value(i[c + k - 1], i[c],D) / k
                        b = b * w / l if l != 0 else b * w
                        f += p[k] * k
                    a += (-1)**(d - 1) * b
    return a

def get_BC_d2(C,d,D=None):
    a = 0
    for i in list(permutations(range(1, C+1), C)):
        for p in partitions(C-1, m = d):
            if sum(p.values()) == d:
                for _ in p.keys():
                    b = 1
                    c = 0
                    f = 0
                    for k in p.keys():
                        w = 1
                        for l in range(p[k]):
                            c = f + l*k
                            for j in range(c, c+k-1):
                                w = w*get_s_value(i[j],i[j+1],D)
                            w = w*get_s_value(i[c+k-1],i[c],D)/k
                        b = b*w/(l+1)
                        f += p[k]*k
                    a+= (-1)**(d)*b
    return a

def get_AC_d(C,d,D=None):
    a = 0
    for i in list(permutations(range(1, C+1), C)):
        for p in partitions(C, m = d):
            if sum(p.values()) == d:
                b = 1
                c = 0
                f = 0
                for k in p.keys():
                    w = 1
                    for l in range(p[k]):
                        c = f + l*k
                        for j in range(c, c+k-1):
                            w = w*get_s_value(i[j],i[j+1],D)
                        w = w*get_s_value(i[c+k-1],i[c],D)/k
                    b = b*w/math.factorial(l+1)
                    f += p[k]*k
                a+=b
    return (-1)**d * a
