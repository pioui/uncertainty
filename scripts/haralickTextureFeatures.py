'''
This script/function calculates all Haralick Features from a GLCM matrix
Source: http://haralick.org/journals/TexturalFeatures.pdf

This code need the gray-level co-occurence matrix that can be calculated using:
skimage.feature.texture.greycomatrix(image, distances, angles, levels=256, symmetric=False, normed=False)

Author: Saloua Chlaily
Date: 24/03/22
'''

import numpy as np
from numpy.matlib import repmat
import multiprocessing as mp
from functools import partial
from itertools import product
import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True)
def my_accum(subs, shape):
    out = np.zeros((shape[0],shape[1]))
    for i, j in product(range(shape[0]), range(shape[1])):
        out[i,j] = np.sum((subs[0,:]==i+1) * (subs[1,:]==j+1))

    return out#.reshape(shape)

@nb.jit(nopython=True, parallel=True)
def get_glcms(r,c,I, NL, iter):
    offset = np.array([[0, iter[0]], [-iter[0], iter[0]], [-iter[0], 0], [-iter[0], -iter[0]]])
    glcm = one_glcm(r,c,offset[iter[1],:], I, NL)
    return get_text_feat(glcm+np.transpose(glcm))

@nb.jit(nopython=True, parallel=True)           
def one_glcm(r,c,offset,si,nl):
    r2 = r+offset[0]
    c2 = c+offset[1]
    nRow, nCol = si.shape
    
    # Subcripts out boundaries
    out_bounds = np.nonzero((c2<0) + (c2>=nCol) + (r2<0) + (r2>=nRow))
    
    # vectors with si(r,c)
    v1 = np.transpose(si).ravel() #np.moveaxis(si, [0,1], [1, 0]).ravel()
    v1 = np.delete(v1, out_bounds)
    r2 = np.delete(r2, out_bounds)
    c2 = np.delete(c2, out_bounds)
    ind = r2 + c2*nRow
    v2 = si.ravel()[ind]#.ravel()
    
    #bad = np.nonzero(v1==np.nan or v2==np.nan)
    Ind = np.vstack((v1, v2))
    
    if Ind.shape==0:
        glcm = np.zeros((nl,nl))
    else:
        glcm = my_accum(Ind, (nl, nl)) #accum(Ind, 1, size = (nl, nl))
    return glcm

@nb.jit(nopython=True, parallel=True)  
def my_grey_comatrix(SI, numOffset = None, directions = 4, NL = 8):
    if numOffset is None:
        numOffset = 1#np.asarray([0, 1])
    
    I = np.floor(SI*NL+1)#.astype(np.uint8)
    I[I>NL] = NL
    I[I<1] = 1
    
    r, c = np.meshgrid(np.arange(0,I.shape[0]), np.arange(0,I.shape[1]), indexing='ij')
    r = r.ravel()
    c = c.ravel()
    GLCMs = np.zeros((14, directions ,numOffset+1))
    
    # pool = mp.Pool(mp.cpu_count())
    # glcm_part = partial(get_glcms, r, c, I, NL)
    # stats = pool.map(glcm_part, product(range(1,numOffset+1), range(4)))
    # GLCMs = np.array(stats).reshape((14, 4, -1))
        
    for i in range(1,numOffset+1):
        offset = np.array([[0, i], [-i, i], [-i, 0], [-i, -i]]) 
        #print(offset.shape)     
        for k in range(offset.shape[0]):
            if len(offset.shape) == 1:
                glcm = one_glcm(r,c,offset, I, NL)
                GLCMs[:,k,i] = get_text_feat(glcm+np.transpose(glcm))
            else:
                glcm = one_glcm(r,c,offset[k,:], I, NL)
                GLCMs[:,k,i] = get_text_feat(glcm+np.transpose(glcm))
    return np.mean(GLCMs, axis = (1,2)) #GLCMs + np.transpose(GLCMs, (1,0,2,3))

@nb.jit(nopython=True)
def get_text_feat(P):
    '''
    Inputs: - P the 4-d grey co-occurence matrix
    Outputs: - the 14 features
    '''
    #! Normalize GLCM
    P_sum = np.sum(P)
    P = P if P_sum == 0 else P/P_sum
    
    #! Coordinate matrices
    cols, rows = np.meshgrid(np.arange(1,P.shape[0]+1), np.arange(1,P.shape[1]+1))
    
    #! Average and standard deviation for correlation and difference variance
    rowMean = np.sum(rows*P)
    colMean = np.sum(cols*P) 
    rowStd = np.sqrt(np.sum((rows-rowMean)**2 *P))
    colStd = np.sqrt(np.sum((cols-colMean)**2 *P))
    # rowMean = np.sum(rows.ravel()*P.ravel())
    # colMean = np.sum(cols.ravel()*P.ravel()) 
    # rowStd = np.sqrt(np.sum((rows.ravel()-rowMean)**2 *P.ravel()))
    # colStd = np.sqrt(np.sum((cols.ravel()-colMean)**2 *P.ravel()))
    
    # rowStd[rowStd==0] = 1
    # colStd[colStd==0] = 1
    
    #! Sum of rows and columns for information measures and correlation coeff
    rowSum = np.sum(P, axis = 1)[None,:] #N_levelx1
    colSum = np.sum(P, axis = 0)[:,None] #1xN_level
    
    #! p_{x+y} for sum average, sum variance, and sum entropy
    m = -P.shape[0]+1
    n = -m
    
    P90 = np.rot90(P)
    p_XplusY = np.zeros((2*n+1, 1))
    
    k = 0
    for i in range(m,n+1):
        p_XplusY[k] = np.sum(np.diag(P90, i))
        k = k+1
        
    #! p_{x-y} for difference variance and difference entropy
    
    p_XminusY = np.zeros((n+1, 1))
    p_XminusY[0] = np.sum(np.diag(P, 0))
    
    k = 1
    for i in range(1,n+1):
        p_XminusY[k] = np.sum(np.vstack((np.diag(P, i), np.diag(P, -i))))
        k = k+1
    
    #! Features calculation
    #* Energy
    energy = np.sum(P**2)
    
    #* Contrast 
    contrast = np.sum(((rows-cols)**2)*P)
    
    #TODO * Correlation
    if (rowStd*colStd) == 0:
        correlation = np.nan #np.sum((rows.ravel()-rowMean)*(cols.ravel()-colMean)*P.ravel())
    else:
        correlation = np.sum((rows-rowMean)*(cols-colMean)*P)/(rowStd*colStd)
    #correlation = np.sum((rows.ravel()-rowMean)*(cols.ravel()-colMean)*P.ravel())
    
    #* Variance
    variance = np.sum(((rows.ravel()-np.mean(P))**2)*P.ravel())
    
    #* Inverse difference moment
    IDM = np.sum(P/(1+(rows-cols)**2))
    
    #* sum average
    sum_average = np.sum(np.arange(2,2*P.shape[0]+1)[:, None]*p_XplusY)
    
    #* Sum_entropy
    sum_entropy = -np.sum(p_XplusY[p_XplusY!=0]*np.log(p_XplusY[p_XplusY!=0]))
    
    #* Sum variance
    sum_variance = np.sum((np.arange(2,2*P.shape[0]+1)[:, None] - sum_entropy)**2*p_XplusY)
    
    #* Entropy
    entropy = -np.sum(P[P!=0] * np.log2(P[P!=0]))
    
    #* Difference variance
    diff_variance = np.sum((np.arange(P.shape[0])[:, None] - np.mean(p_XminusY))**2 * p_XminusY)
    
    #* Difference entropy
    diff_entropy = -np.sum(p_XminusY[p_XminusY!=0] * np.log(p_XminusY[p_XminusY!=0]))
    
    #* Information measure of correlation I
    K = np.transpose(rowSum)@np.transpose(colSum)
    K[np.nonzero(K==0)] = 1
    a = np.log2(K)
    a[np.nonzero(np.isinf(a))] = 0
    Hxy1 = -np.sum(P*a)
    
    K = colSum
    K[np.nonzero(K==0)] = 1
    b = np.log2(K)
    b[np.nonzero(np.isinf(b))] = 0
    Hx = -np.sum(colSum*b)
    
    K = rowSum
    K[np.nonzero(K==0)] = 1
    c = np.log2(K)
    c[np.nonzero(np.isinf(c))] = 0
    Hy = -np.sum(rowSum*c)
    
    if max(Hx, Hy) == 0:
        info_1 = (entropy - Hxy1)
    else:
        info_1 = (entropy - Hxy1)/max(Hx, Hy)
    
    #* Information measure of correlation II
    
    d = np.log2(rowSum*colSum)
    d[np.nonzero(np.isinf(d))] = 0
    Hxy2 = -np.sum((rowSum*colSum)*a)
    
    info_2 = np.sqrt(1-np.exp(-2*(Hxy2-entropy)))
    
    #* Maximal correlation coefficient
    Q = np.zeros((P.shape[0], P.shape[1]))
    for i in range(P.shape[1]):
        denom = repmat((rowSum[:,i]*colSum)[:,0], P.shape[0], 1)
        num = repmat(P[i,:], P.shape[0], 1)*P
        #denom[np.nonzero(denom==0)] = 1
        #num[denom==0] = 0
        Q[i,:] = np.sum(num/denom, axis=1)
    # for i in range(P.shape[0]):
    #     for j in range(P.shape[1]):
    #         Q[i,j] = np.sum((P[i,:]*P[j,:])/(rowSum*colSum))
            
    Q = np.nan_to_num(Q)
    eig = np.linalg.eig(Q)[0]
    if (eig==np.max(eig)).all():
        max_corr_coeff = 0
    else:
        eig = np.delete(eig, np.nonzero(eig==np.max(eig)))
        max_corr_coeff = np.abs(np.sqrt(max(eig)))
        #eig[np.nonzero(eig==np.max(eig))] = []
    
    return energy, contrast, correlation, variance, IDM, sum_average, sum_entropy, sum_variance, entropy, diff_entropy, diff_variance, info_1, info_2, max_corr_coeff