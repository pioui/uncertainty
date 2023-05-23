'''
This script implements the haralick feature extraction
'''


from skimage.feature import greycomatrix, greycoprops
#import h5py
import itertools
import numpy as np
from haralickTextureFeatures import *
import matplotlib.pyplot as plt
#import multiprocessing as mp
#from functools import partial
import seaborn as sns
import pandas as pd
import numba as nb
import imageio
from scipy import io

#from plot_mat import Oil_id

win = 9

@nb.stencil(neighborhood = ((-int((win-1)/2), int((win-1)/2)),(-int((win-1)/2), int((win-1)/2))))
def unit_glcm(G):
    return my_grey_comatrix(G, numOffset = 5, NL = 8)

# @nb.jit('uint8[:,:](uint8[:,:])', nopython=True, parallel=True)
# def glcm(G):
#     return unit_glcm(G)

#@nb.stencil(neighborhood = ((-int((win-1)/2), int((win-1)/2)),(-int((win-1)/2), int((win-1)/2))))
def glcm_skimage(G, distances, angles,glcm_feat):
    NL = 8
    A = np.floor(G*8).astype(np.uint8)
    A[A>=NL] = NL-1
    A[A<0] = 0
    return greycoprops(greycomatrix(A, distances, angles, levels=NL, symmetric=False), prop = glcm_feat)

#@nb.jit(nopython=True, parallel=True)
def glcm(G, distances, angles,glcm_feat, iter):
    return np.mean(glcm_skimage(G[iter[0],iter[1],:,:], distances, angles,glcm_feat))

dataset = "icesar" #"bcss"
#! Load image

if dataset =="bcss":
    data_dir = "/work/saloua/uncertainty-main/datasets/"

    X = np.array(imageio.v2.imread(f"{data_dir}images/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"))
    Ncol, Nrow, D = X.shape 
    gt = np.array(imageio.v2.imread(f"{data_dir}masks/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"), dtype=np.int64).flatten()
    X_new = X.reshape((Ncol*Nrow, D))#[gt!=0,:]
elif dataset =="icesar":
    filename = "/work/saloua/Datasets/ICESAR/"

    L = np.transpose(io.loadmat("".join([f'{filename}ICESAR_L_band.mat']))['L'])
    D1, Ncol, Nrow= L.shape
    C = np.transpose(io.loadmat("".join([f'{filename}ICESAR_C_band.mat']))['C'])
    D2 = C.shape[0]
    D = D1+D2
    X_new = np.transpose(np.concatenate((L,C), axis=0).reshape((5,-1)))
    gt = np.transpose(io.loadmat("".join([f'{filename}ICESAR_training_mask.mat']))['mask'])
    

glcm_feat = ["energy", "contrast", "correlation", "homogeneity", "dissimilarity", "ASM"]

#* Normalize data
mean_values = np.tile(np.mean(X_new, axis=0), (X_new.shape[0], 1))
var_values = np.tile(np.var(X_new, axis=0), (X_new.shape[0], 1))
X_new = (X_new - mean_values) / (np.sqrt(var_values))
X = X_new.reshape((Ncol,Nrow, D))

#! Parameters of glcm features
win = 9 # window size
distances = np.arange(1, 5) 
angles = [0, np.pi/4, np.pi/2, 3*np.pi/2]

#!
stats = np.zeros((X.shape[-1], X.shape[0], X.shape[1], len(glcm_feat)))

# stats = pool.map(glcm_part, itertools.product(range(Y.shape[0]), range(Y.shape[1])))
# glcm_part2 = partial(glcm_skimage, Y, distances, angles)
# stats_skimage = pool.map(glcm_part2, itertools.product(range(Y.shape[0]), range(Y.shape[1])))

for i in range(D):
    #! Add padding
    X_d = np.pad(X[:,:,i], (int((win-1)/2), int((win-1)/2)), mode='constant', constant_values=(np.nan,np.nan)) # M+win-1, N+win-1, win, win

    #! Sliding window
    Y = np.lib.stride_tricks.sliding_window_view (X_d, (win,win)) # M, N, win, win #X.astype(np.uint8)
    for j in range(len(glcm_feat)):
        pool = mp.Pool(mp.cpu_count())
        glcm_part = partial(glcm, Y, distances, angles, glcm_feat[j])
        stats[i,:,:,j] = np.array(pool.map(glcm_part, itertools.product(range(Y.shape[0]), range(Y.shape[1])))).reshape((Ncol, Nrow)) #glcm(X[:,:,i], distances, angles, glcm_feat[j])
        pool.close()
        
np.save(f"/work/saloua/uncertainty-main/outputs/stats_{dataset}.npy", stats)

for j in range(D):
    # create figure
    fig = plt.figure(figsize=(10, 7), frameon=False)
    
    # setting values to rows and column variables
    rows = 2
    columns = 3

    #glcm_feat = ["energy", "contrast", "correlation", "variance", "IDM", "sum_average", "sum_entropy", "sum_variance", "entropy", "diff_entropy", "diff_variance", "info_1", "info_2", "max_corr_coeff"]
    for i in range(len(glcm_feat)):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(stats[j,:,:,i], cmap = 'gray')
        plt.axis('off')
        plt.title(glcm_feat[i], fontsize=7)

    #fig.add_subplot(rows, columns, 15)
    
    # showing image
    #plt.imshow() #, cmap = 'gray')
    #plt.axis('off')
    fig.savefig(f"/work/saloua/uncertainty-main/outputs/glcm_python_{dataset}_{j}.png")
exit()
print(B.shape)
B = np.reshape(B, (B.shape[0]*B.shape[1], B.shape[2]))
#! Plot only distributions
#vv_path = "/work/saloua/Oil_spill/Norne_data/S1/NOFO/PAZ1_SAR__MGD_RE___SC_S_SRA_20210429T165543_20210429T165640/47599/S1A_IW_GRDH_1SDV_20210429T165605_20210429T165630_037668_0471B3_1A8F/features/1x1/Sigma0_VV_db.hdf5"

filename = f"/work/saloua/Oil_spill/Norne_data/Norne_GLCM/{Oil_id}/VV_glcm.mat"
with h5py.File(filename, 'r') as f:
    gt_test = np.asarray(f["Attributes"])
    gt = np.asarray(f["gt"])

print(gt_test.shape)
labels = ["Oil", "Water"]
idx = np.nonzero(gt.flatten())[0]
image = np.transpose(B[idx, :]) #gt_test.reshape((gt_test.shape[0],-1))[:,idx] #

print(image.shape)
data = pd.DataFrame(data = {"classes": gt.flatten()[idx]})

print(np.unique(gt))
#labels = []
for i in range(len(labels)):
    idx = np.nonzero(np.asarray(data["classes"]==i+1))[0]
    #print(len(idx))
    data["classes"][idx] = labels[i]

cols = ["Energy","Contrast","Correlation","Variance","Homogeneity","Sum Average","Sum Variance","Sum Entropy","Entropy","Diff Variance","Diff Entropy","Info Correlation I","Info Correlation II","Max Correlation Coeff","classes"]

# create the figure and axes
fig, axes = plt.subplots(3, 5, figsize=(15,10))
axes = axes.ravel()  # flattening the array makes indexing easier
print(len(axes))
for i in range(len(glcm_feat)):
    data[cols[i]] = image[i,:]
    sns.kdeplot(data=data, x=cols[i], hue="classes", alpha=0.4, fill=True, ax=axes[i], legend=False, hue_order = ["Water", "Oil"])
    
fig.tight_layout()
plt.show()
    
fig.savefig(f"{Oil_id}_glcm_dist_test")
