import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
from scipy import io
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import h5py

from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
    FR_based_uncertainty,
)
filename = "/work/saloua/Datasets/ICESAR/"
outputs_dir = "outputs/icesar/"
images_dir = "outputs/icesar/images/"

location = "bottom"
orientation = "horizontal"
col = 3
borderaxespad =-3.5
columnspacing = 0.5
        
L = np.transpose(io.loadmat("".join([f'{filename}ICESAR_L_band.mat']))['L'])
C = np.transpose(io.loadmat("".join([f'{filename}ICESAR_C_band.mat']))['C'])
X = np.concatenate((L,C), axis=0).reshape((5,-1))

gt = np.transpose(io.loadmat("".join([f'{filename}ICESAR_training_mask.mat']))['mask'])
d_shape = gt.shape

y_true = gt.flatten().astype('float')

color = ["#22181c", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1", "#3b429f"]
labels = ['Open Water', 'Grey-white Ice', 'Level FY-Ice','Deformed FY-Ice', 'Nilas', 'Grey Ice']
model_name = "GP"
print(model_name)
if model_name == "GP_C":
    file = "/work/saloua/Gaussian_Process/Classification/GP_VI/Results/PA_SVGP_tensorflowICESAR_CNone_data_size_all_M_0.01_train_size_0.02_likelihood_Stationary_stoch_False_batch_size_0_qdiag_False_Ztrain_True_Noise_False_run_im.h5"
elif model_name == "GP_L":
    file = "/work/saloua/Gaussian_Process/Classification/GP_VI/Results/PA_SVGP_tensorflowICESAR_LNone_data_size_all_M_0.01_train_size_0.02_likelihood_Stationary_stoch_False_batch_size_0_qdiag_False_Ztrain_True_Noise_False_run_im.h5"
else:
    file = "/work/saloua/Gaussian_Process/Classification/GP_VI/Results/PA_SVGP_tensorflowICESAR_LICESAR_C_data_size_all_M_0.01_train_size_0.2_likelihood_Stationary_stoch_False_batch_size_0_qdiag_False_Ztrain_True_Noise_False_run_im.h5"

with h5py.File(file,'r') as f:
    y_pred_prob = np.array(f['prediction_im_mean'])

y_pred_prob = y_pred_prob/np.tile(np.sum(y_pred_prob, axis = 1)[:,None], (1,6))  
acc_dict = []


y_pred_max_prob = y_pred_prob.max(1)
y_pred = y_pred_prob.argmax(1)
print("y_pred", np.unique(y_pred.ravel()))
#values = np.unique(y_pred.ravel())
values = (np.unique(y_true.ravel())[1:]-1).astype('int')
print(values)
patches = [mpatches.Patch(color=color[i], label=labels[i].format(l=values[i]) ) for i in values] #range(len(values)) ]

L[1,:,:][L[1,:,:]<-35] = -35 
L[1,:,:][L[1,:,:]>-5] = -5

L[0,:,:][L[0,:,:]<-30] = -30 
L[0,:,:][L[0,:,:]>0] = 0

L[2,:,:][L[2,:,:]<-35] = -35 
L[2,:,:][L[2,:,:]>-5] = -5

hh = (L[0,:,:]-np.min(L[0,:,:]))/(np.max(L[0,:,:]) - np.min(L[0,:,:]))
hv = (L[1,:,:]-np.min(L[1,:,:]))/(np.max(L[1,:,:]) - np.min(L[1,:,:]))
vv = (L[2,:,:]-np.min(L[2,:,:]))/(np.max(L[2,:,:]) - np.min(L[2,:,:]))

y_true[y_true==0] = np.nan

L_img = np.moveaxis(np.concatenate((hh[None,:,:], hv[None,:,:], vv[None,:,:]), axis = 0), 0, -1)
print(L_img.shape)
print(y_true.reshape(d_shape).shape)
plt.imshow(L_img)
plt.imshow(
    y_true.reshape(d_shape),
    interpolation="nearest",
    cmap=colors.ListedColormap(color), alpha = 0.8)
plt.axis("off")
plt.legend(handles=patches, loc=8, ncol = col, fontsize='small', borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"

plt.savefig(
    f"L_gt.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

print(np.amin(C[0,:,:]))
print(np.amin(C[1,:,:]))
print(np.amax(C[0,:,:]))
print(np.amax(C[1,:,:]))

C[0,:,:][C[0,:,:]<-35] = -35 
C[0,:,:][C[0,:,:]>-5] = -5

C[1,:,:][C[1,:,:]<-30] = -30 
C[1,:,:][C[1,:,:]>0] = 0

hv = (C[0,:,:]-np.min(C[0,:,:]))/(np.max(C[0,:,:]) - np.min(C[0,:,:]))
hh = (C[1,:,:]-np.min(C[1,:,:]))/(np.max(C[1,:,:]) - np.min(C[1,:,:]))

y_true[y_true==0] = np.nan

C_img = np.moveaxis(np.concatenate((hv[None,:,:], hh[None,:,:], hh[None,:,:]), axis = 0), 0, -1)
print(C_img.shape)
print(y_true.reshape(d_shape).shape)
plt.imshow(C_img)
plt.imshow(
    y_true.reshape(d_shape),
    interpolation="nearest",
    cmap=colors.ListedColormap(color), alpha = 0.8)
plt.axis("off")
plt.legend(handles=patches, loc=8, ncol = col, fontsize='small', borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"

plt.savefig(
    f"C_gt.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.imshow(
    y_pred.reshape(d_shape),
    interpolation="nearest",
    cmap=colors.ListedColormap(color[:-1]),
)
plt.axis("off")
plt.legend(handles=patches, loc=8, ncol = col, fontsize='small', borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"

plt.savefig(
    f"{images_dir}{model_name}_PREDICTIONS.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

y_true[0] = 0
plt.figure(dpi=500)
plt.imshow(
    y_true.reshape(d_shape),
    interpolation="nearest",
    cmap=colors.ListedColormap(color),
)
plt.axis("off")
plt.savefig(
    f"{images_dir}{model_name}_GT.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

GU = geometry_based_uncertainty(y_pred_prob).reshape(d_shape)
print("min GU", np.amin(GU))
print("max GU", np.amax(GU))

plt.figure(dpi=500)
plt.imshow(
    GU,
    cmap="turbo",
    vmin=0,
    vmax=1,
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_GBU.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

plt.figure(dpi=500)
plt.imshow(
    variance(y_pred_prob).reshape(d_shape),
    cmap="turbo",
    vmin=0,
    vmax=1,
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_VARIANCE.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

H = shannon_entropy(y_pred_prob).reshape(d_shape)
plt.figure(dpi=500)
plt.imshow(
    H,
    cmap="turbo",
    vmin=0,
    vmax=1,
)

plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)#location="top"
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_ENTROPY.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# plt.figure(dpi=500)
# plt.imshow(
#     semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
#         d_shape
#     ),
#     cmap="turbo",
#     vmin=0, 
#     vmax=1
# )
# plt.axis("off")
# #cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
# #cbar.ax.tick_params(labelsize=12)
# plt.savefig(
#     f"{images_dir}{model_name}_SBU_manual1.eps",
#     bbox_inches="tight",
#     pad_inches=0.1,
#     dpi=500,
# )

# plt.figure(dpi=500)
# plt.imshow(
#     semantic_based_uncertainty(y_pred_prob, compatibility_matrix1).reshape(
#         d_shape
#     ),
#     cmap="turbo",
#     vmin=0, 
#     vmax=1
# )
# plt.axis("off")
# #cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
# #cbar.ax.tick_params(labelsize=12)
# plt.savefig(
#     f"{images_dir}{model_name}_SBU_manual2.eps",
#     bbox_inches="tight",
#     pad_inches=0.1,
#     dpi=500,
# )


#compatibility_matrix = calculate_compatibility_matrix(X, y, "JS")[1:, 1:]
#compatibility_matrix = compatibility_matrix[1:, 1:]
GU_fr = FR_based_uncertainty(y_pred_prob).reshape(d_shape)
plt.figure(dpi=500)
plt.imshow(
    GU_fr,
    cmap="turbo",
    vmin=0, 
    vmax=1
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_GBU_FR.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)


# compatibility_matrix = calculate_compatibility_matrix(X, y, "KL")[1:, 1:]
# #compatibility_matrix = compatibility_matrix[1:, 1:]
# plt.figure(dpi=500)
# plt.imshow(
#     semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
#         d_shape
#     ),
#     cmap="turbo",
#     #vmin=0, 
#     #vmax=1
# )
# plt.axis("off")
# cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
# cbar.ax.tick_params(labelsize=12)
# plt.savefig(
#     f"{images_dir}{model_name}_SBU_KL.eps",
#     bbox_inches="tight",
#     pad_inches=0.1,
#     dpi=500,
# )

print(X.shape)
if model_name == "GP_L":
    compatibility_matrix = calculate_compatibility_matrix(np.transpose(X[:3,y_true!=0]), y_true[y_true!=0], "energy", len(values))#[1:, 1:]
elif model_name == "GP_C":
    compatibility_matrix = calculate_compatibility_matrix(np.transpose(X[3:,y_true!=0]), y_true[y_true!=0], "energy", len(values))#[1:, 1:]
else:
    compatibility_matrix = calculate_compatibility_matrix(np.transpose(X[:,y_true!=0]), y_true[y_true!=0], "energy", len(values))#[1:, 1:]

np.set_printoptions(precision=2)

print(compatibility_matrix/np.amax(compatibility_matrix))
SU = semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
        d_shape
    )
print("min SU", np.amin(SU))
print("max SU", np.amax(SU))

#compatibility_matrix = compatibility_matrix[1:, 1:]
plt.figure(dpi=500)
su_plt = plt.imshow(
    SU,
    cmap="turbo",
    vmin=0, 
    vmax=1
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_SBU_energy.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)


# draw a new figure and replot the colorbar there
fig,ax = plt.subplots(dpi=500)
cbar = plt.colorbar(su_plt, location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
ax.remove()
plt.savefig('plot_onlycbar.eps',bbox_inches='tight')


plt.figure(dpi=500)
plt.imshow(
    GU - H,
    cmap="turbo",
    vmin=-1,
    vmax=1,
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_DIFF_GBU_H.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

plt.figure(dpi=500)
plt.imshow(
    GU - GU_fr,
    cmap="turbo",
    vmin=-1,
    vmax=1,
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_DIFF_GBU_GUFR.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

plt.figure(dpi=500)
plt.imshow(
    GU - SU,
    cmap="turbo",
    vmin=-1,
    vmax=1,
)
plt.axis("off")
cbar = plt.colorbar(location = location, orientation = orientation, pad = 0.01)
cbar.ax.tick_params(labelsize=12)
plt.savefig(
    f"{images_dir}{model_name}_DIFF_GBU_SU.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
