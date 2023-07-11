"""
Script for generating uncertainty maps and analysis plots for Trento dataset
$ TODO: add all analysis plots and clean the code
Usage:
  python3 scripts/trento_analysis.py 

"""

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
from uncertainty.H_matrix import calculate_H_matrix
import matplotlib.patches as mpatches

from uncertainty.H_matrix import calculate_H_matrix
from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
    FR_based_uncertainty,
)

from trento_config import *
location = "bottom"
orientation = "horizontal"
col = 6
borderaxespad =-2
columnspacing = 0.5


X, y = dataset.full_dataset  
y_true = y.reshape(-1)

for file in os.listdir(classifications_dir):
    if os.path.splitext(file)[-1].lower() == ".npy":
        y_pred_prob = np.load(os.path.join(classifications_dir, file))
    else:
        continue
    
    model_name = os.path.splitext(file)[0]
            
    y_pred = y_pred_prob.argmax(1) + 1
    print(y_pred_prob.shape[1])

    if y_pred_prob.shape[1] == 7:
        y_pred_prob = y_pred_prob[:,:-1]/np.transpose(np.tile(np.sum(y_pred_prob, axis =1), (6,1)))

    values = np.unique(y_pred.ravel())
    patches = [mpatches.Patch(color=color[i+1], label=labels[i+1].format(l=values[i]) ) for i in range(len(values)) ]
    plt.figure(dpi=500)
    plt.imshow(
        y_pred.reshape(dataset.shape),
        interpolation="nearest",
        cmap=colors.ListedColormap(color[1:]),
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
        y_true.reshape(dataset.shape),
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
    
    GU = geometry_based_uncertainty(y_pred_prob).reshape(dataset.shape)

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
        variance(y_pred_prob).reshape(dataset.shape),
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

    H = shannon_entropy(y_pred_prob).reshape(dataset.shape)
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
    #     semantic_based_uncertainty(y_pred_prob, H_matrix).reshape(
    #         dataset.shape
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
    #     pad_inches=0,
    #     dpi=500,
    # )

    # plt.figure(dpi=500)
    # plt.imshow(
    #     semantic_based_uncertainty(y_pred_prob, H_matrix1).reshape(
    #         dataset.shape
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
    #     pad_inches=0,
    #     dpi=500,
    # )


    #H_matrix = calculate_H_matrix(X, y, "JS")[1:, 1:]
    #H_matrix = H_matrix[1:, 1:]
    GU_fr = FR_based_uncertainty(y_pred_prob).reshape(dataset.shape)
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


    # H_matrix = calculate_H_matrix(X, y, "KL")[1:, 1:]
    # #H_matrix = H_matrix[1:, 1:]
    # plt.figure(dpi=500)
    # plt.imshow(
    #     semantic_based_uncertainty(y_pred_prob, H_matrix).reshape(
    #         dataset.shape
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
    #     pad_inches=0,
    #     dpi=500,
    # )

    print(X.shape)
    H_matrix = calculate_H_matrix(X[y_true!=0,:], y[y_true!=0], "energy", len(np.unique(y_pred)))#[1:, 1:]
    SU = semantic_based_uncertainty(y_pred_prob, H_matrix).reshape(
            dataset.shape
        )
    #H_matrix = H_matrix[1:, 1:]
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
    plt.savefig(f"{images_dir}{model_name}_onlycbar.eps" ,bbox_inches='tight')


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