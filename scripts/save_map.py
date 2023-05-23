from ctypes.wintypes import HACCEL
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
import matplotlib.patches as mpatches
from scipy.io import savemat
from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
)


for project_name in os.listdir("outputs/"):
    if project_name == "trento":
        from trento_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            compatibility_matrix1,
            color,
            labels,
        )
        location = "bottom"
        orientation = "horizontal"
        col = 6
        borderaxespad =-2
        columnspacing = 1
    # elif project_name == "bcss":
    #     from bcss_config import (
    #         dataset,
    #         project_name,
    #         images_dir,
    #         outputs_dir,
    #         compatibility_matrix,
    #         compatibility_matrix1,
    #         color,
    #         labels,
    #     )
    #     location = "right"
    #     orientation = "vertical"
    #     col = 3
    #     borderaxespad =-3.25
    #     columnspacing = 1.25
    # elif project_name == "bcss_patched":
    #     from bcss_patched_config import (
    #         dataset,
    #         project_name,
    #         images_dir,
    #         outputs_dir,
    #         compatibility_matrix,
    #         compatibility_matrix1,
    #         color,
    #         labels,
    #     )
    #     location = "right"
    #     orientation = "vertical"
    #     col = 3
    #     borderaxespad =-3.25
    #     columnspacing = 1.25

    else:
        continue

    X, y = dataset.full_dataset  # 15107
    y_true = y.reshape(-1)

    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            model_name = file.split("_")[-1].split(".")[0]
            if model_name == "violin":
                continue
            
            if model_name == "SVM":
                continue
            
            y_pred_prob = np.load(os.path.join(outputs_dir, file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1) + 1
            
            values = np.unique(y_pred.ravel())
            patches_list = [mpatches.Patch(color=color[i+1], label=labels[i+1].format(l=values[i]) ) for i in range(len(values)) ]
            fig, ax = plt.subplots(1, dpi=500)
            plt.imshow(
                y_pred.reshape(dataset.shape),
                interpolation="nearest",
                cmap=colors.ListedColormap(color[1:]),
            )
            #Pixels = [P8, P6, P7, P5, P3, P2, P1, P4]
            x = [449, 420, 420, 302, 212, 157, 61, 225] #, 106, 230, 187
            z = [39, 132, 125, 38, 102, 146, 75, 146] #, 104, 144, 130
            for i in range(len(x)):
                rect = mpatches.Rectangle((x[i]-2.5, z[i]-2.5), 5,5, linewidth = 1, edgecolor = 'r', facecolor = 'none')
                ax.add_patch(rect)
            plt.axis("off")

            plt.legend(handles=patches_list, loc=8, ncol = col, fontsize='small', borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"
            plt.savefig(
                f"{images_dir}{model_name}_PREDICTIONS.eps",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )
            
            
            W = calculate_compatibility_matrix(X, y, "wasserstein")[1:, 1:]
            E = calculate_compatibility_matrix(X, y, "energy")[1:, 1:]
            GU = np.empty(len(x))
            Var = np.empty(len(x))
            H = np.empty(len(x))
            SU = np.empty(len(x))
            SU_W = np.empty(len(x))
            SU_E = np.empty(len(x))
            p = np.empty((len(x),6))
            for i in range(len(x)):
                p[i,:] = y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[z[i],x[i],:]
             
            np.set_printoptions(precision=1)   
            print(p)
            GU = geometry_based_uncertainty(p)
            Var = variance(p)
            H = shannon_entropy(p)
            S = np.array(
                [
                    [0, 3, 3, 2, 1, 3],
                    [3, 0, 3, 3, 3, 2],
                    [3, 3, 0, 3, 3, 2],
                    [2, 3, 3, 0, 2, 3],
                    [1, 3, 3, 2, 0, 3],
                    [3, 2, 2, 3, 3, 0],
                ]
            )/3 #10
            
            SU = semantic_based_uncertainty(p, S)
            S = np.array(
                [
                    [0, 3, 4, 2, 1, 3],
                    [3, 0, 3, 3, 3, 2],
                    [4, 3, 0, 3, 4, 2],
                    [2, 3, 3, 0, 2, 3],
                    [1, 3, 4, 2, 0, 3],
                    [3, 2, 2, 3, 3, 0],
                ]
            )/4 #10
            SU_W = semantic_based_uncertainty(p, S)
            #SU_W = semantic_based_uncertainty(p, WS)
            SU_E = semantic_based_uncertainty(p, E)
            
            print(H)
            print(Var) 
            print(GU)
            print(SU)
            print(SU_W)
            print(SU_E)
                
            data={'p':p, 'H':H, 'GU':GU, 'Var':Var, 'SU':SU, 'SU_W': SU_W, 'SU_E': SU_E}
            np.save(f"{images_dir}{model_name}_points.npy", data)
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[39,449,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[132,420,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[125,449,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[38,302,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[102,212,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[146,157,:])
            # print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[75,61,:])
            
            
            #mdic = {"data": y_pred_prob}
            #savemat(f"{images_dir}{model_name}_PREDICTIONS.mat", mdic)
            

