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
            # print(y_pred_prob[10035,:])
            # print(y_pred_prob[26042,:])
            # print(y_pred_prob[35128,:])
            # print(y_pred_prob[50004,:])
            # print(y_pred_prob[69679,:])
            # print(y_pred_prob[69686,:])
            #print(y_pred_prob[23249,:]) #[74407,:]
            #print(y_pred_prob[74407,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[39,449,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[132,420,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[125,420,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[125,449,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[38,302,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[102,212,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[146,157,:])
            print(y_pred_prob.reshape((dataset.shape[0], dataset.shape[1], -1))[75,61,:])
            #y_pred_max_prob = y_pred_prob.max(1)
            #y_pred = y_pred_prob.argmax(1) + 1
            
            #mdic = {"data": y_pred_prob}
            #savemat(f"{images_dir}{model_name}_PREDICTIONS.mat", mdic)
            

