'''
This script plot the violinplots
'''
import seaborn as sns
import numpy as np
import os
import pandas as pd
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
import matplotlib.pyplot as plt
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
    C = np.unique(y_true)
    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            model_name = file.split("_")[-1].split(".")[0]
            if model_name == "violin":
                continue
            y_pred_prob = np.load(os.path.join(outputs_dir, file))#, allow_pickle = True)
            for i in range(y_pred_prob.shape[1]):
                fig, ax = plt.subplots(1, dpi=500)
                plt.imshow(y_pred_prob[:,i].reshape(dataset.shape))
                plt.axis("off")
                plt.savefig(
                    f"{images_dir}{model_name}_probs_{i}.eps",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=500,
                )            