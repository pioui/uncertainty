import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
import matplotlib.patches as mpatches

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
            color,
            labels,
        )
        location = "bottom"
        orientation = "horizontal"
        col = 6
        borderaxespad =-2
        columnspacing = 1
    elif project_name == "bcss":
        from bcss_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
            labels,
        )
        location = "right"
        orientation = "vertical"
        col = 3
        borderaxespad =-3.25
        columnspacing = 1.25
    elif project_name == "bcss_patched":
        from bcss_patched_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
            labels,
        )
        location = "right"
        orientation = "vertical"
        col = 3
        borderaxespad =-3.25
        columnspacing = 1.25

    else:
        continue

    X, y = dataset.full_dataset  # 15107
    y_true = y.reshape(-1)

    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            model_name = file.split("_")[-1].split(".")[0]

            y_pred_prob = np.load(os.path.join(outputs_dir, file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argsort(1)[:,1] + 1
            print(y_pred.shape)
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
                f"{images_dir}{model_name}_2nd_PREDICTIONS.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )

            