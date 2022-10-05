import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
from uncertainty.compatibility_matrix import calculate_compatibility_matrix

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
        )
    elif project_name == "bcss":
        from bcss_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
        )
    elif project_name == "bcss_patched":
        from bcss_patched_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
        )

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
            y_pred = y_pred_prob.argmax(1) + 1
            plt.figure(dpi=500)
            plt.imshow(
                y_pred.reshape(dataset.shape),
                interpolation="nearest",
                cmap=colors.ListedColormap(color[1:]),
            )
            plt.axis("off")
            plt.savefig(
                f"{images_dir}{model_name}_PREDICTIONS.png",
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
                f"{images_dir}{model_name}_GT.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )

            plt.figure(dpi=500)
            plt.imshow(
                geometry_based_uncertainty(y_pred_prob).reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_GBU.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )

            plt.figure(dpi=500)
            plt.imshow(
                variance(y_pred_prob).reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_VARIANCE.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )

            plt.figure(dpi=500)
            plt.imshow(
                shannon_entropy(y_pred_prob).reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_ENTROPY.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )

            plt.figure(dpi=500)
            plt.imshow(
                semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
                    dataset.shape
                ),
                cmap="coolwarm",
                # vmin=0, 
                # vmax=1
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_SBU_manual.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )

            compatibility_matrix = calculate_compatibility_matrix(X, y, "wasserstein")
            compatibility_matrix = compatibility_matrix[1:, 1:]
            plt.figure(dpi=500)
            plt.imshow(
                semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
                    dataset.shape
                ),
                cmap="coolwarm",
                # vmin=0, 
                # vmax=1
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_SBU_wasserstein.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )

            compatibility_matrix = calculate_compatibility_matrix(X, y, "energy")
            compatibility_matrix = compatibility_matrix[1:, 1:]
            plt.figure(dpi=500)
            plt.imshow(
                semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(
                    dataset.shape
                ),
                cmap="coolwarm",
                # vmin=0, 
                # vmax=1
            )
            plt.axis("off")
            cbar = plt.colorbar(location="top")
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_SBU_energy.png",
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )
