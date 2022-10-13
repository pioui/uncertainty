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

from cifar10_config import (
    dataset,
    project_name,
    images_dir,
    outputs_dir,
    compatibility_matrix,
    color,
)  

X, y = dataset.test_dataset  # 100000
y_true = y.reshape(-1)
acc_dict = []

for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            model_name = file.split("_")[-1].split(".")[0]

            y_pred_prob = np.load(os.path.join(outputs_dir, file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)

            # sort each class horizontally for better representation
            for i in range (0,10000,1000):
                y_pred_class = y_pred[i:i+1000]
                indxsort = np.argsort(y_pred_class)
                y_pred_class = y_pred_class[indxsort]
                y_pred_class = y_pred_class.reshape(10,100)
                y_pred_class = y_pred_class.reshape(100,10)
                y_pred_class = y_pred_class.transpose(1,0)
                y_pred_class = y_pred_class.reshape(1000)
                y_pred[i:i+1000] = y_pred_class

                y_pred_class_prob = y_pred_prob[i:i+1000,:]
                y_pred_class_prob= y_pred_class_prob[indxsort,:]
                y_pred_class_prob = y_pred_class_prob.reshape(10,100,-1)
                y_pred_class_prob = y_pred_class_prob.reshape(100,10,-1)
                y_pred_class_prob = y_pred_class_prob.transpose(1,0,2)
                y_pred_class_prob = y_pred_class_prob.reshape(1000,-1)
                y_pred_prob[i:i+1000] =  y_pred_class_prob              

            plt.figure(dpi=500)
            plt.imshow(
                y_pred.reshape(dataset.shape),
                interpolation="nearest",
                cmap=colors.ListedColormap(color),
            )
            plt.axis("off")
            plt.savefig(
                f"{images_dir}{model_name}_PREDICTIONS.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_PREDICTIONS.npy", y_pred)

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
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_GT.npy", y_true)

            
            y_gbu = geometry_based_uncertainty(y_pred_prob)
            plt.figure(dpi=500)
            plt.imshow(
                y_gbu.reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            # cbar = plt.colorbar(location="top")
            # cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_GBU.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_GBU.npy", y_gbu)


            y_variance = variance(y_pred_prob)
            plt.figure(dpi=500)
            plt.imshow(
                y_variance.reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            # cbar = plt.colorbar(location="top")
            # cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_VARIANCE.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_VARIANCE.npy", y_variance)

            y_entropy = shannon_entropy(y_pred_prob)
            plt.figure(dpi=500)
            plt.imshow(
                y_entropy.reshape(dataset.shape),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            # cbar = plt.colorbar(location="top")
            # cbar.ax.tick_params(labelsize=8)
            plt.savefig(
                f"{images_dir}{model_name}_ENTROPY.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,
            )
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_ENTROPY.npy", y_entropy)


            if os.path.isdir(f"{outputs_dir}uncertainty_npys/{project_name}_energy_distance.npy"):
                compatibility_matrix = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_energy_distance.npy")
            else:
                compatibility_matrix = calculate_compatibility_matrix(dataset.train_dataset, "energy")
                np.save(f"{outputs_dir}uncertainty_npys/{project_name}_energy_distance.npy", compatibility_matrix)

            y_sbu_energy = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)
            plt.figure(dpi=500)
            plt.imshow(
                y_sbu_energy.reshape(dataset.shape),
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
            np.save(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_SBU_ENERGY.npy", y_sbu_energy )

