import numpy as np
import matplotlib.pyplot as plt
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", help="name of dataset to use (trento, bcss)", default="bcss"
)

args = parser.parse_args()
dataset_name = args.dataset

if dataset_name == "bcss":
    from bcss_config import (
        dataset, 
        images_dir, 
        compatibility_matrix
        )

elif dataset_name == "trento":
    from trento_config import (
        dataset, 
        images_dir, 
        compatibility_matrix
        )


X, y = dataset.full_dataset
print(X.shape, y.shape, np.unique(y))
print(dataset.n_classes)

compatibility_matrix = compatibility_matrix / 10
plt.figure(dpi=500)
plt.matshow(compatibility_matrix, cmap="cool")
plt.xticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
plt.yticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
for k in range(len(compatibility_matrix)):
    for l in range(len(compatibility_matrix[k])):
        plt.text(
            k,
            l,
            str(compatibility_matrix[k][l]),
            va="center",
            ha="center",
            fontsize="small",
        )  # trento
plt.savefig(
    f"{images_dir}{dataset_name}_compatibility_manual.png",
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=500,
)

compatibility_matrix = calculate_compatibility_matrix(X, y, "wasserstein")[1:, 1:]
print(compatibility_matrix)
compatibility_matrix = np.around(compatibility_matrix.astype("float"), decimals=2)
plt.figure(dpi=500)
plt.matshow(compatibility_matrix, cmap="cool")
plt.xticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
plt.yticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
for k in range(len(compatibility_matrix)):
    for l in range(len(compatibility_matrix[k])):
        plt.text(
            k,
            l,
            str(compatibility_matrix[k][l]),
            va="center",
            ha="center",
            fontsize="small",
        )  # trento
plt.savefig(
    f"{images_dir}{dataset_name}_compatibility_wasserstein.png",
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=500,
)

compatibility_matrix = calculate_compatibility_matrix(X, y, "energy")[1:, 1:]
print(compatibility_matrix)
compatibility_matrix = np.around(compatibility_matrix.astype("float"), decimals=2)
plt.figure(dpi=500)
plt.matshow(compatibility_matrix, cmap="cool")
plt.xticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
plt.yticks(np.arange(0, dataset.n_classes - 1, 1), range(1, dataset.n_classes))
for k in range(len(compatibility_matrix)):
    for l in range(len(compatibility_matrix[k])):
        plt.text(
            k,
            l,
            str(compatibility_matrix[k][l]),
            va="center",
            ha="center",
            fontsize="small",
        )  # trento
plt.savefig(
    f"{images_dir}{dataset_name}_compatibility_energy.png",
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=500,
)
