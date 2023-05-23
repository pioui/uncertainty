import logging
import os
import numpy as np

from uncertainty.datasets import bcss_patched_dataset

project_name = "bcss_patched"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
data_dir = "/work/saloua/uncertainty-main/datasets/"
patch_size = 7

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

samples_per_class = 100
dataset = bcss_patched_dataset(data_dir=data_dir, samples_per_class=samples_per_class, patch_size=patch_size)

logging.basicConfig(filename=f"{outputs_dir}{project_name}_logs.log")

compatibility_matrix = np.array(
    [
        [0, 1, 1, 1, 3],
        [1, 0, 1, 1, 3],
        [1, 1, 0, 1, 3],
        [1, 1, 1, 0, 3],
        [3, 3, 3, 3, 0],
    ]
)/10

compatibility_matrix1 = np.array(
    [
        [0, 2, 2, 2, 3],
        [2, 0, 1, 1, 3],
        [2, 1, 0, 1, 3],
        [2, 1, 1, 0, 3],
        [3, 3, 3, 3, 0],
    ]
)/10

labels = [
    "Unknown",
    "Tumor",
    "Stroma",
    "Lymphocytic", #_infiltrate",
    "Necrosis", # or Debris",
    "Other",
]

#color = ["black", "palegreen", "lime", "orchid", "green", "steelblue"]
#color = ["#22181c", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1", "#3b429f"]
color = ["#22181c", "#EF476F", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2"]#, "#073B4C"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Patch size: {patch_size} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")
