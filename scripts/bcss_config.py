import logging
import os
import numpy as np

from uncertainty.datasets import bcss_dataset

project_name = "bcss"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
data_dir = "/home/pigi/repos/BCSS/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

samples_per_class = 100
dataset = bcss_dataset(data_dir=data_dir, samples_per_class=samples_per_class)

logging.basicConfig(filename=f"{outputs_dir}{project_name}_logs.log")

compatibility_matrix = np.array(
    [
        [0, 1, 1, 1, 3],
        [1, 0, 1, 1, 3],
        [1, 1, 0, 1, 3],
        [1, 1, 1, 0, 3],
        [3, 3, 3, 3, 0],
    ]
)

labels = [
    "Unknown",
    "Tumor",
    "Stroma",
    "Lymphocytic_infiltrate",
    "Necrosis or Debris",
    "Other",
]
color = ["black", "palegreen", "lime", "orchid", "green", "steelblue"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")
