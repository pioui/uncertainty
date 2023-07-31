"""
Configuration file for Breast Cancer Segmenation sataset

"""

import logging
import os
import numpy as np

from uncertainty.datasets import bcss_dataset

dataset_name = "bcss"
outputs_dir = f"outputs/{dataset_name}/"
images_dir = f"{outputs_dir}images/"
classifications_dir = f"{outputs_dir}classifications/"
uncertainties_dir = f"{outputs_dir}uncertainties/"
H_matrix_file = f"{outputs_dir}{dataset_name}_H.npy"

# data_dir = "/work/saloua/Datasets/bcss/"
data_dir = "/home/pigi/data/bcss/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

samples_per_class = 100
dataset = bcss_dataset(data_dir=data_dir)

logging.basicConfig(filename=f"{outputs_dir}{dataset_name}_logs.log")

labels = [
    "Unknown",
    "Tumor",
    "Stroma",
    "Lymphocytic", # Lymphocytic infiltrate",
    "Necrosis", # or Debris",
    "Other",
]

color = ["#22181c", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1", "#3b429f"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {dataset_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
