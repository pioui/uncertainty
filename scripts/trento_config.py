"""
Configuration file for Trento sataset

"""

import logging
import os
import numpy as np

from uncertainty.datasets import trento_dataset

dataset_name = "trento"
outputs_dir = f"outputs/{dataset_name}/"
images_dir = f"{outputs_dir}images/"
classifications_dir = f"{outputs_dir}classifications/"
uncertainties_dir = f"{outputs_dir}uncertainties/"
compatibility_matrix_file = f"{outputs_dir}{dataset_name}_omegaH.npy"

# data_dir = "/work/saloua/Datasets/Trento/"
data_dir = "/home/pigi/data/trento/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(uncertainties_dir):
    os.makedirs(uncertainties_dir)

if not os.path.exists(classifications_dir):
    os.makedirs(classifications_dir)

samples_per_class = 200
dataset = trento_dataset(data_dir=data_dir, samples_per_class=samples_per_class)

logging.basicConfig(filename=f"{outputs_dir}{dataset_name}_logs.log")

labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]

color = ["#22181c", "#073B4C", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2", "#EF476F"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Dataset name: {dataset_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
