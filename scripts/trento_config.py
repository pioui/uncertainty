import logging
import os
import numpy as np

from uncertainty.datasets import trento_dataset


project_name = "trento"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
data_dir = "/work/saloua/Datasets/Trento/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

samples_per_class = 200
dataset = trento_dataset(data_dir=data_dir, samples_per_class=samples_per_class)

logging.basicConfig(filename=f"{outputs_dir}{project_name}_logs.log")


compatibility_matrix = np.array(
    [
        [0, 3, 3, 2, 1, 3],
        [3, 0, 3, 3, 3, 2],
        [3, 3, 0, 3, 3, 2],
        [2, 3, 3, 0, 2, 3],
        [1, 3, 3, 2, 0, 3],
        [3, 2, 2, 3, 3, 0],
    ]
)/3#/10

compatibility_matrix1 = np.array(
    [
        [0, 3, 4, 2, 1, 3],
        [3, 0, 3, 3, 3, 2],
        [4, 3, 0, 3, 4, 2],
        [2, 3, 3, 0, 2, 3],
        [1, 3, 4, 2, 0, 3],
        [3, 2, 2, 3, 3, 0],
    ]
)/4#/10

labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
#color = ["black", "red", "gray", "blue", "orange", "green", "yellow"]
#color = ["#22181c", "#22181c", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1", "#3b429f"]
color = ["#22181c", "#073B4C", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2", "#EF476F"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")
