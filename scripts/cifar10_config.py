import logging
import os
import numpy as np

from uncertainty.datasets import cifar10_dataset
from uncertainty.utils import unpickle

project_name = "cifar10"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
data_dir = "/home/pigi/data/cifar-10-batches-py/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

dataset = cifar10_dataset(data_dir=data_dir)

logging.basicConfig(filename=f"{outputs_dir}{project_name}_logs.log")

# compatibility_matrix = np.array([
#         [0., 2., 3., 3., 3., 3., 3., 3., 2., 2.],
#         [2., 0., 3., 3., 3., 3., 3., 3., 2., 1.],
#         [3., 3., 0., 2., 2., 2., 2., 2., 3., 3.],
#         [3., 3., 2., 0., 2., 1., 2., 2., 3., 3.],
#         [3., 3., 2., 2., 0., 2., 2., 1., 3., 3.],
#         [3., 3., 2., 1., 3., 0., 2., 2., 3., 3.],
#         [3., 3., 2., 2., 3., 2., 0., 2., 3., 3.],
#         [3., 3., 2., 2., 1., 2., 2., 0., 3., 3.],
#         [2., 2., 3., 3., 3., 3., 3., 3., 0., 2.],
#         [2., 1., 3., 3., 3., 3., 3., 3., 2., 0.]
#     ]
# )/10

compatibility_matrix = np.array(
[ 
[0, 0.76, 0.66, 0.76, 0.94, 0.72, 1, 0.67, 0.39, 0.61],
[0.76, 0, 0.44, 0.41, 0.58, 0.47, 0.51, 0.39, 0.58, 0.33],
[ 0.66, 0.49, 0, 0.27, 0.31, 0.31, 0.4, 0.24, 0.63, 0.57],
[ 0.76, 0.41, 0.27, 0, 0.35, 0.22, 0.33, 0.28, 0.73, 0.6],
[0.94, 0.58, 0.31, 0.35, 0, 0.39, 0.24, 0.36, 0.87, 0.74],
[0.72, 0.47, 0.31, 0.22, 0.38, 0, 0.4, 0.33, 0.7, 0.66],
[1, 0.51, 0.40, 0.33, 0.24, 0.40, 0, 0.44, 0.86, 0.69],
[0.67, 0.39, 0.24, 0.28, 0.36, 0.33, 0.48, 0, 0.66, 0.52],
[0.39, 0.58, 0.63,0.73, 0.87, 0.7, 0.86, 0.66, 0, 0.48],
[0.61, 0.33, 0.57, 0.60, 0.74, 0.66, 0.69, 0.52, 0.48, 0]
] 
)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(labels)
color = ["palegreen", "lime", "orchid", "steelblue","red", "gray", "blue", "orange", "green", "yellow"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")
