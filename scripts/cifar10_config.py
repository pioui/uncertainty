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

compatibility_matrix = np.array([
        [0., 2., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 0., 3., 3., 3., 3., 3., 3., 2., 1.],
        [3., 3., 0., 2., 2., 2., 2., 2., 3., 3.],
        [3., 3., 2., 0., 2., 1., 2., 2., 3., 3.],
        [3., 3., 2., 2., 0., 2., 2., 1., 3., 3.],
        [3., 3., 2., 1., 3., 0., 2., 2., 3., 3.],
        [3., 3., 2., 2., 3., 2., 0., 2., 3., 3.],
        [3., 3., 2., 2., 1., 2., 2., 0., 3., 3.],
        [2., 2., 3., 3., 3., 3., 3., 3., 0., 2.],
        [2., 1., 3., 3., 3., 3., 3., 3., 2., 0.]
    ]
)/10

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(labels)
color = ["palegreen", "lime", "orchid", "steelblue","red", "gray", "blue", "orange", "green", "yellow"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")
