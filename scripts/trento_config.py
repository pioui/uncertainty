import logging
import os

from uncertainty.datasets import trento_dataset

project_name = "trento"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
data_dir = "/home/pigi/data/trento/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

samples_per_class = 200
dataset = trento_dataset(
        data_dir = data_dir,
        samples_per_class=samples_per_class
    )

logging.basicConfig(filename = f'{outputs_dir}{project_name}_logs.log')
