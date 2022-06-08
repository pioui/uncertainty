import logging
import numpy as np

from uncertainty.datasets import bcss_Dataset
images_dir = "/home/pigi/repos/BCSS/images/"
outputs_dir = "/home/pigi/repos/BCSS/outputs/"
data_dir = "/home/pigi/repos/BCSS/"

samples_per_class = 10
project_name = "BCSS_Uncertainty"

logging.basicConfig(filename = f'{outputs_dir}{project_name}_logs.log')

DATASET = bcss_Dataset(
        data_dir = data_dir,
        samples_per_class=samples_per_class
    )

