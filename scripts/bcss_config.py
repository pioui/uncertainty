import logging
import numpy as np

from uncertainty.datasets import bcss_Dataset
images_dir = "/home/pigi/repos/BCSS/images/"
outputs_dir = "/home/pigi/repos/BCSS/outputs/"
data_dir = "/home/pigi/repos/BCSS/"

PROJECT_NAME = "BCSS_Uncertainty"

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

DATASET = bcss_Dataset(
        data_dir = data_dir,
    )

