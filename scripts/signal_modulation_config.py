"""
Configuration file for signal_modulation dataset

"""

import logging
import os
import numpy as np
from scipy.io import loadmat

from uncertainty.datasets import signal_modulation_dataset

project_name = "signal_modulation_SNR_50"
images_dir = f"outputs/{project_name}/images/"
outputs_dir = f"outputs/{project_name}/"
# data_dir = "/work/saloua/Datasets/signal_modulation/"
# data_dir = "/home/pigi/data/signal_modulation/"
data_dir = "/media/pigi/ACD8F9B4D8F97D40/Users/User/Documents/modulation_classification/MCNet_SNR_50/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

dataset = signal_modulation_dataset(data_dir=data_dir)

logging.basicConfig(filename=f"{outputs_dir}{project_name}_logs.log")

# TODO: Define manual compatibility matrix
compatibility_matrix = np.random.rand(11,11)

labels = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]
color = ["#22181c", "#073B4C", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2", "#EF476F", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Project name: {project_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")
logger.info(f"compatibilityy Matrix: {compatibility_matrix}")

rxTestPredNumbers = np.load(os.path.join(data_dir, "rxTestPredNumbers.mat"))
rxTestScores = loadmat(os.path.join(data_dir, "rxTestScores.mat"))["rxTestScores"]
np.save(f"{outputs_dir}{project_name}_CNN.npy", rxTestScores)

rxTestcalScores = np.load(os.path.join(data_dir, "MCNet_SNR_50_cal_preds.npy"))
np.save(f"{outputs_dir}{project_name}_CNN_calibrated.npy", rxTestcalScores)