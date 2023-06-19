"""
Configuration file for signalModulation dataset

"""

import logging
import os
import numpy as np
from scipy.io import loadmat

from uncertainty.datasets import signalModulation_dataset

SNR = 15

dataset_name = f"signalModulation-SNR-{SNR}"
outputs_dir = f"outputs/{dataset_name}/"
images_dir = f"{outputs_dir}images/"
classifications_dir = f"{outputs_dir}classifications/"
uncertainties_dir = f"{outputs_dir}uncertainties/"
compatibility_matrix_file = f"{outputs_dir}{dataset_name}_omegaH.npy"

data_dir = f"/home/pigi/data/modulation_classification/MCNet_SNR_{SNR}/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(uncertainties_dir):
    os.makedirs(uncertainties_dir)

if not os.path.exists(classifications_dir):
    os.makedirs(classifications_dir)

dataset = signalModulation_dataset(data_dir=data_dir)

logging.basicConfig(filename=f"{outputs_dir}{dataset_name}_logs.log")

labels = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]
color = ["#22181c", "#073B4C", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2", "#EF476F", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Dataset name: {dataset_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")


preds_file_here = f"{classifications_dir}/{dataset_name}_CNN-calibrated.npy"
preds_file = os.path.join(data_dir, f"MCNet_SNR_{SNR}_cal_preds.npy")

if not os.path.exists(preds_file_here):
    rxTestcalScores = np.load(preds_file)
    np.save(preds_file_here, rxTestcalScores)