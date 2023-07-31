"""
Configuration file for signalModulation dataset

"""

import logging
import os
import numpy as np
from scipy.io import loadmat

from uncertainty.datasets import signalModulation_dataset

dataSNR = 15
modelSNR = 15

classifier_name = f'CNN-calibrated-SNR-{modelSNR}'
dataset_name = f"signalModulation-SNR-{dataSNR}"
outputs_dir = f"outputs/{dataset_name}/"
images_dir = f"{outputs_dir}images/"
classifications_dir = f"{outputs_dir}classifications/"
uncertainties_dir = f"{outputs_dir}uncertainties/"
H_matrix_file = f"{outputs_dir}{dataset_name}_H.npy"

data_predir = "/home/pigi/data/modulation_classification/"
data_dir = f"{data_predir}MCNet_SNR_{dataSNR}/"
 
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
color =   [ "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#9edae5",   # light blue
    "#d62728",  # red
]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"Dataset name: {dataset_name} ")
logger.info(f"Labels: {labels} ")
logger.info(f"Labels' colors: {color} ")

preds_file_here = f"{classifications_dir}/{dataset_name}_{classifier_name}.npy"
preds_file = os.path.join(f"{data_predir}MCNet_SNR_{modelSNR}/", f"MCNet_SNR_{modelSNR}_MCNet_SNR_{dataSNR}_cal_preds_sum1.npy")

if not os.path.exists(preds_file):
    print(f"Prediction file {preds_file} does not exist.")

if not os.path.exists(preds_file_here):
    rxTestcalScores = np.load(preds_file)
    np.save(preds_file_here, rxTestcalScores)

