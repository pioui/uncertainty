"""
Script for Generating Uncertainty Distribution plots: Signal Modulation Dataset

Usage:
  python3 scripts/calculate_uncertainties_distributions.py 

"""
import os
from sklearn.metrics import accuracy_score
import numpy as np

from uncertainty.compatibility_matrix import calculate_compatibility_matrix
from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
    FR_based_uncertainty,
)

from signal_modulation_config import *

X, y = dataset.test_dataset  
y_true = y.reshape(-1)

for file in os.listdir(os.path.join(outputs_dir)):
    model_name = file.split("_")[-1].split(".")[0]

    if os.path.splitext(file)[-1].lower() == ".npy":
        y_pred_prob = np.load(os.path.join(outputs_dir, file))
    else:
        continue

    y_pred = y_pred_prob.argmax(1) + 1

    print(accuracy_score(y_pred,y_true))

    # Claculate uncertainties and save them
    GU = geometry_based_uncertainty(y_pred_prob)
    np.save(f"{images_dir}{model_name}_GBU.npy", GU)

    H = shannon_entropy(y_pred_prob)
    np.save(f"{images_dir}{model_name}_ENTROPY.npy", H)

    VAR = variance(y_pred_prob)
    np.save(f"{images_dir}{model_name}_VARIANCE.npy", VAR)

    GU_fr = FR_based_uncertainty(y_pred_prob)
    np.save(f"{images_dir}{model_name}_GBU_FR.npy", GU_fr)

    compatibility_matrix = calculate_compatibility_matrix(X[y_true!=0,:], y[y_true!=0], "energy", len(np.unique(y_pred)))#[1:, 1:]
    print(compatibility_matrix)
    
    SU = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)
    np.save(f"{images_dir}{model_name}_SBU_energy.npy", SU)



