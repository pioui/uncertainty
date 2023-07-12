"""
Script for calculate  uncertainties for all classifications on the outputs folder.

Usage:
  python3 scripts/calculate_uncertainties.py 

"""
import os
import numpy as np

from uncertainty.H_matrix import calculate_H_matrix
from uncertainty.uncertainty_measurements import GU, HU, variance


for dataset_name in os.listdir("outputs/"):
    if dataset_name == "trento":
        from trento_config import *

    elif dataset_name == "bcss":
        from bcss_config import *

    elif "signalModulation" in dataset_name:
        from signalModulation_config import *
        print(f"Uncertainties will be calculated only for {dataset_name}, if you want another SNR for this dataset please change the configuration file")
    else:
        print(f'You need to implement the dataset and configuration for {dataset_name} and import it here')
        continue
    
    X, y = dataset.test_dataset
    y_true = y.reshape(-1)

    if not os.path.exists(H_matrix_file):
        print(f"Calculating H matrix ...")
        H = calculate_H_matrix(X[y_true!=0,:], y[y_true!=0], len(np.unique(y_true)))#[1:, 1:]
        np.save(H_matrix_file, H)
    else:
        H = np.load(H_matrix_file)

    # Normalized:
    # maxco = np.max(H)
    # H= H/maxco

    print(f'Dataset: {dataset_name}')
    print(f" H = ")
    print(H)

    print(f"Calculating uncertainties for {dataset_name} predictions...")

    for file in os.listdir(os.path.join(classifications_dir)):
        model_name = file.split(".")[0]
        
        if os.path.splitext(file)[-1].lower() == ".npy":
            y_pred_prob = np.load(os.path.join(classifications_dir, file))
        else:
            continue
        
        uncertainties_folder_dir = os.path.join(uncertainties_dir,model_name)
        if not os.path.exists(uncertainties_folder_dir):
            os.makedirs(uncertainties_folder_dir)

        print('Classification shape:', y_pred_prob.shape)
        
        y_GU = GU(y_pred_prob, d = "euclidean", n = 2)
        np.save(f"{uncertainties_folder_dir}/{model_name}_GBU.npy", y_GU)

        y_H = GU(y_pred_prob, d = "kullbackleibler", n = 1)
        np.save(f"{uncertainties_folder_dir}/{model_name}_ENTROPY.npy", y_H)

        y_VAR = variance(y_pred_prob)
        np.save(f"{uncertainties_folder_dir}/{model_name}_VARIANCE.npy", y_VAR)
        
        y_GU_fr = GU(y_pred_prob, d = "fisherrao", n = 2)
        np.save(f"{uncertainties_folder_dir}/{model_name}_GBU_FR.npy", y_GU_fr)
        
        y_SU = HU(y_pred_prob, H)
        np.save(f"{uncertainties_folder_dir}/{model_name}_SBU_energy.npy", y_SU)
        
    print("-------------------------------------------------------------------")
