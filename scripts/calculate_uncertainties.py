"""
Script for calculate  uncertainties for all classifications on the outputs folder.

Usage:
  python3 scripts/calculate_uncertainties.py 

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


for dataset_name_ in os.listdir("outputs/"):
    if dataset_name_ == "trento":
        from trento_config import *

    elif dataset_name_ == "bcss":
        from bcss_config import *

    elif "signalModulation" in dataset_name_:
        from signalModulation_config import *
        if SNR != int(dataset_name_.split('_')[0].split('-')[-1]):
            print(f"Uncertainties will be calculated only for {dataset_name}, if you want another SNR for this dataset please change the configuration file")
            continue
    else:
        print(f'You need to implement the dataset and configuration for {dataset_name}')
        continue
    
    X, y = dataset.test_dataset
    y_true = y.reshape(-1)

    if not os.path.exists(compatibility_matrix_file):
        print(f"Calculating compatibility matrix ...")
        compatibility_matrix = calculate_compatibility_matrix(X[y_true!=0,:], y[y_true!=0], "energy", len(np.unique(y_true)))#[1:, 1:]
        np.save(compatibility_matrix_file, compatibility_matrix)
    else:
        compatibility_matrix= np.load(compatibility_matrix_file)
    print(f'Dataset: {dataset_name}')
    print(f" Ω_H = ")
    print(compatibility_matrix)

    print(f"Calculating uncertenties for {dataset_name} predictions...")

    for file in os.listdir(os.path.join(classifications_dir)):
        model_name = file.split(".")[0]
        
        if os.path.splitext(file)[-1].lower() == ".npy":
            y_pred_prob = np.load(os.path.join(classifications_dir, file))
        else:
            continue
        
        uncertainties_folder_dir = os.path.join(uncertainties_dir,model_name)
        if not os.path.exists(uncertainties_folder_dir):
            os.makedirs(uncertainties_folder_dir)
        
        GU = geometry_based_uncertainty(y_pred_prob)
        np.save(f"{uncertainties_folder_dir}/{model_name}_GBU.npy", GU)

        H = shannon_entropy(y_pred_prob)
        np.save(f"{uncertainties_folder_dir}/{model_name}_ENTROPY.npy", H)

        VAR = variance(y_pred_prob)
        np.save(f"{uncertainties_folder_dir}/{model_name}_VARIANCE.npy", VAR)
        
        GU_fr = FR_based_uncertainty(y_pred_prob)
        np.save(f"{uncertainties_folder_dir}/{model_name}_GBU_FR.npy", GU_fr)
        
        SU = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)
        np.save(f"{uncertainties_folder_dir}/{model_name}_SBU_energy.npy", SU)
        
    print("-------------------------------------------------------------------")