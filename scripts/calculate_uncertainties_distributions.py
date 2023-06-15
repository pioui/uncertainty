"""
Script for Generating Uncertainty Maps: BCSS and Trento Datasets
With a focus on spatially represented data, these uncertainty maps provide valuable visual representations.

Usage:
  python3 scripts/calculate_uncertainties_distributions.py 

"""



for project_name in os.listdir("outputs/"):
    if "signal_modulation" in project_name:
        from signal_modulation_config import *


    X, y = dataset.full_dataset  
    y_true = y.reshape(-1)

    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            y_pred_prob = np.load(os.path.join(outputs_dir, file))
        else:
            continue
        print(y_pred_prob.shape)

        GU = geometry_based_uncertainty(y_pred_prob).reshape(dataset.shape)
        f"{images_dir}{model_name}_GBU.eps",

        H = shannon_entropy(y_pred_prob).reshape(dataset.shape)
        f"{images_dir}{model_name}_ENTROPY.eps",

        variance(y_pred_prob).reshape(dataset.shape),
        f"{images_dir}{model_name}_VARIANCE.eps",

        GU_fr = FR_based_uncertainty(y_pred_prob).reshape(dataset.shape)
        f"{images_dir}{model_name}_GBU_FR.eps",

        compatibility_matrix = calculate_compatibility_matrix(X[y_true!=0,:], y[y_true!=0], "energy", len(np.unique(y_pred)))#[1:, 1:]
        SU = semantic_based_uncertainty(y_pred_prob, compatibility_matrix).reshape(dataset.shape)
        f"{images_dir}{model_name}_SBU_energy.eps",
