from uncertainty.compatibility_matrix import calculate_compatibility_matrix
from uncertainty.maximum_variance_utils import get_var_max_from_matrix, get_var_max, get_var_opt
from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
)

data = "bcss" # "trento" # 

if data == "bcss":
    from bcss_config import (
                dataset,
                project_name,
                images_dir,
                outputs_dir,
                compatibility_matrix,
                color,
            )
elif data == "trento":
    from trento_config import (
                dataset,
                project_name,
                images_dir,
                outputs_dir,
                compatibility_matrix,
                color,
            )
import os
import numpy as np

X, y = dataset.full_dataset  # 15107
#y_pred_prob = np.load(os.path.join(outputs_dir, f"{data}_RF.npy"))
#y_pred_max_prob = y_pred_prob.max(1)
#y_pred = y_pred_prob.argmax(1) + 1
np.set_printoptions(precision=2)
#print(compatibility_matrix)
#maxVar = get_var_max_from_matrix(compatibility_matrix)
#print(maxVar)
#print(max(semantic_based_uncertainty(y_pred_prob, compatibility_matrix)))
#get_var_max(5)
#compatibility_matrix1 = calculate_compatibility_matrix(X, y, "wasserstein")[1:, 1:]
#print(np.sum(compatibility_matrix, axis=1))
#compatibility_matrix = np.delete(np.delete(compatibility_matrix1[1:, 1:], 2, 0), 2, 1)
#maxVar = get_var_max_from_matrix(compatibility_matrix)
#print(maxVar)
#maxVar = get_var_opt(compatibility_matrix)
#print(maxVar)
# print(max(semantic_based_uncertainty(y_pred_prob, compatibility_matrix)))
#print(y[y!=0].shape)
#print(X[y!=0,:].shape)

compatibility_matrix = calculate_compatibility_matrix(X, y, "energy")[1:, 1:]

print(compatibility_matrix/np.amax(compatibility_matrix))
#print(np.sum(compatibility_matrix, axis=1))
#maxVar = get_var_max_from_matrix(compatibility_matrix)
#print(maxVar)
#maxVar = get_var_opt(compatibility_matrix)
#print(maxVar)
# maxVar = get_var_max_from_matrix(compatibility_matrix)
# print(maxVar)
# print(max(semantic_based_uncertainty(y_pred_prob, compatibility_matrix)))
