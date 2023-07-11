"""
Script for generating uncertainty analysis plots for bcss dataset

Usage:
  python3 scripts/bcss_analysis.py 

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
from bcss_config import *

classifier_name = 'OptRF'
X, y = dataset.test_dataset  
y_true = y.reshape(-1)

# Load predicted probabilities from the classifier
predicted_probs = np.load(f'{classifications_dir}{dataset_name}_{classifier_name}.npy')
uncertainties_dir = f'{uncertainties_dir}{dataset_name}_{classifier_name}/'

# Load uncertainty scores
uncertainties_wrong = []
uncertainties_right = []

uncertainties = []
uncertainties_names_wrong = []
uncertainties_names_right = []

y_pred = np.argmax(predicted_probs, axis=1)+1

# Extract indices of correct and wrong predictions
# Class index of the class of interest
c = 3
misclassified_indices = np.where(y_pred!=y_true)[y_true==c]
wellclassified_indices = np.where(y_pred==y_true)[y_true==c]

total_misclassified = len(misclassified_indices)
total_wellclassified = len(wellclassified_indices)


for file in os.listdir(uncertainties_dir):
    uncertainty = np.load(os.path.join(uncertainties_dir,file))
    uncertainties.append(uncertainty)
    uncertainties_right = np.concatenate((uncertainties_right, uncertainty[wellclassified_indices]))
    uncertainties_wrong = np.concatenate((uncertainties_wrong, uncertainty[misclassified_indices]))
    
    if 'ENTROPY' in file:
        meas_name = 'ENTROPY'

    elif 'GBU_FR' in file:
        meas_name = r'$GU_{2|FR}$'

    elif 'VARIANCE' in file:
        meas_name = 'Var'
        
    elif 'SBU' in file:
        meas_name = r'$HU$'
        
    elif 'GBU' in file:
        meas_name = 'Gini-index'
        
    uncertainties_names_wrong = uncertainties_names_wrong + [meas_name]*total_misclassified
    uncertainties_names_right = uncertainties_names_right + [meas_name]*total_wellclassified
    

# Print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Plot ECDF corresponding to the wrong predictions
borderaxespad =-2
columnspacing = 1

data_wrong = pd.DataFrame(data = {"Uncertainties": uncertainties_wrong, "Measures": uncertainties_names_wrong})

sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'})
sns.set_theme(style="white")

pp = sns.ecdfplot(data=data_wrong, x="Uncertainties", hue = "Measures", hue_order= ["Var", r'$GU_{2|FR}$', "Entropy", 'Gini-index', r'$HU$']) #"Measures") #, y="Uncertainties") #, hue="Classification", data=data, hue_order= ["Right", "Wrong"])#, cut = 0, scale = 'area', split=True)
fig = pp.figure
for lines, marker, legend_handle in zip(pp.lines[::-1], ['*', 'o', '+', 's', '8'], pp.legend_.legend_handles): #, legend_handle, fig.legend.legendHandles
    lines.set_marker(marker)
    lines.set_markevery(0.1)
    legend_handle.set_marker(marker) 
plt.xlim((0,1))
plt.xlabel('Uncertainty')
sns.move_legend(pp, loc=9, ncol = 5, title = "",fontsize='medium', borderaxespad=borderaxespad, columnspacing = columnspacing)
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_misclassified-ecdf_class_{c}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()

# Plot ECDF corresponding to the correct predictions
data_right = pd.DataFrame(data = {"Uncertainties": uncertainties_right, "Measures": uncertainties_names_right})

sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) 
sns.set_theme(style="white")

pp = sns.ecdfplot(data=data_right, x="Uncertainties", hue = "Measures", hue_order= ["Var", r'$GU_{2|FR}$', "Entropy", 'Gini-index', r'$HU$']) 
fig = pp.figure
for lines, marker, legend_handle in zip(pp.lines[::-1], ['*', 'o', '+', 's', '8'], pp.legend_.legend_handles): 
    lines.set_marker(marker)
    lines.set_markevery(0.1)
    legend_handle.set_marker(marker)    
plt.xlim((0,1))
plt.xlabel('Uncertainty')
sns.move_legend(pp, loc=9, ncol = 5, title = "",fontsize='medium', borderaxespad=borderaxespad, columnspacing = columnspacing)
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_wellclassified-ecdf_class_{c}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()
