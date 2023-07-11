"""
Script for generating uncertainty analysis plots for signal modulation dataset

Usage:
  python3 scripts/signalModulation_analysis.py 

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from signalModulation_config import *
from scipy.stats import skew


print(f"Analysis for {dataset_name} tested on {classifier_name}, if you want another model on dataset please change the configuration file")


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
uncertainties_colors = []

y_pred = np.argmax(predicted_probs, axis=1)+1

# Extract indices of correct and wrong predictions
misclassified_indices = np.where(y_pred!=y_true)[0]
wellclassified_indices = np.where(y_pred==y_true)[0]
total_misclassified = len(misclassified_indices)
total_wellclassified = len(wellclassified_indices)


for file in os.listdir(uncertainties_dir):
    uncertainty = np.load(os.path.join(uncertainties_dir,file))
    uncertainties.append(uncertainty)
    uncertainties_right = np.concatenate((uncertainties_right, uncertainty[wellclassified_indices]))
    uncertainties_wrong = np.concatenate((uncertainties_wrong, uncertainty[misclassified_indices]))
    if 'ENTROPY' in file:
        uncertainties_names_wrong = uncertainties_names_wrong + ['Entropy']*total_misclassified
        uncertainties_names_right = uncertainties_names_right + ['Entropy']*total_wellclassified
        uncertainties_colors.append('b')
    elif 'GBU_FR' in file:
        uncertainties_names_wrong = uncertainties_names_wrong + [r'$GU_{2|FR}$']*total_misclassified
        uncertainties_names_right = uncertainties_names_right + [r'$GU_{2|FR}$']*total_wellclassified
        uncertainties_colors.append('g')
    elif 'VARIANCE' in file:
        uncertainties_names_wrong = uncertainties_names_wrong + ['Var']*total_misclassified
        uncertainties_names_right = uncertainties_names_right + ['Var']*total_wellclassified
        uncertainties_colors.append('m')
    elif 'SBU' in file:
        uncertainties_names_wrong = uncertainties_names_wrong + [r'$HU$']*total_misclassified
        uncertainties_names_right = uncertainties_names_right + [r'$HU$']*total_wellclassified
        uncertainties_colors.append('r')
    elif 'GBU' in file:
        uncertainties_names_wrong = uncertainties_names_wrong + ['Gini-index']*total_misclassified
        uncertainties_names_right = uncertainties_names_right + ['Gini-index']*total_wellclassified
        uncertainties_colors.append('c')


data_wrong = pd.DataFrame(data = {"Uncertainties": uncertainties_wrong, "Measures": uncertainties_names_wrong})

# Print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Plot distributions of uncertainties for misclassified points
borderaxespad =-2
columnspacing = 1

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
    f"{images_dir}{dataset_name}_{classifier_name}_misclassified-ecdf.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()

data_right = pd.DataFrame(data = {"Uncertainties": uncertainties_right, "Measures": uncertainties_names_right})

sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) #, 'figure.facecolor':'white'
sns.set_theme(style="white")

pp = sns.kdeplot(data=data_right, x="Uncertainties", hue = "Measures", hue_order= ["Var", r'$GU_{2|FR}$', "Entropy", 'Gini-index', r'$HU$']) #"Measures") #, y="Uncertainties") #, hue="Classification", data=data, hue_order= ["Right", "Wrong"])#, cut = 0, scale = 'area', split=True)
fig = pp.figure
            
plt.xlim((0,1))
plt.xlabel('Uncertainty')
sns.move_legend(pp, loc = "best", title = "", fontsize='medium')
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_wellclassified-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()


#plt.figure(figsize=(3, 6))
sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) #, 'figure.facecolor':'white'
sns.set_theme(style="white")

pp = sns.ecdfplot(data=data_right, x="Uncertainties", hue = "Measures", hue_order= ["Var", r'$GU_{2|FR}$', "Entropy", 'Gini-index', r'$HU$']) #"Measures") #, y="Uncertainties") #, hue="Classification", data=data, hue_order= ["Right", "Wrong"])#, cut = 0, scale = 'area', split=True)
fig = pp.figure
for lines, marker, legend_handle in zip(pp.lines[::-1], ['*', 'o', '+', 's', '8'], pp.legend_.legend_handles): #, legend_handle, fig.legend.legendHandles
    lines.set_marker(marker)
    lines.set_markevery(0.1)
    legend_handle.set_marker(marker)    
plt.xlim((0,1))
plt.xlabel('Uncertainty')
sns.move_legend(pp, loc=9, ncol = 5, title = "",fontsize='medium', borderaxespad=borderaxespad, columnspacing = columnspacing)
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_wellclassified-ecdf.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()
