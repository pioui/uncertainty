"""
Script for generating uncertainty analysis plots for signal modulation dataset

Usage:
  python3 scripts/signalModulation_analysis.py 

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from signalModulation_config import *


print(f"Analysis for {dataset_name} tested on {classifier_name}, if you want another model on dataset please change the configuration file")


X, y = dataset.test_dataset  
y_true = y.reshape(-1)

# Load predicted probabilities from the classifier
predicted_probs = np.load(f'{classifications_dir}{dataset_name}_{classifier_name}.npy')
uncertainties_dir = f'{uncertainties_dir}{dataset_name}_{classifier_name}/'

# Load uncertainty scores for class 5 predictions
uncertainties = []
uncertainties_names = []
uncertainties_colors = []
for file in os.listdir(uncertainties_dir):
    uncertainty = np.load(os.path.join(uncertainties_dir,file))
    uncertainties.append(uncertainty)
    if 'ENTROPY' in file:
        uncertainties_names.append('Entropy')
        uncertainties_colors.append('b')
    elif 'GBU_FR' in file:
        uncertainties_names.append(r'$GU_{2|FR}$')
        uncertainties_colors.append('g')
    elif 'VARIANCE' in file:
        uncertainties_names.append('Variance')
        uncertainties_colors.append('m')
    elif 'SBU' in file:
        uncertainties_names.append(r'$SU_H$')
        uncertainties_colors.append('r')
    elif 'GBU' in file:
        uncertainties_names.append('Gini-index')
        uncertainties_colors.append('c')

y_pred = np.argmax(predicted_probs, axis=1)+1



# Print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Plot distributions of uncertainties for misclassified points
misclassified_indices = np.where(y_pred!=y_true)[0]
total_misclassified = len(misclassified_indices)

plt.figure(figsize=(3, 6))
for i,uncertainty in enumerate(uncertainties):
    density, bins = np.histogram(uncertainty[misclassified_indices], range = (0,1.1), bins=11, density=False)
    plt.plot(bins[:-1], density/total_misclassified, color=uncertainties_colors[i], alpha = 0.5)
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(uncertainties_names)
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_misclassified-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# Plot distribution of the top 10% high uncertainty pixels.
# limit = int(len(y_pred)*0.1)


max_uncertainities = []
plt.figure(figsize=(8, 6))
for i,uncertainty in enumerate(uncertainties):
#     sorted_indices = np.argsort(uncertainty)
#     sorted_uncertainty = uncertainty[sorted_indices]
#     top_uncertainty_indices = sorted_indices[-limit:]

    max_uncertainty = np.max(uncertainty)
    top_uncertainty_indices = uncertainty >= 0.8*max_uncertainty
    max_uncertainities.append(max_uncertainty)

    top_uncertainty_y_pred = y_pred[top_uncertainty_indices]
    top_uncertainty_y_true = y_true[top_uncertainty_indices]
    top_uncertainty_y_pred_probs = predicted_probs[top_uncertainty_indices]

    top_uncertainty_accuracy = accuracy_score(top_uncertainty_y_true, top_uncertainty_y_pred)
    print(f"Highest {uncertainties_names[i]}: ", max_uncertainty)
    print(f"Accuracy of the highest {uncertainties_names[i]} pixels: ", top_uncertainty_accuracy)

    total_top_uncertainty_y_pred = len(top_uncertainty_y_pred)
    density, bins = np.histogram(top_uncertainty_y_pred, range = (0.5,11.5), bins=11, density=False)
    plt.plot(labels, density/total_top_uncertainty_y_pred, marker = ".", color=uncertainties_colors[i], alpha = 0.5)

plt.legend(uncertainties_names)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.ylim((0,1.1))
plt.savefig(
    f"{images_dir}{dataset_name}_{classifier_name}_top_uncertainty_barplot.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)



# Plot distribution of probabilities for the most appearing class in the 10% of highest uncertainties.
for i,uncertainty in enumerate(uncertainties):
    
    # sorted_indices = np.argsort(uncertainty)
    # sorted_uncertainty = uncertainty[sorted_indices]
    # top_uncertainty_indices = sorted_indices[-limit:]

    max_uncertainty = np.max(uncertainty)
    top_uncertainty_indices = uncertainty >= 0.8*max_uncertainty

    top_uncertainty_y_pred = y_pred[top_uncertainty_indices]
    top_uncertainty_y_true = y_true[top_uncertainty_indices]
    top_uncertainty_y_pred_probs = predicted_probs[top_uncertainty_indices]

    density, bins = np.histogram(top_uncertainty_y_pred, range = (0.5,11.5), bins=11, density=False)
    max_class = np.argmax(density)
    max_class_top_uncertainty_y_pred_probs = top_uncertainty_y_pred_probs[top_uncertainty_y_pred==max_class+1]

    plt.figure(figsize=(3, 6))
    for j in range(11):
        prob = max_class_top_uncertainty_y_pred_probs[:,j]
        total_prob = len(prob)
        density, bins = np.histogram(prob, range = (0,1.1), bins=11, density=False)
        plt.plot(bins[:-1], density/total_prob, color=color[j], alpha = 0.7)
        plt.ylim((0,1))
        plt.xlim((0,1))

    plt.xlabel(f"Probabilities of {labels[max_class]}")
    plt.legend(labels)
    plt.ylabel("Frequency")
    plt.savefig(
        f"{images_dir}{dataset_name}_{classifier_name}_top_{uncertainties_names[i]}_max_class_probs.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=500,
    )
 
