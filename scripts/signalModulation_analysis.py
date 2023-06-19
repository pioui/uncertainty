import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from signalModulation_config import *

X, y = dataset.test_dataset  
y_true = y.reshape(-1)
smooth = 100

# Load predicted probabilities from the classifier
classifier_name = 'CNN-calibrated'
predicted_probs = np.load(f'{classifications_dir}/{dataset_name}_{classifier_name}.npy')

# Load uncertainty scores for class 5 predictions
uncertainties = []
names = []
for file in os.listdir(uncertainties_dir):
    uncertainty = np.load(os.path.join(uncertainties_dir,file))
    uncertainties.append(uncertainty)
    if 'ENTROPY' in file:
        names.append('Entropy')
    elif 'GBU_FR' in file:
        names.append(r'$GU_{2|FR}$')
    elif 'VARIANCE' in file:
        names.append('Variance')
    elif 'SBU' in file:
        names.append(r'$SU_H$')
    elif 'GBU' in file:
        names.append('Gini-index')

y_pred = np.argmax(predicted_probs, axis=1)+1

# Plot distributions of uncertainties for misclassified points
misclassified_indices = np.where(y_pred!=y_true)[0]
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    density, bins = np.histogram(uncertainty[misclassified_indices]*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)
plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
plt.savefig(
    f"{images_dir}misclassified-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# Plot distributions of uncertainties for correct points
correct_classified_indices = np.where(y_pred==y_true)[0]
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    density, bins = np.histogram(uncertainty[correct_classified_indices]*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)
plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
plt.savefig(
    f"{images_dir}correct-classified-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# Plot distributions of uncertainties for all points
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    density, bins = np.histogram(uncertainty*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)
plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
plt.savefig(
    f"{images_dir}all-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)



