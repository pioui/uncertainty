import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from signal_modulation_config import *
X, y = dataset.test_dataset  
y_true = y.reshape(-1)
smooth = 100


# Load predicted probabilities and labels
predicted_probs = np.load(f'{outputs_dir}signal_modulation_SNR_{SNR}_CNN-calibrated.npy')

# Load uncertainty scores for class 5 predictions
uncertainties = []
names = []
for file in os.listdir(os.path.join(outputs_dir, "uncertainties-temp")):
    uncertainty = np.load(os.path.join(outputs_dir, "uncertainties-temp",file))
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



# Find points predicted as class 5 but actually belong to class 6
y_pred = np.argmax(predicted_probs, axis=1)+1

# class_7_indices = np.where(y_pred == 7)[0]
# class_not7_indices = np.where(y_true != 7)[0]
# misclassified_indices = np.intersect1d(class_7_indices, class_not7_indices)

misclassified_indices = np.where(y_pred!=y_true)[0]

print(misclassified_indices.shape)
# Plot distributions of uncertainties for misclassified points
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    # sns.kdeplot(uncertainty[misclassified_indices], common_norm=True, alpha=0.5)
    # plt.hist(uncertainty[misclassified_indices]*100, bins=100, range = (0,100), alpha=0.5, density=True)
    density, bins = np.histogram(uncertainty[misclassified_indices]*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)

plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
# plt.show()
plt.savefig(
    f"{images_dir}mis-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

misclassified_indices = np.where(y_pred==y_true)[0]
print(misclassified_indices.shape)
# Plot distributions of uncertainties for misclassified points
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    # sns.kdeplot(uncertainty[misclassified_indices],common_norm=True, alpha=0.5)
    # plt.hist(uncertainty[misclassified_indices]*100, bins=100, range = (0,100), alpha=0.5, density=True)
    density, bins = np.histogram(uncertainty[misclassified_indices]*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)
plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
# plt.show()
plt.savefig(
    f"{images_dir}cor-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# Plot distributions of uncertainties for misclassified points
plt.figure(figsize=(8, 6))
for uncertainty in uncertainties:
    # sns.kdeplot(uncertainty[misclassified_indices],common_norm=True, alpha=0.5)
    # plt.hist(uncertainty[misclassified_indices]*100, bins=100, range = (0,100), alpha=0.5, density=True)
    density, bins = np.histogram(uncertainty[misclassified_indices]*smooth, range = (0,smooth), bins=smooth, density=True)
    plt.plot(bins[:-1]/smooth, density)
plt.xlim((0,1))
plt.xticks((np.arange(0, 1, step=0.1)))
plt.ylim((0,0.4))
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend(names)
# plt.show()
plt.savefig(
    f"{images_dir}all-dist.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)



