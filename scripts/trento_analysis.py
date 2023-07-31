"""
Script for generating uncertainty analysis plots for trento dataset

Usage:
  python3 scripts/trento_analysis.py 

"""

from sklearn.metrics import accuracy_score

import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from trento_config import *
from sklearn.ensemble import IsolationForest

def get_hsi_signature(x, y):
    '''
    Return mean and standard deviation of hyperspectral signatures for each class
    @param x : np.array(N,D) N data points x D channels
    @param y : np.array(N) true labels of the data points
    @return : mean and standard deviation of hyperspectral signatures for each class
    '''

    # Remove lidar's channels
    x = x[:,:-2]
    C = np.unique(y)
    mean_sig = np.zeros((len(C), x.shape[1]))
    std_sig = np.zeros((len(C), x.shape[1]))
    for c in C:
        #Get high density data points to remove outliers
        new_labels = IsolationForest(contamination = 0.4).fit_predict(x[y==c,:])
        x_new = x[y==c,:][new_labels==1,:]
        mean_sig[c-1, :] = np.mean(x_new, axis = 0)
        std_sig[c-1, :] = np.sqrt(np.var(x_new, axis = 0))
    return mean_sig, std_sig

def get_confused_classes(p, y_true, c_true, c_pred):
    '''
    Return indices of data points of label c_true 
    misclassified as c_pred and where the correct 
    class comes as second best estimate
    @param p : np.array(N,C) N data points x C probabilities each for each class
    @param y_true : np.array(N) true labels of the data points
    @c_true: label of the correct class 
    @c_pred: label of the predicted class
    @return : indices of data points misclassified as c_pred where the true class c_true comes as second best estimate
    '''
    K = np.array(p[y_true==c_true, :])
    KK = np.argsort(K, axis = 1)
    KKK = np.argsort(K[KK[:,-2] == c_true-1], axis = 1)
    return np.arange(len(p))[y_true==c_true][KK[:,-2] == c_true-1][KKK[:,-1] == c_pred-1]

X, y = dataset.full_dataset #train_dataset #  # 15107
y = y.reshape(-1)

classifier_name = "RF_OPT" # #other options are "SVM_OPT" and "SVM"

# Load predicted probabilities from the classifier
predicted_probs = np.load(f'{classifications_dir}{dataset_name}_{classifier_name}.npy')[y!=0, :]
y_true = y[y!=0]
C = np.unique(y_true)

uncertainties_dir = f'{uncertainties_dir}{dataset_name}_{classifier_name}/'

# Load uncertainty scores
uncertainties_wrong = []
uncertainties_right = []
uncertainties = []
uncertainties_names_wrong = []
uncertainties_names_right = []
uncertainties_names = []

y_pred = np.argmax(predicted_probs, axis=1)+1

# Get number of confused classes
number_classes_confused = np.sum(predicted_probs>1/len(C), axis = 1)

# Get indices of type of confusion
c_true = 6 # Correct class
c_pred1 = 2 # Class predicted
idx_1 = get_confused_classes(predicted_probs, y_true, c_true, c_pred1) #confusion between classes 6 and 1
c_pred2 = 4 # Class predicted
idx_2 = get_confused_classes(predicted_probs, y_true, c_true, c_pred2) #confusion between classes 6 and 3
c_pred3 = 5 # Class predicted
idx_3 = get_confused_classes(predicted_probs, y_true, c_true, c_pred3) #confusion between classes 6 and 4

conf_unc = [] 
conf_meas = [] 
conf_type = []


# Extract indices of correct and wrong predictions
misclassified_indices = np.where(y_pred!=y_true)[0]
wellclassified_indices = np.where(y_pred==y_true)[0]
total_misclassified = len(misclassified_indices)
total_wellclassified = len(wellclassified_indices)


for file in os.listdir(uncertainties_dir):
    uncertainty = np.load(os.path.join(uncertainties_dir,file))[y!=0]
    uncertainties = np.concatenate((uncertainties, uncertainty))
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
        # Get confused classes
        conf_unc = np.concatenate((conf_unc, uncertainty[idx_1], uncertainty[idx_2], uncertainty[idx_3]))
        conf_meas = conf_meas + [meas_name]*len(idx_1) + [meas_name]*len(idx_2) + [meas_name]*len(idx_3)
        conf_type = conf_type + [labels[c_pred1]]*len(idx_1) + [labels[c_pred2]]*len(idx_2) + [labels[c_pred3]]*len(idx_3)

        
    elif 'GBU' in file:
        meas_name = 'Gini-index' 
        # Get confused classes
        conf_unc = np.concatenate((conf_unc, uncertainty[idx_1], uncertainty[idx_2], uncertainty[idx_3]))
        conf_meas = conf_meas + [meas_name]*len(idx_1) + [meas_name]*len(idx_2) + [meas_name]*len(idx_3)
        conf_type = conf_type + [labels[c_pred1]]*len(idx_1) + [labels[c_pred2]]*len(idx_2) + [labels[c_pred3]]*len(idx_3)
    
        
    # Get correct and wrong predictions
    uncertainties_names_wrong = uncertainties_names_wrong + [meas_name]*total_misclassified
    uncertainties_names_right = uncertainties_names_right + [meas_name]*total_wellclassified
    uncertainties_names = uncertainties_names + [meas_name]*len(uncertainty)

# Print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Plot mean and standard deviation of uncertainties as a function of the number of confused classes
borderaxespad =-2
columnspacing = 1

data = pd.DataFrame(data = {"Uncertainties": uncertainties_names, "Measures": uncertainties, "Classes": list(number_classes_confused) * 5}) #, "Classification": labels, "classes": classes})
sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) #, 'figure.facecolor':'white'
sns.set_theme(style="white")

pp = sns.pointplot(data=data, x="Classes", y="Measures", hue="Uncertainties", errorbar="sd", dodge=True, hue_order= ["Var", r"$GU_{2|FR}$", "entropy", "Gini-index", r"$SU_\mathcal{H}$"])
fig = pp.figure
sns.move_legend(pp, loc=9, ncol = 5, title = "",fontsize='medium', borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"
plt.xlabel("Number of confused classes")
plt.ylabel("Uncertainty")
fig.savefig(f"{images_dir}{dataset_name}_{classifier_name}_number_classes.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,)
plt.close()

# Plot mean and standard deviation of homophily uncertainty as a function of the type of confused classes

data = pd.DataFrame(data = {"Uncertainties": conf_meas, "Measures": conf_unc, "Classes": conf_type}) #, "Classification": labels, "classes": classes})
sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) #, 'figure.facecolor':'white'
sns.set_theme(style="white")

pp = sns.pointplot(data=data, x="Classes", y="Measures", hue="Uncertainties", errorbar="sd", dodge=True)
fig = pp.figure
pp.set(ylim=(-0.2, 1.1))
sns.move_legend(pp, loc="best", ncol = 2, title = "",fontsize='medium')#, borderaxespad=borderaxespad, columnspacing = columnspacing) #, mode = "expand"
plt.xlabel("")
plt.ylabel("Uncertainty")
fig.savefig(f"{images_dir}{dataset_name}_{classifier_name}_confused_classes.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,)
plt.close()

# Plot ECDF corresponding to the wrong predictions

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
    f"{images_dir}{dataset_name}_{classifier_name}_misclassified-ecdf_class.png",
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
    f"{images_dir}{dataset_name}_{classifier_name}_wellclassified-ecdf_class.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
plt.close()

# Outliers plot
idx = 12823 #Index of data point of interest
point = X[y!=0, :-2][idx,:] #Data point of interest
pred_point = predicted_probs[idx,:] #Posterior probability of the point of interest
pred_point = pred_point/np.max(pred_point)

mean_sig, std_sig = get_hsi_signature(X, y)

wav = np.arange(0.40289, 0.98909, 0.0093)[:-1]
for j in range(0,6):
    plt.plot(wav, mean_sig[j,:], label = labels[j+1], color = color[j])
    plt.fill_between(wav, mean_sig[j,:]-std_sig[j,:], mean_sig[j,:]+std_sig[j,:], alpha = 0.5, color = color[j])
plt.legend(loc=9, ncol = 3, borderaxespad=borderaxespad, columnspacing = columnspacing)
plt.xlabel(r'[$\mu$m]')
plt.savefig(f"{images_dir}{dataset_name}_signatures_mean.png")
plt.close()

for j in range(0,6):
    plt.plot(wav, mean_sig[j,:], label = labels[j+1], alpha = pred_point[j], color = color[j])
    plt.fill_between(wav, mean_sig[j,:]-std_sig[j,:], mean_sig[j,:]+std_sig[j,:], alpha = pred_point[j]/2, color = color[j])
plt.plot(wav, np.transpose(point), '--', label = "Data point", color = "black")
plt.legend(loc=9, ncol = 4, borderaxespad=borderaxespad, columnspacing = columnspacing)
plt.xlabel(r'[$\mu$m]') #, fontsize=15
plt.savefig(f"{images_dir}{dataset_name}_{classifier_name}_signatures_outlier.png")
plt.close()