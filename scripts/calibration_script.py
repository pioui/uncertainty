# Load the packages
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io

# Calibration functions from
# https://github.com/markus93/NN_calibration/tree/master

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from scipy.optimize import minimize 
from betacal import BetaCalibration

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)

class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=10):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
def get_bin_info(conf, pred, true, bin_size = 0.1):

    """
    Get accuracy, confidence and elements in bin information for all the bins.
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    accuracies = []
    confidences = []
    bin_lengths = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)
        
        
    return accuracies, confidences, bin_lengths
      
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)
            

def get_pred_conf(y_probs, normalize = True):
    
    y_preds = np.argmax(y_probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        y_confs = np.max(y_probs, axis=1)/np.sum(y_probs, axis=1)
    else:
        y_confs = np.max(y_probs, axis=1)  # Take only maximum confidence
        
    return y_preds, y_confs

def get_predictions(method, y_logits_val, y_val, y_logits_test, y_test, M = 10, name = "", approach = "single", m_kwargs = {}):
    
    bin_size = 1/M
  
    #FILE_PATH = join(path, file)
    #(y_logits_val, y_val), (y_logits_test, y_test) = unpickle_probs(FILE_PATH)

    y_probs_val = softmax(y_logits_val)  # Softmax logits
    y_probs_test = softmax(y_logits_test)
    
    if approach == "single":
        
        K = y_probs_val.shape[1]
        # Go through all the classes
        for k in range(K):
            # Prep class labels (1 fixed true class, 0 other classes)
            #print(np.array(y_val == k, dtype="int"))
            y_cal = np.array(y_val == k, dtype="int")#[:, 0]

            # Train model
            model = method(**m_kwargs)
            #print(y_probs_val[:, k].shape, y_cal.shape)
            model.fit(y_probs_val[:, k], y_cal) # Get only one column with probs for given class "k"

            y_probs_val[:, k] = model.predict(y_probs_val[:, k])  # Predict new values based on the fittting
            y_probs_test[:, k] = model.predict(y_probs_test[:, k])

            # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
            idx_nan = np.where(np.isnan(y_probs_test))
            y_probs_test[idx_nan] = 0

            idx_nan = np.where(np.isnan(y_probs_val))
            y_probs_val[idx_nan] = 0
            
            y_preds_val, y_confs_val = get_pred_conf(y_probs_val, normalize = True)
            y_preds_test, y_confs_test = get_pred_conf(y_probs_test, normalize = True)
    
    else:
        model = method(**m_kwargs)
        model.fit(y_logits_val, y_val)

        y_probs_val = model.predict(y_logits_val) 
        y_probs_test = model.predict(y_logits_test)
    
    return y_probs_test, y_probs_val

def cal_res(method, y_logits_val, y_val, y_logits_test, y_test, M = 10, name = "", approach = "single", m_kwargs = {}):
    
    bin_size = 1/M
  
    y_probs_test, y_probs_val = get_predictions(method, y_logits_val, y_val, y_logits_test, y_test, M = M, name = name, approach = approach, m_kwargs = m_kwargs)
    y_preds_val, y_confs_val = get_pred_conf(y_probs_val, normalize = False)
    y_preds_test, y_confs_test = get_pred_conf(y_probs_test, normalize = False)
    
    accs_val, confs_val, len_bins_val = get_bin_info(y_confs_val, y_preds_val, y_val, bin_size = bin_size)
    accs_test, confs_test, len_bins_test = get_bin_info(y_confs_test, y_preds_test, y_test, bin_size = bin_size)
    
    return (accs_test, confs_test, len_bins_test), (accs_val, confs_val, len_bins_val) #

def get_uncalibrated_res(y_logits_test, y_test, M = 10):
    
    bin_size = 1/M

    #FILE_PATH = join(path, file)
    #(y_logits_val, y_val), (y_logits_test, y_test) = unpickle_probs(FILE_PATH)

    y_probs_test = softmax(y_logits_test)
    y_preds_test, y_confs_test = get_pred_conf(y_probs_test, normalize = False)
    
    return get_bin_info(y_confs_test, y_preds_test, y_test, bin_size = bin_size)

# reliability diagram plotting for subplot case.
def rel_diagram_sub(accs, confs, ax, M = 10, name = "Reliability Diagram", xname = "", yname=""):

    acc_conf = np.column_stack([accs,confs])
    acc_conf.sort(axis=1)
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1/M
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2)

    # Next add error lines
#    for i in range(M):
#        plt.plot([i/M,1], [0, (M-i)/M], color = "red", alpha=0.5, zorder=1)

    #Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "blue", label="Outputs", zorder = 3)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0,1], [0,1], linestyle = "--")
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(name, fontsize=24)
    ax.set_xlabel(xname, fontsize=22, color = "black")
    ax.set_ylabel(yname, fontsize=22, color = "black")

def gen_plots(y_probs_val, y_val, y_probs_test, y_test, plot_names =  [], M = 10, val_set = False):
    
    if val_set:  # Plot Reliability diagrams for validation set
        k = 1
    else:
        k = 0

        
    bin_info_uncal = get_uncalibrated_res(y_probs_val, y_val, M)

    accs_confs = []
    accs_confs.append(cal_res(TemperatureScaling,y_probs_val, y_val, y_probs_test, y_test, M, "", "multi"))
    accs_confs.append(cal_res(HistogramBinning, y_probs_val, y_val, y_probs_test, y_test, M, "", "single", {'M':M}))
    accs_confs.append(cal_res(IsotonicRegression, y_probs_val, y_val, y_probs_test, y_test, M, "", "single", {'y_min':0, 'y_max':1}))
    accs_confs.append(cal_res(BetaCalibration, y_probs_val, y_val, y_probs_test, y_test, M, "", "single", {'parameters':"abm"}))
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(22.5, 4), sharex='col', sharey='row')
    names = [" (Uncal)", " (Temp)", " (Histo)", " (Iso)", " (Beta)"]
    
    # Uncalibrated information
    rel_diagram_sub(bin_info_uncal[0], bin_info_uncal[1], ax[0] , M = M, name = names[0], xname="Confidence")
    for j in range(4):
        rel_diagram_sub(accs_confs[j][k][0], accs_confs[j][k][1], ax[j+1] , M = M, name = names[j+1], xname="Confidence")

    ax[0].set_ylabel("Accuracy", color = "black")
    
    for ax_temp in ax:    
        plt.setp(ax_temp.get_xticklabels(), rotation='horizontal', fontsize=18)
        plt.setp(ax_temp.get_yticklabels(), fontsize=18)

    plt.savefig("Mod_30SNR.pdf", format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.2)
    plt.show()

    return accs_confs

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)
    
def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def evaluate_model(y_true, y_probs, bins = 10, verbose = True):
    """
    Evaluates the model, in addition calculates the calibration errors and 
    
    Parameters:
        model (keras.model): constructed model
        weights (string): path to weights file
        x_test: (numpy.ndarray) with test data
        y_test: (numpy.ndarray) with test data labels
        verbose: (boolean) print out results or just return these
        x_test: (numpy.ndarray) with validation data
        y_test: (numpy.ndarray) with validation data labels

        
    Returns:
        (acc, ece, mce): accuracy of model, ECE and MCE (calibration errors)
    """
    
    y_preds = np.argmax(y_probs, axis=1)
    # Find accuracy and error
    accuracy = accuracy_score(y_true, y_preds) * 100
    error = 100 - accuracy
    
    # Confidence of prediction
    y_confs = np.max(y_probs, axis=1)  # Take only maximum confidence
    
    # Calculate ECE
    ece = ECE(y_confs, y_preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(y_confs, y_preds, y_true, bin_size = 1/bins)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
            
    # Return the basic results
    return (accuracy, ece, mce)

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
      
def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)

modelnoise = "MCNet_SNR_50" 
datanoise = 'MCNet_SNR_15'
mcdirectory = '/home/pigi/data/modulation_classification'
# Load labels and predictions from .mat files

y_test_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxTestLabelsNumbers.mat')
y_test_pred_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxTestPredNumbers.mat')
X_test_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxTestFrames.mat')

y_train_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxTrainLabelsNumbers.mat')
X_train_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxTrainFrames.mat')

y_val_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxValidLabelsNumbers.mat')
X_val_mat = scipy.io.loadmat(f'{mcdirectory}/{datanoise}/rxValidFrames.mat')

# Extract labels and predictions from the loaded data
lb = LabelBinarizer()
X_test = X_test_mat['rxTestFrames']
y_test = y_test_mat['rxTestLabelsNumbers'][:,0] #lb.fit_transform(y_test_mat['rxTestLabelsNumbers'])
y_test_pred = y_test_pred_mat['rxTestPredNumbers'][:,0]

print(np.unique(y_test))
X_train = X_train_mat['rxTrainFrames']
y_train = y_train_mat['rxTrainLabelsNumbers'][:,0]

X_val = X_val_mat['rxValidFrames']
y_val = y_val_mat['rxValidLabelsNumbers'][:,0] #lb.transform(y_val_mat['rxValidLabelsNumbers'])

## Get logits from the model

# Load the model
#! Change the model with the one you trained your data on
model = keras.models.load_model(f"{mcdirectory}/{modelnoise}/{modelnoise}_tfmodel", compile = False)

# Change last activation to linear (instead of softmax)
last_layer = model.layers.pop()
last_layer.activation = keras.activations.linear
i = model.input
o = last_layer(model.layers[-2].output)
model_2 = keras.models.Model(inputs=i, outputs=[o])

# First load in the weights
weights = model.get_weights()
model_2.set_weights(weights)
#model_2.load_weights("outputs/mod/MCNet_SNR_30/trainedNet/weights.h5")
model_2.compile(optimizer="sgd", loss="categorical_crossentropy")

# Next get predictions for X_val
#! Apply predictions on new data point
y_logits_val = model_2.predict(np.moveaxis(X_val,-1,0), verbose=1)
print(y_logits_val.shape)
y_probs_val = softmax(y_logits_val)
y_preds_val = np.argmax(y_probs_val, axis=1)
y_true_val = y_val-1

# Next get predictions for X_test
y_logits_test = model_2.predict(np.moveaxis(X_test,-1,0), verbose=1)
y_probs_test = softmax(y_logits_test)
y_preds_test = np.argmax(y_probs_test, axis=1)
y_true_test = y_test-1

#print(y_logits_test.shape)
#print(y_test.shape)
#exit()
print(accuracy_score(y_true_test, y_test_pred))
print(y_preds_test)
print(y_test)
print(accuracy_score(y_true_test, y_preds_test))
print(y_probs_test.shape)

# Evaluate the uncalibrated model
print("----- Uncalibrated Model -----")
evaluate_model(y_true_test, y_probs_test)
sumprobsum = np.sum(y_probs_test, axis=1)
print(sumprobsum.shape)
print(sumprobsum[sumprobsum>1.01].shape)
print(sumprobsum[sumprobsum<0.99].shape)

# Run and evaluate the calibration methods
M = 10
print("----- Temperature Scaling -----")
y_probs_test_temp, _ = get_predictions(TemperatureScaling, y_logits_val, y_true_val, y_logits_test, y_true_test, M, "", "multi")
evaluate_model(y_true_test, y_probs_test_temp)

print("----- HistogramBinning -----")
y_probs_test_hist, _ = get_predictions(HistogramBinning, y_logits_val, y_true_val, y_logits_test, y_true_test, M, "", "single", {'M':M})
evaluate_model(y_true_test, y_probs_test_hist)
sumprobsum = np.sum(y_probs_test_hist, axis=1)
print(sumprobsum.shape)
print(sumprobsum[sumprobsum>1.01].shape)
print(sumprobsum[sumprobsum<0.99].shape)

print("----- IsotonicRegression -----")
y_probs_test_iso, _ = get_predictions(IsotonicRegression, y_logits_val, y_true_val, y_logits_test, y_true_test, M, "", "single", {'y_min':0, 'y_max':1})
evaluate_model(y_true_test, y_probs_test_iso)
sumprobsum = np.sum(y_probs_test_iso, axis=1)
print(sumprobsum.shape)
print(sumprobsum[sumprobsum>1.01].shape)
print(sumprobsum[sumprobsum<0.99].shape)

print("----- BetaCalibration -----")
y_probs_test_beta, _ = get_predictions(BetaCalibration, y_logits_val, y_true_val, y_logits_test, y_true_test, M, "", "single", {'parameters':"abm"})
evaluate_model(y_true_test, y_probs_test_beta)
sumprobsum = np.sum(y_probs_test_beta, axis=1)
print(sumprobsum.shape)
print(sumprobsum[sumprobsum>1.01].shape)
print(sumprobsum[sumprobsum<0.99].shape)

print("----- Isotonic divided with the sum -----")

probsums = np.sum(y_probs_test_iso, axis=1)
norm_y_probs_test_iso = np.array([probline/probsum for probsum,probline in zip(probsums,y_probs_test_iso)])
print(norm_y_probs_test_iso.shape)
sumprobsum = np.sum(norm_y_probs_test_iso, axis=1)
print(sumprobsum.shape)
print(sumprobsum[sumprobsum>1.01].shape)
print(sumprobsum[sumprobsum<0.99].shape)
print(sumprobsum[sumprobsum==1].shape)
evaluate_model(y_true_test, norm_y_probs_test_iso)


# Save the results
#! Change the name
np.save(f"{mcdirectory}/{modelnoise}/{modelnoise}_{datanoise}_cal_preds_sum1.npy", norm_y_probs_test_iso)

# Plot reliability histograms
accs_confs = gen_plots(y_logits_val, y_true_val, y_logits_test, y_true_test)

