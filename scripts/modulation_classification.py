import os
from scipy.io import loadmat
from sklearn.svm import SVC

# Specify the directory path containing .mat files
directory_path = "/home/pigi/data/signal_modulation"

# Load the .mat file
rxTrainFrames = loadmat(os.path.join(directory_path, "rxTrainFrames.mat"))["rxTrainFrames"][0]
rxTrainLabelsNumbers = loadmat(os.path.join(directory_path, "rxTrainLabelsNumbers.mat"))["rxTrainLabelsNumbers"][:,0]

rxTestFrames = loadmat(os.path.join(directory_path, "rxTestFrames.mat"))["rxTestFrames"][0]
rxTestLabelsNumbers = loadmat(os.path.join(directory_path, "rxTestLabelsNumbers.mat"))["rxTestLabelsNumbers"][:,0]

x_train = rxTrainFrames.transpose(2,0,1)
x_train = x_train.reshape(x_train.shape[0],-1)

x_test = rxTestFrames.transpose(2,0,1)
x_test = x_test.reshape(x_test.shape[0],-1)

# Print the size and type information
print("Shape rxTrainFrames:", rxTrainFrames.shape)
print("Shape rxTrainLabelsNumbers:", rxTrainLabelsNumbers.shape)
print("Shape rxTestFrames:", rxTestFrames.shape)
print("Shape rxTestLabelsNumbers:", rxTestLabelsNumbers.shape)


print("Shape x_train:", x_train.shape)
print("Shape x_test:", x_test.shape)


x_all = np