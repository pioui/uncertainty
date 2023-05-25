from scipy import io
import tifffile
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random
from scipy.io import loadmat
import os

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def normalize(x):
    """
    @param x : array(channels, N)
    @return : array(L, N) normalized x at [0,1] in channels dimention
    """

    # logger.info("Normalize to 0,1")
    x_min = x.min(axis=0)[0]  # [57]
    x_max = x.max(axis=0)[0]  # [57]
    xn = (x - x_min) / (x_max - x_min)
    assert np.unique(xn.min(axis=0)[0] == 0.0)
    assert np.unique(xn.max(axis=0)[0] == 1.0)
    return xn

class signal_modulation_dataset:
    def __init__(
        self, data_dir, samples_per_class=200, train_size=0.2) -> None:
        super().__init__()

        # Load the .mat file
        rxTrainFrames = loadmat(os.path.join(data_dir, "rxTrainFrames.mat"))["rxTrainFrames"][0]
        rxTrainLabelsNumbers = loadmat(os.path.join(data_dir, "rxTrainLabelsNumbers.mat"))["rxTrainLabelsNumbers"][:,0]

        rxTestFrames = loadmat(os.path.join(data_dir, "rxTestFrames.mat"))["rxTestFrames"][0]
        rxTestLabelsNumbers = loadmat(os.path.join(data_dir, "rxTestLabelsNumbers.mat"))["rxTestLabelsNumbers"][:,0]

        x_train = rxTrainFrames.transpose(2,0,1)
        x_train = x_train.reshape(x_train.shape[0],-1)
        y_train = rxTrainLabelsNumbers.reshape(-1)

        x_test = rxTestFrames.transpose(2,0,1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        y_test = rxTestLabelsNumbers.reshape(-1)

        x_all = np.concatenate((x_train,x_test))
        y_all = np.concatenate((y_train,y_test))

        self.shape = y_all.shape
        self.n_classes = len(np.unique(y_all)) 

        self.train_dataset = (x_train, y_train)  # 1 to 5
        logger.info(
            f"Train dataset shape: {x_train.shape}, {y_train.shape}, {np.unique(y_train)}"
        )
        for l in np.unique(y_train):
            logger.info(f"Label {l}: {np.sum(y_train==l)}")

        self.test_dataset = (x_test, y_test)  # 1 to 5
        logger.info(
            f"Test dataset shape: {x_test.shape}, {y_test.shape}, {np.unique(y_test)}"
        )
        for l in np.unique(y_test):
            logger.info(f"Label {l}: {np.sum(y_test==l)}")

        self.full_dataset = (x_all, y_all)  # 1 to 5
        logger.info(f"Dataset shape: {x_all.shape}, {y_all.shape}, {np.unique(y_all)}")
        for l in np.unique(y_all):
            logger.info(f"Label {l}: {np.sum(y_all==l)}")


if __name__ == "__main__":

    DATASET = signal_modulation_dataset(data_dir="/home/pigi/data/signal_modulation/")

    x, y = DATASET.full_dataset  # [5731136] 0 to 20
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.train_dataset  # [1719340] -1 to 19
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.test_dataset  # [4011796] -1 to 19
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

