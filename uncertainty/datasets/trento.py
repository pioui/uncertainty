from scipy import io
import tifffile
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random

from uncertainty.utils import normalize

random.seed(42)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class trento_dataset():
    def __init__(
        self,
        data_dir,
        samples_per_class = 200,
        train_size=0.5,
        do_preprocess=True,
    ) -> None:
        super().__init__()

        image_hyper = np.array(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
        image_lidar = np.array(tifffile.imread(data_dir+"LiDAR_Italy.tif")) # [2,166,600]
        x = np.concatenate((image_hyper,image_lidar), axis = 0) # [65,166,600]
        x_all = x
        x_all = x_all.reshape(len(x_all),-1)
        x_all = x_all.transpose(1,0) # [99600,65]
        
        #Normalize to [0,1]
        if do_preprocess: 
            # x_all = normalize(x_all).astype(float)
            x_all = normalize(x_all)

        y = np.array(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = np.int64) # [166,600] 0 to 6
        y_all = y
        y_all = y_all.reshape(-1) # [99600]

        print(x_all.shape, y_all.shape)
        print(f"HERE{np.unique(y_all)}")

        train_inds = []
        for label in np.unique(y_all):
            label_ind = np.where(y_all == label)[0]
            samples = samples_per_class
            if label == 0:
                continue
            else:
                labelled_exs = np.random.choice(label_ind, size=samples, replace=True)

            train_inds.append(labelled_exs)
        train_inds = np.concatenate(train_inds)

        x_all_train = x_all[train_inds]
        y_all_train = y_all[train_inds]
        
        x_train, x_test, y_train, y_test = train_test_split(
            x_all_train, y_all_train, train_size = train_size, random_state = 42, stratify = y_all_train
        ) # 0 to 5

        self.train_dataset = (x_train, y_train) # 1 to 5
        print(x_train.shape, y_train.shape, np.unique(y_train))
        for l in np.unique(y_train):
            print(f'Label {l}: {np.sum(y_train==l)}')

        self.test_dataset = (x_test, y_test) # 1 to 5
        print(x_test.shape, y_test.shape, np.unique(y))
        for l in np.unique(y_test):
            print(f'Label {l}: {np.sum(y_test==l)}')

        self.full_dataset = (x_all, y_all) # 1 to 5
        print(x_all.shape, y_all.shape, np.unique(y_all))
        for l in np.unique(y_all):
            print(f'Label {l}: {np.sum(y_all==l)}')


if __name__ == "__main__":

    DATASET = trentoDataset(
        data_dir = "/home/pigi/data/trento/",
    )
    x,y = DATASET.train_dataset # 819
    print(x.shape, y.shape, np.unique(y))

    x,y = DATASET.test_dataset # 15107
    print(x.shape, y.shape, np.unique(y))

    x,y = DATASET.full_dataset # 15107
    print(x.shape, y.shape, np.unique(y))