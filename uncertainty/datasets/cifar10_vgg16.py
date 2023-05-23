# CIFAR 10 https://www.cs.toronto.edu/~kriz/cifar.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import random
#import imageio

#from uncertainty.utils import normalize, unpickle

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class cifar10_vgg16_dataset:
    def __init__(
        self, data_dir, train_size=0.5, do_preprocess=True
    ) -> None:
        super().__init__()

        # number_of_batches = 5
        
        # x_train = []
        # y_train= []
        # for batch in range(1,number_of_batches+1):
        #     train_batch_dict = unpickle(data_dir+'/data_batch_'+str(batch))
        #     x_train.append(train_batch_dict[b'data'])
        #     y_train.append(train_batch_dict[b'labels'])

        # meta_dir = data_dir+"/batches.meta"
        # labels_dict = unpickle(meta_dir)
        # x_train = np.array(x_train) # [5, 10000, 3072]
        # y_train = np.array(y_train) # [5, 10000]
        # number_of_batch_samples = y_train.shape[-1]

        # x_train = x_train.reshape((number_of_batches*number_of_batch_samples, -1)) # [50000, 3072]
        # if do_preprocess:
        #     x_train = normalize(x_train)            

        # y_train = y_train.reshape((number_of_batches*number_of_batch_samples)) # [50000]
        # y_train = y_train.reshape(-1)
        
        dataset = np.load(f'{data_dir}CIFAR10_vgg16-keras_features.npz')
        x_train = dataset['features_training']
        x_test = dataset['features_testing']
        y_train = dataset['labels_training']
        y_test = dataset['labels_testing']
        
        self.n_classes = len(np.unique(y_train))

        # x_train, x_test, y_train, y_test = train_test_split(
        #     x_all_train,
        #     y_all_train,
        #     train_size=train_size,
        #     random_state=42,
        #     stratify=y_all_train,
        # )  

        # test_batch_dict = unpickle(data_dir+'/test_batch')
        # x_test = np.array(test_batch_dict[b'data'])
        # y_test = np.array(test_batch_dict[b'labels'])

        # x_test = x_test.reshape((number_of_batch_samples, -1)) # [10000, 3072]
        # if do_preprocess:
        #     x_test = normalize(x_test)

        # y_test = y_test.reshape((number_of_batch_samples)) # [10000]
        # y_test = y_test.reshape(-1)

        x_all = np.concatenate((x_train,x_test))
        y_all = np.concatenate((y_train,y_test))

        if train_size:
            x_train, _, y_train, _ = train_test_split(
                x_train,
                y_train,
                train_size=train_size,
                random_state=42,
                stratify=y_train,
            )  

        self.train_dataset = (x_train, y_train)  
        logger.info(
            f"Train dataset shape: {x_train.shape}, {y_train.shape}, {np.unique(y_train)}"
        )
        for l in np.unique(y_train):
            logger.info(f"Label {l}: {np.sum(y_train==l)}")

        self.test_dataset = (x_test, y_test )  
        logger.info(
            f"Test dataset shape: {x_test.shape}, {y_test.shape}, {np.unique(y_test)}"
        )
        for l in np.unique(y_test):
            logger.info(f"Label {l}: {np.sum(y_test==l)}")
        
        self.full_dataset = (x_all, y_all)  # 0 to 5
        logger.info(f"Dataset shape: {x_all.shape}, {y_all.shape}, {np.unique(y_all)}")
        for l in np.unique(y_all):
            logger.info(f"Label {l}: {np.sum(y_all==l)}")


if __name__ == "__main__":

    DATASET = cifar10_vgg16_dataset(data_dir="/work/saloua/uncertainty-main/outputs/cifar10/")

    x, y = DATASET.train_dataset  
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.test_dataset   
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.full_dataset   
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")