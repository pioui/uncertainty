import numpy as np
from sklearn.model_selection import train_test_split
import logging
import random
import imageio
from uncertainty.utils import normalize
from sklearn.ensemble import IsolationForest

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

label_schema = np.array(
    [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # wanted labels
)

def get_high_density_data(x, y):
    '''
    @param x : np.array(N,D) N data points x D channels
    @param y : np.array(N) true labels of the data points
    @return : high density samples of the dataset
    '''

    C = np.unique(y)
    x_new = np.array([])
    y_new = np.array([])
    for c in C:
        #Get high density data points to remove outliers
        new_labels = IsolationForest(contamination = 0.5, random_state = 42).fit_predict(x[y==c,:])
        if x_new.size == 0:
            x_new = x[y==c,:][new_labels==1,:]
            y_new = y[y==c][new_labels==1]
        else:
            x_new = np.concatenate((x_new, x[y==c,:][new_labels==1,:]))
            y_new = np.concatenate((y_new, y[y==c][new_labels==1]))
    return x_new, y_new

class bcss_dataset:
    def __init__(
        self, data_dir, do_preprocess=True
        ) -> None:
        super().__init__()
        print(data_dir)
        x = np.array(
            imageio.imread(
                f"{data_dir}BCSS/images/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"
            )
        )  # [6941, 5342, 3]
        y = np.array(
            imageio.imread(
                f"{data_dir}BCSS/masks/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"
            ),
            dtype=np.int64,
        )  # [6941, 5342]
        self.shape = y.shape

        x_all = x.reshape(-1, x.shape[-1])  # [37078822, 3]

        y_all = y
        y_all = y_all.reshape(-1)  # [37078822]
        y_all = label_schema[y_all.reshape(-1)] 

        if do_preprocess:
            x_all = normalize(x_all)

        x_train, x_test, y_train, y_test = train_test_split(
                x_all[y_all!=0,:],
                y_all[y_all!=0],
                train_size=0.02,
                random_state=42,
            )

        x_test, _, y_test, _ = train_test_split(
                x_test, 
                y_test,
                train_size=0.02,
                random_state=42,
            )

        # Get high density data
        x_train, y_train = get_high_density_data(x_train, y_train)
        x_test, y_test = get_high_density_data(x_test, y_test)

        self.n_classes = len(np.unique(y_all))


        self.train_dataset = (x_train, y_train)  
        logger.info(
            f"Train dataset shape: {x_train.shape}, {y_train.shape}, {np.unique(y_train)}"
        )
        for l in np.unique(y_train):
            logger.info(f"Label {l}: {np.sum(y_train==l)}")

        self.test_dataset = (x_test, y_test )  
        logger.info(
            f"Test dataset shape: {x_test.shape}, {y_test.shape}, {np.unique(y)}"
        )
        for l in np.unique(y_test):
            logger.info(f"Label {l}: {np.sum(y_test==l)}")

        self.full_dataset = (x_all, y_all)  # 0 to 5
        logger.info(f"Dataset shape: {x_all.shape}, {y_all.shape}, {np.unique(y_all)}")
        for l in np.unique(y_all):
            logger.info(f"Label {l}: {np.sum(y_all==l)}")


if __name__ == "__main__":

    DATASET = bcss_dataset(data_dir = "/media/data/saloua/")

    x, y = DATASET.full_dataset   
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.train_dataset  
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")

    x, y = DATASET.test_dataset   
    print(x.shape, y.shape, np.unique(y))
    for l in np.unique(y):
        print(f"Label {l}: {np.sum(y==l)}")
