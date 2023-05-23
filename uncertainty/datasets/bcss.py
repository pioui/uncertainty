import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random
import imageio

#from uncertainty.utils import normalize

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

label_schema = np.array(
    [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # wanted labels
)


class bcss_dataset:
    def __init__(
        self, data_dir, samples_per_class=500, train_size=0.001, do_preprocess=True
    ) -> None:
        super().__init__()

        x = np.array(imageio.v2.imread(f"{data_dir}images/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"))
        F = np.load("/work/saloua/uncertainty-main/outputs/stats_bcss.npy")[:,:,:,:6]
        for i in range(3):
            x = np.concatenate((x, F[i,:,:,:]), axis = -1)
        
        y = np.array(imageio.v2.imread(f"{data_dir}masks/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"), dtype=np.int64)
        
        #y = np.load(f"{data_dir}masks/labels_bcc.npy")
        self.shape = y.shape

        # x_all = np.moveaxis(x, -1, 0) # [3, 2260, 2545]
        x_all = x.reshape(-1, x.shape[-1])  # [6439719,3]

        if do_preprocess:
            x_all = normalize(x_all)
        y_all = y
        y_all = y_all.reshape(-1)  # [6439719]
        y_all = label_schema[y_all.reshape(-1)]  # [6439719] 0 to 5

        self.n_classes = len(np.unique(y_all))

        # train_inds = []
        # for label in np.unique(y_all):
        #     label_ind = np.where(y_all == label)[0]
        #     samples = samples_per_class
        #     if label == 0:
        #         continue
        #     else:
        #         labelled_exs = np.random.choice(label_ind, size=samples, replace=True)

        #     train_inds.append(labelled_exs)
        # train_inds = np.concatenate(train_inds)

        # x_all_train = x_all[train_inds]
        # y_all_train = y_all[train_inds]

        x_train, x_test, y_train, y_test = train_test_split(
            x_all[y_all!=0,:],
            y_all[y_all!=0],
            train_size=train_size,
            random_state=42,
            stratify=y_all[y_all!=0],
        )  

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

if __name__ == "__main__":

    DATASET = bcss_dataset(data_dir="/work/saloua/uncertainty-main/datasets/")

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
