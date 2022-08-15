import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random
import imageio
from sklearn.feature_extraction import image
from uncertainty.utils import normalize

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

label_schema = np.array(
    [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # wanted labels
)


class bcss_patched_dataset:
    def __init__(
        self, data_dir, samples_per_class=500, train_size=0.5, do_preprocess=True,
        patch_size = 5
    ) -> None:
        super().__init__()

        x = np.array(
            imageio.imread(
                data_dir + "images/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"
            )
        )  # [6941, 5342, 3]
        y = np.array(
            imageio.imread(
                data_dir + "masks/TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500.png"
            ),
            dtype=np.int64,
        )  # [6941, 5342]
        self.shape = y.shape

        x_all = x.reshape(-1, x.shape[-1])  # [37078822, 3]

        if do_preprocess:
            x_all = normalize(x_all)

        # TODO: generalize
        x_all = x.reshape(6941, 5342, -1)  # [6941, 5342, 3]

        # Patching

        x_padded = np.pad(x_all, ((int(patch_size/2),),(int(patch_size/2),),(0,)), 'reflect') # [65,166+p/2, 600+p/2]
        x_patched = image.extract_patches_2d(x_padded, (patch_size, patch_size))
        x_all = x_patched.reshape(x_patched.shape[0],-1)

        y_all = y
        y_all = y_all.reshape(-1)  # [37078822]
        y_all = label_schema[y_all.reshape(-1)]  # [37078822] 0 to 5

        self.n_classes = len(np.unique(y_all))

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
            x_all_train,
            y_all_train,
            train_size=train_size,
            random_state=42,
            stratify=y_all_train,
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


if __name__ == "__main__":

    DATASET = bcss_patched_dataset(data_dir="/home/pigi/repos/BCSS/")

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
