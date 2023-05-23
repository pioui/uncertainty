
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", help="name of dataset to use (trento, bcss)", default="bcss"
)

args = parser.parse_args()
dataset_name = args.dataset

if dataset_name == "bcss":
    from bcss_config import dataset, outputs_dir, project_name
    C = 100
    gamma = 0.1
    n_estimators=300
    criterion="entropy"
    min_samples_leaf=1
    max_depth=10
    #min_samples_split=5,
    bootstrap=False
    max_features="sqrt"
    verbose=False
elif dataset_name == "bcss_patched":
    from bcss_patched_config import dataset, outputs_dir, project_name
elif dataset_name == "trento":
    from trento_config import dataset, outputs_dir, project_name
    C = 100
    gamma = 0.1
    n_estimators=100
    criterion="entropy"
    min_samples_leaf=4
    max_depth=None
    #min_samples_split=5,
    bootstrap=False
    max_features="sqrt"
    verbose=False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

X_train, y_train = dataset.train_dataset
X_test, y_test = dataset.test_dataset
X, y = dataset.full_dataset

logger.info(
    f"Train dataset shape: {X_train.shape}, {y_train.shape}, {np.unique(y_train)}"
)
logger.info(f"Test dataset shape: {X_test.shape}, {y_test.shape}, {np.unique(y_test)}")
logger.info(f"Total dataset shape: {X.shape}, {y.shape}, {np.unique(y)}")


# # # # ----- RF -----#

print("Fitting Optimal RF ...")
  
clf_rf = RandomForestClassifier(
    n_estimators=n_estimators,
    criterion=criterion,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    #min_samples_split=5,
    bootstrap=bootstrap,
    max_features=max_features,
    verbose=verbose,
    n_jobs = 5,
)

clf_rf.fit(X_train, y_train)
rf_accuracy = clf_rf.score(X_test, y_test)

print("Predicting Optimal RF ...")
y_pred = clf_rf.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
rf_confusion_matrix = np.around(
    rf_confusion_matrix.astype("float")
    / rf_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)

y_pred = clf_rf.predict_proba(X)
#np.save(f"{outputs_dir}{project_name}_RF_OPT_test.npy", y_pred)

logger.info(f"RF OPT Accuracy : {rf_accuracy}")
logger.info(f"Confusion Matrix: {rf_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_RF_OPT_confusion_matrix.csv",
    rf_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_RF_OPT_glcm.npy", y_pred)


