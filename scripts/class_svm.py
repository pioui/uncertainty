
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix
import logging
from dask_ml.wrappers import ParallelPostFit
import dask.array as da

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

# ----- SVM -----#
print("Fitting SVM ...")

if dataset_name == "bcss":
    clf_svm = ParallelPostFit(estimator=SVC(probability=True),
                        scoring='accuracy')
else:
    clf_svm = SVC(probability=True)
    
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)

np.save(f"{outputs_dir}{project_name}_clf_svm.npy", clf_svm)

print("Predicting SVM ...")

y_pred = clf_svm.predict(X_test)
if isinstance(y_pred, da.Array):
    y_pred = y_pred.compute()
    
svm_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
svm_confusion_matrix = np.around(
    svm_confusion_matrix.astype("float")
    / svm_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)
np.save(f"{outputs_dir}{project_name}_SVM_test.npy", y_pred)

print("Predicting Image SVM ...")

y_pred = clf_svm.predict_proba(X)
if isinstance(y_pred, da.Array):
    y_pred = y_pred.compute()

    
logger.info(f"SVM Accuracy : {svm_accuracy}")
logger.info(f"Confusion Matrix: {svm_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_SVM_confusion_matrix.csv",
    svm_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_SVM.npy", y_pred)

