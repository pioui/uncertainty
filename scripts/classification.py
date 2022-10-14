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
    "--dataset", "-d", help="name of dataset to use (trento, bcss, etc)", default="bcss"
)

args = parser.parse_args()
dataset_name = args.dataset

if dataset_name == "bcss":
    from bcss_config import dataset, outputs_dir, project_name
elif dataset_name == "bcss_patched":
    from bcss_patched_config import dataset, outputs_dir, project_name
elif dataset_name == "trento":
    from trento_config import dataset, outputs_dir, project_name
elif dataset_name == "cifar10":
    from cifar10_config import dataset, outputs_dir, project_name

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
clf_svm = SVC(C=1, kernel="rbf", verbose=False, probability=True)
clf_svm.fit(X_train, y_train)

print("Predicting SVM ...")
svm_accuracy = clf_svm.score(X_test, y_test)
y_pred = clf_svm.predict(X_test)
svm_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
svm_confusion_matrix = np.around(
    svm_confusion_matrix.astype("float")
    / svm_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)

if dataset_name == "cifar10":
    y_pred = clf_svm.predict_proba(X_test)
else:
    y_pred = clf_svm.predict_proba(X)

logger.info(f"SVM Accuracy : {svm_accuracy}")
rf_accuracy = clf_svm.score(X_train, y_train)
logger.info(f"SVM training Accuracy : {svm_accuracy}")
logger.info(f"Cofusion Matrix: {svm_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_SVM_confusion_matrix.csv",
    svm_confusion_matrix,
    delimiter=",",
)
print(y_pred.shape)
np.save(f"{outputs_dir}{project_name}_SVM.npy", y_pred)

# # ----- RF -----#
# print("Fitting RF ...")
# clf_rf = RandomForestClassifier(
#     n_estimators=300,
#     criterion="gini",
#     min_samples_leaf=2,
#     max_depth=80,
#     min_samples_split=5,
#     bootstrap=True,
#     max_features="sqrt",
#     verbose=False,
# )
# clf_rf.fit(X_train, y_train)

# print("Predicting RF ...")
# rf_accuracy = clf_rf.score(X_test, y_test)
# y_pred = clf_rf.predict(X_test)
# rf_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
# rf_confusion_matrix = np.around(
#     rf_confusion_matrix.astype("float")
#     / rf_confusion_matrix.sum(axis=1)[:, np.newaxis],
#     decimals=2,
# )

# if dataset_name == "cifar10":
#     y_pred = clf_rf.predict_proba(X_test)
# else:
#     y_pred = clf_rf.predict_proba(X)


# logger.info(f"RF Accuracy : {rf_accuracy}")
# rf_accuracy = clf_rf.score(X_train, y_train)
# logger.info(f"RF training Accuracy : {rf_accuracy}")

# logger.info(f"Cofusion Matrix: {rf_confusion_matrix}")
# np.savetxt(
#     f"{outputs_dir}{project_name}_RF_confusion_matrix.csv",
#     rf_confusion_matrix,
#     delimiter=",",
# )
# print(y_pred.shape)
# np.save(f"{outputs_dir}{project_name}_RF.npy", y_pred)