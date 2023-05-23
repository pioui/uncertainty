
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix
import logging
from dask_ml.wrappers import ParallelPostFit

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

if dataset_name == "bcss":
    y_pred = clf_svm.predict(X_test)#.compute()
else:
    y_pred = clf_svm.predict(X_test)
    
# svm_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
# svm_confusion_matrix = np.around(
#     svm_confusion_matrix.astype("float")
#     / svm_confusion_matrix.sum(axis=1)[:, np.newaxis],
#     decimals=2,
# )
np.save(f"{outputs_dir}{project_name}_SVM_test.npy", y_pred)

print("Predicting Image SVM ...")

if dataset_name == "bcss":
    y_pred = clf_svm.predict_proba(X).compute()
else:
    y_pred = clf_svm.predict_proba(X)
    
logger.info(f"SVM Accuracy : {svm_accuracy}")
logger.info(f"Confusion Matrix: {svm_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_SVM_confusion_matrix.csv",
    svm_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_SVM.npy", y_pred)

# ----- SVM -----#
print("Fitting Optimal SVM ...")

if dataset_name == "bcss":
    clf_svm = ParallelPostFit(estimator=SVC(C=C, kernel="rbf", gamma = gamma,  verbose=False, probability=True),
                        scoring='accuracy')
else:
    clf_svm = SVC(C=C, kernel="rbf", gamma = gamma,  verbose=False, probability=True)

clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)

print("Predicting Optimal SVM ...")

if dataset_name == "bcss":
    y_pred = clf_svm.predict(X_test).compute()
else:
    y_pred = clf_svm.predict(X_test)
    
svm_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
svm_confusion_matrix = np.around(
    svm_confusion_matrix.astype("float")
    / svm_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)
np.save(f"{outputs_dir}{project_name}_SVM_OPT_test.npy", y_pred)

print("Predicting Image Optimal SVM ...")

if dataset_name == "bcss":
    y_pred = clf_svm.predict_proba(X).compute()
else:
    y_pred = clf_svm.predict_proba(X)
    
logger.info(f"SVM OPT Accuracy : {svm_accuracy}")
logger.info(f"Confusion Matrix: {svm_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_SVM_OPT_confusion_matrix.csv",
    svm_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_SVM_OPT.npy", y_pred)
# # # # ----- RF -----#

print("Fitting RF ...")
  
clf_rf = RandomForestClassifier(n_jobs = 5)

clf_rf.fit(X_train, y_train)
rf_accuracy = clf_rf.score(X_test, y_test)

print("Predicting RF ...")
y_pred = clf_rf.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
rf_confusion_matrix = np.around(
    rf_confusion_matrix.astype("float")
    / rf_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)

y_pred = clf_rf.predict_proba(X)
np.save(f"{outputs_dir}{project_name}_RF_test.npy", y_pred)

logger.info(f"RF Accuracy : {rf_accuracy}")
logger.info(f"Confusion Matrix: {rf_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_RF_confusion_matrix.csv",
    rf_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_RF.npy", y_pred)

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
np.save(f"{outputs_dir}{project_name}_RF_OPT_test.npy", y_pred)

logger.info(f"RF OPT Accuracy : {rf_accuracy}")
logger.info(f"Confusion Matrix: {rf_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_RF_OPT_confusion_matrix.csv",
    rf_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_RF_OPT.npy", y_pred)



exit()
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
elif dataset_name == "cifar10_vgg16":
    from cifar10_vgg16_config import dataset, outputs_dir, project_name

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
np.save(f"{outputs_dir}{project_name}_SVM_test.npy", y_pred)

#if dataset_name != "cifar10":
y_pred = clf_svm.predict_proba(X)

logger.info(f"SVM Accuracy : {svm_accuracy}")
svm_accuracy = clf_svm.score(X_train, y_train)
logger.info(f"SVM training Accuracy : {svm_accuracy}")
logger.info(f"Cofusion Matrix: {svm_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_SVM_confusion_matrix.csv",
    svm_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_SVM.npy", y_pred)

# ----- RF -----#
print("Fitting RF ...")
clf_rf = RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    min_samples_leaf=2,
    max_depth=80,
    min_samples_split=5,
    bootstrap=True,
    max_features="sqrt",
    verbose=False,
)
clf_rf.fit(X_train, y_train)

print("Predicting RF ...")
rf_accuracy = clf_rf.score(X_test, y_test)
y_pred = clf_rf.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, y_pred, normalize="true")
rf_confusion_matrix = np.around(
    rf_confusion_matrix.astype("float")
    / rf_confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)

np.save(f"{outputs_dir}{project_name}_RF_test.npy", y_pred)

#if dataset_name != "cifar10":
y_pred = clf_rf.predict_proba(X)

logger.info(f"RF Accuracy : {rf_accuracy}")
rf_accuracy = clf_rf.score(X_train, y_train)
logger.info(f"RF training Accuracy : {rf_accuracy}")

logger.info(f"Confusion Matrix: {rf_confusion_matrix}")
np.savetxt(
    f"{outputs_dir}{project_name}_RF_confusion_matrix.csv",
    rf_confusion_matrix,
    delimiter=",",
)
np.save(f"{outputs_dir}{project_name}_RF.npy", y_pred)


