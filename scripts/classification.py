from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d",
    help="name of dataset to use (trento, bcss)",
    default="bcss"
    )

args = parser.parse_args()
dataset = args.dataset    
    
if dataset == 'bcss':
    from bcss_config import (
    dataset
)
elif dataset == 'trento':
    from trento_config import (
    dataset
)

X_train,y_train = dataset.train_dataset 
print(X_train.shape, y_train.shape, np.unique(y_train))
X_test,y_test = dataset.test_dataset 
print(X_test.shape, y_test.shape, np.unique(y_test))
X,y = dataset.full_dataset 
print(X.shape, y.shape, np.unique(y))

# ----- SVM -----#

clf_svm = SVC(
    C=1, 
    kernel="rbf", 
    verbose=True
    )
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)
print(f"SVM Accuracy : {svm_accuracy}")

y_pred = clf_svm.predict(X_test)
svm_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
svm_confusion_matrix = np.around(svm_confusion_matrix.astype('float') / svm_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(f"Cofusion Matrix: {svm_confusion_matrix}")

# y_pred = clf_svm.predict(X)

# np.save(f"{outputs_dir}{PROJECT_NAME}_SVM.npy", y_pred)


# # # ----- RF -----#

clf_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    max_depth=80,
    min_samples_split=5,
    bootstrap=True,
    max_features="sqrt",
    verbose=True
)

clf_rf.fit(X_train, y_train)
rf_accuracy = clf_rf.score(X_test, y_test)
print(f"RF Accuracy : {rf_accuracy}")

y_pred = clf_rf.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
rf_confusion_matrix = np.around(rf_confusion_matrix.astype('float') / rf_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(f"Cofusion Matrix: {rf_confusion_matrix}")

# y_pred = clf_rf.predict(X_test)
# np.save(f"{outputs_dir}{PROJECT_NAME}_RF.npy", y_pred)

