'''
This scripts determines the optimal parameters for svm and random forest using gridsearch
'''

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC 
import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial

def my_classifier(X, y, id):
    kf = KFold(n_splits=5)
    index = kf.split(X)
    clf_svm = SVC(C=id[0], kernel="rbf", gamma = id[1], verbose=False)
    svm_accuracy = np.empty(5)
    for k, (train_index, test_index) in enumerate(index):
        clf_svm.fit(X[train_index], y[train_index])
        svm_accuracy[k] = clf_svm.score(X[test_index], y[test_index])
    print(id[0], id[1], np.mean(svm_accuracy))
    return np.mean(svm_accuracy)
    
# sourcery skip: avoid-builtin-shadow
dataset_name = "bcss"
if dataset_name == "bcss":
    from bcss_config import dataset, outputs_dir, project_name
elif dataset_name == "bcss_patched":
    from bcss_patched_config import dataset, outputs_dir, project_name
elif dataset_name == "trento":
    from trento_config import dataset, outputs_dir, project_name

#X_train, y_train = dataset.train_dataset
#X_test, y_test = dataset.test_dataset
X, y = dataset.full_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

C_list = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 'scale']

#svm_accuracy = np.empty((len(C_list),len(gamma_list),5))
pool = Pool()
f = partial(my_classifier, X_test, y_test)
id = list(itertools.product(C_list, gamma_list))
ans = pool.map(f, id)
pool.close() 
print(ans)
#print(list(ans))
# for i in range(len(C_list)):
#     C = C_list[i]
#     for j in range(len(gamma_list)):
#         gamma = gamma_list[j]
#         clf_svm = SVC(C=C, kernel="rbf", gamma = gamma, verbose=False)
#         k=0
#         for train_index, test_index in index:
#             clf_svm.fit(X[train_index], y[train_index])
#             svm_accuracy[i,j,k] = clf_svm.score(X[test_index], y[test_index])
#             k+=1

print(np.argmax(ans))
exit()       
C = 100
gamma = 1
#parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
#            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf_svm = SVC(C=C, kernel="rbf", gamma = gamma, verbose=False, probability=True)
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)
print(f"SVM Accuracy : {svm_accuracy}")

C = 100
gamma = 'scale'
#parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
#            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf_svm = SVC(C=C, kernel="rbf", gamma = gamma, verbose=False, probability=True)
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)
print(f"SVM Accuracy : {svm_accuracy}")

C = 1
gamma = 1
#parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
#            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf_svm = SVC(C=C, kernel="rbf", gamma = gamma, verbose=False, probability=True)
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)
print(f"SVM Accuracy : {svm_accuracy}")

C = 1
gamma = 'scale'
#parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
#            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf_svm = SVC(C=C, kernel="rbf", gamma = gamma, verbose=False, probability=True)
clf_svm.fit(X_train, y_train)
svm_accuracy = clf_svm.score(X_test, y_test)
print(f"SVM Accuracy : {svm_accuracy}")
#search = GridSearchCV(SVC(), parameters, cv=5, scoring = 'accuracy', n_jobs = -1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#search.fit(X_train, y_train)
#print("Trento")
#print(search.best_params_)
