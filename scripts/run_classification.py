'''
This scripts determines the optimal parameters for svm and random forest using gridsearch
'''

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.svm import SVC 
import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial
from random import randint
from sklearn.ensemble import RandomForestClassifier

def my_classifier(X, y, id):
    kf = KFold(n_splits=5)
    index = kf.split(X)
    clf_svm = SVC(C=id[0], kernel="rbf", gamma = id[1], verbose=False)
    svm_accuracy = np.empty(5)
    for k, (train_index, test_index) in enumerate(index):
        clf_svm.fit(X[train_index], y[train_index])
        svm_accuracy[k] = clf_svm.score(X[test_index], y[test_index])
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

param = {'max_depth': [10, 20, 30, None], 
        'n_estimators':[50, 100, 150, 300], 
        'max_features': ['sqrt', 'log2', None],
        'criterion' : ['gini', 'entropy'],
        'bootstrap':[True, False],
        'min_samples_leaf': [1, 2, 3, 4]}

rnd_search = RandomizedSearchCV(RandomForestClassifier(), param, n_iter =10, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rnd_search.fit(X_test,y_test)
print(rnd_search.best_params_)
print(rnd_search.best_score_)

exit()
clf_rf = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    min_samples_leaf=4,
    max_depth=None,
    #min_samples_split=5,
    bootstrap=False,
    max_features="sqrt",
    verbose=False,
)
kf = KFold(n_splits=5)
index = kf.split(X)
rf_accuracy = np.empty(5)
for k, (train_index, test_index) in enumerate(index):
    clf_rf.fit(X[train_index], y[train_index])
    rf_accuracy[k] = clf_rf.score(X[test_index], y[test_index])

print(np.mean(rf_accuracy))


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

kf = KFold(n_splits=5)
index = kf.split(X)
rf_accuracy = np.empty(5)
for k, (train_index, test_index) in enumerate(index):
    clf_rf.fit(X[train_index], y[train_index])
    rf_accuracy[k] = clf_rf.score(X[test_index], y[test_index])
    
print(np.mean(rf_accuracy))



exit()
C_list = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 'scale']

#svm_accuracy = np.empty((len(C_list),len(gamma_list),5))
pool = Pool()
f = partial(my_classifier, X, y)
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
