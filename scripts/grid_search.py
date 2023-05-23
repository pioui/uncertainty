'''
This scripts determines the optimal parameters for svm and random forest using gridsearch
'''

from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

dataset_name = "trento"
if dataset_name == "bcss":
    from bcss_config import dataset, outputs_dir, project_name
elif dataset_name == "bcss_patched":
    from bcss_patched_config import dataset, outputs_dir, project_name
elif dataset_name == "trento":
    from trento_config import dataset, outputs_dir, project_name

X_train, y_train = dataset.train_dataset
X_test, y_test = dataset.test_dataset
X, y = dataset.full_dataset


parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

search = GridSearchCV(SVC(), parameters, cv=5, scoring = 'accuracy', n_jobs = -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

search.fit(X_train, y_train)
print("Trento")
print(search.best_params_)
