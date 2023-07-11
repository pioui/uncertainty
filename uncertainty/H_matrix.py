import numpy as np
from scipy.stats import energy_distance


def calculate_class_distance(classi, classj):
    _, channels = classi.shape
    channel_distances = []
    distance_metric = energy_distance
    
    channel_distances = [distance_metric(classi[:, channel], classj[:, channel]) for channel in range(channels)]
    
    if np.std(channel_distances)==0:
        return np.mean(channel_distances)
    return np.mean(channel_distances)+np.std(channel_distances)


def calculate_H_matrix(X, y, classes):
    values = np.unique(y)
    C = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(i+1, classes):
            classi = X[y == values[i]]
            classj = X[y == values[j]]
            if len(classi) == 0 or len(classj) == 0:
                C[i, j] = 0
            else:
                C[i, j] = calculate_class_distance(classi, classj)
            C[j, i] = C[i, j]
    return C


