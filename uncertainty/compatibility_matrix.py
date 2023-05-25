import numpy as np
from scipy.stats import norm, wasserstein_distance, energy_distance, entropy
from scipy.special import kl_div
from scipy.spatial.distance import mahalanobis, jensenshannon
import matplotlib.pyplot as plt


def calculate_class_distance(classi, classj, distance_name):
    _, channels = classi.shape
    channel_distances = []
    if distance_name == "wasserstein":
        distance_metric = wasserstein_distance
    if distance_name == "energy":
        distance_metric = energy_distance
    if distance_name == "JS":
        p = np.array(classi)
        q = np.array(classj)
        return jensenshannon(p,q)

    channel_distances = [distance_metric(classi[:, channel], classj[:, channel]) for channel in range(channels)]
    

    if np.std(channel_distances)==0:
        return np.mean(channel_distances)
    return np.mean(channel_distances)+np.std(channel_distances)


def calculate_compatibility_matrix(X, y, distance_name, classes):
    values = np.unique(y)
    C = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(i+1, classes):
            classi = X[y == values[i]]
            classj = X[y == values[j]]
            if len(classi) == 0 or len(classj) == 0:
                C[i, j] = 0
            else:
                C[i, j] = calculate_class_distance(classi, classj, distance_name)
            C[j, i] = C[i, j]
    return C



