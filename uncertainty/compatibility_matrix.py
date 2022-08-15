import numpy as np
from scipy.stats import norm, wasserstein_distance, energy_distance
import matplotlib.pyplot as plt


def calculate_class_distance(classi, classj, distance_name):
    _, channels = classi.shape
    channel_distances = []
    if distance_name == "wasserstein":
        distance_metric = wasserstein_distance
    if distance_name == "energy":
        distance_metric = energy_distance

    for channel in range(channels):
        channel_distances.append(
            distance_metric(classi[:, channel], classj[:, channel])
        )
    return np.mean(channel_distances)


def calculate_compatibility_matrix(X, y, distance_name):
    classes = len(np.unique(y))
    C = np.empty((classes, classes))
    for i in range(classes):
        for j in range(i, classes):
            classi = X[y == i]
            classj = X[y == j]

            C[i, j] = calculate_class_distance(classi, classj, distance_name)
            C[j, i] = C[i, j]
    return C
