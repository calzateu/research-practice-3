import numpy as np


def build_distance_matrix(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in points.index:
        for j in points.index:
            distances[i, j] = np.sqrt((points['lon'][i] - points['lon'][j])**2 + (points['lat'][i] - points['lat'][j])**2)
    return distances
