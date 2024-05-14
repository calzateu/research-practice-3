import numpy as np

def normalize(column):
    return (column - np.min(column))/(np.max(column) - np.min(column))

def denormalize(centroids, points):
    denorm = np.zeros(centroids.shape)
    for i in range(centroids.shape[1]):
        min_i = np.min(points[:,i])
        max_i = np.max(points[:,i])
        denorm[:,i] = list(map(lambda x: x*(max_i-min_i) + min_i, centroids[:,i]))
    return denorm

def build_distance_matrix(points, x_name = 'x', y_name = 'y'):
    n = len(points)
    distances = np.zeros((n, n))
    for i in points.index:
        for j in points.index:
            distances[i, j] = np.sqrt((points[x_name][i] - points[x_name][j])**2 + (points[y_name][i] - points[y_name][j])**2)
    return distances
