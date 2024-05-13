import numpy as np

## Metrics
# p-norm
def pnorm(row, p):
    return pow(np.dot(row, row), 1/p)

# Metric induced by 2-norm
def euclidean_distance(row_1, row_2):
    return pnorm(row_1 - row_2, 2)

# Distances Matrix
def distances_matrix(points1, points2, metric):
    distances = [[metric(points1[j,:], points2[i,:]) for i in range(len(points2))] for j in range(len(points1))]
    return np.array(distances)