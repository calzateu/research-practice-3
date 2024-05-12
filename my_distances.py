import numpy as np

## Metrics
# p-norm
def my_pnorm(row, p):
    return pow(np.dot(row, row), 1/p)

# Metric induced by 2-norm
def my_euclidean_distance(row_1, row_2, in_cov = None):
    return my_pnorm(row_1 - row_2, 2)

# Metric induced by 1-norm
def my_manhattan_distance(row_1, row_2, in_cov = None):
    return my_pnorm(row_1 - row_2, 1)

# Metric induced by 3-norm
def my_3p_distance(row_1, row_2, in_cov = None):
    return my_pnorm(row_1 - row_2, 3)

# Metric induced by inf-norm
def my_inf_distance(row_1, row_2, in_cov = None):
    return np.max(np.abs(row_1 - row_2))       

# Cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
def my_cosine_similarity(row_1, row_2):
    return np.dot(row_1, row_2)/(my_pnorm(row_1, 2)*my_pnorm(row_2, 2))

# Metric induced by cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
def my_cosine_metric(row_1, row_2, in_cov = None):
    return 1 - my_cosine_similarity(row_1, row_2)

# Mahalanobis distance (https://en.wikipedia.org/wiki/Mahalanobis_distance)
def my_mahalanobis_distance(row_1, row_2, in_cov):
    return np.sqrt((row_1 - row_2).T @ in_cov @ (row_1 - row_2))

## Distances Matrices
# Distances of points from themselves
def my_distances_matrix_themselves(points, metric, cov = None):
    if cov is None:
        distances = [[metric(points.iloc[i,:], points.iloc[j,:]) if i >= j else 0 for i in range(len(points))] for j in range(len(points))]
    else:
        distances = [[metric(points.iloc[i,:], points.iloc[j,:], cov) if i >= j else 0 for i in range(len(points))] for j in range(len(points))]
    distances = np.array(distances) + np.array(distances).transpose()
    return distances

# Distances of points from other points
def my_distances_matrix_other(points1, points2, metric, cov = None):
    if cov is None:
        distances = [[metric(points1.iloc[i,:], points2.iloc[j,:]) for i in range(len(points1))] for j in range(len(points2))]
    else:
        distances = [[metric(points1.iloc[i,:], points2.iloc[j,:], cov) for i in range(len(points1))] for j in range(len(points2))]
    return np.array(distances)