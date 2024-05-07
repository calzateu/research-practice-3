import my_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculates the distance between a given point and its corresponding centroid in a dataset.
def distance_to_centroid(point, centroids, metric):
    return metric(point.iloc[:-1], centroids.iloc[int(point['labels']),:])

#  Calculates the total sum of distances from data points to their respective centroids based on the given labels and metric
def total_distances(data, labels, centroids, metric):
    temp_data = data.copy()
    temp_data['labels'] = labels
    temp_data['distance_centroid'] = temp_data.apply(lambda x: distance_to_centroid(x, centroids, metric), axis=1)
    
    sum = temp_data[['labels', 'distance_centroid']].groupby('labels').sum().sum()
    return sum

# Implements the k-means clustering algorithm.
def centroids_search(data, k, max_iterations, metric):
    dim = data.shape[1]
    proposed_centroids = pd.DataFrame(np.random.rand(k, dim), columns=[f'X{i}' for i in range(dim)])
    
    i = 0
    while i < max_iterations:
        distances_matrix = pd.DataFrame(my_distances.distances_matrix(data.to_numpy(), proposed_centroids.to_numpy(), metric))
        data['labels'] = distances_matrix.idxmin(axis=1)
        
        new_centroids = data.groupby('labels').mean().values
        if new_centroids.shape[0] < k:
            new_centroids = np.concatenate((new_centroids, np.random.rand(k - new_centroids.shape[0], dim)))
        
        data.drop('labels', axis=1, inplace=True)
        proposed_labels = distances_matrix.idxmin(axis=1)
        
        if (proposed_centroids == new_centroids).min().min():
            break
        else:
            proposed_centroids = pd.DataFrame(new_centroids, columns = proposed_centroids.columns)
        i += 1
    return proposed_centroids, proposed_labels

# Performs K-means clustering with a specified number of repetitions.
def my_kmeans(data, k, max_iterations, repetition_number, metric):
    centroids, labels = centroids_search(data, k, max_iterations, metric)
    
    for i in range(repetition_number):
        new_centroids, new_labels = centroids_search(data, k, max_iterations, metric)
        if (total_distances(data, labels, centroids, metric) < total_distances(data, labels, centroids, metric))['distance_centroid']:
            centroids, labels = new_centroids, new_labels
        else:
            continue

    return centroids, labels