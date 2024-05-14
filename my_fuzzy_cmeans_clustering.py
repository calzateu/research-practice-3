import my_distances
import numpy as np
import pandas as pd

# Calculate the centroids using the fuzzy matrix (It is called U in the book)
def get_centroids(data, fuzzy_matrix, m):
    centroids = np.zeros((fuzzy_matrix.shape[0], data.shape[1]))
    for i in range(fuzzy_matrix.shape[0]):
        for j in range(data.shape[0]):
            centroids[i] = centroids[i] + data[j]*(fuzzy_matrix[i][j])**m
        centroids[i] = centroids[i]/(fuzzy_matrix[i]**m).sum()    
    return centroids

# Update an individual element of the fuzzy matrix
def update_element_of_fuzzy_matrix(distances, i, j, m):
    dij = distances[i,j]*np.ones(distances.shape[0])
    quotient = dij/distances[:,j]
    return quotient**(2/(m-1))

# Update the complete fuzzy matrix
def update_fuzzy_matrix(distances, m):
    return np.array([[1/(update_element_of_fuzzy_matrix(distances, i, j, m).sum()) for j in range(distances.shape[1])] for i in range(distances.shape[0])])

# Calculate the cost function
def cost_function(distances, fuzzy_matrix, m):
    cost_function = 0
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            cost_function += (fuzzy_matrix[i,j]**m)*(distances[i,j]**2)
    return cost_function

# Condunted Fuzzy c means algorithm
def my_fuzzy_c_means(data, c, metric, m=2, epsilon=0.01, max_ite=100, verbose = False, **kwargs):
    dim = data.shape[1]
    points_number = data.shape[0]
    
    fuzzy_matrix = np.random.rand(c, points_number)
    fuzzy_matrix /= fuzzy_matrix.sum(axis=1, keepdims=True)
    
    current_cost_function = np.inf

    i = 0
    while i < max_ite:
        if verbose:
            print(f"Iteration {i}: Cost Function: ", current_cost_function)
            
        centroids = get_centroids(data, fuzzy_matrix, m)
        distances = my_distances.distances_matrix(centroids, data, metric)
        fuzzy_matrix = update_fuzzy_matrix(distances, m)
        
        next_cost_function = cost_function(distances, fuzzy_matrix, m)
        
        if np.abs(current_cost_function - next_cost_function) < epsilon:
            break
        else:
            current_cost_function = next_cost_function
        i += 1
        
    labels = np.argmin(distances, axis=0)
        
    return centroids, labels, current_cost_function