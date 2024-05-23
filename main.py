import optimization as go
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

import processing as pr
import read_instances as ri

from sklearn.cluster import KMeans
import my_fuzzy_cmeans_clustering as fcm
import my_kmeans_clustering as km
import my_distances as md

import time


def run_clustering(instance_name: str, clustering_method: callable, 
                   verbose: bool = False, **kwargs):
    
    data, problem_info = ri.obtain_instance_data(instance_name)
    
    points = data[['x', 'y']]
    deposit = points.iloc[[0]]
    points_df = points.iloc[1:,:].copy()
    normalized_points = points.apply(pr.normalize, axis=0).to_numpy()[1:]
    normalized_df = points.apply(pr.normalize, axis=0).iloc[1:].reset_index(drop=True)
    points = points.to_numpy()
    points = np.delete(points, 0, axis=0)
    print(normalized_df)

    ti = time.time()
    normalized_fc, labels_fc, _ = clustering_method(data=normalized_points, 
                                                 k = problem_info['n_vehicles'],
                                                 c = problem_info['n_vehicles'],
                                                 data_df = normalized_df,               
                                                 **kwargs)
    print(f"Clustering time: {time.time() - ti}")
    
    centroids_fc = pr.denormalize(normalized_fc, points)
        
    # Build matrix to tell if the point i is in cluster j
    membership_matrix = np.zeros((len(points), len(centroids_fc)))

    for i in range(len(points)):
        membership_matrix[i][labels_fc[i]] = 1
    
    Q_c = np.array([problem_info['q_vehicles'] 
                   for _ in range(problem_info['n_vehicles'])])
    
    q = data['Demand'].to_numpy()[1:]
    
    ti = time.time()
    # model_pulp = go.relocate_pulp(membership_matrix, Q_c, q, points_df)
    # variables = model_pulp.variablesDict()
    #
    # # Copy point into points_pulp
    # labels_fc_realoc = labels_fc.copy()
    # points_pulp = points_df.copy()
    # for i in range(len(labels_fc_realoc)):
    #     for j in range(len(centroids_fc)):
    #         if variables['x_ic_add__(%d,_%d)' % (i, j)].value() == 1:
    #             labels_fc_realoc[i] = j
    #             # print(i, j)

    model_gurobi, x_ic_add = go.relocate(membership_matrix, Q_c, q, points_df)
    print(x_ic_add[(0, 0)].X)
    labels_fc_realoc = labels_fc.copy()
    for i in range(len(labels_fc_realoc)):
        for j in range(len(centroids_fc)):
            if x_ic_add[(i, j)].X == 1:
                labels_fc_realoc[i] = j
                # print(i, j)

    print(f"Relocation time: {time.time() - ti}")
                
    if verbose:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 2, 1)
        scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc, cmap = "viridis")
        ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
        for i, txt in enumerate(np.unique(labels_fc)):
            ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
        ax.set_title('Visualization Clusters', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax = fig.add_subplot(1, 2, 2)
        scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc_realoc, cmap = "viridis")
        ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
        for i, txt in enumerate(np.unique(labels_fc_realoc)):
            ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
        ax.set_title('Visualization Clusters Reallocted', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        
    clusters = list(range(len(centroids_fc)))
    
    ti = time.time()
    distances = pr.build_distance_matrix(pd.concat([deposit, points_df]))
    costs = []
    for cluster in clusters:
        print(f"Cluster {cluster}:")
        points_cluster = points_df[labels_fc == cluster]
        if 0 not in points_cluster.index:
            points_cluster = pd.concat([deposit, points_cluster])

        nodes = list(points_cluster.index)
        print(nodes)

        cost, x = go.tsp(nodes, distances)
        costs.append(cost)
        if verbose:
            print(f"Cost for cluster {cluster}: {cost}")
            print("##########################################################")
            
    total_cost = np.sum(costs)
    print(f"TSP time: {time.time() - ti}")
    print("Total cost: ", total_cost)
    
    return total_cost


if __name__ == "__main__":
    verbose = True
    instance_name = 'p01'
    
    algorithms_params = {
        "metric": md.euclidean_distance, 
        "m": 2,
        "epsilon": 0.01,
        "max_ite": 100,
        "max_iterations": 100,
        "repetition_number": 100
    }
    
    fc_method = fcm.my_fuzzy_c_means
    km_method = km.centroids_search
    
    cost = run_clustering(instance_name, km_method, verbose,
                          **algorithms_params)
    print(cost)
    
    # data, problem_info = ri.obtain_instance_data(instance_name)
    
    # points = data[['x', 'y']]
    # deposit = points.iloc[[0]]
    # points_df = points.iloc[1:,:].copy()
    # normalized_points = points.apply(pr.normalize, axis=0).to_numpy()[1:]
    # points = points.to_numpy()
    # warehouse = points[0]
    # points = np.delete(points, 0, axis=0)
        
    # normalized_fc, labels_fc, _ = fcm.my_fuzzy_c_means(
    #                         normalized_points, problem_info['n_vehicles'], 
    #                         md.euclidean_distance)
    
    # centroids_fc = pr.denormalize(normalized_fc, points)
        
    # # Build matrix to tell if the point i is in cluster j
    # membership_matrix = np.zeros((len(points), len(centroids_fc)))

    # for i in range(len(points)):
    #     membership_matrix[i][labels_fc[i]] = 1
    
    # Q_c = np.array([problem_info['q_vehicles'] 
    #                for _ in range(problem_info['n_vehicles'])])
    
    # q = data['Demand'].to_numpy()[1:]
    
    # model_pulp = go.relocate_pulp(membership_matrix, Q_c, q, points_df)
    # variables = model_pulp.variablesDict()

    # # Copy point into points_pulp
    # labels_fc_realoc = labels_fc.copy()
    # points_pulp = points_df.copy()
    # for i in range(len(labels_fc_realoc)):
    #     for j in range(len(centroids_fc)):
    #         if variables['x_ic_add__(%d,_%d)' % (i, j)].value() == 1:
    #             labels_fc_realoc[i] = j
    #             # print(i, j)
                
    # if verbose:
    #     fig = plt.figure(figsize=(16,8))
    #     ax = fig.add_subplot(1, 2, 1)
    #     scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc, cmap = "viridis")
    #     ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
    #     for i, txt in enumerate(np.unique(labels_fc)):
    #         ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
    #     legend = ax.legend(*[scatter.legend_elements()[0], np.unique(labels_fc)], title="Clusters")
    #     ax.add_artist(legend)
    #     ax.set_title('Visualization fcmeans', fontsize=10)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
        
    #     ax = fig.add_subplot(1, 2, 2)
    #     scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc_realoc, cmap = "viridis")
    #     ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
    #     for i, txt in enumerate(np.unique(labels_fc_realoc)):
    #         ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
    #     legend = ax.legend(*[scatter.legend_elements()[0], np.unique(labels_fc_realoc)], title="Clusters")
    #     ax.add_artist(legend)
    #     ax.set_title('Visualization fcmeans Reallocted', fontsize=10)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     plt.show()
        
    # clusters = list(range(len(centroids_fc)))
    
    # distances = pr.build_distance_matrix(pd.concat([deposit, points_df]))
    # costs = []
    # for cluster in clusters:
    #     print(f"Cluster {cluster}:")
    #     points_cluster = points_df[labels_fc == cluster]
    #     if 0 not in points_cluster.index:
    #         points_cluster = pd.concat([deposit, points_cluster])

    #     nodes = list(points_cluster.index)
    #     print(nodes)

    #     cost, x = go.tsp(nodes, distances)
    #     costs.append(cost)
    #     print(f"Cost for cluster {cluster}: {cost}")
    #     print("##########################################################")
        
    # print("Total cost: ", np.sum(costs))
