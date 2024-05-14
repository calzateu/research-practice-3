import optimization as go
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import processing as pr
import read_instances as ri

from sklearn.cluster import KMeans
import my_fuzzy_cmeans_clustering as fcm
import my_kmeans_clustering as km
import my_distances as md

# exit()
if __name__ == "__main__":
    verbose = True
    instance_name = 'p01'
    
    data, problem_info = ri.obtain_instance_data(instance_name)
    
    points = data[['x', 'y']]
    deposit = points.iloc[[0]]
    points_df = points.iloc[1:,:].copy()
    normalized_points = points.apply(pr.normalize, axis=0).to_numpy()[1:]
    points = points.to_numpy()
    warehouse = points[0]
    points = np.delete(points, 0, axis=0)
    
    if verbose:
        plt.plot(points[:,0], points[:,1], 'o', color='b')
        plt.title(instance_name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
    cost_functions_fcmeans = []
    c_clusters = range(2, 11)
    for c in c_clusters:
        _, _, j = fcm.my_fuzzy_c_means(
                  normalized_points, c, md.euclidean_distance)
        cost_functions_fcmeans.append(j)
    
    if verbose:        
        plt.plot(c_clusters, cost_functions_fcmeans)
        plt.title('Elbow FC-means')
        plt.xlabel('c')
        plt.ylabel('Cost Function')
        plt.show()
        
    # 5 clusters for p01
    
    normalized_fc, labels_fc, _ = fcm.my_fuzzy_c_means(
                            normalized_points, 5, md.euclidean_distance)
    
    centroids_fc = pr.denormalize(normalized_fc, points)
        
    # Build matrix to tell if the point i is in cluster j
    membership_matrix = np.zeros((len(points), len(centroids_fc)))

    for i in range(len(points)):
        membership_matrix[i][labels_fc[i]] = 1
    
    Q_c = np.array([problem_info['q_vehicles'] 
                   for _ in range(problem_info['n_vehicles'])])
    
    q = data['Demand'].to_numpy()[1:]
    
    model_pulp = go.relocate_pulp(membership_matrix, Q_c, q, points_df)
    variables = model_pulp.variablesDict()

    # Copy point into points_pulp
    labels_fc_realoc = labels_fc.copy()
    points_pulp = points_df.copy()
    for i in range(len(labels_fc_realoc)):
        for j in range(len(centroids_fc)):
            if variables['x_ic_add__(%d,_%d)' % (i, j)].value() == 1:
                labels_fc_realoc[i] = j
                # print(i, j)
                
    if verbose:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 2, 1)
        scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc, cmap = "viridis")
        ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
        for i, txt in enumerate(np.unique(labels_fc)):
            ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
        legend = ax.legend(*[scatter.legend_elements()[0], np.unique(labels_fc)], title="Clusters")
        ax.add_artist(legend)
        ax.set_title('Visualization fcmeans', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax = fig.add_subplot(1, 2, 2)
        scatter = ax.scatter(points[:,0], points[:,1], c = labels_fc_realoc, cmap = "viridis")
        ax.scatter(centroids_fc[:,0], centroids_fc[:,1], color='gray', marker='*', linewidths=15, alpha=0.8)
        for i, txt in enumerate(np.unique(labels_fc_realoc)):
            ax.annotate(txt, (centroids_fc[:,0][i], centroids_fc[:,1][i]), fontsize=12, ha='center', va='center')
        legend = ax.legend(*[scatter.legend_elements()[0], np.unique(labels_fc_realoc)], title="Clusters")
        ax.add_artist(legend)
        ax.set_title('Visualization fcmeans Reallocted', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        
    clusters = list(range(len(centroids_fc)))
    
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
        print(f"Cost for cluster {cluster}: {cost}")
        print("##########################################################")
        
    print("Total cost: ", np.sum(costs))
    
exit()    
if __name__ == "__main__":
    path = os.getcwd()
    data = pd.read_excel(path + '/data/1_n_128_dis_c_3_ln_8_seed_23032023_node_data.xlsx')
    points = data[data['Type'] == 'Vd'][['lon', 'lat']]

    plt.plot(points['lon'], points['lat'], 'o', color='b')
    plt.show()

    K_clusters = range(1, 10)
    kmeans_models = [KMeans(n_clusters=i) for i in K_clusters]
    score = [kmeans_models[i].fit(points).score(points) for i in range(len(kmeans_models))]
    # Plotting
    plt.plot(K_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

    kmeans = KMeans(n_clusters=4, init='k-means++')
    kmeans.fit(points[points.columns[0:2]])  # Compute k-means clustering.
    points['cluster_label'] = kmeans.labels_

    centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
    labels = points['cluster_label']  # Labels of each point

    points.plot.scatter(x='lon', y='lat', c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    # Build matrix to tell if the point i is in cluster j
    membership_matrix = np.zeros((len(points), len(centers)))

    for i in range(len(points)):
        for j in range(len(centers)):
            if points['cluster_label'][i] == j:
                membership_matrix[i][j] = 1
            else:
                membership_matrix[i][j] = 0

    print(membership_matrix)

    Q_c = np.array([32, 32, 32, 32])
    q = np.array([1 for i in range(len(points))])

    model_pulp = go.relocate_pulp(membership_matrix, Q_c, q, points)

    variables = model_pulp.variablesDict()

    # Copy point into points_pulp
    points_pulp = points.copy()
    for i in range(len(points_pulp)):
        for j in range(len(centers)):
            if variables['x_ic_add__(%d,_%d)' % (i, j)].value() == 1:
                points_pulp.loc[i, 'cluster_label'] = j
                print(i, j)

    # model = go.relocate(membership_matrix, Q_c, q, points)

    points_pulp.plot.scatter(x='lon', y='lat', c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    clusters = list(range(len(centers)))

    for cluster in clusters:
        print(f"Cluster {cluster}:")
        points_cluster = points_pulp[points_pulp['cluster_label'] == cluster]
        if 0 not in points_cluster['cluster_label']:
            deposit = points_pulp.iloc[0]
            points_cluster = pd.concat([pd.DataFrame(deposit).T, points_cluster])

        distances = pr.build_distance_matrix(points_pulp)

        nodes = list(points_cluster.index)
        print(nodes)

        cost, x = go.tsp(nodes, distances)
        print(f"Cost for cluster {cluster}: {cost}")
        # print(x)

    # variables = model.getVars()
    # points_gurobi = points.copy()
    # for var in variables:
    #     # if var.X == 1:
    #     if var.VarName.startswith("x_ic_add__") and var.X == 1:
    #         node, cluster = var.VarName.split('_')[-1].split(',')
    #         node = int(node)
    #         cluster = int(cluster[:-1])
    #         points_gurobi.loc[node, 'cluster_label'] = cluster
    #         print(node, cluster)
    #
    # points_gurobi.plot.scatter(x='lon', y='lat', c=labels, s=50, cmap='viridis')
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.show()
