import optimization as go
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


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

    model = go.relocate(membership_matrix, Q_c, q, points)

    points_pulp.plot.scatter(x='lon', y='lat', c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

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
