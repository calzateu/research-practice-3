import gurobipy as gp
from gurobipy import GRB
import numpy as np


def relocate(a_matrix, q_c_vector, q, points):
    num_nodes = len(a_matrix)
    num_clusters = len(a_matrix[0])
    nodes = list(range(num_nodes))
    clusters = list(range(num_clusters))

    model = gp.Model("Model")

    # Variables
    x_ic_add = model.addVars(nodes, clusters, vtype=GRB.BINARY, name="x_ic_add")
    x_ic_rem = model.addVars(nodes, clusters, vtype=GRB.BINARY, name="x_ic_rem")

    # Objective
    # model.setObjective(
    #     gp.quicksum(
    #         gp.quicksum(
    #             np.sqrt(
    #                 (points.iloc[i]['lon'] - np.sum(
    #                     [points.iloc[j]['lon'] * A[j, c] for j in range(num_nodes)]
    #                 )) ** 2 +
    #                 (points.iloc[i]['lat'] - np.sum(
    #                     [points.iloc[j]['lat'] * A[j, c] for j in range(num_nodes)]
    #                 )) ** 2
    #             ) * (x_ic_add[i, c] - x_ic_rem[i, c])
    #             for c in range(num_clusters)
    #         ) for i in range(num_nodes)
    #     ),
    #     GRB.MINIMIZE
    # )

    model.setObjective(
        gp.quicksum(
            gp.quicksum(
                np.sqrt(
                    (points.iloc[i]['lon'] - np.sum(
                        [points.iloc[j]['lon'] * a_matrix[j, c] for j in range(num_nodes)]
                    )) ** 2 +
                    (points.iloc[i]['lat'] - np.sum(
                        [points.iloc[j]['lat'] * a_matrix[j, c] for j in range(num_nodes)]
                    )) ** 2
                ) * (x_ic_add[i, c] - x_ic_rem[i, c])
                for c in range(num_clusters)
            ) for i in range(num_nodes)
        ),
        GRB.MINIMIZE
    )

    # Constraints
    for node in range(num_nodes):
        for cluster in range(num_clusters):
            model.addConstr(x_ic_rem[node, cluster] <= a_matrix[node, cluster])

    for node in range(num_nodes):
        model.addConstr(0 == gp.quicksum(x_ic_rem[node, cluster] - x_ic_add[node, cluster]
                                         for cluster in range(num_clusters)))

    for node in range(num_nodes):
        for cluster_prime in range(num_clusters):
            if a_matrix[node, cluster_prime] == 1:
                model.addConstr(gp.quicksum(
                    x_ic_add[node, cluster] for cluster in range(num_clusters) if cluster != cluster_prime
                ) == x_ic_rem[node, cluster_prime])

    for cluster in range(num_clusters):
        model.addConstr(gp.quicksum(x_ic_add[node, cluster] for node in range(num_nodes)) <= q_c_vector[cluster])

    for cluster in range(num_clusters):
        model.addConstr(gp.quicksum(
            q[node] * (a_matrix[node, cluster] + x_ic_add[node, cluster] -
                       x_ic_rem[node, cluster]) for node in range(num_nodes)
        ) <= q_c_vector[cluster])

    for cluster in range(num_clusters):
        if np.sum([a_matrix[node, cluster] * q[node] for node in range(num_nodes)]) <= q_c_vector[cluster]:
            model.addConstr(0 == gp.quicksum(x_ic_rem[node, cluster] for node in range(num_nodes)))

    model.optimize()

    return model


def tsp(nodes, distances):
    n = len(nodes)
    model = gp.Model('tsp')

    # Variables
    x = model.addVars(nodes, nodes, vtype=GRB.BINARY, name='x')
    l = model.addVars(nodes, nodes, vtype=GRB.CONTINUOUS, name='l', lb=0.0, ub=n*1.0)

    # Objective
    model.setObjective(gp.quicksum(distances[i, j] * x[i, j] for i in nodes for j in nodes), GRB.MINIMIZE)

    # Constraints
    for i in nodes:
        model.addConstr(1 == gp.quicksum(x[i, j] for j in nodes))
        model.addConstr(1 == gp.quicksum(x[j, i] for j in nodes))

    for j in nodes:
        if j != 0:
            model.addConstr(1 == gp.quicksum(l[i, j] - l[j, i] for i in nodes))

    for i in nodes:
        for j in nodes:
            model.addConstr(l[i, j] <= (n-1) * x[i, j])

    model.optimize()

    return model, x
