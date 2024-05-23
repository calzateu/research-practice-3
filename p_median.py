import numpy as np
import gurobipy as gp
from gurobipy import GRB
import my_distances


def p_median_clustering(data, p, metric, **kwargs):
    n = data.shape[0]
    distances_matrix = my_distances.distances_matrix(data, data, metric)

    model = gp.Model("p-median")

    # Create variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # Objective function: minimize the sum of distances
    model.setObjective(gp.quicksum(distances_matrix[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Constraint 1: exactly p facilities must be placed
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == p, "R1")

    # Constraint 2: each demand point must be assigned to one facility
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, f"R2_{i}")

    # Constraint 3: demand points can only be assigned to existing facilities
    for i in range(n):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], f"R3_{i}_{j}")

    # Optimize the model
    model.optimize()

    # Extract centroids and labels
    centroids = []
    labels = np.zeros(n, dtype=int)

    for j in range(n):
        if y[j].x > 0.5:
            centroids.append(data[j])
        for i in range(n):
            if x[i, j].x > 0.5:
                labels[i] = len(centroids) - 1

    centroids = np.array(centroids)

    return centroids, labels, None
