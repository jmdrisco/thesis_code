import math
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
#
# def get_e(file):
#     ex = pd.read_csv(file)
#     ex.columns = ["X", "Y", "color"]
#     ex = ex[["X", "Y"]]
#     ex.to_numpy()
#
#     # Compute the k nearest neighbors
#     neighbors = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(ex)
#     distances, indices = neighbors.kneighbors(ex)
#     nearest_neighbor_list = np.array([point_list[1] for point_list in distances])
#
#     x_values = [i for i in range(len(ex))]
#     nearest_neighbor_list.sort()
#     kneedle = KneeLocator(x_values, nearest_neighbor_list, S=1.0, curve="convex", direction="increasing")
#
#     print(round(kneedle.knee_y, 3))
#     # print((round(kneedle.knee_y, 3)*(10000-len(ex))))
#     x = 1/(.6*math.log(len(ex)-9500, 10)) +.35
#     print(x)
#     return x*round(kneedle.knee_y, 3)

def get_e(file):
    ex = pd.read_csv(file)
    print(ex.describe())
    ex = ex.loc[(ex["gata6_normalized"] < 0.4795166009607743) | (ex["nanog_normalized"] > 1.35671874885376)]
    ex = ex[["X", "Y"]]
    # ex.columns = ["X", "Y", "color"]
    # ex = ex.loc[ex["color"] == 55]
    ex = ex[["X", "Y"]]
    ex.to_numpy()

    # Compute the k nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(ex)
    distances, indices = neighbors.kneighbors(ex)
    nearest_neighbor_list = np.array([point_list[1] for point_list in distances])

    x_values = [i for i in range(len(ex))]
    nearest_neighbor_list.sort()
    kneedle = KneeLocator(x_values, nearest_neighbor_list, S=1.0, curve="convex", direction="increasing")

    # print(nearest_neighbor_list[m])
    # round(kneedle.knee_y, 3)
    # return nearest_neighbor_list[m]
    print(round(kneedle.knee_y, 3)/19)
    return round(kneedle.knee_y, 3)

# print(get_e("temporal_exp_data_Jackie/Day1/D1_1_50_m.csv"))

#
# # def get_nn
