import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

ex = pd.read_csv("NetLogo_50sim_03062023/sim1_144.csv")
# ex = pd.read_csv("temporal_exp_data_Jackie/Day2/D2_2_50_rt.csv")
print(len(ex))
ex.columns = ["x", "y", "color"]
# ex = ex.loc[ex["color"] == 55]
# ex = ex[["X", "Y"]]
ex = ex[["x", "y"]]
ex.to_numpy()

# Compute the k nearest neighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(ex)
distances, indices = neighbors.kneighbors(ex)

nearest_neighbor_list = np.array([point_list[4] for point_list in distances])

x_values = [i for i in range(len(ex))]

nearest_neighbor_list.sort()

kneedle = KneeLocator(x_values, nearest_neighbor_list, S=1.0, curve="convex", direction="increasing")
print(round(kneedle.knee_y, 3))
# print(round(kneedle.knee, 3))
kneedle.plot_knee()
# print(len(ex))
# print(nearest_neighbor_list[-5716])

plt.xlabel("Points")
plt.ylabel("Distance to 5th Nearest Neighbor")
plt.show()



# 1.69, 2.58, 1.77, 2.55
# 2.08, 2.86, 3.16, 3.05
# 2.43, 3.24, 4.02, 3.44