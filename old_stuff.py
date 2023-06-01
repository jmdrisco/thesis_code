
"""

functions for creating the adjacency matrix A

"""


#
# def driver(file_name, show_pd, c):
#
#     # prep data. decide on colors to be evaluated.
#     ex = pd.read_csv(file_name)
#     ex.columns = ["x", "y", "color"]
#     if c == "red":
#         ex = ex.loc[ex["color"] != 55]
#     elif c == "green":
#         ex = ex.loc[ex["color"] == 55]
#     ex = ex.sort_values(by=["x"])
#
#
#     # filtration rate
#     d_max = 201
#     d = 0
#     rate = 1
#
#     # create adj matrix
#     adj_matrix = get_A_fast(ex, e, 4)
#
#     remaining_pts = [i for i in range(len(ex))]
#     living_com = []
#     dead_com = [[], []]
#
#     while d < d_max:
#         remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, ex)
#         living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
#         d = d + rate
#
#     if show_pd:
#         to_plot = [[], []]
#
#         for c in living_com:
#             to_plot[0].append(c[1])
#             to_plot[1].append(d_max)
#         for i in range(len(dead_com[0])):
#             to_plot[0].append(dead_com[0][i])
#             to_plot[1].append(dead_com[1][i])
#
#         plt.scatter(to_plot[0], to_plot[1])
#         plt.xlim(0, 201)
#         plt.ylim(0, 201)
#         plt.show()
#
#     num_of_nodes = len(ex)
#
#     return dead_com, num_of_nodes



#
# # orders points from left to right such that we may cut computation time. Uses DBSCAN.
# def get_A_fast(df, epsilon, min_pts):
#     size = len(df)
#     to_remove = []
#     A = np.zeros((size, size), dtype=int)
#     df.sort_values(by=['x'])
#     for i in range(size):
#         for j in range(i+1, size):
#             if abs(df.iloc[i,0] - df.iloc[j,0]) >= e:
#                 break
#             if math.sqrt((df.iloc[i, 0] - df.iloc[j, 0]) ** 2 + (df.iloc[i, 1] - df.iloc[j, 1]) ** 2) < epsilon:
#                 A[i][j] = 1
#                 A[j][i] = 1
#         # what follows is DBSAN (checking if a point is within min_pts)
#         # if a pt is not adj to at least min_pts, then it should not be included in any component
#         # we will remove all edges between this set of points at the end
#         if count_edges_on_pt(A, i, size, min_pts):
#             to_remove.append(i)
#     for pt in to_remove:
#         A = change_row_col_to_zeros(A, pt, size)
#     return A

#
# # helper function for get_A_fast.
# # sets the row and column to zero for a certain point, if point is determined to be noise
# def change_row_col_to_zeros(adj_matrix, pt, size):
#     for j in range(size):
#         adj_matrix[pt][j] = 0
#         adj_matrix[j][pt] = 0
#     return adj_matrix
#
#
# # helper function for get_A_fast. determines if a point is noise (if it's within min_pts)
# def count_edges_on_pt(adj_matrix, pt, size, min_pts):
#     counter = 0
#     for j in range(size):
#         if adj_matrix[pt][j] == 1:
#             counter = counter + 1
#     if counter < min_pts:
#         return True
#     return False


# # orders points from left to right such that we may cut computation time. but no DBSCAN.
# def get_A_fast_regular(df, epsilon):
#     size = len(df)
#     print(size)
#     A = np.zeros((size, size), dtype=int)
#     df.sort_values(by=['x'])
#     for i in range(size):
#         for j in range(i + 1, size):
#             if abs(df.iloc[i, 0] - df.iloc[j, 0]) >= epsilon:
#                 break
#             if check_dis(df.iloc[i, 0], df.iloc[i, 1], df.iloc[j, 0], df.iloc[j, 1], epsilon):
#                 A[i][j] = 1
#                 A[j][i] = 1
#     return A
#
# # helper function for get_A_fast_regular. checks if two points are within e
# def check_dis(x1, y1, x2, y2, epsilon):
#     if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < epsilon:
#         return True
#     else:
#         return False




# def help_drive_sum_of_bar(sim, c, mapper):
#     sum_of_bar_list = []
#     for tick in sim:
#         if mapper:
#             dead_c, num_of_clusters = drive_mapper(tick, show_sim_com=False, show_pd=False, c=c)
#         else:
#             dead_c, num_of_clusters = driver(tick, show_pd=False, c=c)
#
#         # normalize by number of clusters (# of living components at the end of filtration)
#         sum_of_bar_list.append(sum_of_bar(dead_c) / num_of_clusters)
#
#         # normalize by number of cells (or in the case of mapper, number of nodes)
#         # sum_of_bar_list.append(sum_of_bar(dead_c)/num_of_nodes)
#
#         # don't normalize
#         # sum_of_bar_list.append(sum_of_bar(dead_c))
#     return sum_of_bar_list


# def drive_sum_of_bar(sims, c, mapper):
#     list_of_sim_lines = [help_drive_sum_of_bar(sim, c, mapper) for sim in sims]
#     graph_sum_of_bar(list_of_sim_lines)



# in tumor paper, they "sum up all the short bars" and then divide by the number of vessel segments.
# dividing by number of vessel segments doesn't make sense here, as we would divide by number of components
# but plently of components are really small or just clumps
# makes more sense to divide by number of cells
# def sum_of_bar(dead_com):
#     persist = []
#     for pt in range(len(dead_com[0])):
#         persist.append(abs(dead_com[0][pt] - dead_com[1][pt]))
#     return sum(persist)


# def get_sims():
#     ticks = get_ticks()
#     sims = []
#     for i in range(50):
#         sims.append(ticks[4*i:4*(i+1)])
#     return sims[1]


# # Finds the biggest cluster found from mapper
# def find_biggest_cluster(graph, ex):
#     biggest = ''
#     longest = 0
#     for c in graph["nodes"]:
#         if len(graph["nodes"][c]) > longest:
#             biggest = c
#             longest = len(graph["nodes"][c])
#     cluster_id = biggest
#     if cluster_id in graph["nodes"]:
#         cluster_members = graph["nodes"][cluster_id]
#         cluster_members_data = []
#         for pt_index in cluster_members:
#             cluster_members_data.append(ex.iloc[pt_index])
#         df = pd.DataFrame(cluster_members_data)
#         graph_data(df, 1)




# def graph_all_components(liv_com, df):
#     for i in liv_com:
#         if len(i[0]) > 1:
#             x = []
#             y = []
#             for pt in i[0]:
#                 x.append(df.iloc[pt,0])
#                 y.append(df.iloc[pt,1])
#             plt.scatter(x, y, s=1)
#     plt.xlim(-100, 100)
#     plt.ylim(-100,100)
#     plt.show()


# helper function for graph_big_persist
# # finds the length of the point with biggest persistence
# def find_big_persist(dead_c):
#     persist = [abs(dead_c[0][pt]-dead_c[1][pt]) for pt in range(len(dead_c[0]))]
#     return max(persist)


# # Finds the biggest component created by the adjacency matrix
# def find_big_comp(liv_com, df):
#     max_length = max([len(i[0]) for i in liv_com])
#     next_biggest = []
#     for i in liv_com:
#         if len(i[0]) == max_length:
#             next_biggest.append(i[0])
#     x = []
#     y = []
#     for pt in next_biggest[0]:
#         x.append(df.iloc[pt,0])
#         y.append(df.iloc[pt,1])
#     return x, y


# brute force of nearest neighbors
# ex = pd.read_csv(file_name)
# ex.columns = ["x", "y", "color"]
# ex = ex.sort_values(by=["x"])
#
# epsilon = 3
#
# size = len(ex)
# nearest_neighbor_list = np.array([epsilon]*size, dtype=float)
# second_nearest = np.array([epsilon]*size, dtype=float)
# third_nearest = np.array([epsilon]*size, dtype=float)
# fourth = np.array([epsilon]*size, dtype=float)
#
#
# for i in range(size):
#     for j in range(i+1, size):
#         if abs(ex.iloc[i,0] - ex.iloc[j,0]) >= epsilon:
#             break
#         distance = round(filtration.get_dis(ex.iloc[i, 0], ex.iloc[i, 1], ex.iloc[j, 0], ex.iloc[j, 1]),epsilon)
#         if distance < epsilon:
#             if nearest_neighbor_list[i] <= distance:
#                 if second_nearest[i] <= distance:
#                     if third_nearest[i] <= min(third_nearest[i], distance):
#                         fourth[i] = min(fourth[i], distance)
#                     else:
#                         third_nearest[i] = distance
#                 else:
#                     second_nearest[i] = distance
#             else:
#                 nearest_neighbor_list[i] = distance
#
#             if nearest_neighbor_list[j] <= distance:
#                 if second_nearest[j] <= distance:
#                     if third_nearest[i] <= min(third_nearest[i], distance):
#                         fourth[i] = min(fourth[i], distance)
#                     else:
#                         third_nearest[i] = distance
#                 else:
#                     second_nearest[j] = distance
#             else:
#                 nearest_neighbor_list[j] = distance
#
#
# print("done with list")
#
# ok = [i for i in range(size)]
#
# nearest_neighbor_list.sort()
# second_nearest.sort()
# third_nearest.sort()
# fourth.sort()
# plt.plot(ok, nearest_neighbor_list, "r")
# plt.plot(ok, second_nearest, "y")
# plt.plot(ok, third_nearest, "g")
# plt.plot(ok, fourth, "m")
# plt.show()