import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kmapper as km
import sklearn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# orders points from left to right such that we may cut computation time. Uses DBSCAN.
def get_A_fast(df, epsilon, min_pts):
    size = len(df)
    print(size)
    to_remove = []
    A = np.zeros((size, size), dtype=int)
    df.sort_values(by=['x'])
    for i in range(size):
        for j in range(i+1, size):
            if abs(df.iloc[i,0] - df.iloc[j,0]) >= e:
                break
            if check_dis(df.iloc[i, 0], df.iloc[i, 1], df.iloc[j, 0], df.iloc[j, 1]):
                A[i][j] = 1
                A[j][i] = 1
        # what follows is DBSAN (checking if a point is within min_pts)
        # if a pt is not adj to at least min_pts, then it should not be included in any component
        # we will remove all edges between this set of points at the end
        if count_edges_on_pt(A, i, size, min_pts):
            to_remove.append(i)
    for pt in to_remove:
        A = change_row_col_to_zeros(A, pt, size)
    return A


# orders points from left to right such that we may cut computation time. but no DBSCAN.
def get_A_fast_regular(df, epsilon):
    size = len(df)
    print(size)
    A = np.zeros((size, size), dtype=int)
    df.sort_values(by=['x'])
    for i in range(size):
        for j in range(i + 1, size):
            if abs(df.iloc[i, 0] - df.iloc[j, 0]) >= epsilon:
                break
            if check_dis(df.iloc[i, 0], df.iloc[i, 1], df.iloc[j, 0], df.iloc[j, 1], epsilon):
                A[i][j] = 1
                A[j][i] = 1
    return A


# helper function for get_A_fast.
# sets the row and column to zero for a certain point, if point is determined to be noise
def change_row_col_to_zeros(adj_matrix, pt, size):
    for j in range(size):
        adj_matrix[pt][j] = 0
        adj_matrix[j][pt] = 0
    return adj_matrix


# helper function for get_A_fast. determines if a point is noise (if it's within min_pts)
def count_edges_on_pt(adj_matrix, pt, size, min_pts):
    counter = 0
    for j in range(size):
        if adj_matrix[pt][j] == 1:
            counter = counter + 1
    if counter < min_pts:
        return True
    return False


# helper function for get_A_fast_regular. checks if two points are within e
def check_dis(x1, y1, x2, y2, epsilon):
    if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < epsilon:
        return True
    else:
        return False


def get_dis(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# helper function that tells us where sweeping will start, and determines distances.
def get_min_height(df):
    return -100


# helper function that tells us where sweeping will start, and determines distances.
def get_max_height(df):
    return 100


# checks if two points are within delta
def check_delta_dis(x1, y1, x2, y2, d):
    if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < d:
        return True
    else:
        return False


# center of mass used for radial filtration
def get_center(df):
    return [np.mean(df[0]), np.mean(df[1])]


# creates list of indexes for points in df
def create_pt_list(df):
    size = len(df)
    pt_list = []
    for i in range(size):
        pt_list.append(i)
    return pt_list


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_top_down(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if get_max_height(df) - df.iloc[remaining_pts[i], 1] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_right_left(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if get_max_height(df) - df.iloc[remaining_pts[i], 0] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_left_right(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df.iloc[remaining_pts[i], 0] - get_min_height(df) < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_bottom_up(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df.iloc[remaining_pts[i], 1] - get_min_height(df) < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_radial(remaining_pts, delta, center, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if check_delta_dis(center[0], center[1], df.iloc[remaining_pts[i], 0], df.iloc[remaining_pts[i], 1], delta):
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# takes current points from eval_delta and adds them to living components.
# returns the new list of living components and dead components
def add_to_com(current_pts, living_com, A, dead_com, delta):
    # first, must connect all new points together
    # keep track of which components each new point is adj to
    current_components = []
    for pt in range(len(current_pts)):
        connections = [current_pts[pt]]
        adj_components = []

        # find all components said point is adjacent to, and add to adj component list
        # if current components is empty, we will pass over this step
        for i in range(len(current_components)):
            if current_components[i] is not None:
                # for each point in each component, see if it is adjacent to our current point
                for component_pt_index in range(len(current_components[i])):
                    # if the points are adjacent, add the component to the adjacent component list
                    # and move on to see if the next component is adjacent
                    if A[current_pts[pt]][current_components[i][component_pt_index]] == 1:
                        adj_components.append(current_components[i])
                        # replace the component that is being taken out of the list with None
                        current_components[i] = None
                        break

        # join together all adjacent components to form new component
        for c in adj_components:
            for com_pt in c:
                connections.append(com_pt)

        # add new component to list
        current_components.append(connections)

    # get rid of all nones once finished
    current_components = [i for i in current_components if i is not None]

    # now cross-check with all living components using similar logic
    # keep track of which components each current component is adj to
    for i in range(len(current_components)):
        connections = current_components[i]
        adj_components = []

        # find all components said component is adjacent to, and add to adj component list
        # if current components is empty, we will pass over this step
        for pt in range(len(current_components[i])):  # for each point in our current component
            for component_index in range(len(living_com)):  # check with each living component
                if living_com[component_index] is not None:

                    # for each point in each living component
                    for component_pt_index in range(len(living_com[component_index][0])):
                        # if the points are adjacent, add the living component to the list of adjacent components
                        # and then move onto checking the next living component
                        if A[current_components[i][pt]][living_com[component_index][0][component_pt_index]] == 1:
                            adj_components.append(living_com[component_index])
                            living_com[component_index] = None
                            break

        # if adj_components is empty, this is our base case or the current component is not adj to anything
        # so it is a new component and its birth is the current delta
        birthday, youngest_com = find_min_birth(adj_components, delta)
        # join together all connections to form new component
        for c in adj_components:
            for com_pt in c[0]:
                connections.append(com_pt)
            # if not the youngest component, add birthday and death day to list of dying components
            if c != youngest_com:
                dead_com[0].append(c[1])
                dead_com[1].append(delta)

        # add new component to list
        living_com.append([connections, birthday])

    # get rid of all nones once finished
    # return living and dead
    living_com = [i for i in living_com if i is not None]
    return living_com, dead_com


def find_min_birth(component_list, delta):
    if not component_list:
        return delta, None
    else:
        youngest_com = None
        mini = delta
        for c in component_list:
            if c[1] < mini:
                mini = c[1]
                youngest_com = c
        return mini, youngest_com


def driver(file_name, show_pd, c):

    # prep data. decide on colors to be evaluated.
    ex = pd.read_csv(file_name)
    ex.columns = ["x", "y", "color"]
    if c == "red":
        ex = ex.loc[ex["color"] != 55]
    elif c == "green":
        ex = ex.loc[ex["color"] == 55]
    ex = ex.sort_values(by=["x"])

    # filtration rate
    d_max = 210
    d = -100
    rate = 1

    # DBSCAN or no DBSCAN
    adj_matrix = get_A_fast(ex, e, 3)

    remaining_pts = create_pt_list(ex)
    living_com = []
    dead_com = [[], []]

    start = time.time()
    while d < d_max:
        remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, ex)
        living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
        d = d + rate
    end = time.time()
    print(f"the val of x is {end-start}")

    if show_pd:
        to_plot = [[], []]

        for c in living_com:
            to_plot[0].append(c[1])
            to_plot[1].append(d_max)
        for i in range(len(dead_com[0])):
            to_plot[0].append(dead_com[0][i])
            to_plot[1].append(dead_com[1][i])

        plt.scatter(to_plot[0], to_plot[1])
        plt.xlim(0, 210)
        plt.ylim(0, 210)
        plt.show()

    # num_of_nodes = len(ex)
    num_of_clusters = len(living_com)

    return dead_com, num_of_clusters


"""the following functions are applicable for mapper"""


# if we want to visualize cells
def graph_data(dataframe, size):
    dataframe.columns = ["x", "y"]
    plt.scatter(dataframe["x"], dataframe["y"], s=size)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()


# will find the biggest cluster found from mapper
def find_biggest_cluster(graph, ex):
    biggest = ''
    longest = 0
    for c in graph["nodes"]:
        if len(graph["nodes"][c]) > longest:
            biggest = c
            longest = len(graph["nodes"][c])
    cluster_id = biggest
    if cluster_id in graph["nodes"]:
        cluster_members = graph["nodes"][cluster_id]
        cluster_members_data = []
        for cluster in cluster_members:
            cluster_members_data.append(ex.iloc[cluster])
        df = pd.DataFrame(cluster_members_data)
        graph_data(df, 1)


# helper function to determine the location of node for each cluster
def find_cluster_data_mean(c, graph, ex):
    cluster_members = graph["nodes"][c]
    cluster_members_data = []
    for cluster in cluster_members:
        cluster_members_data.append(ex.iloc[cluster])
    df = pd.DataFrame(cluster_members_data)
    df.columns = ["x", "y"]
    df_mean = df[["x", "y"]].mean()
    return df_mean


# returns list of all edges in simp. comp. in [ [x_start, x_finish], [y_start, y_finish] ] form
def find_edges(graph, ex):
    edges = []
    for c in graph["links"]:
        x_start, y_start = find_cluster_data_mean(c, graph, ex)
        for connected_node in graph["links"][c]:
            x_finish, y_finish = find_cluster_data_mean(connected_node, graph, ex)
            edges.append([[x_start, x_finish], [y_start, y_finish]])
    return edges


# build adjacency matrix to be used for sweeping from graph. A is indexed by cluster ID
def build_A_from_clusters(graph):
    start = time.time()
    size = len(graph["nodes"])
    node_list = []
    for c in graph["nodes"]:
        node_list.append(c)
    A = np.zeros((size, size), dtype=int)
    for c in graph["nodes"]:
        if c in graph["links"]:
            for connected_com in graph["links"][c]:
                A[node_list.index(c)][node_list.index(connected_com)] = 1
                A[node_list.index(connected_com)][node_list.index(c)] = 1
    end = time.time()
    print(f"done with A {end-start}")
    return A


# will graph the simplicial complex output of mapper with location preserved
def draw_sim_com(graph, ex, show_sim_com):
    # graph resulting sim complex
    df = pd.DataFrame({'x': [], 'y': []})
    for c in graph["nodes"]:
        df = df.append(find_cluster_data_mean(c, graph, ex), ignore_index=True)
    if show_sim_com:
        edges = find_edges(graph, ex)
        graph_sim_com(df, edges, 1)
    return df


# helper function for draw sim com
# dataframe is a df of the nodes, edges is the list of edges
# one edge looks like: [ [x_start, x_finish], [y_start, y_finish] ]
def graph_sim_com(dataframe, edges, size):
    dataframe.columns = ["x", "y"]
    plt.scatter(dataframe["x"], dataframe["y"], s=size)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    for edge in edges:
        plt.plot(edge[0], edge[1])
    plt.show()


def drive_mapper(test_file, show_sim_com, show_pd, c):
    ex = pd.read_csv(test_file)
    ex.columns = ["x", "y", "color"]
    if c == "red":
        ex = ex.loc[ex["color"] != 55]
    if c == "green":
        ex = ex.loc[ex["color"] == 55]
    ex = ex[["x", "y"]]
    ex.to_numpy()

    # Initialize
    mapper = km.KeplerMapper(verbose=0)

    # Create a cover with n x n elements. perc_overlap will give more edges for higher number
    cover = km.Cover(n_cubes=60, perc_overlap=.25)

    # Create dictionary called 'graph' with nodes, edges and meta-information
    # have lens = original data if no projection was done
    # graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.KMeans(1), remove_duplicate_nodes=True)
    # DBSCAN was chosen because it makes more sense in context
    graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.DBSCAN(
        min_samples=3, metric="euclidean", eps=1.7), remove_duplicate_nodes=True)

    if show_sim_com:
        # Visualize it
        mapper.visualize(graph, path_html="simplical_com.html",
                         title="simpl_com)")

        # plot the sim complex
        nodes_locations_df = draw_sim_com(graph, ex, True)
    else:
        nodes_locations_df = draw_sim_com(graph, ex, False)

    # now calc bendiness from sim com
    d_max = 210
    d = -100
    rate = 1
    adj_matrix = build_A_from_clusters(graph)

    # num_of_nodes = len(graph["nodes"])
    # print("num of nodes:", num_of_nodes)

    remaining_pts = create_pt_list(graph["nodes"])
    living_com = []
    dead_com = [[], []]

    start = time.time()
    while d < d_max:
        remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, nodes_locations_df)
        living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
        d = d + rate
    end = time.time()
    print(f"done with components {end-start}")

    # print(len(living_com))
    print("number of dead components:", len(dead_com[0]))

    if show_pd:
        to_plot = [[], []]

        for c in living_com:
            to_plot[0].append(c[1])
            to_plot[1].append(d_max)
        for i in range(len(dead_com[0])):
            to_plot[0].append(dead_com[0][i])
            to_plot[1].append(dead_com[1][i])

        plt.scatter(to_plot[0], to_plot[1])
        plt.xlim(0, 210)
        plt.ylim(0, 210)
        plt.show()

    num_of_clusters = len(living_com)

    return dead_com, num_of_clusters


"""functions for analysis of results. need to consider need for normalization."""

"""vector analysis: take the n most prominent features and puts into a vector"""


# input the dead component list and which of the most prominent
# features you will choose to analyze (starting at start (integer) and not including finish (integer)
def create_vector(dead_com, start, finish):
    # order dead_com from most to least important
    persist = []
    for pt in range(len(dead_com[0])):
        persist.append(abs(dead_com[0][pt] - dead_com[1][pt]))
    p_values = sorted(persist, reverse=True)
    # return only the ones we want
    return p_values[start:finish:1]


def help_graph_vectors(tick, c, mapper):
    if mapper:
        dead_c, num_of_nodes = drive_mapper(tick, show_sim_com=False, show_pd=False, c=c)
    else:
        dead_c, num_of_nodes = driver(tick, show_pd=False, c=c)
    p_values = create_vector(dead_c, 0, 150)
    x_values = []
    for i in range(len(p_values)):
        x_values.append(i)
    return [x_values, p_values]


# input is a list of ticks grouped by sim in order of 0->192
def graph_vectors(ticks, c, mapper):
    list_of_lines = [help_graph_vectors(tick, c=c, mapper=mapper) for tick in ticks]
    tick_num = 0
    for tick in list_of_lines:
        plt.plot(tick[0], tick[1], color=find_color(tick_num))
        if tick_num == 192:
            tick_num = 0
        else:
            tick_num = tick_num + 48
    plt.show()


def find_color(tick_num):
    if tick_num == 0:
        return "r"
    if tick_num == 48:
        return "y"
    if tick_num == 96:
        return "g"
    if tick_num == 144:
        return "b"
    if tick_num == 192:
        return "m"


# in tumor paper, they "sum up all the short bars" and then divide by the number of vessel segments.
# dividing by number of vessel segments doesn't make sense here, as we would divide by number of components
# but plently of components are really small or just clumps
# makes more sense to divide by number of cells
def sum_of_bar(dead_com):
    persist = []
    for pt in range(len(dead_com[0])):
        persist.append(abs(dead_com[0][pt] - dead_com[1][pt]))
    return sum(persist)


# graphs the sum of bar percentages against the different ticks
# where each line is a different sim
# list_of_sim_lines is a list of sims, where each sim is a list of the percentage at each tick
# ticks are ordered from least to greatest
def graph_sum_of_bar(list_of_sim_lines):
    for sim in list_of_sim_lines:
        plt.plot([0, 48, 96, 144, 192], sim)
    plt.show()


def help_drive_sum_of_bar(sim, c, mapper):
    sum_of_bar_list = []
    for tick in sim:
        if mapper:
            dead_c, num_of_clusters = drive_mapper(tick, show_sim_com=False, show_pd=False, c=c)
        else:
            dead_c, num_of_clusters = driver(tick, show_pd=False, c=c)

        # normalize by number of clusters (# of living components at the end of filtration)
        sum_of_bar_list.append(sum_of_bar(dead_c) / num_of_clusters)

        # normalize by number of cells (or in the case of mapper, number of nodes)
        # sum_of_bar_list.append(sum_of_bar(dead_c)/num_of_nodes)

        # don't normalize
        # sum_of_bar_list.append(sum_of_bar(dead_c))
    return sum_of_bar_list


def drive_sum_of_bar(sims, c, mapper):
    list_of_sim_lines = [help_drive_sum_of_bar(sim, c, mapper) for sim in sims]
    graph_sum_of_bar(list_of_sim_lines)


# note: this is not normalized for sum of bars
def drive_bar_and_vector(ticks, c, mapper):
    list_of_lines = [help_graph_vectors(tick, c=c, mapper=mapper) for tick in ticks]
    list_of_sums = [[], [], []]
    tick_num = 0
    sim_num = 0
    for tick in list_of_lines:
        plt.plot(tick[0], tick[1], color=find_color(tick_num))
        if tick_num == 192:
            tick_num = 0
        else:
            tick_num = tick_num + 48

        if sim_num < 5:
            list_of_sums[0].append(sum(tick[1]))
        elif sim_num < 10:
            list_of_sums[1].append(sum(tick[1]))
        elif sim_num < 15:
            list_of_sums[2].append(sum(tick[1]))
        sim_num = sim_num + 1
    plt.show()
    graph_sum_of_bar(list_of_sums)


e = 1.7

sim1 = ["NetLogo-Simulations/sim1/sim1-tick0.csv",
        "NetLogo-Simulations/sim1/sim1-tick48.csv",
        "NetLogo-Simulations/sim1/sim1-tick96.csv",
        "NetLogo-Simulations/sim1/sim1-tick144.csv",
        "NetLogo-Simulations/sim1/sim1-tick192.csv"]
sim2 = ["NetLogo-Simulations/sim2/sim2-tick0.csv",
        "NetLogo-Simulations/sim2/sim2-tick48.csv",
        "NetLogo-Simulations/sim2/sim2-tick96.csv",
        "NetLogo-Simulations/sim2/sim2-tick144.csv",
        "NetLogo-Simulations/sim2/sim2-tick192.csv"]
sim3 = ["NetLogo-Simulations/sim3/sim3-tick0.csv",
        "NetLogo-Simulations/sim3/sim3-tick48.csv",
        "NetLogo-Simulations/sim3/sim3-tick96.csv",
        "NetLogo-Simulations/sim3/sim3-tick144.csv",
        "NetLogo-Simulations/sim3/sim3-tick192.csv"]

ticks = ["NetLogo-Simulations/sim1/sim1-tick0.csv",
         "NetLogo-Simulations/sim1/sim1-tick48.csv",
         "NetLogo-Simulations/sim1/sim1-tick96.csv",
         "NetLogo-Simulations/sim1/sim1-tick144.csv",
         "NetLogo-Simulations/sim1/sim1-tick192.csv",
         "NetLogo-Simulations/sim2/sim2-tick0.csv",
         "NetLogo-Simulations/sim2/sim2-tick48.csv",
         "NetLogo-Simulations/sim2/sim2-tick96.csv",
         "NetLogo-Simulations/sim2/sim2-tick144.csv",
         "NetLogo-Simulations/sim2/sim2-tick192.csv",
         "NetLogo-Simulations/sim3/sim3-tick0.csv",
         "NetLogo-Simulations/sim3/sim3-tick48.csv",
         "NetLogo-Simulations/sim3/sim3-tick96.csv",
         "NetLogo-Simulations/sim3/sim3-tick144.csv",
         "NetLogo-Simulations/sim3/sim3-tick192.csv"]

# drive_sum_of_bar([sim1, sim2, sim3], "green", False)

# drive_bar_and_vector(ticks, "green", True)

dead_c, n = drive_mapper("NetLogo-Simulations/sim1/sim1-tick192.csv", show_sim_com=True, show_pd=False, c="green")
# print(create_vector(dead_c, 0, 30))
# print(create_vector(dead_c, 0, 30))
# print(sum_of_bar(dead_c))
#
# dead_c, n = drive_mapper("NetLogo-Simulations/sim3/sim3-tick144.csv", show_sim_com=False, show_pd=False)
# print(create_vector(dead_c, 0, 30))
# print(sum_of_bar(dead_c))
#
# dead_c, n = drive_mapper("NetLogo-Simulations/sim3/sim3-tick96.csv", show_sim_com=False, show_pd=False)
# print(create_vector(dead_c, 0, 30))
# print(sum_of_bar(dead_c))

# dead_c, n = drive_mapper("NetLogo-Simulations/sim3/sim3-tick48.csv", show_sim_com=False, show_pd=False)
# print(create_vector(dead_c, 0, 30))
# print(sum_of_bar(dead_c))
#
# dead_c , n = drive_mapper("NetLogo-Simulations/sim3/sim3-tick0.csv", show_sim_com=False, show_pd=False)
# print(create_vector(dead_c, 0, 20))
# print(sum_of_bar(dead_c))
