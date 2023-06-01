import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kmapper as km
import sklearn
import warnings
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from persim.landscapes import PersLandscapeApprox
from persim.landscapes.visuals import plot_landscape_simple
from persim.landscapes.tools import *
import matplotlib as mpl

warnings.filterwarnings("ignore", category=FutureWarning)
from get_epsilon import *

"""

functions for creating tortuosity measurement

"""


# select pts within delta. return pts leftover and pts to be added to components
# remaining points is in indexes of pts in df
def eval_delta_top_down(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df["Y"].max() - df.iloc[remaining_pts[i], 1] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


def eval_delta_right_left(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df["X"].max() - df.iloc[remaining_pts[i], 0] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


def eval_delta_left_right(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df.iloc[remaining_pts[i], 0] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


def eval_delta_bottom_up(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df.iloc[remaining_pts[i], 1] - df["Y"].min() < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


def eval_delta_radial(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if math.sqrt((df["x"].mean() - df.iloc[remaining_pts[i], 0]) ** 2 + (
                df["y"].mean() - df.iloc[remaining_pts[i], 1]) ** 2) < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# current_pts : list of points just passed through filtration.
# Entries are the index value of a point in master df (also the index of same point in labels)
# living_com : list of current living components already through filtration
# A: adjacency matrix of for points
# dead_com : list of components that have died due to filration
# delta : current delta in our filtration
def add_to_com(current_pts, living_com, A, dead_com, delta):
    # First, find components among all new points
    # by keeping track of which components each point is adjacent to
    current_components = []
    for pt in range(len(current_pts)):
        connections = [current_pts[pt]]
        adj_components = []

        # Find all components said point is adjacent to, and add to adj component list
        # if current components is empty, we will pass over this step
        for i in range(len(current_components)):
            if current_components[i] is not None:
                # For each point in each component, see if it is adjacent to our current point
                for component_pt_index in range(len(current_components[i])):
                    # if the points are adjacent, add the component to the adjacent component list
                    # and move on to see if the next component is adjacent
                    if A[current_pts[pt]][current_components[i][component_pt_index]] == 1:
                        adj_components.append(current_components[i])
                        # Replace the component that is being taken out of the list with None
                        current_components[i] = None
                        break

        # Join together all adjacent components to form new component
        for c in adj_components:
            for com_pt in c:
                connections.append(com_pt)

        # Add new component to list
        current_components.append(connections)

    # Get rid of all nones once finished
    current_components = [i for i in current_components if i is not None]

    # Now cross-check with all living components using similar logic
    # Keep track of which components each current component is adj to
    for i in range(len(current_components)):
        connections = current_components[i]
        adj_components = []

        # Find all components said component is adjacent to, and add to adj component list
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

        # Ff adj_components is empty, this is our base case or the current component is not adj to anything
        # So it is a new component and its birth is the current delta
        birthday, oldest_com = find_min_birth(adj_components, delta)
        # join together all connections to form new component
        for c in adj_components:
            for com_pt in c[0]:
                connections.append(com_pt)
            # if not the oldest component, add birthday and death day to list of dying components
            if c != oldest_com:
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
        oldest_com = None
        mini = delta
        for c in component_list:
            if c[1] < mini:
                mini = c[1]
                oldest_com = c
        return mini, oldest_com


# Takes in the list of labels from driver and returns a list of clusters
# Where each cluster is a list of the indices in ex
def sort_clusters(labels):
    num_of_clusters = max(labels) + 1
    to_return = [[] for i in range(num_of_clusters)]
    for index in range(len(labels)):
        if labels[index] != -1:
            to_return[labels[index]].append(index)
    return to_return


# Helps create adjacency matrix for driver.
# Input two points indexes that are in the same cluster and check if they are within epsilon
def check_distance(pt1, pt2, df, eps):
    if math.sqrt((df.iloc[pt1, 0] - df.iloc[pt2, 0]) ** 2 + (df.iloc[pt1, 1] - df.iloc[pt2, 1]) ** 2) < eps:
        return True


def driver(test_file, eps, ms, cube, overlap, show_graph, show_pd, c):
    # prep data based on color we want to analyze
    ex = pd.read_csv(test_file)
    # ex.columns = ["x", "y", "color"]
    # if c == "red":
    #     ex = ex.loc[ex["color"] != 55]
    # if c == "green":
    #     ex = ex.loc[ex["color"] == 55]
    ex = ex[["X", "Y"]]
    # ex = ex.loc[ex["X"] >= 1000]
    # ex = ex.loc[ex["Y"] >= 1000]
    ex.to_numpy()

    # create clusters using DBSCAN
    cl = DBSCAN(eps=eps, min_samples=ms)
    clusters = cl.fit(ex)
    # labels is a list of which cluster each point is in; -1 means a point is noise
    labels = clusters.labels_

    sorted_clusters = sort_clusters(labels)

    # plots clusters in different colors
    if show_graph:
        for cluster in sorted_clusters:
            cluster_mem_data = []
            for pt_index in cluster:
                cluster_mem_data.append(ex.iloc[pt_index])
            df = pd.DataFrame(cluster_mem_data)
            df.columns = ["x", "y"]
            plt.scatter(df["x"], df["y"], s=1.5)
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        plt.xlim(1000, 2850)
        plt.ylim(1000, 2850)
        plt.axis("off")
        plt.show()

    # Filtration rate for bendiness
    d_max = 210
    d = 0
    rate = 1

    # Create adjacency matrix
    size = len(ex)
    A = np.zeros((size, size), dtype=int)
    for cluster in sorted_clusters:
        for pt_index in cluster:
            for pt2_index in cluster:
                if pt_index != pt2_index:
                    if check_distance(pt_index, pt2_index, ex, eps):
                        A[pt_index][pt2_index] = 1

    # Calculate bendiness
    remaining_pts = [i for i in range(len(ex))]
    living_com = []
    dead_com = [[], []]

    while d < d_max:
        remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, ex)
        living_com, dead_com = add_to_com(current_pts, living_com, A, dead_com, d)
        d = d + rate

    # Plot the persistence diagram
    if show_pd:
        to_plot = [[], []]

        for c in living_com:
            to_plot[0].append(c[1])
            to_plot[1].append(d_max)
        for i in range(len(dead_com[0])):
            to_plot[0].append(dead_com[0][i])
            to_plot[1].append(dead_com[1][i])

        plt.scatter(to_plot[0], to_plot[1])
        plt.xlim(0, 205)
        plt.ylim(0, 205)
        plt.show()

    # Include the number of nodes (used for normalization)
    num_of_nodes = len(ex)
    return dead_com, num_of_nodes


# To visualize cells
def graph_data(dataframe, size):
    plt.scatter(dataframe["X"], dataframe["Y"], s=size, color='black')
    plt.xlim(0, 2850)
    plt.ylim(0, 2850)
    plt.show()


"""

the following functions are applicable for mapper


"""


# Helper function to determine the location of node for each cluster
def find_cluster_data_mean(c, graph, ex):
    cluster_members = graph["nodes"][c]
    cluster_members_data = []
    for cluster in cluster_members:
        cluster_members_data.append(ex.iloc[cluster])
    df = pd.DataFrame(cluster_members_data)
    df.columns = ["X", "Y"]
    df_mean = df[["X", "Y"]].mean()
    return df_mean


# Returns list of all edges in simp. comp. in [ [x_start, x_finish], [y_start, y_finish] ] form
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
    return A


# Graphs the simplicial complex output of mapper with location preserved
# Returns a data frame of the x and y coordinates of each node
def draw_sim_com(graph, ex, show_sim_com):
    df = pd.DataFrame({'X': [], 'Y': []})
    for c in graph["nodes"]:
        df = df.append(find_cluster_data_mean(c, graph, ex), ignore_index=True)
    if show_sim_com:
        edges = find_edges(graph, ex)
        graph_sim_com(df, edges, 1)
    return df


# Helper function for draw sim com
# Dataframe is a df of the nodes, edges is the list of edges
# One edge looks like: [ [x_start, x_finish], [y_start, y_finish] ]
def graph_sim_com(dataframe, edges, size):
    dataframe.columns = ["X", "Y"]
    plt.scatter(dataframe["X"], dataframe["Y"], s=size)
    plt.axis("off")
    for edge in edges:
        plt.plot(edge[0], edge[1])
    plt.show()

def get_ms(n):
    if n<475:
        # print(6)
        return 6
    if n>725:
        # print(8)
        return 8
    # print(7)
    return 7


def drive_mapper(ex, eps, ms, cube, overlap, show_sim_com, show_pd, c):
    ex = ex.loc[(ex["gata6_normalized"] < 0.4800010305034226) | (ex["nanog_normalized"] > 1.357431938575777)]
    ex = ex[["X", "Y"]]
    ex.to_numpy()

    # Initialize
    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=cube, perc_overlap=overlap)
    graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.DBSCAN(
        min_samples=get_ms(len(ex)), metric="euclidean", eps=eps), remove_duplicate_nodes=True)

    if show_sim_com:
        # Visualize it
        mapper.visualize(graph, path_html="simplical_com.html",
                         title="simpl_com)")

        # plot the sim complex
        nodes_locations_df = draw_sim_com(graph, ex, True)
    else:
        nodes_locations_df = draw_sim_com(graph, ex, False)

    # now calc bendiness from sim com
    d_max = ex["X"].max() - ex["X"].min()
    d = 0
    rate = d_max // 100
    adj_matrix = build_A_from_clusters(graph)

    num_of_nodes = len(ex)

    remaining_pts = [i for i in range(len(graph["nodes"]))]
    living_com = []
    dead_com = [[], []]

    while d < d_max:
        remaining_pts, current_pts = eval_delta_top_down(remaining_pts, d, nodes_locations_df)
        living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
        d = d + rate

    if show_pd:
        to_plot = [[], []]

        for c in living_com:
            to_plot[0].append(c[1])
            to_plot[1].append(d_max)
        for i in range(len(dead_com[0])):
            to_plot[0].append(dead_com[0][i])
            to_plot[1].append(dead_com[1][i])

        plt.scatter(to_plot[0], to_plot[1])
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

    return dead_com, num_of_nodes


"""

functions for analysis of results.

 need to consider need for normalization


 """


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


def help_graph_vectors(tick, eps, ms, cube, overlap, c, mapper):
    if mapper:
        dead_c, num_of_nodes = drive_mapper(tick, eps, ms, cube, overlap, show_sim_com=False, show_pd=False, c=c)
    else:
        dead_c, num_of_nodes = driver(tick, eps, ms, cube, overlap, show_pd=False, c=c, show_graph=False)
    p_values = create_vector(dead_c, 0, 200)
    x_values = [i for i in range(len(p_values))]

    # worried about edge case for csv files: where this would return empty lists
    if len(p_values) == 0:
        return [[0], [0]], num_of_nodes
    else:
        return [x_values, p_values], num_of_nodes


# input is a list of ticks grouped by sim in order of 0->144
def graph_vectors(ticks, eps, ms, cube, overlap, c, mapper):
    list_of_lines = [[help_graph_vectors(tick, eps, ms, cube, overlap, c=c, mapper=mapper)] for tick in ticks]
    tick_num = 0
    for tick in list_of_lines:
        plt.plot(tick[0][0], tick[0][1], color=find_color(tick_num))
        if tick_num == 144:
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


# graphs the sum of bar percentages against the different ticks
# where each line is a different sim
# list_of_sim_lines is a list of sims, where each sim is a list of the percentage at each tick
# ticks are ordered from least to greatest
def graph_sum_of_bar(list_of_sim_lines):
    for sim in list_of_sim_lines:
        plt.plot([0, 48, 96, 144], sim)
    plt.show()


# note: this is now normalized for sum_of_bars
def drive_bar_and_vector(ticks, eps, ms, cube, overlap, c, mapper):
    list_of_lines = [help_graph_vectors(tick, eps, ms, cube, overlap, c=c, mapper=mapper) for tick in ticks]
    list_normalizers = [normalizer[1] for normalizer in list_of_lines]
    list_of_lines = [lines[0] for lines in list_of_lines]
    list_of_sums = []
    tick_num = 0
    sim_num = 0
    sim_counter_helper = 0
    current_sum_list = []
    n = 0
    csv_vectors = []
    for tick in list_of_lines:
        plt.plot(tick[0], tick[1], color=find_color(tick_num))
        csv_vectors.append(tick[1])
        if tick_num == 144:
            tick_num = 0
        else:
            tick_num = tick_num + 48
        current_sum_list.append((sum(tick[1])) / list_normalizers[n])
        n = n + 1
        if sim_counter_helper == 3:
            sim_counter_helper = 0
            sim_num = sim_num + 1
            list_of_sums.append(current_sum_list)
            current_sum_list = []
        else:
            sim_counter_helper = sim_counter_helper + 1
    plt.show()
    graph_sum_of_bar(list_of_sums)

    # put list of sums into a csv file to be used for further analysis
    csv_sums = pd.DataFrame(list_of_sums, columns=["0", "48", "96", "144"])
    csv_sums.to_csv("sums_test.csv")

    # put list of vectors into a csv file to be used for further analysis
    # vectors are not marked by the tick number they come from,
    # but are in the order of 0,48,96,144,0...
    csv_vectors_df = pd.DataFrame(csv_vectors)

    # create a list to be inserted as a column that will describe the
    # corresponding tick number of each row
    biggest_vector_len = len(csv_vectors_df)
    tick_list = []
    start = 0
    for i in range(biggest_vector_len):
        tick_list.append(start % 192)
        start = start + 48

    csv_vectors_df["tick number"] = tick_list
    csv_vectors_df.to_csv("vectors_test.csv")


""" 


function for getting p norms of each tick and sending outputs to a file


"""


def get_p_norms(ticks, eps, ms, cube, overlap, c):
    tick_list = []
    start = 0
    csv_norms = []

    for tick in ticks:
        # print(tick)
        csv_norms.append(get_p_norm_helper(tick, eps, ms, cube, overlap, c))
        tick_list.append(start % 192)
        start = start + 48

    csv_norms_df = pd.DataFrame(csv_norms)
    csv_norms_df["tick number"] = tick_list
    csv_norms_df.to_csv("16_6_35_36_all.csv")


# input file name, create death pairs, create the landscape, and then calculate the p norm
def get_p_norm_helper(test_file, eps, ms, cube, overlap, c):
    dead_com, num_of_nodes = drive_mapper(test_file, eps, ms, cube, overlap, show_sim_com=True, show_pd=False, c=c)
    if len(dead_com[0]) > 0:
        # divide by 14 to normalize to simulated data
        correct_format = [[dead_com[0][i]/14, dead_com[1][i]/14] for i in range(len(dead_com[0]))]

        correct_format = [np.array(correct_format)]
        P = PersLandscapeApprox(dgms=correct_format, hom_deg=0)
        mpl.rcParams['text.usetex'] = False
        plot_landscape_simple(P)
        plt.show()
        norm = P.p_norm(2)
        print(norm / num_of_nodes)
        return norm / num_of_nodes
    else:
        return 0


# computes and graphs the average persistence landscape for each time tick
def get_p_landscapes(ticks, eps, ms, cube, overlap, c):
    lands = [get_p_landscape_helper(tick, eps, ms, cube, overlap, c) for tick in ticks]

    tick48 = [lands[(4 * i) + 1] for i in range(100)]
    tick96 = [lands[(4 * i) + 2] for i in range(100)]
    tick144 = [lands[(4 * i) + 3] for i in range(100)]

    avg48 = average_approx(tick48)
    plot_landscape_simple(avg48)
    plt.show()

    avg96 = average_approx(tick96)
    plot_landscape_simple(avg96)
    plt.show()

    avg144 = average_approx(tick144)
    plot_landscape_simple(avg144)
    plt.show()


# input file name, create death pairs, create the landscape
def get_p_landscape_helper(test_file, eps, ms, cube, overlap, p):
    print(test_file)
    dead_com, num_of_nodes = drive_mapper(test_file, eps, ms, cube, overlap, show_sim_com=False, show_pd=False, c=p)
    if len(dead_com[0]) > 0:
        correct_format = [[dead_com[0][i], dead_com[1][i]] for i in range(len(dead_com[0]))]
        correct_format = [np.array(correct_format)]
        P = PersLandscapeApprox(dgms=correct_format, hom_deg=0)
        return P
    else:
        return 0


ticks = ["temporal_exp_data_Jackie/Day1/D1_1_50_ld.csv",
          "temporal_exp_data_Jackie/Day1/D1_1_50_lt.csv",
          "temporal_exp_data_Jackie/Day1/D1_1_50_m.csv",
         "temporal_exp_data_Jackie/Day1/D1_1_50_rt.csv",
         "temporal_exp_data_Jackie/Day1/D1_2_50_ld.csv",
         "temporal_exp_data_Jackie/Day1/D1_2_50_lt.csv",
         "temporal_exp_data_Jackie/Day1/D1_2_50_m.csv",
         "temporal_exp_data_Jackie/Day1/D1_2_50_rt.csv",
         "temporal_exp_data_Jackie/Day2/D2_1_50_ld.csv",
         "temporal_exp_data_Jackie/Day2/D2_1_50_lt.csv",
         "temporal_exp_data_Jackie/Day2/D2_1_50_m.csv",
         "temporal_exp_data_Jackie/Day2/D2_1_50_rt.csv",
         "temporal_exp_data_Jackie/Day2/D2_2_50_ld.csv",
         "temporal_exp_data_Jackie/Day2/D2_2_50_lt.csv",
         "temporal_exp_data_Jackie/Day2/D2_2_50_m.csv",
         "temporal_exp_data_Jackie/Day2/D2_2_50_rt.csv"
         ]


mpl.rcParams['text.usetex'] = False

e = 23.8
ms= 7
c=30
o=.3

split_ticks = []
for tick in ticks:
    ex = pd.read_csv(tick)
    split_ticks.append(ex.loc[(ex["X"] <= 1400) & (ex["Y"] >= 1400)])
    split_ticks.append(ex.loc[(ex["X"] > 1400) & (ex["Y"] >= 1400)])
    split_ticks.append(ex.loc[(ex["X"] > 1400) & (ex["Y"] < 1400)])
    split_ticks.append(ex.loc[(ex["X"] <= 1400) & (ex["Y"] < 1400)])


# # gets 75th percentile of gata6 from day 1
# p=[]
# for tick in split_ticks[:32]:
#     p.append(tick["gata6_normalized"].describe()[6])
# print(sum(p)/32) # 0.4800010305034226
#
# p=[]
# for tick in split_ticks[32:]:
#     p.append(tick["nanog_normalized"].describe()[6])
# print(sum(p)/32) # 1.357431938575777



p_norms = [get_p_norm_helper(tick, e, ms, c, o, c="all") for tick in split_ticks]

