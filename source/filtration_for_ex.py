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
        if df.iloc[remaining_pts[i], 0] - df["X"].min() < delta:
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
    ex["X"] = (ex["X"] / 19) -75
    ex["Y"] = (ex["Y"]/19)-75
    graph_data(ex, 2)
    ex = ex.loc[(ex["gata6_normalized"] < 0.4795166009607743) | (ex["nanog_normalized"] > 1.35671874885376)]
    ex = ex[["X", "Y"]]

    ex.to_numpy()

    # create clusters using DBSCAN
    cl = DBSCAN(eps=eps, min_samples=get_ms(len(ex)))
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
        plt.xlim(-75, 75)
        plt.ylim(-75, 75)
        plt.axis("off")
        plt.show()

    # Filtration rate for bendiness
    d_max = ex["X"].max() - ex["X"].min()
    d = 0
    rate = d_max // 150

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
        plt.xlim(0, 155)
        plt.ylim(0, 155)
        plt.show()

    # Include the number of nodes (used for normalization)
    num_of_nodes = len(ex)
    return dead_com, num_of_nodes


# To visualize cells
def graph_data(dataframe, size):
    plt.scatter(dataframe["X"], dataframe["Y"], s=size, color='black')
    plt.xlim(-75, 75)
    plt.ylim(-75, 75)
    plt.axis("on")
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
    if n<17000:
        return 3
    if n>21500:
        if n>28000:
            return 6
        return 5
    return 4


def drive_mapper(test_file, eps, ms, cube, overlap, show_sim_com, show_pd, c, filter="lr"):
    print(test_file)
    ex = pd.read_csv(test_file)
    ex = ex.loc[(ex["gata6_normalized"] < c) | (ex["nanog_normalized"] > 1.35671874885376)]
    # scale down to width of 150X150
    ex["X"] = (ex["X"] / 19)-75
    ex["Y"] = (ex["Y"]/ 19) - 75
    ex = ex[["X", "Y"]]
    ex.to_numpy()
    # print(len(ex))

    # Initialize
    mapper = km.KeplerMapper(verbose=0)

    # Create a cover with n x n elements. perc_overlap will give more edges for higher number
    cover = km.Cover(n_cubes=cube, perc_overlap=overlap)

    # eps = get_e(test_file)

    # Create dictionary called 'graph' with nodes, edges and meta-information
    # have lens = original data if no projection was done
    # DBSCAN was chosen because it makes more sense in context
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
    d_max = 151
    d = 0
    rate = 1
    adj_matrix = build_A_from_clusters(graph)

    num_of_nodes = len(ex)

    remaining_pts = [i for i in range(len(graph["nodes"]))]
    living_com = []
    dead_com = [[], []]

    while d < d_max:
        if filter=="lr":
            remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, nodes_locations_df)
        elif filter=="rl":
            remaining_pts, current_pts = eval_delta_right_left(remaining_pts, d, nodes_locations_df)
        elif filter=="td":
            remaining_pts, current_pts = eval_delta_top_down(remaining_pts, d, nodes_locations_df)
        elif filter=="bu":
            remaining_pts, current_pts = eval_delta_bottom_up(remaining_pts, d, nodes_locations_df)
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
        plt.xlim(0, 155)
        plt.ylim(0, 155)
        plt.show()

    return dead_com, num_of_nodes


""" 


function for getting p norms of each tick and sending outputs to a file


"""


# input file name, create death pairs, create the landscape, and then calculate the p norm
def get_p_norm_helper(test_file, eps, ms, cube, overlap, c, f="lr"):
    dead_com, num_of_nodes = drive_mapper(test_file, eps, ms, cube, overlap, show_sim_com=False, show_pd=False, c=c, filter=f)
    if len(dead_com[0]) > 0:
        correct_format = [[dead_com[0][i], dead_com[1][i]] for i in range(len(dead_com[0]))]
        correct_format = [np.array(correct_format)]
        P = PersLandscapeApprox(dgms=correct_format, hom_deg=0)
        mpl.rcParams['text.usetex'] = False
        # plot_landscape_simple(P)
        # plt.show()
        norm = P.p_norm(2)
        print(norm / num_of_nodes)
        return norm / num_of_nodes
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

e = 1.1
ms= 4
c=50
o=.3

p = 0.4795166009607743
q = 1.35671874885376



# lr = [get_p_norm_helper(tick, e, ms, c, o, c=p, f="lr") for tick in ticks]


# exit()


# num_red = []
# for i in range(len(ticks)):
#     df = pd.read_csv(ticks[i])
    # df["ratio col"] = df["gata6_normalized"] / df["nanog_normalized"]
    # df = df.loc[df["ratio col"] >= p]
    # df = df.loc[(df["gata6_normalized"] < p) | (df["nanog_normalized"] > q)]
    # print(len(df))
    # num_red.append(len(df))


# plt.scatter(num_red, p_norms)
# plt.show()

# exit()

try_this = [[1.1, 4, 50, .3]]
# 3, 6, 7
# [1.7, 6, 40, .3], [1.7, 6, 50, .35], [1.7, 6, 40, .35]
for parameters in try_this:
    e = parameters[0]
    ms = parameters[1]
    c = parameters[2]
    o = parameters[3]
    print(e, ms, c, o)

    lr = [get_p_norm_helper(tick, e, ms, c, o, c=p, f="lr") for tick in ticks]
    rl = [get_p_norm_helper(tick, e, ms, c, o, c=p, f="rl") for tick in ticks]
    td = [get_p_norm_helper(tick, e, ms, c, o, c=p, f="td") for tick in ticks]
    bu = [get_p_norm_helper(tick, e, ms, c, o, c=p, f="bu") for tick in ticks]

    averages = []
    for i in range(16):
        averages.append(((lr[i])+rl[i]+bu[i]+td[i])/4)
    print(averages)

    print(max(averages[:8]), min(averages[8:]))


# gets 75th percentile of gata6 from day 1
# p=[]
# for tick in ticks[:8]:
#     df = pd.read_csv(tick)
#     p.append(df["gata6_normalized"].describe()[6])
# print(sum(p)/8) # 0.4795166009607743
# #
# p=[]
# for tick in ticks[8:]:
#     df = pd.read_csv(tick)
#     print(df["nanog_normalized"].describe())
#     p.append(df["nanog_normalized"].describe()[6])
# print(sum(p)/8) #1.35671874885376

# Lr:
"""

[0.0007670842182452239, 
0.0008234518491224505, 
0.0015630869068510328, 
0.0020103687169728355, 

0.0012109318794797648, 
0.0012762706125845201, 
0.002539464707858512, 
0.001966639464310534, 

0.003682958515467843, 
0.00272495908093696, 
0.005307224971182173,
 0.005201918295659698, 
 
 0.005357385742954655, 
 0.003250572925996157, 
 0.00289828897557927, 
 0.00313020440943319]



"""