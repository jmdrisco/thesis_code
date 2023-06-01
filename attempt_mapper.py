# Import the class
import kmapper as km
import matplotlib.pyplot as plt
import sklearn
import warnings

warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd

ex = pd.read_csv("NetLogo-Simulations/sim1/sim1-tick192.csv")
ex.columns = ["x", "y", "color"]
# ex = ex.loc[ex["color"] == 55]
ex = ex[["x", "y"]]
ex.to_numpy()


# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data, can skip this step
# projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis

# Create a cover with n x n elements. perc_overlap will give more edges for higher number
cover = km.Cover(n_cubes=50, perc_overlap=.2)

# Create dictionary called 'graph' with nodes, edges and meta-information
# have lens = original data if no projection was done
# graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.KMeans(1), remove_duplicate_nodes=True)
# DBSCAN was chosen because it makes more sense in context
graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.DBSCAN(min_samples=3, metric="euclidean", eps= 1.7), remove_duplicate_nodes=True)


# Visualize it
mapper.visualize(graph, path_html="simplical_com.html",
                 title="simpl_com)")


def graph_data(dataframe, size):
    dataframe.columns = ["x", "y"]
    plt.scatter(dataframe["x"], dataframe["y"], s=size)
    plt.xlim(-100, 100)
    plt.ylim(-100,100)
    plt.show()

# will graph the simplicial complex output of mapper with location preserved
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

def find_biggest_cluster(graph):
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


def find_cluster_data_mean(c):
    cluster_members = graph["nodes"][c]
    cluster_members_data = []
    for cluster in cluster_members:
        cluster_members_data.append(ex.iloc[cluster])
    df = pd.DataFrame(cluster_members_data)
    df.columns = ["x", "y"]
    df_mean = df[["x", "y"]].mean()
    return(df_mean)

# returns list of all edges in simp. comp. in [ [x_start, x_finish], [y_start, y_finish] ] form
def find_edges(graph):
    edges = []
    for c in graph["links"]:
        x_start, y_start = find_cluster_data_mean(c)
        for connected_node in graph["links"][c]:
            x_finish, y_finish = find_cluster_data_mean(connected_node)
            edges.append([[x_start, x_finish], [y_start, y_finish]])
    return edges


# build adjacency matrix to be used for sweeping from graph. A is indexed by cluster ID
def build_A(graph):
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


def driver(graph):
    df = pd.DataFrame({'x': [], 'y': []})
    for c in graph["nodes"]:
            df = df.append(find_cluster_data_mean(c), ignore_index=True)
    edges = find_edges(graph)
    graph_sim_com(df, edges, 1)



driver(graph)

