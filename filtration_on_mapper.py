import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kmapper as km
import sklearn
import warnings
import math

warnings.filterwarnings("ignore", category=FutureWarning)


def get_min_height(df):
    return -100
    # return np.min(df[1])


def get_max_height(df):
    return 100


# checks if two points are within delta
def check_delta_dis(x1, y1, x2, y2, d):
    if math.sqrt((x1-x2)**2+(y1-y2)**2) < d:
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
def eval_delta_radial(remaining_pts, delta, center, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if check_delta_dis(center[0], center[1], df.iloc[remaining_pts[i], 0], df.iloc[remaining_pts[i], 1], delta):
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# (remaining points is indexes of pts in df)
def eval_delta_top_down(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if  get_max_height(df) - df.iloc[remaining_pts[i],1] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# (remaining points is indexes of pts in df)
def eval_delta_right_left(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if  get_max_height(df) - df.iloc[remaining_pts[i],0] < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# (remaining points is indexes of pts in df)
def eval_delta_left_right(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if  df.iloc[remaining_pts[i],0] - get_min_height(df) < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# select pts within delta. return pts leftover and pts to be added to components
# (remaining points is indexes of pts in df)
def eval_delta_bottom_up(remaining_pts, delta, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if df.iloc[remaining_pts[i],1] - get_min_height(df) < delta:
            current_pts.append(remaining_pts[i])
            remaining_pts[i] = None
    return [i for i in remaining_pts if i is not None], current_pts


# takes current points from eval_delta and adds them to living components. for base case: living/dead = []
def add_to_com(current_pts, living_com, A, dead_com, delta):
    # first, must connect all current points together
    # keep track of which components each current point is adj to
    current_components = []
    for pt in range(len(current_pts)):
        connections = [current_pts[pt]]
        adj_components = [] # index locations of the adj components

        # find all components said point is adjacent to, and add to adj component list
        for component_index in range(len(current_components)):
            if current_components[component_index] is not None:
                for component_pt_index in range(len(current_components[component_index])):
                    if A[current_pts[pt]][current_components[component_index][component_pt_index]] == 1:
                        adj_components.append(current_components[component_index])
                        current_components[component_index] = None
                        break

        # join together all connections to form new component
        for c in adj_components:
            for com_pt in c:
                connections.append(com_pt)

        # add new component to list
        current_components.append(connections)

    # get rid of all nones once finished and add in birthdays
    current_components = [i for i in current_components if i is not None]

    # now cross-check with all living components using similar logic
    # keep track of which components each current component is adj to
    for curr_com_index in range(len(current_components)):
        connections = current_components[curr_com_index]
        adj_components = []  # index locations of the adj components
        for pt in range(len(current_components[curr_com_index])):
            # find all components said point is adjacent to, and add to adj component list
            for component_index in range(len(living_com)): # for each living component
                if living_com[component_index] is not None:
                    for component_pt_index in range(len(living_com[component_index][0])):
                        # for each point in each living component
                        if A[current_components[curr_com_index][pt]][living_com[component_index][0][component_pt_index]] == 1:
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
    # print(living_com)
    # print(dead_com)
    return living_com, dead_com


def find_min_birth(component_list, delta):
    if component_list == []:
        return delta, None
    else:
        youngest_com = None
        mini = delta
        for c in component_list:
            if c[1]<mini:
                mini = c[1]
                youngest_com = c
        return mini, youngest_com


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


def find_cluster_data_mean(c, graph, ex):
    cluster_members = graph["nodes"][c]
    cluster_members_data = []
    for cluster in cluster_members:
        cluster_members_data.append(ex.iloc[cluster])
    df = pd.DataFrame(cluster_members_data)
    df.columns = ["x", "y"]
    df_mean = df[["x", "y"]].mean()
    return(df_mean)


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


def draw_sim_com(graph, ex):
    # graph resulting sim complex
    df = pd.DataFrame({'x': [], 'y': []})
    for c in graph["nodes"]:
        df = df.append(find_cluster_data_mean(c, graph, ex), ignore_index=True)
    edges = find_edges(graph, ex)
    graph_sim_com(df, edges, 1)
    return df


def driver(test_file):
    ex = pd.read_csv(test_file)
    ex.columns = ["x", "y", "color"]
    ex = ex.loc[ex["color"] == 55]
    ex = ex[["x", "y"]]
    ex.to_numpy()

    # Initialize
    mapper = km.KeplerMapper(verbose=1)

    # Create a cover with n x n elements. perc_overlap will give more edges for higher number
    cover = km.Cover(n_cubes=50, perc_overlap=.2)

    # Create dictionary called 'graph' with nodes, edges and meta-information
    # have lens = original data if no projection was done
    # graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.KMeans(1), remove_duplicate_nodes=True)
    # DBSCAN was chosen because it makes more sense in context
    graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.DBSCAN(
        min_samples=3, metric="euclidean", eps=1.7), remove_duplicate_nodes=True)


    # Visualize it
    mapper.visualize(graph, path_html="simplical_com.html",
                 title="simpl_com)")

    # plot the sim complex
    nodes_locations_df = draw_sim_com(graph, ex)

    # now calc bendiness
    d_max = 210
    d = -100
    rate = 1
    adj_matrix = build_A(graph)

    remaining_pts = create_pt_list(graph["nodes"])
    living_com = []
    dead_com = [[],[]]

    while d < d_max:
        remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, nodes_locations_df)
        living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
        d = d + rate

    print(len(living_com))
    print(len(dead_com[0]))

    for c in living_com:
        dead_com[0].append(c[1])
        dead_com[1].append(d_max)

    plt.scatter(dead_com[0], dead_com[1])
    plt.xlim(0, 210)
    plt.ylim(0,210)
    plt.show()

# driver("NetLogo-Simulations/sim1/sim1-tick0.csv")
# driver("NetLogo-Simulations/sim1/sim1-tick48.csv")
driver("NetLogo-Simulations/sim3/sim3-tick144.csv")
# driver("NetLogo-Simulations/sim1/sim1-tick144.csv")
# driver("NetLogo-Simulations/sim1/sim1-tick192.csv")
