from filtration_for_50 import *


# Graphs the points that correspond to a bend larger than max_persist
# Used to debug model and get an idea for which bends are picked up by filtration
# This function is very similar to add_to_com with the addition of graphing large persistences
def graph_big_persist(current_pts, living_com, A, dead_com, delta, max_persist, df):
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
        biggest_bend = False
        for c in adj_components:
            for com_pt in c[0]:
                connections.append(com_pt)
            # if not the youngest component, add birthday and death day to list of dying components
            if c != youngest_com:
                dead_com[0].append(c[1])
                dead_com[1].append(delta)
                if abs(c[1]-delta) >= max_persist:
                    biggest_bend = True

        if biggest_bend is True:
            x = []
            y = []
            for pt in connections:
                x.append(df.iloc[pt, 0])
                y.append(df.iloc[pt, 1])
            plt.scatter(x, y, s=1)
            plt.xlim(-100, 100)
            plt.ylim(-100, 100)
            plt.show()

        # add new component to list
        living_com.append([connections, birthday])

    # get rid of all nones once finished
    # return living and dead
    living_com = [i for i in living_com if i is not None]
    return living_com, dead_com


ex = pd.read_csv("NetLogo_50sim_03062023/sim1_144.csv")
ex.columns = ["x", "y", "color"]
ex = ex.loc[ex["x"] >= 50]
ex = ex.loc[ex["y"] >= 50]
ex = ex[["x", "y"]]
ex.to_numpy()

# create clusters using DBSCAN
cl = DBSCAN(eps=e, min_samples=5)
clusters = cl.fit(ex)
# labels is a list of which cluster each point is in; -1 means a point is noise
labels = clusters.labels_

sorted_clusters = sort_clusters(labels)

# plots clusters in different colors
for cluster in sorted_clusters:
    cluster_mem_data = []
    for pt_index in cluster:
        cluster_mem_data.append(ex.iloc[pt_index])
    df = pd.DataFrame(cluster_mem_data)
    df.columns = ["x", "y"]
    plt.scatter(df["x"], df["y"], s=.5)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.show()


# Filtration rate for bendiness
d_max = 105
d = 0
rate = 1

# Create adjacency matrix
size = len(ex)
A = np.zeros((size, size), dtype=int)
for cluster in sorted_clusters:
    for pt_index in cluster:
        for pt2_index in cluster:
            if pt_index != pt2_index:
                if check_distance(pt_index, pt2_index, ex):
                    A[pt_index][pt2_index] = 1

# Calculate bendiness
remaining_pts = [i for i in range(len(ex))]
living_com = []
dead_com = [[], []]

while d < d_max:
    remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, ex)
    living_com, dead_com = graph_big_persist(current_pts, living_com, A, dead_com, d, 3, ex)
    d = d + rate
