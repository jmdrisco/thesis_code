import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# matrix indexed by the indexes of points in df. 1 corresponds to those points having a distance <e
def get_A(df):
  size = len(df)
  A = np.zeros((size, size), dtype=int)
  for i in range(size):
    for j in range(i+1, size):
        if check_dis(df.iloc[i,0], df.iloc[i,1], df.iloc[j,0], df.iloc[j,1]):
            A[i][j] = 1
            A[j][i] = 1
  return A


# same as get_A but implemented so it runs much faster
def get_A_fast(df):
    size = len(df)
    print(size)
    A = np.zeros((size,size), dtype=int)
    df.sort_values(by=['x'])
    for i in range(size):
        for j in range(i+1, size):
            if abs(df.iloc[i,0] - df.iloc[j,0]) >= e:
                break
            if check_dis(df.iloc[i, 0], df.iloc[i, 1], df.iloc[j, 0], df.iloc[j, 1]):
                A[i][j] = 1
                A[j][i] = 1
    return A


# checks if two points are within e
def check_dis(x1, y1, x2, y2):
    if math.sqrt((x1-x2)**2+(y1-y2)**2) < e:
        return True
    else:
        return False


# checks if two points are within delta
def get_dist(x1, y1, x2, y2, d):
    if math.sqrt((x1-x2)**2+(y1-y2)**2) < d:
        return True
    else:
        return False


# center of mass used for filtration
def get_center(df):
    return [np.mean(df[0]), np.mean(df[1])]


# creates list of indexes for points in df
def create_pt_list(df):
    size = len(df)
    pt_list = []
    for i in range(size):
        pt_list.append(i)
    return pt_list


# select pts within delta. return pts leftover and pts to be added to components (remaining points is indexes of pts in df)
def eval_delta(remaining_pts, delta, center, df):
    current_pts = []
    for i in range(len(remaining_pts)):
        if get_dist(center[0], center[1], df.iloc[remaining_pts[i],0], df.iloc[remaining_pts[i],1], delta):
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


ex = pd.read_csv("NetLogo-Simulations/sim1/sim1-tick0.csv")
ex.columns = ["x", "y", "color"]
ex = ex.loc[ex["color"] == 55]
# print(df.head)
# print(df.iloc[2, 0])
ex =ex.sort_values(by=["x"])
# print(df.head)
# print(df.iloc[2, 0])

# print(df.describe())
# data = [[-1,0], [1,0], [-1,1], [1,1], [-1,2],[1,2], [-.5,2.3], [.5,2.3]]
# ex = pd.DataFrame(data)

e = 1.3
d_max = 150
d = 2
rate = 1

adj_matrix = get_A_fast(ex)
# adj_matrix = get_A(df)

# print(df.describe())
center = [0, 0]

remaining_pts = create_pt_list(ex)
living_com = []
dead_com = [[],[]]

while d < d_max:
    remaining_pts, current_pts = eval_delta(remaining_pts, d, center, ex)
    living_com, dead_com = add_to_com(current_pts, living_com, adj_matrix, dead_com, d)
    d = d + rate

# # check living is unique
#     check = [None]*5000
#     for c in living_com:
#         for pt in c[0]:
#             if check[pt] is not None:
#                 print("grrr")
#             else:
#                 check[pt] = "ok"

print(len(living_com))
print(len(dead_com[0]))

for c in living_com:
    dead_com[0].append(c[1])
    dead_com[1].append(d_max)


plt.scatter(dead_com[0], dead_com[1])
plt.xlim(0, 150)
plt.ylim(0,150)
plt.show()
