import numpy as np
import pandas as pd
import random
import kmapper as km
import sklearn
from sklearn.cluster import DBSCAN
from get_epsilon import *


# calculates the instability of mapper for a certain set of parameters
def calc_instability(file_name, k, cube, overlap, eps, ms):

    # set up data
    df = pd.read_csv(file_name)
    df.columns = ["x", "y", "color"]
    df = df.loc[df["color"] == 55]
    df = df[["x", "y"]]
    df = df.loc[df["x"] >= 40]
    df = df.loc[df["y"] >= 40]
    # shuffles rows
    df = df.sample(frac=1)
    df = df.values

    # make length of data divisible by k
    extra = len(df) % k
    for i in range(extra):
        byebye = random.sample([i for i in range(len(df))], 1)
        df = np.delete(df, byebye[0], axis=0)

    # make k samples
    m = len(df)//k
    subsamples = []
    for i in range(k):
        # k_i = df[m*(i):m*(i+1)]
        k_i = np.concatenate((df[:m*(i)], df[m*(i+1):]), axis=0)
        subsamples.append(k_i)
    print(k)
    # for every pair of subsamples, compute their mapper difference
    total_distance = 0
    open_cover = get_cover(cube, overlap)
    for i in range(len(subsamples)):
        for j in range((i+1),len(subsamples)):
            intersection_of_pts = np.concatenate((df[:m*(i)], df[m*(i+1):m*(j)], df[m*(j+1):]), axis=0)
            cluster1 = get_cluster1(np.array(subsamples[i]), open_cover, eps, ms, intersection_of_pts)
            cluster2, num_of_clusters = get_cluster2(np.array(subsamples[j]), open_cover, eps, ms, intersection_of_pts)

            total_distance = total_distance + (calc_dist(cluster1, (k-1)*m, 0, [], cluster2)/((k-2)*m))
            print(total_distance)
    average_dist = (2*total_distance)/((k-1)*k)
    return average_dist


# takes in two clusterings and calculates their mapper difference
def calc_dist(cluster1, bound, p, mismatch, cluster2):
    # select the current cluster in the first cluster and get the bin number
    try:
        bin_num = cluster1[p][-1][0]
    except IndexError:
        return bound

    # the number of clusters in the same bin from the other mapper output
    num_clusters_in_bin = len(cluster2[bin_num])

    # must account for the bin being empty (there is no cluster to match with)
    if num_clusters_in_bin == 0:
        # since the bin is empty, there are no clusters to match with, and all the points go unmatched
        new_mismatch = get_union(mismatch, cluster1[p][:len(cluster1[p])-1])
        cardinality = len(new_mismatch)
        if cardinality < bound:
            if p + 1 == len(cluster1):
                bound = cardinality
            else:
                # if not at the end, continue with the next cluster from cluster1
                bound = calc_dist(cluster1, bound, p + 1, new_mismatch, cluster2)

    # for every cluster in cluster 2 taken from the same bin, find the mismatches
    for i in range(num_clusters_in_bin):
        try:
            S = cluster2[bin_num][i]
        except IndexError:
            continue
        # find the new points that aren't matched
        new_mismatch = get_union(mismatch, get_s_diff(cluster1[p][:len(cluster1[p])-1], S))
        cardinality = len(new_mismatch)
        if cardinality < bound:
            if p + 1 == len(cluster1):
                bound = cardinality
            else:
                left_over_clusters = []
                for c in cluster2[bin_num]:
                    if c != S:
                        left_over_clusters.append(c)
                replace_c2 = [i for i in cluster2]
                replace_c2[bin_num] = left_over_clusters
                bound = calc_dist(cluster1, bound, p+1, new_mismatch, replace_c2)
    return bound


# input two lists, turn into sets, find union, and turn back into list
def get_union(list1, list2):
    comp = [pt for pt in list1]
    for pt in list2:
        if not pt in list1:
            comp.append(pt)
    return comp


# compute the symmetiric difference
def get_s_diff(list1, list2):
    diff = []
    for pt in list1:
        if not pt in list2:
            diff.append(pt)
    for pt in list2:
        if not pt in list1:
            diff.append(pt)
    return diff

# inpute points and open cover
# returns clusters as a np arrary of lists
# with a tuple at the end of each cluster list describing the bin number and cluster number
# also orders clusters by magnitude
def get_cluster1(subsample, open_cover, eps, ms, intersection):
    clusters = []
    count_open_cover = 0
    for i in open_cover:
        points = []
        # select only the points that are in the bin of the open cover
        for pt in subsample:
            if i[0][0] <= pt[0] <= i[0][1]:
                if i[1][0] <= pt[1] <= i[1][1]:
                    points.append([pt[0], pt[1]])

        # get new clusters, and append them to all cluster list with
        # (i, j) that describes the bin and cluster number at the end
        # this is so length can be computed still
        new_clusters = get_dbscan(points, eps, ms)
        for j in range(len(new_clusters)):
            new = []
            for pt in new_clusters[j]:
                if pt in intersection:
                    new.append(pt)
            new.append((count_open_cover,j))
            clusters.append(new)
        count_open_cover = count_open_cover + 1
    # sorts the clusters by magnitude in descending order
    clusters.sort(reverse=True, key=len)
    return clusters


# input points and open cover
# returns clusters as a np arrary of lists
def get_cluster2(subsample, open_cover, eps, ms, intersection):
    clusters = []
    num_of_clusters = 0
    for i in open_cover:
        points = []

        # select only the points that are in the bin of the open cover
        for pt in subsample:
            if i[0][0] <= pt[0] <= i[0][1]:
                if i[1][0] <= pt[1] <= i[1][1]:
                    points.append([pt[0], pt[1]])

        # get new clusters, and append them to all cluster list
        # the index of this list refers to the bin number
        if points == []:
            clusters.append([])
        else:
            new_clusters = get_dbscan(points, eps, ms)
            num_of_clusters = num_of_clusters + len(new_clusters)
            new2 = []
            for c in new_clusters:
                for pt in c:
                    if pt not in intersection:
                        c.remove(pt)
                new2.append(c)
            clusters.append(new2)
    return clusters, num_of_clusters


# calculates the open cover for a certain df with set parameters
# has the (x,x), (y,y) bounds of each bin going from left to right, bottom to top
def get_cover(cube, overlap):
    l = (201)/((cube-1)*(1-overlap)+1)
    open_cover = []
    for i in range(cube):
        for j in range(cube):
            x_1 = -100.5 + (i * (1 - overlap) * l)
            y_1 = -100.5 + (j * (1 - overlap) * l)
            open_cover.append([[x_1, x_1+l], [y_1, y_1+l]])
    return open_cover


# takes in the points from each bin and computes the clusters
def get_dbscan(points, eps, ms):
    if len(points) == 0:
        return []

    points = np.array(points)
    # Create clusters using DBSCAN
    cl = DBSCAN(eps=eps, min_samples=ms)
    clusters = cl.fit(points)
    labels = clusters.labels_
    # Takes in the list of labels from DBSCAN and returns a np array of list of clusters
    num_of_clusters = max(labels)+1
    to_return = [[] for i in range(num_of_clusters)]
    for i in range(len(labels)):
        if labels[i] != -1:
            to_return[labels[i]].append([points[i][0], points[i][1]])
    return to_return


# computes the instability n number of times and returns the average instability
def compute_avg_instability(file_name, k, n, cube, overlap, eps, ms):
    inst = [calc_instability(file_name, k, cube, overlap, eps, ms) for i in range(n)]
    return sum(inst)/len(inst)


def find_best(file_name, k, n):

    # range of parameters to try
    cubes = [40, 45, 50, 55, 60]
    overlaps = [.35, .4, .5]
    eps = [1.45, 1.5, 1.55, 1.6, 1.65]
    ms = [5, 6]

    for cube in cubes:
        for overlap in overlaps:
            for e in eps:
                for m in ms:
                    inst = compute_avg_instability(file_name, k, n, cube, overlap, e, m)
                    print(f'\n-----\nCubes: {cube}\nOverlap: {overlap}\nEpsilon: {e}\nMin Samples: {m}\nInstability: {inst}\n-----')
    return


# print(compute_avg_instability("NetLogo_50sim_03062023/sim25_144.csv", 40,1, 20,.35,1.8,7))
# find_best("NetLogo_50sim_03062023/sim22_96.csv", 10, 3)

print(compute_avg_instability("NetLogo_05272023_25/sim1_0.csv", 40,1,50,.3,1.1,4))

# print(compute_avg_instability("NetLogo_50sim_03062023/sim25_96.csv", 40,1,40,.35,1.6,6)) # 6.812%
