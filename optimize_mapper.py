import random

from persim.landscapes.tools import snap_pl
from filtration_for_50 import *


# Similar to driver, but altered to can compute parameter optimization
def drive_opt_mapper(test_file, show_pd, c, e, min_samp):
    ex = pd.read_csv(test_file)
    ex.columns = ["x", "y", "color"]
    if c == "red":
        ex = ex.loc[ex["color"] != 55]
    if c == "green":
        ex = ex.loc[ex["color"] == 55]
    ex = ex[["x", "y"]]
    ex = ex.loc[ex["x"] >= 0]
    ex = ex.loc[ex["y"] >= 0]
    ex.to_numpy()

    # create clusters using DBSCAN
    cl = DBSCAN(eps=e, min_samples=min_samp)
    clusters = cl.fit(ex)
    # labels is a list of which cluster each point is in; -1 means a point is noise
    labels = clusters.labels_

    sorted_clusters = sort_clusters(labels)

    # Filtration rate for bendiness
    d_max = 76
    d = 0
    rate = 1

    # Create adjacency matrix
    size = len(ex)
    A = np.zeros((size, size), dtype=int)
    for cluster in sorted_clusters:
        for pt_index in cluster:
            for pt2_index in cluster:
                if pt_index != pt2_index:
                    if check_distance(pt_index, pt2_index, ex, e):
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


# Similar to drive_mapper, but altered to can compute parameter optimization
def drive_mapper_opt_mapper(test_file, show_sim_com, show_pd, c, cube, over, e, min_samp):
    ex = pd.read_csv(test_file)
    ex.columns = ["x", "y", "color"]
    if c == "red":
        ex = ex.loc[ex["color"] != 55]
    if c == "green":
        ex = ex.loc[ex["color"] == 55]
    ex = ex[["x", "y"]]
    ex = ex.loc[ex["x"] >= 0]
    ex = ex.loc[ex["y"] >= 0]
    ex.to_numpy()

    # Initialize
    mapper = km.KeplerMapper(verbose=0)

    # Create a cover with n x n elements. perc_overlap will give more edges for higher number
    cover = km.Cover(n_cubes=cube, perc_overlap=over)

    # Create dictionary called 'graph' with nodes, edges and meta-information
    # have lens = original data if no projection was done
    # graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.KMeans(1), remove_duplicate_nodes=True)
    # DBSCAN was chosen because it makes more sense in context
    graph = mapper.map(ex, ex, cover=cover, clusterer=sklearn.cluster.DBSCAN(
        min_samples=min_samp, metric="euclidean", eps=e), remove_duplicate_nodes=True)

    if show_sim_com:
        # Visualize it
        mapper.visualize(graph, path_html="simplical_com.html",
                         title="simpl_com)")

        # plot the sim complex
        nodes_locations_df = draw_sim_com(graph, ex, True)
    else:
        nodes_locations_df = draw_sim_com(graph, ex, False)

    # now calc bendiness from sim com
    d_max = 76
    d = 0
    rate = 1
    adj_matrix = build_A_from_clusters(graph)

    num_of_nodes = len(graph["nodes"])

    remaining_pts = [i for i in range(len(graph["nodes"]))]
    living_com = []
    dead_com = [[], []]

    while d < d_max:
        remaining_pts, current_pts = eval_delta_left_right(remaining_pts, d, nodes_locations_df)
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
        plt.xlim(0, 205)
        plt.ylim(0, 205)
        plt.show()

    return dead_com, num_of_nodes


# Takes in the birth, death pairs from non-mapper output and mapper output
# and computes the difference in their persistence landscapes
def optimize_mapper(dgm1, dgm2):
    correct_format = []
    for i in range(len(dgm1[0])):
        correct_format.append([dgm1[0][i], dgm1[1][i]])
    correct_format = [np.array(correct_format)]
    P1 = PersLandscapeApprox(dgms=correct_format, hom_deg=0)
    correct_format = []
    for i in range(len(dgm2[0])):
        correct_format.append([dgm2[0][i], dgm2[1][i]])
    correct_format = [np.array(correct_format)]
    P2 = PersLandscapeApprox(dgms=correct_format, hom_deg=0)

    mpl.rcParams['text.usetex'] = False
    [snapped_P_1, snapped_P_2] = snap_pl([P1, P2])
    difference = snapped_P_1 - snapped_P_2
    return difference.p_norm(p=2)


def find_best(e, min_samp):
    dead_com_base, num_of_nodes = drive_opt_mapper("NetLogo_05272023_25/sim1_144.csv", show_pd=False, c="green", e=e, min_samp=min_samp)

    cubes = [15, 20, 25, 30

         ]
    overlaps = [.25, .3, .35, .4
            ]

    norms = []
    for cube in cubes:
        for overlap in overlaps:
            dead_com_opt, num_of_nodes = drive_mapper_opt_mapper("NetLogo_05272023_25/sim1_144.csv", show_sim_com=False, show_pd=False, c="green", cube=cube, over=overlap, e=e, min_samp=min_samp)
            this_norm = optimize_mapper(dead_com_base, dead_com_opt)
            norms.append(this_norm)

            # print(f'\n-----\nCubes: {cube}\nOverlap: {overlap}\nNorm dif: {this_norm}\n-----')


    min_norm_params = np.argmin(norms)
    num_of_overlaps = len(overlaps)
    overlap_min = min_norm_params%num_of_overlaps
    cube_min = int((min_norm_params - overlap_min)/num_of_overlaps)

    # print(f'Minimum params with norm of {np.min(norms)} - {cubes[cube_min]} {overlaps[overlap_min]}')
    # print("done")
    return (2*cubes[cube_min], overlaps[overlap_min])

def get_overlap():
    l = [[1.1, 4], [1.15, 4], [1.2, 4], [1.25, 5], [1.3, 5]]
    # .0012863359461599002 1.1 4 50 0.3
    # 1.15 4 50 0.25 .010462613462583101
    # 1.2 4 60 0.4 0146096789636396
    # 1.25 5 60 0.3 005688548931218999
    # 1.3 5 60 0.25 0048359812449934


    for pair in l:
        e = pair[0]
        m = pair[1]
        c, o = find_best(e, m)
        print(e, m, c, o)
        get_p_norms(get_ticks_25(), e, m, c, o, c="green")
        all_norms = pd.read_csv("25sim_lr.csv")

        train = [i for i in range(25)]

        over = 0

        tick_0 = all_norms.loc[all_norms["tick number"] == 0]
        train_tick_0 = [tick_0.iloc[i].loc["0"] for i in train]

        tick_48 = all_norms.loc[all_norms["tick number"] == 48]
        train_tick_48 = [tick_48.iloc[i].loc["0"] for i in train]

        tick_96 = all_norms.loc[all_norms["tick number"] == 96]
        train_tick_96 = [tick_96.iloc[i].loc["0"] for i in train]

        tick_144 = all_norms.loc[all_norms["tick number"] == 144]
        train_tick_144 = [tick_144.iloc[i].loc["0"] for i in train]

        if (min(train_tick_48) - max(train_tick_0)) < 0:
            over = over + min(train_tick_48) - max(train_tick_0)

        if (min(train_tick_96) - max(train_tick_48)) < 0:
            over = over + min(train_tick_96) - max(train_tick_48)

        if (min(train_tick_144) - max(train_tick_96)) < 0:
            over = over + min(train_tick_144) - max(train_tick_96)
        print(over)


get_overlap()

# print(find_best(1.7,6))

