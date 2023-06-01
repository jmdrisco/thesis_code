import pandas as pd
import numpy as np
from ripser import ripser
from ripser import Rips
from persim import plot_diagrams
from persim.landscapes import PersLandscapeApprox
import matplotlib as mpl
from persim.landscapes.visuals import plot_landscape_simple
from persim.landscapes.tools import *
import matplotlib.pyplot as plt
import os
mpl.rcParams['text.usetex'] = False



# x and y coordinates in float64
# data points range from 16600 to 4000
sim1 = [pd.read_csv("NetLogo-Simulations/sim1/sim1-tick0.csv"),
        pd.read_csv("NetLogo-Simulations/sim1/sim1-tick48.csv"),
        pd.read_csv("NetLogo-Simulations/sim1/sim1-tick96.csv"),
        pd.read_csv("NetLogo-Simulations/sim1/sim1-tick144.csv"),
        pd.read_csv("NetLogo-Simulations/sim1/sim1-tick192.csv")]
sim2 = [pd.read_csv("NetLogo-Simulations/sim2/sim2-tick0.csv"),
        pd.read_csv("NetLogo-Simulations/sim2/sim2-tick48.csv"),
        pd.read_csv("NetLogo-Simulations/sim2/sim2-tick96.csv"),
        pd.read_csv("NetLogo-Simulations/sim2/sim2-tick144.csv"),
        pd.read_csv("NetLogo-Simulations/sim2/sim2-tick192.csv")]
sim3 = [pd.read_csv("NetLogo-Simulations/sim3/sim3-tick0.csv"),
        pd.read_csv("NetLogo-Simulations/sim3/sim3-tick48.csv"),
        pd.read_csv("NetLogo-Simulations/sim3/sim3-tick96.csv"),
        pd.read_csv("NetLogo-Simulations/sim3/sim3-tick144.csv"),
        pd.read_csv("NetLogo-Simulations/sim3/sim3-tick192.csv")]


def get_all(sim_number):
    placeholder = []
    for df in sim_number:
        df.columns = ["x", "y", "color"]
        df = df[["x", "y"]]
        df = np.array(df)
        placeholder.append(df)
    return placeholder


def get_all_dg(sim1, sim2, sim3, threshold):
    diagrams_sim1 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim1]
    diagrams_sim2 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim2]
    diagrams_sim3 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim3]

    plt.figure(figsize=(15,9))

    start=351
    for i in range(15):
        plt.subplot(3,5,i+1)
        if i<5:
            plot_diagrams(diagrams_sim1[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        elif i<10:
            plot_diagrams(diagrams_sim2[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        else:
            plot_diagrams(diagrams_sim3[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
    plt.show()


def get_green(sim_number):
    placeholder = []
    for i in range(len(sim_number)):
        sim_number[i].columns = ["x", "y", "color"]
        df = sim_number[i].loc[sim_number[i]["color"] == 55]
        placeholder.append(df)
    return placeholder


def get_red(sim_number):
    placeholder = []
    for i in range(2, len(sim_number)):
        sim_number[i].columns = ["x", "y", "color"]
        df = sim_number[i].loc[sim_number[i]["color"] != 55]
        placeholder.append(np.array(df))
    return placeholder


def get_green_dg(sim1, sim2, sim3, threshold):
    diagrams_sim1 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim1]
    diagrams_sim2 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim2]
    diagrams_sim3 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim3]

    plt.figure(figsize=(15,9))

    start=351
    for i in range(15):
        plt.subplot(3,5,i+1)
        if i<5:
            plot_diagrams(diagrams_sim1[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        elif i<10:
            plot_diagrams(diagrams_sim2[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        else:
            plot_diagrams(diagrams_sim3[i%5], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)

    plt.show()


def get_red_dg(sim1, sim2, sim3, threshold):
    diagrams_sim1 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim1]
    diagrams_sim2 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim2]
    diagrams_sim3 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim3]

    plt.figure(figsize=(12,12))

    start = 331
    for i in range(9):
        plt.subplot(3,3,i+1)
        if i<3:
            plot_diagrams(diagrams_sim1[i%3], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        elif i<6:
            plot_diagrams(diagrams_sim2[i%3], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        else:
            plot_diagrams(diagrams_sim3[i%3], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
    plt.show()

# sim1 = get_green(sim1)
# sim2 = get_green(sim2)
# sim3 = get_green(sim3)
# get_green_dg(sim1, sim2, sim3, 10)

# sim1 = get_red(sim1)
# sim2 = get_red(sim2)
# sim3 = get_red(sim3)
# get_red_dg(sim1, sim2, sim3, 40)

# sim1 = get_all(sim1)
# sim2 = get_all(sim2)
# sim3 = get_all(sim3)
# get_all_dg(sim1, sim2, sim3, 22)


""" below are the functions written for calculating 
the persistence landscapes of the set of 100 simulations"""

# gets the names of all the simulations
def get_ticks():
    data_dir = "/Users/jackiedriscoll/Documents/TDA/NetLogo_50sim_03062023"
    ticks2 = [i for i in os.listdir(data_dir) if i[-3:] == "csv"]
    sort_key1 = lambda s:int(s.split("_")[1][:-4])
    sort_key2 = lambda s:int(s.split("_")[0][3:])
    ticks2.sort(key=sort_key1)
    ticks2.sort(key=sort_key2)
    ticks2 = [os.path.join(data_dir, i) for i in ticks2]

    data_dir = "/Users/jackiedriscoll/Documents/TDA/NetLogo_50sim_03232023"
    ticks3 = [i for i in os.listdir(data_dir) if i[-3:] == "csv"]
    sort_key1 = lambda s: int(s.split("_")[1][:-4])
    sort_key2 = lambda s: int(s.split("_")[0][3:])
    ticks3.sort(key=sort_key1)
    ticks3.sort(key=sort_key2)
    ticks3 = [os.path.join(data_dir, i) for i in ticks3]

    ticks2.extend(ticks3)
    return ticks2


# gets only the data for one simulation
def get_sim_number(n):
    sim = [get_ticks()[4*n-4+i] for i in range(4)]
    return [pd.read_csv(tick) for tick in sim]


# shows the persistence diagrams for one simulation
def show_pd_for_sim(n, threshold):
    sim = get_sim_number(n)
    sim = get_all(sim)
    diagrams_sim1 = [ripser(df, maxdim=1, thresh=threshold)['dgms'] for df in sim]

    plt.figure(figsize=(8, 8))

    start = 221
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i < 2:
            plot_diagrams(diagrams_sim1[i], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
        else:
            plot_diagrams(diagrams_sim1[i], show=False)
            plt.xlim(-1, threshold)
            plt.ylim(-1, threshold)
    plt.show()


# calculates the persistence landscapes for desired homology of all data sets
# plots the average persistence landscape for each time tick
# puts the normalized p norms into a csv file to be used for analysis
def find_per_land(ticks):
    ticks = [pd.read_csv(tick) for tick in ticks]
    normalizers = [len(tick) for tick in ticks]
    l = (len(ticks)//4)

    # compute persistence diagram
    ticks_prepped = get_all(ticks)
    diagrams = [ripser(tick, maxdim=1, thresh=20)['dgms'] for tick in ticks_prepped]
    print("done with ripser")

    # compute persistence landscapes
    pl_estimate = [PersLandscapeApprox(dgms=dgm, hom_deg=0) for dgm in diagrams]
    print("done with landscapes")

    # compute p norms and store them in csv file with tick number
    tick_list = [(48*i) % 192 for i in range(len(pl_estimate))]
    csv_norms = [pl_estimate[i].p_norm()/normalizers[i] for i in range(len(pl_estimate))]
    csv_norms_df = pd.DataFrame(csv_norms)
    csv_norms_df["tick number"] = tick_list
    csv_norms_df.to_csv("h0_p_norms.csv")

    # compute average persistence landscape and plot
    avg0 = average_approx([pl_estimate[(4 * i)] for i in range(l)])
    plot_landscape_simple(avg0)
    plt.show()
    avg48 = average_approx([pl_estimate[(4 * i) + 1] for i in range(l)])
    plot_landscape_simple(avg48)
    plt.show()
    avg96 = average_approx([pl_estimate[(4 * i) + 2] for i in range(l)])
    plot_landscape_simple(avg96)
    plt.show()
    avg144 = average_approx([pl_estimate[(4 * i) + 3] for i in range(l)])
    plot_landscape_simple(avg144)
    plt.show()


# find_per_land(get_ticks()[:4])
show_pd_for_sim(2, 15)