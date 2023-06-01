import os

import pandas as pd
from matplotlib import pyplot as plt
import random

# input the mean of a df and format into a list ready to plot
def format_list(df):
    list = []
    x_values = []
    for i in range(len(df) - 2):
        x_values.append(i)
        list.append(df.iloc[i+1])
    return x_values, list


# input the mean of a df and format into a list ready to plot
def format_list_sums(df):
    list = []
    x_values = []
    for i in range(len(df) - 1):
        x_values.append(i)
        list.append(df.iloc[i+1])
    return x_values, list


# input the mean of a df and format into a list ready to plot
def format_list_norms(df):
    list = []
    for i in range(len(df) - 1):
        list.append(df.iloc[i+1])
    return list



"""

train test split analysis

"""


def predict_tick(pnorm):
    if pnorm <= divider1:
        return 0
    if pnorm <= divider2:
        return 48
    if pnorm <= divider3:
        return 96
    return 144

perc_0 = []
perc_48 = []
perc_96 = []
perc_144 = []
perc_all = []


for k in range(100):
    # creates random list of simulations to be used for train test split (70/30)
    train = [i for i in range(25)]
    test = random.sample(train, 0)
    for j in test:
        train.remove(j)


    # all_norms = pd.read_csv("17_7_60_3_all.csv")
    # leftright = pd.read_csv("17_7_60_3_all_bu.csv")
    # another = pd.read_csv("17_7_60_3_all_rl")
    # anotheranother = pd.read_csv("17_7_60_3_all_td.csv")
    #
    all_norms = pd.read_csv("25sim_lr.csv")
    leftright = pd.read_csv("25sim_rl.csv")
    another = pd.read_csv("25sim_td.csv")
    anotheranother = pd.read_csv("25sim_bu.csv")
    all_norms["0"] = (all_norms["0"] + another["0"]+leftright["0"]+anotheranother["0"])/4


    tick_0 = all_norms.loc[all_norms["tick number"] == 0]
    train_tick_0 = [tick_0.iloc[i].loc["0"] for i in train]
    test_tick_0 = [tick_0.iloc[i].loc["0"] for i in test]

    tick_48 = all_norms.loc[all_norms["tick number"] == 48]
    train_tick_48 = [tick_48.iloc[i].loc["0"] for i in train]
    test_tick_48 = [tick_48.iloc[i].loc["0"] for i in test]

    tick_96 = all_norms.loc[all_norms["tick number"] == 96]
    train_tick_96 = [tick_96.iloc[i].loc["0"] for i in train]
    test_tick_96 = [tick_96.iloc[i].loc["0"] for i in test]

    tick_144 = all_norms.loc[all_norms["tick number"] == 144]
    train_tick_144 = [tick_144.iloc[i].loc["0"] for i in train]
    test_tick_144 = [tick_144.iloc[i].loc["0"] for i in test]



    divider1 = (min(train_tick_48) + max(train_tick_0))/2
    # if (min(train_tick_48) - max(train_tick_0)) < 0:
    #     print(f'overlap between 0 and 48 of {min(train_tick_48) - max(train_tick_0)}')

    divider2 = (min(train_tick_96) + max(train_tick_48))/2
    # if (min(train_tick_96) - max(train_tick_48)) < 0:
    #     print(f'overlap between 96 and 48 of {min(train_tick_96) - max(train_tick_48)}')

    divider3 = (min(train_tick_144) + max(train_tick_96))/2
    # if (min(train_tick_144) - max(train_tick_96)) < 0:
    #     print(f'overlap between 96 and 144 of {min(train_tick_144) - max(train_tick_96)}')


    correct_0 = 0
    for tick in test_tick_0:
        if predict_tick(tick) == 0:
            correct_0 = correct_0 + 1
    perc_0.append(correct_0)

    correct_48 = 0
    for tick in test_tick_48:
        if predict_tick(tick) == 48:
            correct_48 = correct_48 + 1
    perc_48.append(correct_48)

    correct_96 = 0
    for tick in test_tick_96:
        if predict_tick(tick) == 96:
            correct_96 = correct_96 + 1
    perc_96.append(correct_96)

    correct_144 = 0
    for tick in test_tick_144:
        if predict_tick(tick) == 144:
            correct_144 = correct_144 + 1
    perc_144.append(correct_144)


    perc_all.append(correct_0+correct_48+correct_96+correct_144)

print(sum(perc_0)/300)
print(sum(perc_48)/300)
print(sum(perc_96)/300)
print(sum(perc_144)/300)
print(sum(perc_all)/1200)



# exit()


"""

train test split analysis

"""


def predict_tick(pnorm):
    if pnorm <= divider1:
        return 0
    if pnorm <= divider2:
        return 48
    if pnorm <= divider3:
        return 96
    return 144

perc_0 = []
perc_48 = []
perc_96 = []
perc_144 = []
perc_all = []



"""

graphs the p norms from simulations 1-100, and the histogram for each time tick

"""

tick_0 = all_norms.loc[all_norms["tick number"] == 0]
tick_48 = all_norms.loc[all_norms["tick number"] == 48]
tick_96 = all_norms.loc[all_norms["tick number"] == 96]
tick_144 = all_norms.loc[all_norms["tick number"] == 144]

size = 3
plt.scatter(tick_0["0"], tick_0["tick number"], s=size)
plt.scatter(tick_48["0"], tick_48["tick number"], s=size)
plt.scatter(tick_96["0"], tick_96["tick number"], s=size)
plt.scatter(tick_144["0"], tick_144["tick number"], s=size)

# plots the midlines
plt.plot([divider1, divider1], [-10, 200], "black")
plt.plot([divider2, divider2], [-10, 200], "black")
plt.plot([divider3, divider3], [-10, 200], "black")

plt.ylim(0, 150)
plt.ylabel("Time Tick")
plt.xlabel("Persistence Landscape Norms")
plt.show()

# # exit()
#
# plt.hist(tick_0["0"])
# plt.show()
#
# plt.hist(tick_48["0"])
# plt.show()
#
# plt.hist(tick_96["0"])
# plt.show()
#
# plt.hist(tick_144["0"])
# plt.show()
#
# print(tick_0.mean(), tick_0.std())
# print(tick_48.mean(), tick_48.std())
# print(tick_96.mean(), tick_96.std())
# print(tick_144.mean(), tick_144.std())
#
#
# # prints the index of the outlier simulations
# for i in range(len(tick_144)):
#     if tick_144.iloc[i]["0"] > 375:
#         print(i)



"""

graphs the p norm for each simulation

"""

train_tick_0 = [tick_0.iloc[i].loc["0"] for i in range(25)]
train_tick_48 = [tick_48.iloc[i].loc["0"] for i in range(25)]
train_tick_96 = [tick_96.iloc[i].loc["0"] for i in range(25)]
train_tick_144 = [tick_144.iloc[i].loc["0"] for i in range(25)]

for i in range(25):
    plt.plot([0, 48, 96, 144], [train_tick_0[i], train_tick_48[i], train_tick_96[i], train_tick_144[i]])
plt.xlabel("Time Tick")
plt.ylabel("Norm of Persistence Landscape")
plt.show()
print(type(train_tick_0))

plt.plot([0, 48, 96, 144],[sum(train_tick_0)/25, sum(train_tick_48)/25, sum(train_tick_96)/25, sum(train_tick_144)/25])
plt.xlabel("Time Tick")
plt.ylabel("Average Norm of Persistence Landscape")
plt.show()





"""

graphs the vectors and sums from simulations 1-100

"""

#
# all_vectors = pd.read_csv("vectors_152_6.csv")
# all_vectors.fillna(0, inplace=True)
#
# tick_0 = all_vectors.loc[all_vectors["tick number"] == 0]
# tick_48 = all_vectors.loc[all_vectors["tick number"] == 48]
# tick_96 = all_vectors.loc[all_vectors["tick number"] == 96]
# tick_144 = all_vectors.loc[all_vectors["tick number"] == 144]
#
# tick_0_mean = format_list(tick_0.mean())
# tick_48_mean = format_list(tick_48.mean())
# tick_96_mean = format_list(tick_96.mean())
# tick_144_mean = format_list(tick_144.mean())
#
# sums = pd.read_csv("sums_normalized_lr.csv")
# sums_means = format_list_sums(sums.mean())
#
# # plt.plot(tick_0_mean[0], tick_0_mean[1])
# # plt.plot(tick_48_mean[0], tick_48_mean[1])
# # plt.plot(tick_96_mean[0], tick_96_mean[1])
# # plt.plot(tick_144_mean[0], tick_144_mean[1])
# # plt.show()
#
# plt.plot(sums_means[0], sums_means[1])
# plt.xlabel("Time Tick")
# plt.ylabel("Average Sum")
# plt.show()
# exit()