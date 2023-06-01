import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# takes in data set and splits it up into n by n disjoint cubes
# will color in cube if at least m cells are in cube
def make_cubes(file_name, n, m):
    # organize file
    df = pd.read_csv(file_name)
    df.columns = ["x", "y", "color"]
    df = df[["x", "y"]]
    print(len(df))

    max_x = df["x"].max()
    min_x = df["x"].min()
    max_y = df["y"].max()
    min_y = df["y"].min()
    range_x = max_x - min_x
    range_y = max_y - min_y

    # length of the side of each cube/square
    len_x = range_x/n
    len_y = range_y/n

    # location of each of the lines dividing the data set
    x_values = [min_x + (i*len_x) for i in range(n+1)]
    y_values = [min_y + (i*len_y) for i in range(n+1)]

    # set up plot
    fig, ax = plt.subplots()
    ax.set_xlim([-101, 101])
    ax.set_ylim([-101, 101])

    count = 0
    for i in range(n):
        for j in range(n):
            if help_make_cube(x_values[i], x_values[i+1], y_values[j], y_values[j+1], df) >= m:
                square = Rectangle((x_values[i], y_values[j]), len_x, len_y, fill=True, color='green')
                count = count + 1
                ax.add_patch(square)
    print(count)
    plt.show()



def help_make_cube(x1, x2, y1, y2, df):
    df = df.loc[df["x"] >= x1]
    df = df.loc[df["x"] < x2]
    df = df.loc[df["y"] >= y1]
    df = df.loc[df["y"] < y2]
    return len(df)

make_cubes("NetLogo_50sim_03232023/sim51_144.csv", 150, 2)