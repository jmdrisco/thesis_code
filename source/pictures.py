import matplotlib.pyplot as plt
import numpy as np

import random

def generate_random_coordinates(n):
    coordinates = []
    for _ in range(n):
        x = random.uniform(0, 2)
        y = random.uniform(0, 2)
        coordinates.append((x, y))
    return coordinates

def plot_circle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 100)  # Generate 100 points on the circle

    # Compute the coordinates of the circle
    circle_x = r * np.cos(theta) + x
    circle_y = r * np.sin(theta) + y

    # Plot the circle
    plt.plot(circle_x, circle_y)
    plt.plot(x, y, 'o', label='Center', color=plt.gca().lines[-1].get_color())
    plt.axis('equal')  # Set equal scaling on x and y axes
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Circle Plot')
    plt.axis("off")
    plt.grid(False)

e=1
x=[]
y =[]
for n in generate_random_coordinates(6):
    x.append(n[0])
    y.append(n[1])
    plot_circle(n[0], n[1], e)

plt.show()


exit()

z = [1.1, 1.3, 1.5, 2.5, 3, 4.5, 6.3, 5, 4.3, 3.1, 4.9, 6.5, 8.7, 8.8, 7.5]
y = [1.5, .5, 2.1, 1.8, 1.7, 1.9, 2.4, 3.1, 2.9, 3.9, 3.7, 4.3, 3.5, 3.2, 2.9]

plt.scatter([1, 1.2], [1.5, .5], s=10)
plt.scatter([1.5, 2.5, 3, 4.5, 6.2, 5, 4.3], [2, 1.7, 1.6, 1.8, 2.4, 3, 2.8], s=10)
plt.scatter([3, 5, 6.5, 8.6, 8.7, 7.6], [3.8, 3.6, 4.2, 3.5, 3.2, 2.8], s=10)

n = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

for i, txt in enumerate(n):
    plt.annotate(txt, (z[i], y[i]))

plt.xlim(0, 10)
plt.ylim(0, 5)
plt.plot([1, 1.2], [1.5, .5])
plt.plot([1.5, 2.5, 3, 4.5, 6.2, 5, 4.3], [2, 1.7, 1.6, 1.8, 2.4, 3, 2.8])
plt.plot([3, 5, 6.5, 8.6, 8.7, 7.6], [3.8, 3.6, 4.2, 3.5, 3.2, 2.8])


# deltas
plt.plot([2, 2], [-1, 11], "black")
plt.plot([4, 4], [-1, 11], "black")
plt.plot([6, 6], [-1, 11], "black")
plt.plot([8, 8], [-1, 11], "black")

plt.show()


exit()

from matplotlib.patches import Rectangle

def draw_square(x, y, z):
    """
    Draws the outline of a square of length z with its upper left corner at location (x,y)
    """
    fig, ax = plt.subplots()
    square = Rectangle((x, y), z, z, fill=False, lw=3, color='blue')

    ax.set_xlim([x-5, x+z+5])
    ax.set_ylim([y-5, y+z+5])
    ex = pd.read_csv("NetLogo_50sim_03062023/sim22_96.csv")
    ex.columns = ["x", "y", "color"]
    ex = ex[["x", "y"]]
    ex.to_numpy()
    ex.columns = ["x", "y"]
    plt.scatter(ex["x"], ex["y"], s=.5, color='black')
    ax.add_patch(square)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.axis("off")
    plt.savefig('flow2.png')
    plt.show()

# 19 cubes and .5 overlap --> 20
# cube 7, 5  has location: (-30, -10), (-50, -30)
# for cube 4, 3: [

# draw_square(-30, -50, 20)
