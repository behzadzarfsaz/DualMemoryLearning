from math import exp
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    go = 100
    points = []
    points_go = []
    for u in range(1, 1000):
        if u < go:
            points_go.append(1.0)
        else:
            points_go.append(exp(-1 * ((10 * (u - go)) / 1000)))
        points.append(exp(-1 * (10 * u / 1000)))

    plt.grid(True)
    plt.title('Decay Reduction with Global-Ordering=' + str(go))
    plt.plot(np.array(points), linewidth=3)
    plt.plot(np.array(points_go), linewidth=3)
    plt.legend(["Decay without GO", "Decay with GO"])
    plt.show()
