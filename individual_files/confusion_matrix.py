import numpy as np
from lib.plotter import Plotter

if __name__ == "__main__":
    values = np.arange(0, 100).astype("int32").reshape([10, 10])
    values[5, 1] = 999
    Plotter.plot_cm(values).show()
