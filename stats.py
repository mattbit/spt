import pandas
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.graph_objs import Heatmap, Histogram, Scatter

DATAFILE = "data_125ms.csv"

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])

"""
#######################################
# Things to do                        #
#######################################

1. Plot histogram with global distribution for the step size.
2. Plot heatmap of local mean step size, to find regions with
   higher drift/diffusivity.
3. Compare with the drift/diffusivity empirical coefficients
   (average step size and drift/diffuivity must be correlated)
4. Try to find pathways with high directed drift.


"""

# Ensure data is sorted
data.sort_index()

# Utils

def plot_trajectories(trajectories):
    """Plot the trajectories."""

    data = []

    for track_id, traj in trajectories:
        data.append(Scatter(
            x=traj["x"].values,
            y=traj["y"].values,
            name=track_id))

    py.plot(data)

#####################################################################
# Plot a histogram with the distribution of the step size.          #
#####################################################################
"""
stepsize = np.array([])

for track_id, traj in data.groupby("track id"):
    t1 = np.delete(traj.values, 0, axis=0)
    t2 = np.delete(traj.values, -1, axis=0)
    
    stepsize = np.concatenate((stepsize, [np.nan], np.linalg.norm(t1 - t2, axis=1)))

py.plot([Histogram(x=stepsize, histnorm="probability")], filename='step_size_dist.html')
"""

#####################################################################
# Divide the system in squares.                                     #
#####################################################################
x_min, x_max = data["x"].min(), data["x"].max()
y_min, y_max = data["y"].min(), data["y"].max()

L = max(x_max - x_min, y_max - y_min)
DIVISIONS = 10

x = np.linspace(min(x_min, y_min), max(x_max, y_max), DIVISIONS + 1)


# plot_trajectories(data.head(10000).groupby("track id"))


