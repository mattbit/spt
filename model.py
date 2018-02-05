import pandas
import numpy as np
import scipy.stats
import scipy.optimize
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Histogram, Scatter, Layout, Figure

from grid import Grid

DATAFILE = "data_125ms.csv"
TIMESTEP = 125e-3

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])


# %%
#####################################################################
# Create the grid object.                                           #
#####################################################################

grid = Grid(data, binsize=0.5)

# %%
#####################################################################
# Plot the global distribution of the step size.                    #
#####################################################################

steps = np.linalg.norm(grid.data.loc[:, ("step_x", "step_y")].values, axis=1)

py.iplot([Histogram(x=steps[~np.isnan(steps)], histnorm="probability")])

steps_x = grid.data.loc[:, "step_x"].values
steps_y = grid.data.loc[:, "step_y"].values

py.iplot([Histogram2d(
    x=steps_x,
    y=steps_y,
    histnorm="probability"
)])



# %%
#####################################################################
# Plot a heatmap of data point count.                               #
#####################################################################

grid.heatmap(len, "Number of datapoints", threshold=100)

# %%
#####################################################################
# Plot a heatmap of the step size.                                  #
#####################################################################

def step_norm(data):
    values = data.loc[:, ("step_x", "step_y")].values
    steps = np.linalg.norm(values, axis=1)

    return np.nanmean(steps) if len(steps) > 0 else np.nan

grid.heatmap(step_norm, "Step modulus", threshold=100)


# %%
#####################################################################
# Estimate the diffusion.                                           #
#####################################################################

def diff_tensor_xx(data):
    return np.nanmean(data.loc[:, "step_x"]**2) / TIMESTEP

def diff_tensor_xy(data):
    return np.nanmean(data.loc[:, "step_x"]*data.loc[:, "step_y"]) / TIMESTEP

def diff_tensor_yy(data):
    return np.nanmean(data.loc[:, "step_y"]**2) / TIMESTEP

diff_xx = grid.apply(diff_tensor_xx, threshold=100)
diff_xy = grid.apply(diff_tensor_xy, threshold=100)
diff_yy = grid.apply(diff_tensor_yy, threshold=100)

D = 0.5*diff_xx + 0.5*diff_yy

grid.heatmap(D, title="Diffusion coefficient")

grid.heatmap(abs(diff_xy/D), title="Isotropy of diffusion")


# %%
#####################################################################
# Drift vector field.                                               #
#####################################################################

from utils import estimate_drift

xx, yy, u, v = estimate_drift(grid, TIMESTEP, threshold=100)

grid.heatmap(np.hypot(u, v), threshold=100, title="Drift norm")
