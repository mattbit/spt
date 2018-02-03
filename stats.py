import pandas
import numpy as np
import scipy.stats
import scipy.optimize
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Histogram, Scatter, Layout, Figure, Histogram2d

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

grid = Grid(data, binsize=0.1)

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


std = np.nanstd(steps_x, ddof=1)
mean = np.nanmean(steps_x)
x = np.linspace(-0.5, 0.5, num=100)
y = normal.pdf(x, loc=mean, scale=0.5*std)/820

py.iplot([
    Histogram(x=steps_x, histnorm="probability", name="Empirical"),
    Scatter(x=x, y=y, name="Fit")])



py.iplot([Histogram(x=steps_y, histnorm="probability")])




# %%
#####################################################################
# Plot a heatmap of data point count.                               #
#####################################################################

grid.heatmap(len, "Number of datapoints", threshold=1)

# %%
#####################################################################
# Plot a heatmap of the step size.                                  #
#####################################################################

def step_norm(data):
    values = data.loc[:, ("step_x", "step_y")].values
    steps = np.linalg.norm(values, axis=1)

    return np.nanmean(steps) if len(steps) > 0 else np.nan

grid.heatmap(step_norm, "Step modulus", threshold=10)


# %%
#####################################################################
# Estimate the drift.                                               #
#####################################################################

def drift_norm(data):
    vals = data.loc[:, ("step_x", "step_y")].values

    return np.linalg.norm(np.nanmean(vals))

grid.heatmap(drift_norm, "Drift modulus", threshold=100)


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

grid.heatmap(0.5*diff_xx + 0.5*diff_yy)


# %%
#####################################################################
# Drift vector field.                                               #
#####################################################################

def estimate_drift(grid, threshold=20):
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")

    u = np.full((grid.ndiv, grid.ndiv), np.nan)
    v = u.copy()

    for ij, data in grid.data.groupby("bin"):

        if len(data) < threshold:
            continue

        drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                           axis=0) / TIMESTEP

        u[ij] = drift[0]
        v[ij] = drift[1]

    return xx, yy, u, v

xx, yy, u, v = estimate_drift(grid, threshold=100)

fig = ff.create_quiver(xx, yy, u, v, scale=10)


x_min = grid.data["x"].min()
y_min = grid.data["y"].min()

lyt = Layout(
    title="Drift field",
    height=600,
    width=600,
    yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
               autotick=False, dtick=grid.binsize, tick0=y_min,
               showticklabels=False),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, dtick=grid.binsize, tick0=x_min,
               showticklabels=False)
)

fig.update(layout=lyt)
py.iplot(fig)

np.nanmean(np.sqrt(u**2 + v**2))

drift = np.sqrt(u**2 + v**2)

grid.heatmap(drift < 0.035, threshold=100)

# %%
# Fit attractors

def bfit(x, A, r):
    return -2*A/r**2 * np.sum(x)

np.sqrt(2 * 0.05/(200*0.015))
