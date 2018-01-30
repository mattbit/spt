import pandas
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Histogram, Scatter, Layout, Figure, Histogram2d

import scipy.stats as stats
from scipy.stats import norm as normal

DATAFILE = "data_125ms.csv"
TIMESTEP = 125e-3

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])


class Grid(object):
    def __init__(self, data, ndiv):
        """Initializes the grid.
        Precalculates block indices and the steps between datapoints
        of the trajectories.
        """
        self.data = data.sort_index()  # Ensure data is sorted.

        # Calculates the displacement vector.
        def step_vector(data):
            steps = np.empty((len(data), 2))

            i = 0
            for track_id, traj in data.loc[:, ("x", "y")].groupby("track id"):
                for j in range(len(traj) - 1):
                    steps[i+j] = traj.values[j+1] - traj.values[j]

                i += len(traj)
                steps[i-1] = np.nan

            return steps

        # Add step to the next position (NaN for last point of traj).
        steps = step_vector(self.data)
        self.data["step_x"], self.data["step_y"] = steps[:, 0], steps[:, 1]

        # Set the grid and add square indices.
        self.set_grid(ndiv)


    def set_grid(self, ndiv):
        """Recalculates the grid with ndiv divisions."""
        self.ndiv = ndiv

        x_min, x_max = self.data["x"].min(), self.data["x"].max()
        y_min, y_max = self.data["y"].min(), self.data["y"].max()

        self.L = max(x_max - x_min, y_max - y_min)
        deltaL = self.L/self.ndiv
        self.x = np.linspace(x_min + deltaL/2,
                             x_min + self.L - deltaL/2, ndiv)
        self.y = np.linspace(y_min + deltaL/2,
                             y_min + self.L - deltaL/2, ndiv)

        # Calculates the square index given the coordinates.
        def square_index(x, y):
            ii = ((x - x_min - 1e-6) // deltaL).astype(int, copy=False)
            jj = ((y - y_min - 1e-6) // deltaL).astype(int, copy=False)

            return list(zip(ii, jj))

        # Add the square index column.
        self.data["square"] = square_index(self.data["x"], self.data["y"])


    def apply(self, func, threshold=0):
        """Applies a function on each non-empty grid square."""
        z = np.full((self.ndiv, self.ndiv), np.nan)

        for ij, data in self.data.groupby("square"):
            if len(data) > threshold:
                z[ij] = func(data)

        return z

    def heatmap(self, data, title="", threshold=None):
        if callable(data):
            data = self.apply(data, threshold)

        htm = Heatmap(z=data.T)

        lyt = self._plot_layout(title)

        fig = Figure(data=[htm], layout=lyt)
        py.iplot(fig)

    def count(self, threshold):
        n = self.apply(lambda d: 1, threshold)

        return np.nansum(n)

    def _plot_layout(self, title):
        # x_min = self.data["x"].min()
        # y_min = self.data["y"].min()
        # deltaL = self.L / self.ndiv

        return Layout(
            title=title,
            height=600,
            width=600,
            yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
                       autotick=False, dtick=1, tick0=-0.5,
                       showticklabels=False),
            xaxis=dict(showgrid=True, zeroline=False,
                       autotick=False, dtick=1, tick0=-0.5,
                       showticklabels=False)
        )

# %%
#####################################################################
# Create the grid object.                                           #
#####################################################################

grid = Grid(data, 40)

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
grid.heatmap(len, "Number of datapoints")

# %%
#####################################################################
# Plot a heatmap of the step size.                                  #
#####################################################################

def step_norm(data):
    values = data.loc[:, ("step_x", "step_y")].values
    steps = np.linalg.norm(values, axis=1)

    return np.nanmean(steps) if len(steps) > 0 else np.nan


grid.heatmap(step_norm, "Step modulus", threshold=200)


# %%
#####################################################################
# Estimate the drift.                                               #
#####################################################################

def drift_norm(data):
    vals = data.loc[:, ("step_x", "step_y")].values

    return np.linalg.norm(np.nanmean(vals))

grid.heatmap(drift_norm, "Drift modulus", threshold=200)


# %%
#####################################################################
# Estimate the diffusion (assumed isotropic).                       #
#####################################################################

def diff_norm(data):
    vals = data.loc[:, ("step_x", "step_y")].values

    for i in range(2):
        for j in range(2):


    return np.sqrt(np.nanmean(np.sum(vals**2, axis=1)))

grid.heatmap(diff_norm, "Diffusion coefficient", threshold=200)

# %%
#####################################################################
# Drift vector field.                                               #
#####################################################################

def estimate_drift(grid, threshold=0):
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")

    u = np.full((grid.ndiv, grid.ndiv), np.nan)
    v = u.copy()

    for ij, data in grid.data.groupby("square"):

        if len(data) < threshold:
            continue

        drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                           axis=0)

        u[ij] = drift[0]
        v[ij] = drift[1]

    return xx, yy, u, v

xx, yy, u, v = estimate_drift(grid)

fig = ff.create_quiver(xx, yy, u, v, scale=100)

lyt = grid._plot_layout("Drift field")
fig.update(layout=lyt)
py.iplot(fig)


# %%
#####################################################################
# Standard error of the mean drift.                                 #
#####################################################################


grid.heatmap(len, threshold=0)

# %%
data = grid.data.loc[grid.data["square"] == (7, 16)]
ss = data["step_x"].values
x = np.linspace(-0.4, 0.4, 1000)
y = normal.pdf(x, loc=np.nanmean(ss), scale=0.5*np.nanstd(ss))

py.iplot([
    Histogram(x=ss),
    Scatter(x=x, y=y*50)
])




# %%
#####################################################################
# Test normality of step distribution inside each square.           #
#####################################################################

# My question: is the noise really Gaussian?

ALPHA = 1e-3  # This is quite small

u, v = grid.apply(estimate_drift)

def test_normal(data):
    x = data.loc[:, "step_x"].values
    _, p = stats.normaltest(x, nan_policy="omit")

    if p < ALPHA:
        # Null hypothesis rejected (not normal).
        return 0

    return 1

grid.set_grid(40)
count = grid.apply(len, threshold=0)
grid.heatmap(test_normal, threshold=50)


ndivs = range(25, 201, 25)
n50 = []
n200 = []
acceptance = []
for ndiv in ndivs:
    grid.set_grid(ndiv)
    count = grid.apply(len, threshold=0)
    n1 = grid.apply(test_normal, threshold=200)

    n50.append(np.nansum(n1) / grid.count(0))

    acceptance.append(grid.count(50) / grid.count(0))

    # n2 = grid.apply(test_normal, threshold=200)
    # n200.append(np.nansum(n2) / np.sum(count[~np.isnan(count)] > 200))

py.iplot([
    Scatter(x=list(ndivs), y=n50, name="Normal"),
    Scatter(x=list(ndivs), y=acceptance, name="Acceptance"),
])


# %%
# Estimate the error
D = grid.apply(diff_norm, threshold=50) / 2
N = grid.apply(len)

TIMESTEP

np.nanmax(D)

def drift_error(threshold):
    D = grid.apply(diff_norm, threshold) / 2
    N = grid.apply(len)

    return np.sqrt(2*D / (TIMESTEP * N))

xx, yy, u, v = estimate_drift(grid, threshold=200)
np.nanmean(np.sqrt(u**2 + v**2))
np.nanmax(drift_error(500))


# Test

testdata = pandas.read_csv("test_dataset.csv", sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])

testgrid = Grid(testdata, 40)
testgrid.L
D = testgrid.apply(diff_norm, threshold=200)
np.nanmean(D)

dd = testgrid.data.loc[testgrid.data["square"] == (5, 5)]

steps = dd.loc[:, ("step_x", "step_y")].values
steps_x = dd.loc[:, ("step_x")].values
steps_y = dd.loc[:, ("step_y")].values

D[0, 0] = np.nanmean(steps_x**2) / TIMESTEP
D[0, 1] = D[1, 0] = np.nanmean(steps_x*steps_y) / TIMESTEP
D[1, 1] = np.nanmean(steps_y**2) / TIMESTEP
np.matmul(D, D.T)
np.nanmean(steps[:, 0]*steps[:, 1])

D = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        D[i, j]
