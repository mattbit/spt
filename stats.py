import pandas
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Histogram, Scatter, Layout, Figure

DATAFILE = "data_125ms.csv"

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
        self.ndiv = ndiv

        x_min, x_max = data["x"].min(), data["x"].max()
        y_min, y_max = data["y"].min(), data["y"].max()

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

        # Add the square index column.
        self.data["square"] = square_index(self.data["x"], self.data["y"])

        # Add step to the next position (NaN for last point of traj).
        steps = step_vector(self.data)
        self.data["step_x"], self.data["step_y"] = steps[:, 0], steps[:, 1]


    def apply(self, func):
        """Applies a function on each non-empty grid square."""
        z = np.full((self.ndiv, self.ndiv), np.nan)

        for ij, data in self.data.groupby("square"):
            z[ij] = func(data)

        return z

    def heatmap(self, func, title=""):
        htm = Heatmap(z=self.apply(func).T, x=self.x, y=self.y)

        lyt = self._plot_layout(title)

        fig = Figure(data=[htm], layout=lyt)
        py.iplot(fig)

    def _plot_layout(self, title):
        x_min = self.data["x"].min()
        y_min = self.data["y"].min()
        deltaL = self.L / self.ndiv

        return Layout(
            title=title,
            height=600,
            width=600,
            yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
                       autotick=False, dtick=deltaL, tick0=y_min),
            xaxis=dict(showgrid=True, zeroline=False,
                       autotick=False, dtick=deltaL, tick0=x_min)
        )

#%%

#####################################################################
# Utilities.                                                        #
#####################################################################

def plot_trajectories(trajectories):
    """Plot the trajectories."""

    traces = []

    for track_id, traj in trajectories:
        traces.append(Scatter(
            x=traj["x"].values,
            y=traj["y"].values,
            name=track_id))

    py.plot(traces)

# %%
#####################################################################
# Create the grid object.                                           #
#####################################################################

grid = Grid(data, 50)

# %%
#####################################################################
# Plot the global distribution of the step size.                    #
#####################################################################

steps = np.linalg.norm(grid.data.loc[:, ("step_x", "step_y")].values, axis=1)

py.iplot([Histogram(x=steps[~np.isnan(steps)], histnorm="probability")])


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


grid.heatmap(step_norm, "Step modulus")


# %%
#####################################################################
# Estimate the drift.                                               #
#####################################################################

def drift_norm(data):
    vals = data.loc[:, ("step_x", "step_y")].values

    return np.linalg.norm(np.nanmean(vals))

grid.heatmap(drift_norm, "Drift modulus")


# %%
#####################################################################
# Estimate the diffusion (assumed isotropic).                       #
#####################################################################

def diff_norm(data):
    vals = data.loc[:, ("step_x", "step_y")].values

    return np.sqrt(np.nanmean(np.sum(vals**2, axis=1)))

grid.heatmap(diff_norm, "Diffusion coefficient")

# %%
#####################################################################
# Drift vector field.                                               #
#####################################################################

xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")

u = np.full((grid.ndiv, grid.ndiv), np.nan)
v = u.copy()

for ij, data in grid.data.groupby("square"):

    if len(data) < 200:
        continue

    drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                       axis=0)

    u[ij] = drift[0]
    v[ij] = drift[1]

fig = ff.create_quiver(xx, yy, u, v, scale=100)

lyt = grid._plot_layout("Drift field")
fig.update(layout=lyt)
py.iplot(fig)
