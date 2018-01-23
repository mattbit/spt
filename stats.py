import pandas
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Histogram, Scatter, Layout, Figure

py.init_notebook_mode()

DATAFILE = "data_125ms.csv"

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])

# Ensure data is sorted
data.sort_index()


#####################################################################
# Divide the space in squares.                                      #
#####################################################################

x_min, x_max = data["x"].min(), data["x"].max()
y_min, y_max = data["y"].min(), data["y"].max()

L = max(x_max - x_min, y_max - y_min)
NUM_DIVISIONS = 20

delta = L / NUM_DIVISIONS

def square_index(x, y):
    ii = ((x - x_min) // delta).astype(int, copy=False)
    jj = ((y - y_min) // delta).astype(int, copy=False)

    return list(zip(ii, jj))


def step_vector(data):
    steps = np.empty((len(data), 2))

    i = 0
    for track_id, traj in data.loc[:, ("x", "y")].groupby("track id"):
        for j in range(len(traj) - 1):
            steps[i+j] = traj.values[j+1] - traj.values[j]

        i += len(traj)
        steps[i-1] = np.nan

    return steps

# Add a square index
data["square"] = square_index(data["x"], data["y"])

# Add step size to the next position (NaN for the last point)
steps = step_vector(data)
data["step_x"], data["step_y"] = steps[:, 0], steps[:, 1]


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


def heatmap(z, title=""):
    delta = L/NUM_DIVISIONS
    x = np.linspace(x_min + delta/2, x_min + L - delta/2, NUM_DIVISIONS)
    y = np.linspace(y_min + delta/2, y_min + L - delta/2, NUM_DIVISIONS)
    htm = Heatmap(z=z.T, x=x, y=y)
    lyt = Layout(
        title=title,
        height=600,
        width=600,
        yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
                   autotick=False, dtick=delta, tick0=y_min),
        xaxis=dict(showgrid=True, zeroline=False,
                   autotick=False, dtick=delta, tick0=x_min)
    )

    fig = Figure(data=[htm], layout=lyt)
    py.iplot(fig)


#####################################################################
# Plot the global distribution of the step size.                    #
#####################################################################

steps = np.linalg.norm(data.loc[:, ("step_x", "step_y")].values, axis=1)

py.iplot([Histogram(x=steps[~np.isnan(steps)], histnorm="probability")])


#####################################################################
# Plot a heatmap of data point count.                               #
#####################################################################

count = np.zeros((NUM_DIVISIONS, NUM_DIVISIONS))
for i in range(NUM_DIVISIONS):
    for j in range(NUM_DIVISIONS):
        num = len(data.loc[data["square"] == (i, j)])
        # Use NaN just to be nice and avoid clutter
        count[(i, j)] = num if num > 0 else np.nan


heatmap(count)

#####################################################################
# Plot a heatmap of the step size.                                  #
#####################################################################

stepnorm = np.zeros((NUM_DIVISIONS, NUM_DIVISIONS), dtype=np.float)
for i in range(NUM_DIVISIONS):
    for j in range(NUM_DIVISIONS):
        values = data.loc[data["square"] == (i, j), ("step_x", "step_y")].values
        steps = np.linalg.norm(values, axis=1)

        stepnorm[(i, j)] = np.nanmean(steps) if len(steps) > 0 else np.nan

heatmap(stepnorm, "Step size")


#####################################################################
# Estimate the drift.                                               #
#####################################################################
driftnorm = np.empty((NUM_DIVISIONS, NUM_DIVISIONS))
for i in range(NUM_DIVISIONS):
    for j in range(NUM_DIVISIONS):
        drift = np.nanmean(
            data.loc[data["square"] == (i, j), ("step_x", "step_y")].values,
            axis=0
        )

        driftnorm[(i, j)] = np.linalg.norm(drift)

heatmap(driftnorm, "Drift modulus")


#####################################################################
# Estimate the diffusion (assumed isotropic).                       #
#####################################################################

diffnorm = np.empty((NUM_DIVISIONS, NUM_DIVISIONS))
for i in range(NUM_DIVISIONS):
    for j in range(NUM_DIVISIONS):
        steps = data.loc[data["square"] == (i, j), ("step_x", "step_y")].values
        diffnorm[(i, j)] = np.sqrt(np.nanmean(np.sum(steps**2, axis=1)))

heatmap(diffnorm, "Diffusion coefficient")


#####################################################################
# Drift vector field.                                               #
#####################################################################

# Prepare the grid
# @TODO: refactor this mess
delta = L/NUM_DIVISIONS
x = np.linspace(x_min + delta/2, x_min + L - delta/2, NUM_DIVISIONS)
y = np.linspace(y_min + delta/2, y_min + L - delta/2, NUM_DIVISIONS)
xx, yy = np.meshgrid(x, y, indexing="ij")  # @TODO: check indexing


u = np.zeros((NUM_DIVISIONS, NUM_DIVISIONS))
v = np.zeros_like(u)

for i in range(NUM_DIVISIONS):
    for j in range(NUM_DIVISIONS):
        drift = np.nanmean(
            data.loc[data["square"] == (i, j), ("step_x", "step_y")].values,
            axis=0
        )

        u[(i, j)] = drift[0]
        v[(i, j)] = drift[1]

fig = ff.create_quiver(xx, yy, u, v, scale=100)


lyt = Layout(
    title="Drift field",
    height=600,
    width=600,
    yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
               autotick=False, dtick=delta, tick0=y_min),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, dtick=delta, tick0=x_min)
)

fig.update(layout=lyt)

py.iplot(fig)
