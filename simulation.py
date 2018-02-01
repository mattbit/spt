import pandas
import numpy as np
from grid import Grid
from utils import estimate_drift
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.graph_objs import Heatmap, Layout, Figure, Scatter, Contour

DATAFILE = "data_125ms.csv"
TIMESTEP = 125e-3

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])


grid = Grid(data, binsize=0.5)

# Add threshold and filter out the isolated bins
THRESHOLD = 50

tdata = grid.data.groupby("bin").filter(lambda x: len(x) > THRESHOLD)
bins = sorted(tdata["bin"].unique())
filtered_bins = []


for b in bins:
    if ((b[0] + 1, b[1]) in bins or (b[0] - 1, b[1]) in bins
       or (b[0], b[1] + 1) in bins or (b[0], b[1] - 1) in bins):
       filtered_bins.append(b)


def neighbours(b):
    return [(b[0] + 1, b[1]), (b[0] - 1, b[1]),
            (b[0], b[1] + 1), (b[0], b[1] - 1)]

extended_bins = filtered_bins.copy()
for b in filtered_bins:
    for n in neighbours(b):
        if n not in extended_bins:
            extended_bins.append(n)

z = np.full([grid.ndiv+1, grid.ndiv+1], np.nan)
for b in extended_bins:
    z[b] = 1

for b in filtered_bins:
    z[b] += 1

grid.plot(Heatmap(z=z.T, colorscale=[[0, "rgb(120,120,120)"], [1, "rgb(0,0,0)"]]), "Domain")

# Estimate the drift field
xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")

u = np.zeros((grid.ndiv, grid.ndiv))
v = u.copy()

for b, data in grid.data.groupby("bin"):
    if b not in filtered_bins:
        continue

    drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                       axis=0) / TIMESTEP

    u[b] = drift[0]
    v[b] = drift[1]


grid.heatmap(np.sqrt(u**2 + v**2), "Estimated drift")

# Smooth the field

us = np.full((grid.ndiv+1, grid.ndiv+1), np.nan)
vs = us.copy()

for b in extended_bins:
    nsum_u = 0
    nsum_v = 0
    for n in neighbours(b):
        if b[0] < grid.ndiv and b[1] < grid.ndiv:
            nsum_u += u[n]
            nsum_v += v[n]

    us[b] = 0.5*u[b] + 0.125*nsum_u
    vs[b] = 0.5*v[b] + 0.125*nsum_v


# Use 0 instead of NaN to get a better plot.
plot_us = us.copy()
plot_us[np.isnan(plot_us)] = 0.
plot_vs = vs.copy()
plot_vs[np.isnan(plot_vs)] = 0.

grid.heatmap(np.sqrt(plot_us**2 + plot_vs**2), "Smoothed drift field")


# Now wait for last year.

NSTEPS = 1000
dt = 0.1 * grid.binsize / np.mean(np.sqrt(u**2 + v**2))
x_min = min(grid.data["x"])
y_min = min(grid.data["y"])

x = x_min + np.random.random(1000000) * grid.L
y = y_min + np.random.random(1000000) * grid.L

def bin_index(x, y):
    ii = ((x - x_min) // grid.binsize).astype(int, copy=False)
    jj = ((y - y_min) // grid.binsize).astype(int, copy=False)

    return ii, jj


def density(x, y):
    h, _, _ = np.histogram2d(x[~np.isnan(x)], y[~np.isnan(y)],
                  bins=grid.ndiv,
                  range=[[x_min, x_min + grid.L],
                         [y_min, y_min + grid.L]],
                  normed=True)

    return h

for n in range(1000):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    ii, jj = bin_index(x, y)
    mask = np.logical_and(ii < grid.ndiv, jj < grid.ndiv)
    x[mask] += us[ii[mask], jj[mask]] * dt
    y[mask] += vs[ii[mask], jj[mask]] * dt

p = density(x, y)
grid.heatmap(p)

grid.heatmap((p > 0.10)*1)


# Attractors with the biggest basin.

bins = np.empty((grid.ndiv, grid.ndiv), dtype=tuple)
for i in range(grid.ndiv):
    for j in range(grid.ndiv):
        bins[i, j] = (i, j)

p.shape
bins.shape

# Select the biggest attractors.
bins[p > 0.15]


def mask2d(grid, x, r=5):
    mask = np.full((grid.ndiv, grid.ndiv), False)

    for i in range(max(x[0] - r, 0), min(x[0] + r, grid.ndiv)):
        for j in range(max(x[1] - r, 0), min(x[1] + r, grid.ndiv)):
            mask[i, j] = True


    return mask

def mask1d(array, x, r=5):
    i_min = max(x - r, 0)
    i_max = min(x + r, len(array))

    return array[i_min:i_max]


def mask2d(array, x, r=5):
    i_min = max(x[0] - r, 0)
    i_max = min(x[0] + r, len(array))
    j_min = max(x[1] - r, 0)
    j_max = min(x[1] + r, len(array))

    return array[i_min:i_max, j_min:j_max]

A = (18, 43)


xxm = mask2d(xx, A, 10)
yym = mask2d(yy, A, 10)
um = mask2d(u, A, 10)
vm = mask2d(v, A, 10)
xm = mask1d(grid.x, A[0], 10)
ym = mask1d(grid.y, A[1], 10)


lyt = Layout(
    height=600,
    width=600,
    yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
               autotick=False, dtick=grid.binsize, tick0=0,
               showticklabels=False),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, dtick=grid.binsize, tick0=0,
               showticklabels=False)
)

grid.x[A[0]]
grid.y[A[1]]
fig = ff.create_quiver(xxm, yym, um, vm, scale=2)
fig["data"].append(Scatter(x=[grid.x[A[0]]], y=[grid.y[A[1]]], mode="markers"))
fig.layout = lyt
py.iplot(fig)


RADIUS = 2*grid.binsize
def MSE(params, x, y, u, v):
    x0, y0, A, r = params
    xx, yy = np.meshgrid(x, y)

    return np.mean((u + 2*A/r**2 * (xx - x0))**2 + (v + 2*A/r**2 * (yy - y0))**2)

x0 = grid.x[A[0]]
y0 = grid.y[A[1]]

initial = [x0, y0, 1., 5*grid.binsize]
bounds = [(x0-grid.binsize, x0+grid.binsize),
          (y0-grid.binsize, y0+grid.binsize),
          (-10, 10),
          (grid.binsize, 5*grid.binsize)]
res = scipy.optimize.minimize(MSE, initial, args=(xm, ym, um, vm), bounds=bounds)

res

fig = ff.create_quiver(xxm, yym, um, vm, scale=2)
fig["data"].append(Scatter(x=[res.x[0]], y=[res.x[1]], mode="markers"))
fig.layout = lyt
fig.layout.shapes = [{
            'type': 'circle',
            'xref': 'x',
            'yref': 'y',
            'x0': res.x[0] - res.x[3],
            'y0': res.x[1] - res.x[3],
            'x1': res.x[0] + res.x[3],
            'y1': res.x[1] + res.x[3],
            'line': {'color': 'rgba(50, 171, 96, 1)'}
}]
py.iplot(fig)
