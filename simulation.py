import pandas
import scipy
import numpy as np
from grid import Grid
from utils import estimate_drift
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.plotly import image as pyimg
import scipy.ndimage.filters as filters
from plotly.graph_objs import Heatmap, Layout, Figure, Scatter, Contour

DATAFILE = "data_125ms.csv"
TIMESTEP = 125e-3

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])


# %%
#####################################################################
# Utilities.                                                        #
#####################################################################

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


def neighbours(b):
    return [(b[0] + 1, b[1]), (b[0] - 1, b[1]),
            (b[0], b[1] + 1), (b[0], b[1] - 1),
            (b[0] + 1, b[1] + 1), (b[0] + 1, b[1] - 1),
            (b[0] - 1, b[1] + 1), (b[0] - 1, b[1] - 1)]


def bin_index(x, y):
    ii = ((x - x_min) // grid.binsize).astype(int, copy=False)
    jj = ((y - y_min) // grid.binsize).astype(int, copy=False)

    return ii, jj


def density(x, y):
    h, _, _ = np.histogram2d(x[~np.isnan(x)], y[~np.isnan(y)],
                  bins=grid.ndiv,
                  range=[[x_min, x_min + grid.L],
                         [y_min, y_min + grid.L]])

    return h

# %%
#####################################################################
# Recover the drift field.                                          #
#####################################################################

grid = Grid(data, binsize=0.5)
THRESHOLD = 50

tdata = grid.data.groupby("bin").filter(lambda x: len(x) > THRESHOLD)
bins = sorted(tdata["bin"].unique())

# Filter out the isolated bins.
isolated_bins = []
for b in bins:
    if set(neighbours(b)).isdisjoint(bins):
        isolated_bins.append(b)

filtered_bins = list(set(bins) - set(isolated_bins))

# Extend the bins considering their neighbours.
extended_bins = []
for b in filtered_bins:
    extended_bins.append(b)
    for n in neighbours(b):
        if n not in extended_bins:
            extended_bins.append(n)

z = np.full([grid.ndiv+1, grid.ndiv+1], np.nan)
for b in extended_bins:
    z[b] = -1

for b in list(set(bins) - set(isolated_bins)):
    z[b] = -2

for b in isolated_bins:
    z[b] = 0

# Plot
fig = Figure(data=[Heatmap(z=z[0:60, 15:75].T, colorscale="Portland",
                           showscale=False)],
             layout=grid._layout())
pyimg.save_as(fig, filename="img/03_bins.png")

# Estimate the drift field
xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
u = np.zeros((grid.ndiv + 1, grid.ndiv + 1))
v = u.copy()

for b, data in grid.data.groupby("bin"):
    if b not in filtered_bins:
        continue

    drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                       axis=0) / TIMESTEP

    u[b] = drift[0]
    v[b] = drift[1]

# Plot
drift_norm = np.sqrt(u**2 + v**2)[0:60, 15:75]
lyt = grid._layout()
fig = Figure(data=[Heatmap(z=drift_norm.T, zmin=0, zmax=0.3,
                   colorscale="Portland", showscale=False)],
             layout=lyt)
pyimg.save_as(fig, filename="img/03_empirical_drift.png")


# Smoothen the field
K = 1/16 * np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 2]])

us = filters.convolve(u, K, mode="constant", cval=0.0)
vs = filters.convolve(v, K, mode="constant", cval=0.0)


# Plot
smooth_drift = np.sqrt(us**2 + vs**2)[0:60, 15:75]
fig = Figure(data=[Heatmap(z=smooth_drift.T, zmin=0, zmax=0.3,
                   colorscale="Portland", showscale=False)],
             layout=grid._layout())
pyimg.save_as(fig, filename="img/03_smooth_drift.png")


# %%
#####################################################################
# Run the simulation.                                               #
#####################################################################

dt = 0.1 * grid.binsize / np.mean(np.sqrt(us**2 + vs**2))
x_min = min(grid.data["x"])
y_min = min(grid.data["y"])

# Uniformly spaced initial position
xr, yr = np.meshgrid(np.linspace(x_min, x_min + grid.L, num=10*grid.ndiv),
                     np.linspace(y_min, y_min + grid.L, num=10*grid.ndiv))

x = xr.ravel()
y = yr.ravel()

p = density(x, y)[0:60, 15:75]
fig = Figure(data=[Heatmap(z=p.T, zmin=0, zmax=5000, colorscale="Portland", showscale=False)],
             layout=grid._layout())
pyimg.save_as(fig, filename="img/03_000_sim.png")


for n in range(201):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    ii, jj = bin_index(x, y)
    mask = np.logical_and(ii < grid.ndiv, jj < grid.ndiv)
    x[mask] += us[ii[mask], jj[mask]] * dt
    y[mask] += vs[ii[mask], jj[mask]] * dt

    if n in [2, 5, 10, 200]:
        p = density(x, y)[0:60, 15:75]
        fig = Figure(data=[Heatmap(z=p.T, zmin=0, zmax=5000, colorscale="Portland", showscale=False)],
                     layout=grid._layout())
        pyimg.save_as(fig, filename="img/03_{:03}_sim.png".format(n))


# %%
# Attractors with the biggest basin.

bins = np.empty((grid.ndiv, grid.ndiv), dtype=tuple)
for i in range(grid.ndiv):
    for j in range(grid.ndiv):
        bins[i, j] = (i, j)

p = density(x, y)
bins.shape

# Select the biggest attractors.
attractors = bins[p > 10000]

attractors[0]

lyt = Layout(
    height=1000,
    width=1000,
    yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
               autotick=False, ticks="", dtick=0.5, tick0=x_min,
               showticklabels=False),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, ticks="", dtick=0.5, tick0=y_min,
               showticklabels=False))

xx, yy = np.meshgrid(grid.x, grid.y)
fig = ff.create_quiver(xx[0:60, 30:90], yy[0:60, 30:90], u[0:60, 30:90], v[0:60, 30:90], scale=5)
fig.layout = lyt
py.iplot(fig)

grid.heatmap(u[0:60, 30:90])

A = (34, 23)

xx, yy = np.meshgrid(grid.x, grid.y)

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
               autotick=False, dtick=grid.binsize, tick0=x_min,
               showticklabels=False),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, dtick=grid.binsize, tick0=y_min,
               showticklabels=False))

fig = ff.create_quiver(xxm, yym, um, vm, scale=2)
fig["data"].append(Scatter(x=[grid.x[A[0]]], y=[grid.y[A[1]]], mode="markers"))
fig.layout = lyt
py.iplot(fig)


def MSE(params, x, y, u, v):
    x0, y0, A, r = params
    xx, yy = np.meshgrid(x, y)

    return np.mean((u + 2*A/r**2 * (xx - x0))**2 + (v + 2*A/r**2 * (yy - y0))**2)

x0 = grid.x[A[0]]
y0 = grid.y[A[1]]

initial = [x0, y0, 0.1, 5*grid.binsize]
bounds = [(x0-grid.binsize, x0+grid.binsize),
          (y0-grid.binsize, y0+grid.binsize),
          (0, 10),
          (grid.binsize, 5*grid.binsize)]
res = scipy.optimize.minimize(MSE, initial, args=(xm, ym, um, vm), bounds=bounds)
res
def c(r, x, y, u, v):
    xx, yy = np.meshgrid(x, y)

    return -0.5*(r**2) * np.sum(xx*um + yy*vm) / np.sum(xx**2 + yy**2)


MSE(initial, xm, ym, um, vm)

def cMSE(r, x0, y0, x, y, u, v):
    xx, yy = np.meshgrid(x, y)

    return np.mean((u + 2*c(r, x, y, u, v)/r**2 * (xx - x0))**2 + (v + 2*c(r, x, y, u, v)/r**2 * (yy - y0))**2)


cMSE(0.100, x0, y0, xm, ym, um, vm)


scipy.optimize.minimize(cMSE, (100*grid.binsize), args=(x0, y0, xm, ym, um, vm))

fig = ff.create_quiver(xxm, yym, um, vm, scale=2)
fig.layout = lyt
fig.layout.shapes = [{
            'type': 'circle',
            'xref': 'x',
            'yref': 'y',
            'x0': x0 - res.x[3],
            'y0': y0 - res.x[3],
            'x1': x0 + res.x[3],
            'y1': y0 + res.x[3],
            'line': {'color': 'rgba(50, 171, 96, 1)'}
}]
py.iplot(fig)

xx.shape
u.shape
fig = ff.create_quiver(xx, yy, u[:grid.ndiv, :grid.ndiv], v[:grid.ndiv, :grid.ndiv], scale=5)
fig.layout = lyt
fig.layout.shapes = [{
            'type': 'circle',
            'xref': 'x',
            'yref': 'y',
            'x0': x0 - res.x[3],
            'y0': y0 - res.x[3],
            'x1': x0 + res.x[3],
            'y1': y0 + res.x[3],
            'line': {'color': 'rgba(50, 171, 96, 1)'}
}]
py.iplot(fig)



# Simulation from attractor


NSTEPS = 100
NSAMPLES = 100
dt = 0.1 * grid.binsize / np.mean(np.sqrt(u**2 + v**2))

x = np.zeros((NSTEPS, NSAMPLES))
y = np.zeros((NSTEPS, NSAMPLES))

x[0] = x0 + np.random.random(size=NSAMPLES) * grid.binsize
y[0] = y0 + np.random.random(size=NSAMPLES) * grid.binsize

for t in range(1, NSTEPS):
    ii, jj = bin_index(x[t-1], y[t-1])
    mask = np.logical_and(ii < grid.ndiv, jj < grid.ndiv)
    x[t][mask] = x[t-1][mask] + u[ii[mask], jj[mask]] * dt + np.random.normal(scale=grid.binsize, size=np.sum(mask))
    y[t][mask] = x[t-1][mask] + v[ii[mask], jj[mask]] * dt + np.random.normal(scale=grid.binsize, size=np.sum(mask))

    x[t][~mask] = x[t-1][~mask]
    y[t][~mask] = y[t-1][~mask]


data = []

for i in range(NSAMPLES):
    data.append(Scatter(x=x.T[i], y=y.T[i], name=i))


py.iplot(data)
A
p = density(x[-1], y[-1])
grid.heatmap(p)
