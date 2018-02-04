import pandas
import scipy
import numpy as np
from grid import Grid
from utils import estimate_drift, neighbours
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


grid = Grid(data, binsize=0.5)
THRESHOLD = 100

xx, yy, u, v = estimate_drift(grid, TIMESTEP, threshold=100)

K = 1/16 * np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 2]])

uc = u.copy()
vc = v.copy()
uc[np.isnan(uc)] = 0.
vc[np.isnan(vc)] = 0.
us = filters.convolve(uc, K, mode="constant", cval=0.0)
vs = filters.convolve(vc, K, mode="constant", cval=0.0)

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


U = np.full((grid.ndiv, grid.ndiv), np.nan)
for b in extended_bins:
    U[b] = 1

grid.heatmap(U)

for i in range(grid.ndiv - 1):
    U[i + 1, 0] = -u[i, 0] * grid.binsize + U[i, 0]


def out_of_bounds(grid, i):
    return i < 0 or i >= grid.ndiv

import plotly.graph_objs as go

py.plot([go.Surface(z=U.T)])

def adjacents(b):
    return [(b[0] + 1, b[1]), (b[0] - 1, b[1]),
            (b[0], b[1] + 1), (b[0], b[1] - 1)]

def BFS(source, domain):
    explored = [source]
    new = set([source])
    while len(new) > 0:
        adjs = []
        for n in new:
            adjs += adjacents(n)

        new = set(adjs).intersection(domain) - set(explored)
        explored += list(new)

    return explored


def recover_potential(grid, u, v, source):
    U = np.full((grid.ndiv, grid.ndiv), np.nan)
    U[source] = 0.

    domain = []
    for b in extended_bins:
        domain += adjacents(b)

    bins = BFS(source, domain)

    for b in bins[1:]:
        i, j = b[0], b[1]
        s = []
        if not out_of_bounds(grid, i+1) and not np.isnan(U[i+1, j]):
            s.append(u[i, j] * grid.binsize + U[i+1, j])

        if not out_of_bounds(grid, i-1) and not np.isnan(U[i-1, j]):
            s.append(-u[i-1, j] * grid.binsize + U[i-1, j])

        if not out_of_bounds(grid, j+1) and not np.isnan(U[i, j+1]):
            s.append(v[i, j] * grid.binsize + U[i, j+1])

        if not out_of_bounds(grid, j-1) and not np.isnan(U[i, j-1]):
            s.append(-v[i, j-1] * grid.binsize + U[i, j-1])

        if len(s) > 0:
            U[i, j] = np.nanmean(s)

    U = U - np.nanmin(U)

    return U


domain = []
for b in extended_bins:
    domain += adjacents(b)

bins = BFS((30, 40), domain)
z = np.zeros([grid.ndiv+1, grid.ndiv+1])
bins[0]
for i, b in enumerate(extended_bins):
    z[b] = i

grid.heatmap(z)

U = np.full((grid.ndiv, grid.ndiv), 0.)
bins = BFS((30, 40), extended_bins)
for b in bins:
    U += recover_potential(grid, us, vs, b)

U = U/len(bins)
grid.heatmap(U)

py.plot([go.Surface(z=U.T)])

U1 = recover_potential(grid, us, vs, (24, 4))
U2 = recover_potential(grid, us, vs, (75, 99))
U3 = recover_potential(grid, us, vs, (97, 86))
fig = Figure(data=[Heatmap(z=U1.T), Heatmap(z=U2.T), Heatmap(z=U3.T)], layout=grid._layout())

py.plot([go.Surface(z=Up.T)])

us[29, 6]
vs[29, 6]

xx, yy = np.meshgrid(grid.x[20:50], grid.y[0:30])
fig = ff.create_quiver(xx, yy, us[20:50, 0:30].T, vs[20:50, 0:30].T, scale=5)
fig.layout = grid._layout()
py.plot(fig)
