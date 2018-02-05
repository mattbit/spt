import pandas
import scipy
import numpy as np
from grid import Grid
from utils import *
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.plotly import image as pyimg
import scipy.ndimage.filters as filters
from plotly.graph_objs import Heatmap, Layout, Figure, Scatter, Surface

DATAFILE = "data_125ms.csv"
TIMESTEP = 125e-3

data = pandas.read_csv(DATAFILE, sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])

# %%
#####################################################################
# Load the data and estimate drift.                                 #
#####################################################################

grid = Grid(data, binsize=0.5)
THRESHOLD = 100

xx, yy, u, v = estimate_drift(grid, TIMESTEP, threshold=100)

# Gaussian kernel
K = 1/16 * np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 2]])

uc = u.copy()
vc = v.copy()
uc[np.isnan(uc)] = 0.
vc[np.isnan(vc)] = 0.
us = filters.convolve(uc, K, mode="constant", cval=0.0)
vs = filters.convolve(vc, K, mode="constant", cval=0.0)

# %%
#####################################################################
# Filter the bins.                                                  #
#####################################################################

tdata = grid.data.groupby("bin").filter(lambda x: len(x) > THRESHOLD)
bins = sorted(tdata["bin"].unique())

isolated_bins = []
for b in bins:
    if set(neighbours(b)).isdisjoint(bins):
        isolated_bins.append(b)

filtered_bins = list(set(bins) - set(isolated_bins))

extended_bins = []
for b in filtered_bins:
    extended_bins.append(b)
    for n in neighbours(b):
        if n not in extended_bins:
            extended_bins.append(n)

# %%
#####################################################################
# Build the potential from the field.                               #
#####################################################################

def build_potential(grid, u, v, source):
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


# Plot the potential
U = build_potential(grid, us, vs, (24, 3))
grid.heatmap(U, title="Potential cluster")

grid.heatmap(U[20:40, 0:20], title="Potential cluster", inline=False)

# And the corresponding quiver
xx, yy = np.meshgrid(grid.x[20:40], grid.y[0:20])
fig = ff.create_quiver(xx, yy, us[20:40, 0:20].T, vs[20:40, 0:20].T, scale=10)
fig.layout = grid._layout()
fig.layout.xaxis.tick0 = grid.x[0] - grid.binsize/2
fig.layout.yaxis.tick0 = grid.y[0] - grid.binsize/2
fig.layout.xaxis.dtick = grid.binsize
fig.layout.yaxis.dtick = grid.binsize
fig.layout.width = 1000
fig.layout.height = 1000
py.plot(fig)

py.plot([Surface(z=U.T)])

# Compare the attractors found with the simulation.
A, B = (3, 45), (18, 43)
C = (15, 45)

UA = build_potential(grid, us, vs, C)

fig = Figure(data=[Heatmap(z=mask2d(UA, C, 15).T, zmin=0, zmax=0.5, colorscale="Portland")],
             layout=grid._layout())
py.plot(fig)
