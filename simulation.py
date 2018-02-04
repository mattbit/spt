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
    i_max = min(x + r + 1, len(array))

    return array[i_min:i_max]


def mask2d(array, x, r=5):
    i_min = max(x[0] - r, 0)
    i_max = min(x[0] + r + 1, len(array))
    j_min = max(x[1] - r, 0)
    j_max = min(x[1] + r + 1, len(array))

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
THRESHOLD = 100


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
# pyimg.save_as(fig, filename="img/03_bins.png")
py.plot(fig, filename="img/03_bins.html")

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
# pyimg.save_as(fig, filename="img/03_empirical_drift.png")
py.plot(fig, filename="img/03_empirical_drift.html")


# Smoothen the field
K = 1/16 * np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 2]])
#
# K = 1/8 * np.array([[0, 1, 0],
#                     [1, 4, 1],
#                     [0, 1, 0]])

us = filters.convolve(u, K, mode="constant", cval=0.0)
vs = filters.convolve(v, K, mode="constant", cval=0.0)


# Plot
smooth_drift = np.sqrt(us**2 + vs**2)[0:60, 15:75]
fig = Figure(data=[Heatmap(z=smooth_drift.T, zmin=0, zmax=0.3,
                   colorscale="Portland", showscale=False)],
             layout=grid._layout())
py.plot(fig, filename="img/03_smooth_drift.html")
# pyimg.save_as(fig, filename="img/03_smooth_drift.png")


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
py.plot(fig, filename="img/03_000_sim.html")
# pyimg.save_as(fig, filename="img/03_000_sim.png")


# Use NaN values for unknown bins.
uc = np.full_like(us, np.nan)
vc = np.full_like(vs, np.nan)

for b in extended_bins:
    uc[b] = us[b]
    vc[b] = vs[b]

# Start the simulation.
for n in range(201):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    ii, jj = bin_index(x, y)
    mask = np.logical_and(ii < grid.ndiv, jj < grid.ndiv)
    x[mask] += uc[ii[mask], jj[mask]] * dt
    y[mask] += vc[ii[mask], jj[mask]] * dt

    if n in [5, 10, 200]:
        p = density(x, y)[0:60, 15:75]
        fig = Figure(data=[Heatmap(z=p.T, zmin=0, zmax=5000, colorscale="Portland", showscale=False)],
                     layout=grid._layout())
        py.plot(fig, filename="img/03_{:03}_sim.html".format(n))
        # pyimg.save_as(fig, filename="img/03_{:03}_sim.png".format(n))


# %%
# Attractors with the biggest basin.


import matplotlib.pyplot as plt
import plotly.tools as tls


fig = Figure(data=[Heatmap(z=np.hypot(u, v).T)], layout=grid._layout())
py.iplot(fig)


def quiver(u, v, x_range, y_range):
    plt.close()
    fig = plt.figure(figsize=(15, 15))
    un = u[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vn = v[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    un[un == 0] = np.nan
    vn[vn == 0] = np.nan
    plt.quiver(grid.x[x_range[0]:x_range[1]], grid.y[y_range[0]:y_range[1]], un.T, vn.T, np.hypot(un, vn).T, scale=10)
    plt.grid(True)
    plt.xticks(np.arange(-0.5, x_range[1] - x_range[0]))
    plt.yticks(np.arange(-0.5, y_range[1] - y_range[0]))
    plt.gca().tick_params(labelbottom=False, labelleft=False)

    return fig


bins = np.empty((grid.ndiv, grid.ndiv), dtype=tuple)
for i in range(grid.ndiv):
    for j in range(grid.ndiv):
        bins[i, j] = (i, j)

p = density(x, y)
grid.heatmap(p)

# Select the biggest attractors.
A = bins[p>4000][0]

lyt = Layout(
    height=1000,
    width=1000,
    yaxis=dict(scaleanchor="x", showgrid=True, zeroline=False,
               autotick=False, ticks="", dtick=1, tick0=-0.5,
               showticklabels=False),
    xaxis=dict(showgrid=True, zeroline=False,
               autotick=False, ticks="", dtick=1, tick0=-0.5,
               showticklabels=False))


# xx, yy = np.meshgrid(mask1d(grid.x, A[0], 10), mask1d(grid.x, A[1], 10))
xx, yy = np.meshgrid(mask1d(range(127), A[0], 10), mask1d(range(127), A[1], 10))

fig = ff.create_quiver(xx, yy, mask2d(u, A, 10).T, mask2d(v, A, 10).T, scale=4)
fig.layout = lyt
# fig["data"].append(Scatter(x=[grid.x[A[0]]], y=[grid.y[A[1]]], mode="markers"))
fig["data"].append(Scatter(x=[A[0]], y=[A[1]], mode="markers"))
py.iplot(fig)

grid.heatmap(p)
grid.heatmap(np.hypot(us, vs))

def well_weight(grid, A, r, u, v):
    xx, yy = np.meshgrid(mask1d(grid.x, A[0], r), mask1d(grid.x, A[1], r))
    um = mask2d(u, A, 1)
    vm = mask2d(v, A, 1)

    return -0.5*(r*grid.binsize)**2 * np.sum(xx*um + yy*vm) / np.sum(xx**2 + yy**2)

bins[p > 4000]
for a in bins[p > 4000]:
    print(well_weight(grid, a, 1, u, v))

ww = np.zeros((grid.ndiv, grid.ndiv))
for i in range(2, grid.ndiv-2):
    for j in range(2, grid.ndiv-2):
        ww[i, j] = well_weight(grid, (i, j), 1, u, v)


A = bins[p > 4000][5]
B = bins[p > 4000][5]
C = A
xx, yy = np.meshgrid(mask1d(range(127), C[0], 10), mask1d(range(127), C[1], 10))

fig = ff.create_quiver(xx, yy, mask2d(us, C, 10).T, mask2d(vs, C, 10).T, scale=10)
lyt.shapes = [{
            'type': 'circle',
            'xref': 'x',
            'yref': 'y',
            'x0': A[0] - 1.5,
            'y0': A[1] - 1.5,
            'x1': A[0] + 1.5,
            'y1': A[1] + 1.5,
            'line': {'color': 'rgba(50, 171, 96, 1)'}}]
fig.layout = lyt
# fig["data"].append(Scatter(x=[grid.x[A[0]]], y=[grid.y[A[1]]], mode="markers"))
# fig["data"].append(Scatter(x=[A[0], B[0]], y=[A[1], B[1]], mode="markers"))
py.plot(fig, filename="img/03_attractor.html")





xx, yy = np.meshgrid(mask1d(grid.x, A[0], 1), mask1d(grid.x, A[1], 1))
um = mask2d(u, A, 1)
vm = mask2d(v, A, 1)



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

diff_xx = grid.apply(diff_tensor_xx, threshold=THRESHOLD)
diff_xy = grid.apply(diff_tensor_xy, threshold=THRESHOLD)
diff_yy = grid.apply(diff_tensor_yy, threshold=THRESHOLD)

diff_xx[np.isnan(diff_xx)] = 0
diff_yy[np.isnan(diff_yy)] = 0
diff_xy[np.isnan(diff_xy)] = 0
diffs_xx = filters.convolve(diff_xx, K, mode="constant", cval=0.0)
diffs_yy = filters.convolve(diff_yy, K, mode="constant", cval=0.0)
diffs_xy = filters.convolve(diff_xy, K, mode="constant", cval=0.0)

smooth_diff = 0.5*(diffs_xx + diffs_yy)
fig = Figure(data=[Heatmap(z=smooth_diff.T,
                   colorscale="Portland", showscale=False)],
             layout=grid._layout())
py.plot(fig, filename="img/04_smooth_diff.html")

# %%
#####################################################################
# Pathways.                                               #
#####################################################################

p = density(x, y)

NSTEPS = 1000
NSAMPLES = 1000000
dt = 0.1 * grid.binsize / np.mean(np.sqrt(u**2 + v**2))

x = np.zeros((NSTEPS, NSAMPLES))
y = np.zeros((NSTEPS, NSAMPLES))

A = bins[p > 4000][0]
x0, y0 = grid.x[A[0]], grid.y[A[1]]

x[0] = x0 + (-0.5 + np.random.random(size=NSAMPLES)) * grid.binsize
y[0] = y0 + (-0.5 + np.random.random(size=NSAMPLES)) * grid.binsize

np.dot(np.array([[1, 2],[3, 4]]), np.array([1, 2]))

# Use NaN values for unknown bins.
uc = np.full_like(us, np.nan)
vc = np.full_like(vs, np.nan)

for b in extended_bins:
    uc[b] = us[b]
    vc[b] = vs[b]


for t in range(1, NSTEPS):
    x[t-1][np.isnan(x[t-1])] = x[t-2][np.isnan(x[t-1])]
    y[t-1][np.isnan(y[t-1])] = y[t-2][np.isnan(y[t-1])]
    ii, jj = bin_index(x[t-1], y[t-1])
    mask = np.logical_and(ii < grid.ndiv, jj < grid.ndiv)
    eta = np.random.normal(size=2)
    x[t][mask] = x[t-1][mask] + uc[ii[mask], jj[mask]] * dt + np.sqrt(2*smooth_diff[ii[mask], jj[mask]])*eta[0]
    y[t][mask] = x[t-1][mask] + vc[ii[mask], jj[mask]] * dt + np.sqrt(2*smooth_diff[ii[mask], jj[mask]])*eta[0]

    x[t][~mask] = x[t-1][~mask]
    y[t][~mask] = y[t-1][~mask]


grid.heatmap(density(x[-2], y[-2]))

x
data = []

for i in range(NSAMPLES):
    data.append(Scatter(x=x.T[i], y=y.T[i], name=i))


py.iplot(data)
A
p = density(x[-1], y[-1])
grid.heatmap(p)
