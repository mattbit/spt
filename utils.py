import numpy as np

def estimate_drift(grid, timestep, threshold=20):
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")

    u = np.full((grid.ndiv, grid.ndiv), np.nan)
    v = u.copy()

    for ij, data in grid.data.groupby("bin"):

        if len(data) < threshold:
            continue

        drift = np.nanmean(data.loc[:, ("step_x", "step_y")].values,
                           axis=0) / timestep

        u[ij] = drift[0]
        v[ij] = drift[1]

    return xx, yy, u, v



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


def bin_index(grid, x, y):
    x_min = min(grid.data["x"])
    y_min = min(grid.data["y"])

    ii = ((x - x_min) // grid.binsize).astype(int, copy=False)
    jj = ((y - y_min) // grid.binsize).astype(int, copy=False)

    return ii, jj


def density(grid, x, y):
    x_min = min(grid.data["x"])
    y_min = min(grid.data["y"])

    h, _, _ = np.histogram2d(x[~np.isnan(x)], y[~np.isnan(y)],
                  bins=grid.ndiv,
                  range=[[x_min, x_min + grid.L],
                         [y_min, y_min + grid.L]])

    return h


def out_of_bounds(grid, i):
    return i < 0 or i >= grid.ndiv


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
