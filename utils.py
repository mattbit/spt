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
