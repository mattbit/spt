import numpy as np
import plotly.offline as py
from plotly.graph_objs import Heatmap, Layout, Figure

class Grid(object):
    def __init__(self, data, binsize=1.0):
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

        # Set the grid and add bin indices.
        self.set_binsize(binsize)


    def set_binsize(self, binsize):
        """Recalculates the grid with bins of given size."""
        self.binsize = binsize

        x_min, x_max = self.data["x"].min(), self.data["x"].max()
        y_min, y_max = self.data["y"].min(), self.data["y"].max()

        self.L = max(x_max - x_min, y_max - y_min)

        r = np.arange(self.binsize/2, self.L + self.binsize, step=self.binsize)
        self.ndiv = len(r)
        self.x = r + x_min
        self.y = r + y_min


        # Calculates the bin index given the coordinates.
        def bin_index(x, y):
            ii = ((x - x_min) // self.binsize).astype(int, copy=False)
            jj = ((y - y_min) // self.binsize).astype(int, copy=False)

            return list(zip(ii, jj))

        # Add the bin index column.
        self.data["bin"] = bin_index(self.data["x"], self.data["y"])


    def apply(self, func, threshold=0):
        """Applies a scalar function on each non-empty grid bin."""
        z = np.full((self.ndiv, self.ndiv), np.nan)

        for ij, data in self.data.groupby("bin"):
            if len(data) > threshold:
                z[ij] = func(data)

        return z

    def heatmap(self, data, title="", threshold=0):
        if callable(data):
            z = self.apply(data, threshold)
        else:
            z = data

        htm = Heatmap(z=z.T, colorscale="Portland")

        return self.plot(htm, title)

    def count(self, threshold):
        n = self.apply(lambda d: 1, threshold)

        return np.nansum(n)

    def plot(self, trace, title):
        lyt = Layout(
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

        fig = Figure(data=[trace], layout=lyt)
        py.plot(fig)
