import pandas
import numpy as np
from grid import Grid

# Test

testdata = pandas.read_csv("test_dataset.csv", sep=";", decimal=",",
                       index_col=("track id", "t"),
                       usecols=["track id", "x", "y", "t"])

testgrid = Grid(testdata, 40)
testgrid.L

def diff_tensor_xx(data):
    return np.nanmean(data.loc[:, "step_x"]**2)

def diff_tensor_xy(data):
    return np.nanmean(data.loc[:, "step_x"]*data.loc[:, "step_y"])

def diff_tensor_yy(data):
    return np.nanmean(data.loc[:, "step_y"]**2)

diff_xx = testgrid.apply(diff_tensor_xx, threshold=200)
diff_xy = testgrid.apply(diff_tensor_xy, threshold=200)
diff_yy = testgrid.apply(diff_tensor_yy, threshold=200)

np.nanmean(diff_xx)
np.nanmean(diff_yy)
np.nanmean(diff_xy)
