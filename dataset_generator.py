import csv
import locale
import numpy as np

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')

N = 10000
TIMESTEP = 0.125
N_STEPS = 100

def drift(x, y):
    return 0, 0

def diffusion(x, y):
    return 1, 1


# %%
# Run for N_STEPS steps
xx = np.zeros((N_STEPS, N))
yy = np.zeros_like(xx)

# Initial positions
xx[0] = np.random.random(N) * 60
yy[0] = np.random.random(N) * 60

for i in range(1, N_STEPS):
    drift_x, drift_y = drift(xx[i-1], yy[i-1])

    xx[i] = xx[i-1] + drift_x*TIMESTEP + diff_x*np.random.normal(size=N)
    yy[i] = yy[i-1] + drift_y*TIMESTEP + diff_y*np.random.normal(size=N)

# Create CSV

with open("test_dataset.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=";")

    # Header
    writer.writerow(["track id", "t", "x", "y"])

    for n in range(N):
        for t in range(N_STEPS):
            x = locale.format("%f", xx[t, n])
            y = locale.format("%f", yy[t, n])

            writer.writerow([n, t, x, y])
