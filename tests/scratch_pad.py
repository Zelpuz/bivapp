import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import bivapp, numpy as np
import matplotlib.pyplot as plt

# %% Import data
df = bivapp.ImportOpenairDataExample()

# %% Test plot
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(df["so2"], color="blue")
axs[0].set_ylabel(r"SO$_{2}$ [ppbv?]")
axs[1].plot(df["ws"], color="black")
axs[1].set_ylabel("Wind Speed [m/s]")
axs[2].plot(df["wd"], ".", color="red")
axs[2].set_ylabel("Wind Direction [degrees]")
fig.set_figheight(10)
fig.set_figwidth(8)

# %% Define plotting functions

fig, axs = bivapp.BivariatePlotRaw(
    df["so2"],
    df["ws"],
    df["wd"],
    positive=True,
    vmin=None,
    vmax=df["so2"].quantile(0.9),
    cmap=cm.batlowK,
    colourbar_label="SO$_2$ [ppbv?]",
    scatter_kwds={"s": 2},
)

fig, axs = bivapp.BivariatePlotGrid(
    df["so2"],
    df["ws"],
    df["wd"],
    resolution=73,
    agg_method="mean",
    interpolate=False,
    positive=True,
    vmin=None,
    vmax=df["so2"].quantile(0.9),
    cmap=cm.batlowK,
    colourbar_label="SO$_2$ [ppbv?]",
)

fig, axs = bivapp.BivariatePlotRawGAM(
    df["so2"],
    df["ws"],
    df["wd"],
    pred_res=200,
    positive=True,
    vmin=None,
    vmax=df["so2"].quantile(0.9),
    cmap=cm.batlowK,
    colourbar_label="SO$_2$ [ppbv?]",
    masking_method="ch",
    near_dist=1,
)

# %% scratch pad: testing _excludeTooFar
res = 100
true_grid = np.meshgrid(np.arange(-2, 2 + 1, 1), np.arange(-5, 5 + 1, 1))
true_points = np.vstack([x.ravel() for x in true_grid]).T
pred_grid = np.meshgrid(np.linspace(-10, 10, res), np.linspace(-10, 10, res))
pred_points = np.vstack([x.ravel() for x in pred_grid]).T
masked_pred_points = bivapp._excludeTooFar(true_points, pred_points, 1)

fig, ax = plt.subplots(1, 1)
_x, _y = np.vstack([x.ravel() for x in pred_grid])
ax.scatter(_x, _y, c="k", s=1)
_x, _y = masked_pred_points.T
ax.scatter(_x, _y, c="r", s=1)
_x, _y = np.vstack([x.ravel() for x in true_grid])
ax.scatter(_x, _y, c="g", s=10)

fig.set_figwidth(8)
fig.set_figheight(7.5)
