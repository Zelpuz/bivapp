# First, monkey patches to get pygam working
def to_array(self):
    return self.toarray()


import scipy.sparse

scipy.sparse.spmatrix.A = property(to_array)

import numpy as np

np.int = int

# Now for the real imports
import matplotlib.pyplot as plt
import pandas as pd, os
import cmcrameri.cm as cm
from scipy.interpolate import griddata
from shapely import MultiPoint, Point, intersection, convex_hull

from pygam import GAM, te  # pyGAM maintanance lapsed. Waiting for update.

np.random.seed(117)


def _makeScatterPlot(xs, ys, zs, vmin, vmax, cmap, colourbar_label, scatter_kwds):
    # Create and return the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

    lowlim = np.min([np.min(xs), np.min(ys)])
    highlim = np.max([np.max(xs), np.max(ys)])
    squarelim = np.max([np.abs(lowlim), np.abs(highlim)])

    if vmin is None:
        vmin = np.nanmin(zs)
    if vmax is None:
        vmax = np.nanmax(zs)

    raw = ax.scatter(xs, ys, c=zs, cmap=cmap, vmin=vmin, vmax=vmax, **scatter_kwds)
    fig.colorbar(raw, ax=ax, label=colourbar_label, shrink=0.5)
    ax.set_xlim(-squarelim, squarelim)
    ax.set_ylim(-squarelim, squarelim)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    return fig, ax


def _makeImagePlot(reshaped, resolution, wind_bins, vmin, vmax, cmap, colourbar_label):
    # Create and return the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

    if vmin is None:
        vmin = np.nanmin(reshaped)
    if vmax is None:
        vmax = np.nanmax(reshaped)

    raw = ax.imshow(reshaped, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(raw, ax=ax, label=colourbar_label, shrink=0.5)
    ax.set_xlim(0, resolution - 1)
    ax.set_ylim(0, resolution - 1)
    ax.set_xticks(np.linspace(0, resolution - 1, 5))
    ax.set_yticks(np.linspace(0, resolution - 1, 5))
    ax.set_xticklabels([round(wind_bins[x]) for x in ax.get_xticks().astype(int)])
    ax.set_yticklabels([round(wind_bins[x]) for x in ax.get_xticks().astype(int)])
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # ax.set_xlabel("Wind x-component [m/s]")
    # ax.set_ylabel("Wind y-component [m/s]")
    return fig, ax


def _getWindComponents(wind_speed, wind_dir):
    """Get wind components from wind speed and direction."""
    wind_u = wind_speed * np.sin(np.deg2rad(wind_dir))
    wind_v = wind_speed * np.cos(np.deg2rad(wind_dir))

    return wind_u, wind_v


def _interpGrid(sparse_grid, xs, ys, interpolation_method):
    masked_grid = np.ma.masked_invalid(sparse_grid)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    points = np.vstack([x.ravel(), y.ravel()]).T
    interp = griddata(
        (x[~masked_grid.mask], y[~masked_grid.mask]),
        masked_grid[~masked_grid.mask].ravel(),
        points,
        method=interpolation_method,
    )
    return interp.reshape(len(xs) - 1, len(ys) - 1)


def _aggValuesToGrid(wind_u, wind_v, values, wind_bins, resolution, agg_method):
    df = pd.DataFrame([wind_u, wind_v, values]).T
    df.columns = ["wind_u", "wind_v", "values"]
    wind_u_cut = pd.cut(
        df["wind_u"],
        bins=wind_bins,
        include_lowest=True,
        labels=np.arange(0, resolution - 1, 1),
    )
    wind_v_cut = pd.cut(
        df["wind_v"],
        bins=wind_bins,
        include_lowest=True,
        labels=np.arange(0, resolution - 1, 1),
    )
    agged_values = df.groupby([wind_u_cut, wind_v_cut], observed=False).agg(
        {"values": agg_method}
    )
    return agged_values.values.reshape(resolution - 1, resolution - 1)


def _excludeTooFar(true_points, pred_points, dist):
    """Returns a copy of pred_grid where values further than dist from the nearest point
    in true_grid are masked to NaN. true_grid and pred_grid should both be shaped like
    the output of np.meshgrid. dist is a number (float or int).

    If any values in true_points or pred_points are NaN, they will be ignored."""
    masked_pred_points = np.copy(pred_points)

    if dist < 0:
        raise Exception("dist must be > 0.")
    # true_grid_list = np.vstack([x.ravel() for x in true_grid]).T
    # pred_grid_list = np.vstack([x.ravel() for x in pred_grid]).T

    norms = np.array(
        [
            np.nanmin(np.linalg.norm(true_points - point, axis=1))
            for point in pred_points
        ]
    )
    # min_dists = np.min(norms, axis=0)
    min_dists = np.where(norms > dist, False, True)

    masked_pred_points[~min_dists] = np.nan

    return masked_pred_points


def BivariatePlotRaw(
    values,
    wind_speed,
    wind_dir,
    positive=True,
    vmin=None,
    vmax=None,
    cmap=cm.batlow,
    colourbar_label=None,
    scatter_kwds=None,
):
    """Make a bivariate polar plot of the raw data, with optional interpolation."""
    # Get wind components
    wind_u, wind_v = _getWindComponents(wind_speed, wind_dir)

    # Force positive if requested
    if positive:
        values = np.where(values < 0, 0, values)

    _df = pd.DataFrame([wind_u, wind_v, values]).T
    _df.columns = ["wind_u", "wind_v", "values"]
    _df = _df.dropna()

    # Create and return plot of reshaped
    return _makeScatterPlot(
        _df["wind_u"],
        _df["wind_v"],
        _df["values"],
        vmin,
        vmax,
        cmap,
        colourbar_label,
        scatter_kwds,
    )


def BivariatePlotGrid(
    values,
    wind_speed,
    wind_dir,
    resolution=101,
    agg_method="mean",
    interpolate=False,
    interpolation_method="linear",
    positive=True,
    vmin=None,
    vmax=None,
    cmap=cm.batlow,
    colourbar_label=None,
):
    """Make a bivariate polar plot of the raw data, with optional interpolation."""
    # Get wind components
    wind_u, wind_v = _getWindComponents(wind_speed, wind_dir)

    # Find largest absolute wind speed component and define bins
    wind_comp_abs_max = np.max(
        [np.abs(np.nanmin([wind_u, wind_v])), np.nanmax([wind_u, wind_v])]
    )
    wind_bins = np.linspace(-wind_comp_abs_max, wind_comp_abs_max, resolution)

    # Aggregate values into bins. For convenience, we use pandas cut and groupby methods.
    reshaped = _aggValuesToGrid(
        wind_v, wind_u, values, wind_bins, resolution, agg_method
    )

    # Interpolate, if requested
    if interpolate:
        reshaped = _interpGrid(
            reshaped, wind_bins[:-1], wind_bins[:-1], interpolation_method
        )

    # Force positive if requested
    if positive:
        reshaped = np.where(reshaped < 0, 0, reshaped)

    # Create and return plot of reshaped
    return _makeImagePlot(
        reshaped, resolution, wind_bins, vmin, vmax, cmap, colourbar_label
    )


def _fitGAM(X, y, degfree, lam):
    gam = GAM(
        (te(0, 1, n_splines=degfree, spline_order=3, lam=lam, penalties="derivative")),
        distribution="normal",
        link="identity",
        fit_intercept=True,
    )
    gam.fit(X, y)
    return gam


def BivariatePlotRawGAM(
    values,
    wind_speed,
    wind_dir,
    pred_res=500,
    positive=True,
    degfree=30,
    lam=0.6,
    vmin=None,
    vmax=None,
    cmap=cm.batlow,
    colourbar_label=None,
    masking_method="near",
    near_dist=1,
):
    """Fits a GAM to the raw data, similar to the R openair package. Specifically fits
    values ** 0.5 ~ s(wind_u) + s(wind_v), where s are smoothing splines, and returns
    a plot of the fitted values.

    Note that openair fits a GAM to binned data, but this function fits a GAM to the raw data.

    Masking method chooses how to cut GAM predictions to a reasonable boundary around the raw
    data. Options are:

    "near": points within a set distance (1 m/s) of the raw data are preserved. This is similar to
    how openair limits data. However, the implementation here can be slow for large datasets.
    "ch": a convext hull is drawn around the raw data and only predictions falling within this hull
    are presented.
    "grid": raw data is gridded to a resolution of pred_res and only grid cells with raw data present
    are preserved. This is best used with smaller values of pred_res and will produce an image that
    looks like a smoothed version of the image produced by BivariatePlotGrid. Note that due to the
    way the gridding calculation works, this method drops the final prediction along each axis. This
    usually isn't a problem as the edges are often excluded after gridding anyways.

    near_dist is only used if masking_method is "near".
    """
    # Get wind components
    wind_u, wind_v = _getWindComponents(wind_speed, wind_dir)

    df = pd.DataFrame([wind_u, wind_v, values]).T
    df.columns = ["wind_u", "wind_v", "values"]
    df = df.dropna()
    df["values"] = df["values"] ** 0.5
    X = df[["wind_v", "wind_u"]]
    y = df["values"]

    gam = _fitGAM(X, y, degfree, lam)

    # Find largest absolute wind speed component and define bins
    wind_comp_abs_max = np.max(
        [np.abs(np.nanmin([wind_u, wind_v])), np.nanmax([wind_u, wind_v])]
    )
    wind_bins = np.linspace(-wind_comp_abs_max, wind_comp_abs_max, pred_res)
    points_x, points_y = np.meshgrid(wind_bins, wind_bins, indexing="ij")
    points = np.vstack([points_x.ravel(), points_y.ravel()]).T
    pred = (
        gam.predict(points).reshape(pred_res, pred_res) ** 2
    )  # squared; we took a root earlier

    if masking_method == "near":
        masked_points = _excludeTooFar(np.array([wind_u, wind_v]).T, points, near_dist)
        mask = (np.isnan(masked_points)[:, 0] | np.isnan(masked_points)[:, 1]).reshape(
            pred_res, pred_res
        )
    elif masking_method == "grid":
        grid = _aggValuesToGrid(wind_v, wind_u, values, wind_bins, pred_res, "mean")
        mask = np.ma.masked_invalid(grid).mask
        pred = pred[:-1, :-1]
    elif masking_method == "ch":
        hull = convex_hull(MultiPoint(np.array([wind_u, wind_v]).T))
        mask = np.array([intersection(Point(point), hull).is_empty for point in points])
        mask = mask.reshape(pred_res, pred_res)
    else:
        mask = np.full((pred_res, pred_res), False)

    pred[mask] = np.nan

    if positive:
        pred[pred < 0] = 0

    return _makeImagePlot(pred, pred_res, wind_bins, vmin, vmax, cmap, colourbar_label)
