# Bivariate polar plots in Python
What it says on the tin. This repo provides functions for producing bivariate polar plots, a useful graphical analysis tool in air pollution research. This implementation is largely based on the R package [`openair`](https://github.com/openair-project/openair/tree/master) and Carslaw and Beevers (2013) - these sources also provide some excellent example cases of bivariate polar plots in practice.

## What bivapp does and does not provide
`bivapp` is intended to provide bivariate polar plots similar to the implementation in [`openair`](https://github.com/openair-project/openair/tree/master) and as described by Carslaw and Beevers (2013). It is not intended to be a full-featured alternative to `openair`, as that package provides enough features that it is effectively a complete data analysis suite for air pollution studies. Many of `openair`'s features are already available in other popular Python libraries. For example, `openair` provides a function for calculating Theil-Sen slopes, but [`scikit-learn`](https://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html) and [`scipy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html) already feature such tools.

`bivapp` currently also does not support producing windroses. See [`windrose`](https://github.com/python-windrose/windrose) instead. This may change in the future.

## Documentation
At this early stage functions are only self-documented. Proper documentation is planned.

## Getting started
Install from PyPI: `pip install bivapp`. 

Note that because the dependency `pyGAM` has fallen behind on maintenance `bivapp` depends on specific versions of some common dependencies, so you might want to use it in a dedicated virtual environment for the time being.

Here's an example of a basic plotting setup.
```python
from bivapp.sampledata import ImportOpenairDataExample
import bivapp.plots as bp

df = ImportOpenairDataExample()
fig, ax = bp.BivariatePlotRawGAM(
    df["so2"], 
    df['ws'], 
    df['wd'], 
    pred_res=200, 
    vmax=df['so2'].quantile(0.9),
    colourbar_label="SO$_2$ [ppbv]"
)
fig.set_figwidth(6)
fig.set_figheight(4.5)
```
![Example from BivariatePlotRawGAM](/examples/images/example_openair_so2_raw_gam.png)

## Existing solutions
The [`openair`](https://github.com/openair-project/openair/tree/master) package for R provides all these features, but is obviously in R and not Python. The topic of bivariate polar plots in Python also pops up occasionally, like [here](https://stackoverflow.com/questions/61940629/bivariate-polar-plots-in-python), [here](https://stackoverflow.com/questions/61702585/pollution-rose-plot-gridded), [here](https://stackoverflow.com/questions/9071084/how-to-create-a-polar-contour-plot), and [here](https://blog.rtwilson.com/producing-polar-contour-plots-with-matplotlib/). Lastly, there is the existing [`windrose`](https://github.com/python-windrose/windrose) library, but it lacks bivariate polar plots.

## Differences from `openair`
Users should be aware that the implementation of smoothed bivariate polar plots in this library differs from `openair`. `openair` uses the `mgcv` R package to fit a thin-plate spline GAM to smooth their bivariate polar plots. In their implementation, they bin input data by wind direction and speed, and then fit the GAM to this binned data. In `bivapp` there is currently only one method that fits a GAM, `BivariatePlotRawGAM`. This method differs from `openair`'s in a couple ways: first, the GAM is fit to the raw measurements rather than binned measurements; second, due to differences in GAM libraries (and their documentation), we are not exactly replicating the thin-plate spline approach. Instead, `bivapp` fits a GAM to a tensor product of the $u$ and $v$ components of the input wind data. Thus, the GAM-smoothed bivariate polar plot in `bivapp` is not a perfect replication of `openair`'s smoothed plots, but does appear to achieve the same goal of producing a reasonably smoothed plot.
