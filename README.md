# Bivariate polar plots in Python
What it says on the tin. Largely based on the R package [`openair`](https://github.com/openair-project/openair/tree/master) and Carslaw and Beevers (2013).

## What bivapp does and does not provide
This package is intended to provide bivariate polar plots similar to the implementation in [`openair`](https://github.com/openair-project/openair/tree/master) and as described by Carslaw and Beevers (2013).

bivapp is not intended to be a full-featured alternative to `openair`. That package provides enough features that it is effectively a complete software package for data analysis in air pollution studies. Many of its features are already available in other popular Python libraries. For example, `openair` provides a function for calculating Theil-Sen slopes, but [`scikit-learn`](https://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html) and [`scipy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html) already feature such tools.

bivapp currently also does not support producing windroses. See [`windrose`](https://github.com/python-windrose/windrose) instead. This may change in the future.

## Documentation
Functions are currently self-documented. That is, you may refer to their docstrings.

## Existing solutions
The [`openair`](https://github.com/openair-project/openair/tree/master) package for R provides all these features, but is obviously in R and not Python.

The topic of bivariate polar plots in Python has also popped up occasionally online:
https://stackoverflow.com/questions/61940629/bivariate-polar-plots-in-python
https://stackoverflow.com/questions/61702585/pollution-rose-plot-gridded
https://stackoverflow.com/questions/9071084/how-to-create-a-polar-contour-plot
https://blog.rtwilson.com/producing-polar-contour-plots-with-matplotlib/

Lastly, there is the existing [`windrose`](https://github.com/python-windrose/windrose) library, but it lacks bivariate polar plots.