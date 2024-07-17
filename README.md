# Ensemble weighting of C3S seasonal forecasting systems

We present a methodology to improve seasonal forecasts, based on using information on climate variability patterns to weight the ensemble members of different seasonal forecasting systems.

A two-steps approach is used: (1) a first forecast of 4 climate variability patterns and (2) a second forecast of surface variables (temperatura and precipitation) in which the ensemble members of the different seasonal forecasting systems are weighted according to their similarity to the predicted variability patterns.

Different verification scores are computed in order to compare the skill of the proposed methodology with the skill of the non-processed seasonal forecasting systems. The calculation of the verification scores is based on the Copernicus Climate Change Service (C3S) [Seasonal Forecast Verification Tutorial](https://ecmwf-projects.github.io/copernicus-training-c3s/sf-verification.html).

IMPORTANT: These scripts use data that was downloaded/calculated in [Seasonal Verification](https://github.com/mSenande/SEAS-Verification).

## Index

Depending on how we define the variability patterns, we will obtaing different results. Here we test 3 different options, corresponding with the following directories:

* 1-DJF_patterns: We use for all seasons the same four varability patterns, obtained as the four main Empirical Orthogonal Functions of the December-January-February (DJF) ERA5 500 hPa geopotential height climatology (North Atlantic Oscillation, East Atlantic, East Atlantic / Western Russia and Scandinavian Pattern).
* 2-EOFg500_patterns: We use for each season a different set of varability patterns, obtained as the four main Empirical Orthogonal Functions of each season ERA5 500 hPa geopotential height climatology.
* 3-EOFmslp_patterns: We use for each season a different set of varability patterns, obtained as the four main Empirical Orthogonal Functions of each season ERA5 mean sea level pressure climatology.
* 0-Visualization: Html interface for an interactive visualization of results.
