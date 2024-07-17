# Ensemble weighting with EOFs

Verification of seasonal forecasts of surface variables (2m temperature, total precipitation and mean sea level pressure) for the Iberian Peninsula and MedCOF region. A methodology to improve seasonal forecasts is applied, based on using information on climate variability patterns to weight the ensemble members of different seasonal forecasting systems.

A two-steps approach is used: (1) a first forecast of 4 climate variability patterns and (2) a second forecast of surface variables (temperatura and precipitation) in which the ensemble members of the different seasonal forecasting systems are weighted according to their similarity to the predicted variability patterns.

We use for each season a different set of varability patterns, obtained as the four main Empirical Orthogonal Functions of each season ERA5 mean sea level pressure climatology.

The calculation of the verification scores is based on the Copernicus Climate Change Service (C3S) [Seasonal Forecast Verification Tutorial](https://ecmwf-projects.github.io/copernicus-training-c3s/sf-verification.html).

## Index

Depending on how we perform the variability patterns forecast, we will obtaing different results. Here we test 2 different options, corresponding with the following directories:

* 1-Perfect_forecast: We use ERA5 variability patterns data as forecasts ("perfect" forecasts) to quantify the theoretical potential of the methodology for improving seasonal forecasts.
* 2-Three_best_forecasts: We use for each season the three forecasting systems with the best skill in predicting the variability patterns (according to the Ranked Probability Skill Score). Each of the three best forecasting systems corresponds to a different lead time, so all possible lead times are covered.
