#!/bin/bash -l

aggr=$1

for month in {1..12}
do
    echo "Month: $month"
    python -u 1-PCs_forecast.py $aggr $month
done

