#!/bin/bash -l

startmonth=$1


while IFS="," read -r institution name
do
    echo "Model: $institution $name"
#    python -u 1-Download_data.py "$institution" "$name" $startmonth
    python -u 2-Postprocess.py "$institution" "$name" $startmonth
#    python -u 3-Verification_plots.py "$institution" "$name" $startmonth "$aggr" $fcmonth
done < <(tail -n +2 models.csv)
#python -u 4-Multi-System_Verification_plots.py $startmonth "$aggr" $fcmonth "corr"
#python -u 4-Multi-System_Verification_plots.py $startmonth "$aggr" $fcmonth "roc"
#python -u 4-Multi-System_Verification_plots.py $startmonth "$aggr" $fcmonth "rpss"

