#!/bin/bash -l

startmonth=$1
#aggr=$2
#fcmonth=$3

echo "Scores"
while IFS="," read -r institution name
do
    echo "Model: $institution $name"
    python 2-Postprocess.py "$institution" "$name" $startmonth
done < <(tail -n +2 models.csv)
# echo "Score cards"
# python 6-Multi-Season_score-card.py "t2m" "3m" "corr"
# python 6-Multi-Season_score-card.py "t2m" "3m" "roc"
# python 6-Multi-Season_score-card.py "t2m" "3m" "rpss"
# python 6-Multi-Season_score-card.py "tprate" "3m" "corr"
# python 6-Multi-Season_score-card.py "tprate" "3m" "roc"
# python 6-Multi-Season_score-card.py "tprate" "3m" "rpss"
# echo "End"
