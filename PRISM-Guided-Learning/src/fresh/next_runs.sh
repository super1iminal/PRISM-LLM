#!/usr/bin/env bash
# LLM experiments queued until the OpenRouter budget is back (run from PRISM-Guided-Learning/, in bash).
# At most 2 heavy processes at a time: each method.py run is light while it waits on Haiku and heavy while it repairs.
set -e
PY=C:/Users/asher/Documents/Coding/llmtrust/PRISM-LLM/.venv/Scripts/python
set -a && . C:/Users/asher/Documents/Coding/llmtrust/PRISM-LLM/.env && set +a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step=${1:?usage: next_runs.sh inloop|grids K|grids_noclose K|mer K|mer_n K|onechain|nolines}

case $step in
  inloop)   # the verifier resolves each candidate's `any` (best of 24 orders) before scoring and feedback
    for f in rooms crossing adjacent; do
      $PY -u src/fresh/method.py $f --chains 8 --rounds 6 --resolve-in-loop --out out/fresh/attr/inloop > out/fresh/attr/inloop_$f.log 2>&1
    done ;;
  grids|grids_noclose)   # random grids: one list per training set; my vocabulary (hazard hidden; closer too for _noclose)
    k=${2:?repeat}; hide=hazard; [ $step = grids_noclose ] && hide=hazard,closer
    out=out/fresh/$step/run$k; mkdir -p $out
    sets=("0,4,8" "1,5,9" "2,6,10" "3,7,11")
    for i in 0 1 2 3; do
      $PY -u src/fresh/method.py set$i --kind random --data grid_20_large.csv --ids ${sets[$i]} --hide $hide \
          --chains 8 --rounds 6 --out $out > $out/set$i.log 2>&1
    done
    $PY src/fresh/certify_grids.py $out/cert --hide $hide --run-transfer \
        --lists $out/set0_summary.json $out/set1_summary.json $out/set2_summary.json $out/set3_summary.json ;;
  mer|mer_n)   # MER across x (mer_transfer.csv) or across n (mer_n_transfer.csv): train ids 0-2, validate 3-8
    k=${2:?repeat}; ds=mer_transfer.csv; [ $step = mer_n ] && ds=mer_n_transfer.csv
    $PY -u src/fresh/mer_method.py --dataset $ds --chains 8 --rounds 6 --out out/fresh/$step/run$k > out/fresh/$step/run$k.log 2>&1 ;;
  onechain)   # one chain with the same total number of answers as 8 chains x 7 (56)
    for f in rooms crossing adjacent; do
      $PY -u src/fresh/method.py $f --chains 1 --rounds 55 --out out/fresh/attr/onechain > out/fresh/attr/onechain_$f.log 2>&1
    done ;;
  nolines)   # feedback without the per-action sensor and risk lines
    for f in rooms crossing adjacent; do
      $PY -u src/fresh/method.py $f --chains 8 --rounds 6 --no-action-lines --out out/fresh/attr/nolines > out/fresh/attr/nolines_$f.log 2>&1
    done ;;
esac
