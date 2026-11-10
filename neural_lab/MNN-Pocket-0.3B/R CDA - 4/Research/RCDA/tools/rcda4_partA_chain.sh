#!/usr/bin/env bash
# RCDA-4 Part A chain: base benchmark, then edited benchmark (sequential, RAM-bound).
set -u
cd /home/z/my-project/rcda_work/rcda4
echo "== base run start $(date +%H:%M:%S) =="
python3 /home/z/my-project/scripts/rcda4_partA_bench.py \
  --model-dir /home/z/my-project/rcda_work/rcda4/models/base \
  --tag base --out-prefix partA > partA_base.log 2>&1
echo "== base rc=$? $(date +%H:%M:%S) =="
python3 /home/z/my-project/scripts/rcda4_partA_bench.py \
  --model-dir /home/z/my-project/rcda_work/rcda4/models/edited \
  --tag edited --out-prefix partA > partA_edited.log 2>&1
echo "== edited rc=$? $(date +%H:%M:%S) =="
echo PART_A_CHAIN_FINISHED
