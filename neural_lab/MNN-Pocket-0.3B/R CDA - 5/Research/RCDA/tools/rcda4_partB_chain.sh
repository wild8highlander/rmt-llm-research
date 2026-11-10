#!/usr/bin/env bash
# RCDA-4 Part B chain: blind arm, aided arm, then final evals (RAM-sequential).
set -u
S=/home/z/my-project/scripts
B=/home/z/my-project/rcda_work/rcda4/partB
cd "$B"
rm -f edited_state.safetensors partB_blind_transcript.json partB_aided_transcript.json

echo "== B blind start $(date +%H:%M:%S) =="
python3 "$S/rcda4_wsp.py" --system-file system_b_blind.txt --task-file task_b_blind.txt \
  --boot boot_b.py --cwd "$B" --max-turns 14 --max-cells 25 --cell-timeout 480 \
  --out partB_blind > partB_blind.log 2>&1
echo "== B blind rc=$? $(date +%H:%M:%S) =="
if [ -f edited_state.safetensors ]; then
  mv edited_state.safetensors edited_state_blind.safetensors
  python3 "$S/rcda4_eval_b.py" --edited-state "$B/edited_state_blind.safetensors" \
    --tag blind --out "$B/partB" > partB_eval_blind.log 2>&1
  echo "== B blind eval rc=$? =="
else
  echo "== B blind: NO edited_state.safetensors (agent did not call save_edited) =="
fi

echo "== B aided start $(date +%H:%M:%S) =="
python3 "$S/rcda4_wsp.py" --system-file system_b_aided.txt --task-file task_b_aided.txt \
  --boot boot_b.py --cwd "$B" --max-turns 14 --max-cells 25 --cell-timeout 480 \
  --out partB_aided > partB_aided.log 2>&1
echo "== B aided rc=$? $(date +%H:%M:%S) =="
if [ -f edited_state.safetensors ]; then
  mv edited_state.safetensors edited_state_aided.safetensors
  python3 "$S/rcda4_eval_b.py" --edited-state "$B/edited_state_aided.safetensors" \
    --tag aided --out "$B/partB" > partB_eval_aided.log 2>&1
  echo "== B aided eval rc=$? =="
else
  echo "== B aided: NO edited_state.safetensors =="
fi
echo PART_B_CHAIN_FINISHED
