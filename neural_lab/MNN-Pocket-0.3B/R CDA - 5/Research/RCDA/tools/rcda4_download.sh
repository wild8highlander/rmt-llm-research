#!/usr/bin/env bash
# RCDA-4 Part A: download base + edited Qwen3.5-0.8B from HuggingFace (resumable).
# Usage: dl <outdir> <repo> <filename>
set -u
BASE_DIR=/home/z/my-project/rcda_work/rcda4/models/base
EDIT_DIR=/home/z/my-project/rcda_work/rcda4/models/edited
mkdir -p "$BASE_DIR" "$EDIT_DIR"

dl () {
  local out="$1/$3" repo="$2"
  rm -rf "$1/$2" 2>/dev/null  # cleanup of the buggy first run
  if [ -s "$out" ] && [ ! -f "$out.dl" ]; then echo "SKIP $repo/$3"; return 0; fi
  touch "$out.dl"
  curl -sL -C - -o "$out.part" "https://huggingface.co/$repo/resolve/main/$3" \
    && mv "$out.part" "$out" && rm -f "$out.dl" && echo "DONE $repo/$3" \
    || echo "FAIL $repo/$3"
}

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt chat_template.jinja model.safetensors.index.json; do
  dl "$BASE_DIR" "Qwen/Qwen3.5-0.8B" "$f"
  dl "$EDIT_DIR" "RikiyaT/Qwen3.5-0.8B-weight-space-programming" "$f"
done

echo "== edited repo tree =="
curl -sL "https://huggingface.co/api/models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/tree/main" \
  | python3 -c "
import json,sys
try:
    for x in json.load(sys.stdin): print(' ', x['path'], x.get('size', 0))
except Exception as e: print('tree API failed:', e)"

dl "$BASE_DIR" "Qwen/Qwen3.5-0.8B" "model.safetensors"
dl "$EDIT_DIR" "RikiyaT/Qwen3.5-0.8B-weight-space-programming" "model.safetensors"

echo "== sizes =="
du -sh "$BASE_DIR" "$EDIT_DIR"
echo ALL_DOWNLOADS_FINISHED
