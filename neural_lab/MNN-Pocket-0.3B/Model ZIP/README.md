# Model ZIP — split archive of the intervention kit

`rmt-pocket-interventions` is stored as a multi-volume archive to stay
under the 100 MB per-file GitHub limit:

```text
rmt-pocket-interventions.z01 ... rmt-pocket-interventions.z06
rmt-pocket-interventions.zip      <- last volume, keep next to the others
```

To unpack: put ALL volumes in one directory, then

```bash
zip -s 0 rmt-pocket-interventions.zip --out combined.zip
unzip combined.zip
```

or run `../Code/reassemble_model.py`. Every volume is required — a missing
`.zNN` makes the archive unreadable. Step-by-step text:
[`../Docs/HOW_TO_UNPACK.txt`](../Docs/HOW_TO_UNPACK.txt).
