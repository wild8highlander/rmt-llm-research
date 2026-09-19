# Running the Scaling Experiment on Android: Termux + proot-distro Ubuntu

> A step-by-step guide to running the "formulas vs hallucinations" A/B
> experiment (`research/tinygpt_formula/scaling/`) right on your phone:
> Termux → proot-distro → Ubuntu → Python 3 + NumPy → TinyGPT v3 training
> (S 447K, optionally M ~1.1M). Everything is free, no root, no external server.

## 0. Why proot-distro Ubuntu instead of "bare" Termux

| | Termux (native) | Termux + proot Ubuntu |
|---|---|---|
| Python/NumPy | `pkg install python-numpy` (bionic build) | `apt install python3-numpy` (official Debian/Ubuntu arm64 debs) |
| Script compatibility | occasionally different paths/symbols | **full glibc, same as desktop/CI** |
| git, make, compilers | yes, but Termux packages | yes, from the standard repository |
| Speed | ~100% | ~90–95% (proot syscall overhead) |

The experiment only needs Python 3.10+ and NumPy, so both paths work;
proot-distro provides a predictable desktop-like environment and is convenient
for pushing to GitHub.

## 1. Installing Termux

1. Install Termux **from F-Droid** (https://f-droid.org) or from GitHub
   Releases (github.com/termux/termux-app/releases). **The Google Play version
   is outdated and unmaintained.**
2. First launch: allow notifications (needed for the wakelock session).

## 2. Access to shared storage (to unzip a ZIP from Downloads)

```bash
termux-setup-storage          # allow storage access
```

Afterwards `~/storage/downloads` = the phone's Downloads folder.

## 3. Installing Ubuntu via proot-distro

```bash
pkg update -y && pkg upgrade -y
pkg install -y proot-distro    # guest distro manager
proot-distro install ubuntu    # ~25 MB, no root
proot-distro login ubuntu      # enter Ubuntu (the prompt becomes root@localhost)
```

To leave Ubuntu and return to Termux: run `exit`.

## 4. Inside Ubuntu: Python + NumPy + git

```bash
apt update -y
apt install -y python3 python3-numpy python3-pip git coreutils
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

`python3-numpy` from apt is a ready-made package — no compilation needed.

## 5. Getting the repository — two ways

### Option A: git clone (if scaling/ is already pushed to GitHub)

```bash
git clone --depth 1 https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research/research/tinygpt_formula
```

### Option B: unpack the rmt-llm-push.zip package from Downloads

```bash
# in Termux (before entering Ubuntu): the ZIP from Downloads is visible via ~/storage/downloads
pkg install -y unzip
cp ~/storage/downloads/rmt-llm-push.zip ~/
cd ~ && unzip -q rmt-llm-push.zip
# inside: rmt-llm-push/rmt-llm-research + push_to_github.sh + INSTRUCTIONS_TERMUX.md

# enter Ubuntu and navigate to the folder (the Termux home is visible in proot as /host-rootfs/home/...):
proot-distro login ubuntu --shared-tmp
cd /host-rootfs/data/data/com.termux/files/home/rmt-llm-push/rmt-llm-research/research/tinygpt_formula
```

If the `/host-rootfs/...` path is unavailable — just copy the folder into the
Ubuntu home:

```bash
# in Termux:
proot-distro login ubuntu -- mkdir -p /root/work
cp -r ~/rmt-llm-push /sdcard/Download/ 2>/dev/null || true
# then in Ubuntu: cp -r /sdcard/Download/rmt-llm-push /root/work/ (requires termux-setup-storage + bind)
```

A proven minimal route: `pkg install -y nano` is unnecessary — simply keep the
ZIP in `~/` and unpack it INSIDE Ubuntu after copying via `/sdcard`:

```bash
# Termux: cp ~/rmt-llm-push.zip /sdcard/Download/
# Ubuntu: apt install -y unzip && cp /sdcard/Download/rmt-llm-push.zip . && unzip -q rmt-llm-push.zip
```

## 6. Running the experiment (S — the base point, ~20–40 min per arm on a phone)

```bash
cd research/tinygpt_formula   # (or scaling/ entirely — see the path above)

# 1) control corpus (seconds)
python3 scaling/build_control_corpus.py

# 2) quick pipeline check (2 epochs, ~2–3 min)
SCALING_SMOKE=1 python3 scaling/scaling_experiment.py
rm -rf scaling/runs           # smoke results are not included in the summary

# 3) the real S run: both arms (interruption is safe — resume)
SCALING_CELLS=S_formula,S_control SCALING_MAX_MINUTES=38 python3 scaling/scaling_experiment.py
# over budget? just run the same command again (it continues)

# results:
cat scaling/SCALING_RESULTS.md
```

Optionally overnight — size M (~1.1M): leave the phone charging:

```bash
# in Termux (before entering Ubuntu) prevent sleep:
termux-wake-lock
# then in Ubuntu:
SCALING_CELLS=M_formula,M_control SCALING_MAX_MINUTES=480 python3 scaling/scaling_experiment.py
# after completion (in Termux): termux-wake-unlock
```

## 7. Pushing the results to GitHub from proot-Ubuntu

```bash
# inside Ubuntu:
apt install -y git
cd rmt-llm-research
git config user.name  "wild8highlander"
git config user.email "wild8highlander@users.noreply.github.com"
git add research/tinygpt_formula/scaling/SCALING_RESULTS.md research/tinygpt_formula/scaling/scaling_report.json
git commit -m "feat(scaling): S-point A/B results from Termux/Ubuntu run"
git remote -v   # origin must point to your repository
# push with a Personal Access Token (create one at github.com → Settings → Developer settings → PAT, repo scope):
git push https://<YOUR_PAT>@github.com/wild8highlander/rmt-llm-research.git main
```

The token in the command line is not stored in the git config. Alternatively,
use the ready-made `push_to_github.sh` from the ZIP package (it validates the
token and finds the repository on its own).

## 8. What to expect time-wise (guidelines)

| Stage | 2020s flagship (8 cores) | Average phone (4–6 cores) |
|---|---|---|
| SMOKE (2 S epochs) | ~2 min | ~4 min |
| S_formula (40 epochs) | ~15–20 min | ~30–40 min |
| S_control (40 epochs) | ~15–20 min | ~30–40 min |
| M (60 epochs, both arms) | ~3–5 h | overnight on charger |
| L / XL | Colab/Kaggle only | Colab/Kaggle only |

## 9. Troubleshooting

- **`proot-distro: command not found`** — run `pkg update` and
  `pkg install proot-distro` again.
- **`numpy` won't install in native Termux** — use proot-Ubuntu (steps 3–4),
  where it installs from apt.
- **The phone gets hot / the battery drains** — normal for 40 epochs; keep it
  charging, `termux-wake-lock`, remove the case.
- **Android kills the session** — keep Termux in the background with a
  wakelock, don't swipe it away from recents; big battery + battery
  optimization disabled for Termux in Android settings.
- **`/sdcard` is empty in Ubuntu** — run `termux-setup-storage` in Termux and
  log in via `proot-distro login ubuntu --bind /sdcard` (or copy files via
  `~/storage`).
- **Training interrupted** — no problem: weights and history are written every
  epoch; a new run continues (`RESUME from epoch N`).
- **Low on space** — `runs/` with weights is ~2 MB per cell; the ZIP package
  is ~200 MB (mostly `.git` and repo test artifacts).

## 10. What to read next

- `research/tinygpt_formula/README.md` — what the research has already done
  (training, benchmark, open questions).
- `research/tinygpt_formula/scaling/SCALING_PLAN.md` — hypotheses H1–H4 and
  the ladder design.
- `INSTRUCTIONS_TERMUX.md` (in the ZIP package root) — pushing the whole
  repository to GitHub from a phone.
- `colab/Scaling_Ladder_Hallucinations.ipynb` — the same experiment on free
  Colab CPU/GPU.
