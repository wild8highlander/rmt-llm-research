# C++ Laboratory

C++17 port of the lab pipeline (`lab_en/` English, `lab_ru/` Russian).
Header-only research engine (`research_3d.hpp`) plus a console front-end,
optionally using Eigen when available.

```bash
cd laboratory/cpp/lab_en
make                     # or: g++ -std=c++17 -O2 main.cpp -o lab
./lab                    # interactive menu
```

The implementation is kept numerically consistent with the Python reference
(`mp_bounds`, BBP thresholds, Tracy-Widom constants) — the same values that
the CI cross-implementation job asserts.
