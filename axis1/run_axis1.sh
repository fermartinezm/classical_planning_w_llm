#!/usr/bin/env bash
# Axis 1 end to end, as in the team plan: L0-L3 x (R1 x 3 seeds + V3 x 1 seed), n = 10,
# selection on the training tasks (60 s), tests on the 240-task subset (5 min, 4 GiB),
# h_FF / blind controls at every level, then tables and F1.
#
# Every step skips work that is already done, so the script can be stopped and rerun.
# Steps can also be run separately (see axis1/README.md), e.g. on different laptops
# with --levels / --domains and the result files merged afterwards (cat *.jsonl).
#
#   export DEEPSEEK_API_KEY=...
#   JOBS=6 bash axis1/run_axis1.sh
set -euo pipefail
JOBS=${JOBS:-$(( $(nproc) - 1 ))}

python -m axis1.obfuscate                          # benchmarks/obfuscated/L0-L3
python -m axis1.check_invariance --n-tasks 10      # obfuscator sanity check
python -m axis1.generate --jobs 4                  # LLM candidates (API)
python -m axis1.evaluate train --jobs "$JOBS"      # ~100k runs of <= 60 s
python -m axis1.select_best
python -m axis1.evaluate test --jobs "$JOBS"       # 16 configs x 240 tasks
python -m axis1.evaluate control --jobs "$JOBS"    # h_FF and blind, 4 levels x 240 tasks
python -m axis1.summarize | tee axis1/results/summary.md
python -m axis1.plot_f1
