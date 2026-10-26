#!/usr/bin/env bash
# Nov 2 gate: the whole Axis 1 pipeline end to end on one domain and one level.
# Blocksworld, L1, R1, n = 10, one seed, 30 test tasks. Needs DEEPSEEK_API_KEY
# (about $0.25 of API calls). Run from the repository root, e.g. `bash axis1/smoke_test.sh`.
set -euo pipefail
JOBS=${JOBS:-4}

python -m axis1.obfuscate --domains blocksworld
python -m axis1.generate --models deepseek-r1 --levels L0 L1 --seeds 1 --domains blocksworld --jobs 4
python -m axis1.evaluate train   --models deepseek-r1 --levels L0 L1 --seeds 1 --domains blocksworld --jobs "$JOBS"
python -m axis1.select_best
python -m axis1.evaluate test    --models deepseek-r1 --levels L0 L1 --seeds 1 --domains blocksworld --jobs "$JOBS"
python -m axis1.evaluate control --levels L0 L1 L2 L3 --domains blocksworld --jobs "$JOBS"
python -m axis1.summarize
python -m axis1.plot_f1
