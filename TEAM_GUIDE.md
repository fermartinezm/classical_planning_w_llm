# Step-by-Step Team Execution Guide (Fast, Low-Cost Laptop Protocol)

This guide defines the exact **Commands**, **Inputs**, and **Outputs** for each team member so all results merge into one table and plotting pipeline with zero manual editing.

---

## Shared Protocol Summary (Locked for Nov 9 Freeze)
* **Planner & Search:** Pyperplan `-s gbf` (`-s gbf_alt` for Member 3's open-list alternation).
* **Canonical Splits ([`benchmarks/splits.json`](file:///Users/sahmed/code/classical_planning_w_llm/benchmarks/splits.json)):**
  * **Training (`--split train`):** 20 tasks/domain (160 total), **60s** timeout, **4 GiB** RAM limit.
  * **Testing (`--split test`):** 30 tasks/domain (`files[0::3]`, 240 total), **300s (5 min)** timeout, **4 GiB** RAM limit.
* **Fast & Cheap Generation Rule:**
  * **1 seed × $n = 10$ candidates** per configuration (cuts compute & API cost by 3×).
  * Reuse pre-generated heuristics in [`archive-heuristics/`](file:///Users/sahmed/code/classical_planning_w_llm/archive-heuristics) ($0 API cost) wherever possible (`deepseek-r1`, `deepseek-v3-0324`, `deepseek-r1-distill-qwen-14b`, `gemini-2.0-flash`, and Blocksworld L1 `2025-07-23-hidden-blocks-r1`).

---

## Step 0: One-Time Setup (All 4 Members)

* **Input:** Cloned repository + Python 3.10+ (and `pypy3` + `validate` (VAL) on `PATH` if available).
* **Commands:**
  ```bash
  pip install -r requirements.txt
  # Smoke-test 2 Blocksworld tasks (takes ~2 seconds):
  python3 run_experiments.py --preset baselines --domains blocksworld --max-tasks 2 --out results/smoke_test.jsonl
  ```
* **Output:** `results/smoke_test.jsonl` containing 6 validated JSONL lines (`pyperplan-blind`, `pyperplan-hff`, `paper-r1`).

---

## Member 1: Axis 1 — Semantic Obfuscation (L0–L3)

> [!IMPORTANT]
> **Cost & Time Shortcut:**
> * **L0 (Original):** Already in repo! Member 4 re-runs `paper-r1` and `paper-v3` on L0 ($0 cost).
> * **Blocksworld L1 (R1):** Already in repo at `archive-heuristics/2025-07-23-hidden-blocks-r1` ($0 cost).
> * For remaining L1, L2, L3 runs: use **1 seed (`--num-candidates 10`)** with **`deepseek-v3`** (or free `gemini-2.5-flash`), which costs **<$1.50 total** and generates 10× faster than R1.

### Step 1.1: Build `obfuscate.py` to generate L1, L2, L3 benchmarks
* **Input:** Original domains & tasks in `benchmarks/ipc2023-learning/{training,testing}/<domain>/`
  * **Crucial:** Also obfuscate `example-state.out` and `example-static.out` in each `training/<domain>/` folder using the exact same symbol mapping so the LLM prompt stays consistent!
* **Output:** Obfuscated folders:
  * `benchmarks/obfuscated/L1/{training,testing}/<domain>/` (renamed predicates, actions, types)
  * `benchmarks/obfuscated/L2/{training,testing}/<domain>/` (L1 + shuffled order + object names)
  * `benchmarks/obfuscated/L3/{training,testing}/<domain>/` (misleading cross-domain names)

### Step 1.2: Verify `h_FF` invariance on L1–L3 (Correctness Check)
* **Input:** Obfuscated test directories `benchmarks/obfuscated/L{1,2,3}/testing`
* **Command:**
  ```bash
  python3 run_experiments.py --preset pyperplan-hff --obf-level L1 \
    --benchmarks-dir benchmarks/obfuscated/L1/testing \
    --out results/member1_hff_check.jsonl
  ```
* **Output:** `results/member1_hff_check.jsonl` (coverage per domain must match Member 4's L0 `pyperplan-hff` coverage).

### Step 1.3: Generate $n=10$ candidates per domain on L1, L2, L3
* **Input:** Obfuscated training folder `benchmarks/obfuscated/L1/training/<domain>`
* **Command (run for each domain & level L1, L2, L3):**
  ```bash
  python3 llm-heuristics.py --domain blocksworld \
    --framework deepseek --model deepseek-v3 \
    --suite-dir benchmarks/obfuscated/L1/training/blocksworld \
    --num-candidates 10 \
    --output-dir generated/axis1_L1_v3
  ```
* **Output:** Candidate Python files `generated/axis1_L1_v3/<domain>1_10.py` ... `<domain>10_10.py`.

### Step 1.4: Evaluate candidates on Training (60s), pick best, and run on Testing (300s)
* **Input:** `generated/axis1_L1_v3/`
* **Commands:**
  ```bash
  # 1. Evaluate all 10 candidates on the 20 training tasks (60s limit)
  python3 run_experiments.py --heuristic-dir generated/axis1_L1_v3 --max-candidates 10 \
    --split train --obf-level L1 --config axis1-L1-v3-train --model deepseek-v3 \
    --benchmarks-dir benchmarks/obfuscated/L1/training \
    --out results/member1_axis1_train.jsonl

  # 2. Print the selected best candidate ID per domain
  python3 run_experiments.py --select-from results/member1_axis1_train.jsonl --select-rule paper_rule

  # 3. Copy the selected best file per domain into `generated/axis1_L1_v3_selected/` and evaluate on 30 test tasks (300s limit)
  python3 run_experiments.py --heuristic-dir generated/axis1_L1_v3_selected --max-candidates 1 \
    --split test --obf-level L1 --config axis1-L1-v3 --model deepseek-v3 \
    --benchmarks-dir benchmarks/obfuscated/L1/testing \
    --out results/member1_axis1_test.jsonl
  ```
* **Final Deliverable to Repo:** `results/member1_axis1_test.jsonl` (feeds directly into Figure **F1**).

---

## Member 2: Axis 2 — Model Scale & Failure Taxonomy

> [!IMPORTANT]
> **Cost & Time Shortcut:**
> * **671B (`deepseek-r1` & `deepseek-v3`)** and **14B (`deepseek-r1-distill-qwen-14b`)** are **already generated** in `archive-heuristics/` ($0 API cost!).
> * You only need to generate **2 small open-weight sizes** (e.g., `qwen2.5-coder:3b` and `qwen2.5-coder:7b` locally via Ollama for $0, or via OpenRouter/Groq for <$0.50 total) using **1 seed × $n=10$ candidates**.

### Step 2.1: Generate $n=10$ candidates for 2 small open-weight models
* **Input:** Standard L0 domains (`--domain <domain>`) + local Ollama (`http://localhost:11434/v1`) or OpenRouter/Groq endpoint.
* **Command (repeat for each of the 8 domains):**
  ```bash
  # Example using local Ollama (free) or set OPENAI_BASE_URL / OPENAI_COMPATIBLE_API_KEY for Groq/OpenRouter:
  python3 llm-heuristics.py --domain blocksworld \
    --framework openai-compatible --base-url http://localhost:11434/v1 \
    --model qwen2.5-coder:7b --num-candidates 10 \
    --output-dir generated/axis2_qwen7b
  ```
* **Output:** 80 candidate files (`8 domains × 10 candidates`) in `generated/axis2_qwen7b/` (and `generated/axis2_qwen3b/`).

### Step 2.2: Evaluate archived 14B + new small models on Training (60s) & Testing (300s)
* **Input:**
  * Archived 14B (`--preset paper-qwen14b` or `--heuristic-dir archive-heuristics/2025-03-13-deepseek-r1-distill-qwen-14b --max-candidates 10`)
  * Your new `generated/axis2_qwen7b` and `generated/axis2_qwen3b` folders.
* **Commands:**
  ```bash
  # 1. Run archived 14B selected heuristics on the 240 test tasks ($0 API cost):
  python3 run_experiments.py --preset paper-qwen14b --out results/member2_axis2_test.jsonl

  # 2. Run all 10 candidates of your 7B model on training (logs crash_compile, crash_runtime, infinite_h, timeout):
  python3 run_experiments.py --heuristic-dir generated/axis2_qwen7b --max-candidates 10 \
    --split train --config axis2-qwen7b-train --model qwen-7b \
    --out results/member2_axis2_train.jsonl

  # 3. Select the best candidate per domain:
  python3 run_experiments.py --select-from results/member2_axis2_train.jsonl --select-rule paper_rule

  # 4. Run the selected best 7B candidate per domain on the 240 test tasks:
  python3 run_experiments.py --heuristic-dir generated/axis2_qwen7b_selected --max-candidates 1 \
    --split test --config axis2-qwen7b --model qwen-7b \
    --out results/member2_axis2_test.jsonl
  ```
* **Final Deliverable to Repo:** `results/member2_axis2_train.jsonl` (for the failure taxonomy stacked bar) and `results/member2_axis2_test.jsonl` (for the scaling curve in Figure **F2**).

---

## Member 3: Axis 3 — Candidate Selection Rules & Open-List Alternation

> [!IMPORTANT]
> **Cost & Time Shortcut:**
> * **Zero LLM API calls ($0).** All 25 candidates per domain for `deepseek-r1` are in `archive-heuristics/2025-03-10-deepseek-r1`.
> * Open-list alternation (`-s gbf_alt` via comma-separated `-H h1.py,h2.py,h3.py`) and all 4 selection rules (`paper_rule`, `largest_train_only`, `fewest_expansions`, `oracle`) are **already implemented** in `src/pyperplan/search/a_star.py` and `run_experiments.py`!

### Step 3.1: Evaluate archived R1 candidates on Training (60s) and Testing (300s)
* **Input:** `archive-heuristics/2025-03-10-deepseek-r1`
* **Commands:**
  ```bash
  # 1. Evaluate archived R1 candidates on the 20 training tasks (60s limit)
  python3 run_experiments.py --preset archive-r1-candidates --split train \
    --config axis3-r1-train --out results/member3_axis3_train.jsonl

  # 2. Evaluate archived R1 candidates on the 30 test tasks (300s limit)
  python3 run_experiments.py --preset archive-r1-candidates --split test \
    --config axis3-r1-test --out results/member3_axis3_test.jsonl
  ```
* **Output:** `results/member3_axis3_train.jsonl` and `results/member3_axis3_test.jsonl`.

### Step 3.2: Compare Selection Rules & Run Top-3 Open-List Alternation
* **Input:** `results/member3_axis3_train.jsonl` and `results/member3_axis3_test.jsonl`
* **Commands:**
  ```bash
  # Compare picks from each rule:
  python3 run_experiments.py --select-from results/member3_axis3_train.jsonl --select-rule paper_rule --select-top-k 3
  python3 run_experiments.py --select-from results/member3_axis3_train.jsonl --select-rule largest_train_only
  python3 run_experiments.py --select-from results/member3_axis3_train.jsonl --select-rule fewest_expansions
  python3 run_experiments.py --select-from results/member3_axis3_test.jsonl  --select-rule oracle

  # Run LAMA-style round-robin open-list alternation over the top-3 training candidates for a domain:
  python3 run_experiments.py --domains blocksworld --split test \
    --config axis3-alt3 --model deepseek-r1 \
    --heuristic archive-heuristics/2025-03-10-deepseek-r1/blocksworld9_10.py,archive-heuristics/2025-03-10-deepseek-r1/blocksworld1_10.py,archive-heuristics/2025-03-10-deepseek-r1/blocksworld2_10.py \
    --out results/member3_axis3_alt.jsonl
  ```
* **Final Deliverable to Repo:** `results/member3_axis3_train.jsonl`, `results/member3_axis3_test.jsonl`, and `results/member3_axis3_alt.jsonl` (feeds directly into Figure **F3**).

---

## Member 4 (You): Baselines, Comparability Re-runs, Plots & Tables

### Step 4.1: Run All Baselines & Paper L0 Re-runs on the 240 Test Tasks
* **Input:** `benchmarks/splits.json` + `archive-heuristics/`
* **Commands:**
  ```bash
  # Runs pyperplan-blind, pyperplan-hff, paper-r1 (and fd-hff / fd-lama-first if fast-downward.py is on PATH):
  python3 run_experiments.py --preset baselines --split test --workers 4 \
    --out results/member4_baselines.jsonl

  # Also run paper-v3 L0 baseline (so Member 1 has the exact L0 reference for DeepSeek-V3):
  python3 run_experiments.py --preset paper-v3 --split test --workers 4 \
    --out results/member4_baselines.jsonl
  ```
* **Output:** `results/member4_baselines.jsonl` (240 tasks × each baseline configuration).

### Step 4.2: Generate All 5 Figures (F1–F5) & LaTeX Summary Table
* **Input:** All `results/*.jsonl` files committed by Members 1–4.
* **Command:**
  ```bash
  python3 plots/make_plots_and_tables.py --inputs "results/*.jsonl" --out-dir plots/output
  ```
* **Output:**
  * `plots/output/F1_obfuscation.pdf` (.png)
  * `plots/output/F2_model_scaling.pdf` (.png)
  * `plots/output/F3_candidate_selection.pdf` (.png)
  * `plots/output/F4_expansions_scatter.pdf` (.png)
  * `plots/output/F5_cactus.pdf` (.png)
  * `plots/output/summary_table.tex`

