# Axis 1 results summary

## h_FF sanity check

Renaming must not change what h_FF solves. 'same exp.' counts tasks whose number of expanded nodes is identical to L0; differences come only from tie-breaking, which depends on names (L1, L3) and on the order of actions and facts (L2).

### train split

| domain | solved L0 | solved L1 | solved L2 | solved L3 | same exp. L1 | same exp. L2 | same exp. L3 |
|---|---|---|---|---|---|---|---|
| blocksworld | 20/20 | 20/20 | 20/20 | 20/20 | 16/20 | 13/20 | 17/20 |
| childsnack | 20/20 | 20/20 | 20/20 | 20/20 | 13/20 | 6/20 | 13/20 |
| floortile | 20/20 | 20/20 | 20/20 | 20/20 | 5/20 | 5/20 | 5/20 |
| miconic | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 20/20 | 17/20 |
| rovers | 20/20 | 20/20 | 20/20 | 20/20 | 15/20 | 10/20 | 15/20 |
| sokoban | 20/20 | 20/20 | 20/20 | 20/20 | 10/20 | 12/20 | 10/20 |
| spanner | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 |
| transport | 20/20 | 20/20 | 20/20 | 20/20 | 10/20 | 10/20 | 10/20 |

## Candidates on training tasks (20 tasks/domain, 60 s)

Cell = training coverage of the best candidate (working candidates / generated), where a working candidate solves at least one training task.

| level | blocksworld | childsnack | floortile | miconic | rovers | sokoban | spanner | transport |
|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini L0 | 20 (9/10) | 20 (9/10) | 18 (10/10) | 20 (7/10) | 20 (8/10) | 20 (8/10) | 20 (8/10) | 20 (9/10) |
| gpt-4.1-mini L1 | 20 (10/10) | 20 (10/10) | 17 (6/10) | 20 (10/10) | 19 (10/10) | 20 (10/10) | 20 (10/10) | 20 (9/10) |
| gpt-4.1-mini L2 | 20 (8/10) | 20 (10/10) | 17 (10/10) | 20 (10/10) | 20 (9/10) | 20 (9/10) | 20 (9/10) | 20 (10/10) |
| gpt-4.1-mini L3 | 20 (8/10) | 20 (10/10) | 17 (9/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) |
| gpt-5-mini L0 | 20 (10/10) | 20 (9/10) | 17 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) |
| gpt-5-mini L1 | 20 (10/10) | 20 (10/10) | 19 (9/10) | 20 (9/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) |
| gpt-5-mini L2 | 20 (10/10) | 20 (10/10) | 17 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) |
| gpt-5-mini L3 | 20 (10/10) | 20 (10/10) | 19 (9/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) | 20 (10/10) |

Outcome of every candidate x task run:

| level | runs | solved | crash_compile | crash_runtime | infinite_h | oom | timeout | unsolved |
|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini L0 | 1600 | 78.0% | 0.0% | 18.2% | 0.0% | 0.0% | 3.8% | 0.0% |
| gpt-4.1-mini L1 | 1600 | 88.3% | 0.0% | 8.6% | 1.1% | 0.0% | 2.0% | 0.0% |
| gpt-4.1-mini L2 | 1600 | 90.5% | 2.5% | 3.8% | 0.6% | 0.0% | 2.6% | 0.1% |
| gpt-4.1-mini L3 | 1600 | 93.4% | 0.0% | 4.2% | 0.0% | 0.0% | 2.4% | 0.0% |
| gpt-5-mini L0 | 1600 | 94.1% | 0.0% | 3.1% | 0.0% | 0.0% | 2.8% | 0.0% |
| gpt-5-mini L1 | 1600 | 95.6% | 0.0% | 2.5% | 0.0% | 0.0% | 1.9% | 0.0% |
| gpt-5-mini L2 | 1600 | 95.9% | 0.0% | 1.8% | 0.0% | 0.0% | 2.3% | 0.0% |
| gpt-5-mini L3 | 1600 | 95.8% | 0.0% | 1.2% | 0.0% | 0.0% | 3.0% | 0.0% |

## Selected candidates

| config | domain | candidate | train |
|---|---|---|---|
| axis1-L0-g5mini | blocksworld | 9_10 | 20/20 |
| axis1-L0-g5mini | childsnack | 9_10 | 20/20 |
| axis1-L0-g5mini | floortile | 1_10 | 17/20 |
| axis1-L0-g5mini | miconic | 9_10 | 20/20 |
| axis1-L0-g5mini | rovers | 4_10 | 20/20 |
| axis1-L0-g5mini | sokoban | 9_10 | 20/20 |
| axis1-L0-g5mini | spanner | 9_10 | 20/20 |
| axis1-L0-g5mini | transport | 9_10 | 20/20 |
| axis1-L1-g5mini | blocksworld | 9_10 | 20/20 |
| axis1-L1-g5mini | childsnack | 8_10 | 20/20 |
| axis1-L1-g5mini | floortile | 8_10 | 19/20 |
| axis1-L1-g5mini | miconic | 9_10 | 20/20 |
| axis1-L1-g5mini | rovers | 4_10 | 20/20 |
| axis1-L1-g5mini | sokoban | 9_10 | 20/20 |
| axis1-L1-g5mini | spanner | 9_10 | 20/20 |
| axis1-L1-g5mini | transport | 9_10 | 20/20 |
| axis1-L2-g5mini | blocksworld | 9_10 | 20/20 |
| axis1-L2-g5mini | childsnack | 10_10 | 20/20 |
| axis1-L2-g5mini | floortile | 7_10 | 17/20 |
| axis1-L2-g5mini | miconic | 9_10 | 20/20 |
| axis1-L2-g5mini | rovers | 7_10 | 20/20 |
| axis1-L2-g5mini | sokoban | 9_10 | 20/20 |
| axis1-L2-g5mini | spanner | 9_10 | 20/20 |
| axis1-L2-g5mini | transport | 7_10 | 20/20 |
| axis1-L3-g5mini | blocksworld | 9_10 | 20/20 |
| axis1-L3-g5mini | childsnack | 10_10 | 20/20 |
| axis1-L3-g5mini | floortile | 9_10 | 19/20 |
| axis1-L3-g5mini | miconic | 9_10 | 20/20 |
| axis1-L3-g5mini | rovers | 5_10 | 20/20 |
| axis1-L3-g5mini | sokoban | 9_10 | 20/20 |
| axis1-L3-g5mini | spanner | 9_10 | 20/20 |
| axis1-L3-g5mini | transport | 4_10 | 20/20 |
| axis1-L0-g41mini | blocksworld | 7_10 | 20/20 |
| axis1-L0-g41mini | childsnack | 9_10 | 20/20 |
| axis1-L0-g41mini | floortile | 7_10 | 18/20 |
| axis1-L0-g41mini | miconic | 7_10 | 20/20 |
| axis1-L0-g41mini | rovers | 5_10 | 20/20 |
| axis1-L0-g41mini | sokoban | 9_10 | 20/20 |
| axis1-L0-g41mini | spanner | 9_10 | 20/20 |
| axis1-L0-g41mini | transport | 9_10 | 20/20 |
| axis1-L1-g41mini | blocksworld | 8_10 | 20/20 |
| axis1-L1-g41mini | childsnack | 7_10 | 20/20 |
| axis1-L1-g41mini | floortile | 2_10 | 17/20 |
| axis1-L1-g41mini | miconic | 7_10 | 20/20 |
| axis1-L1-g41mini | rovers | 5_10 | 19/20 |
| axis1-L1-g41mini | sokoban | 8_10 | 20/20 |
| axis1-L1-g41mini | spanner | 9_10 | 20/20 |
| axis1-L1-g41mini | transport | 7_10 | 20/20 |
| axis1-L2-g41mini | blocksworld | 8_10 | 20/20 |
| axis1-L2-g41mini | childsnack | 10_10 | 20/20 |
| axis1-L2-g41mini | floortile | 3_10 | 17/20 |
| axis1-L2-g41mini | miconic | 9_10 | 20/20 |
| axis1-L2-g41mini | rovers | 5_10 | 20/20 |
| axis1-L2-g41mini | sokoban | 6_10 | 20/20 |
| axis1-L2-g41mini | spanner | 9_10 | 20/20 |
| axis1-L2-g41mini | transport | 5_10 | 20/20 |
| axis1-L3-g41mini | blocksworld | 8_10 | 20/20 |
| axis1-L3-g41mini | childsnack | 9_10 | 20/20 |
| axis1-L3-g41mini | floortile | 5_10 | 17/20 |
| axis1-L3-g41mini | miconic | 9_10 | 20/20 |
| axis1-L3-g41mini | rovers | 2_10 | 20/20 |
| axis1-L3-g41mini | sokoban | 8_10 | 20/20 |
| axis1-L3-g41mini | spanner | 9_10 | 20/20 |
| axis1-L3-g41mini | transport | 5_10 | 20/20 |

## Test coverage (30 tasks/domain, 300 s)

| level | blocksworld | childsnack | floortile | miconic | rovers | sokoban | spanner | transport | total |
|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini L0 | 5 | 8 | 1 | 11 | 4 | 6 | 10 | 6 | 51 |
| gpt-4.1-mini L1 | 5 | 8 | 1 | 9 | 6 | 8 | 10 | 6 | 53 |
| gpt-4.1-mini L2 | 8 | 5 | 1 | 10 | 6 | 8 | 10 | 11 | 59 |
| gpt-4.1-mini L3 | 8 | 5 | 1 | 10 | 6 | 6 | 10 | 10 | 56 |
| gpt-5-mini L0 | 5 | 7 | 1 | 9 | 6 | 8 | 10 | 6 | 52 |
| gpt-5-mini L1 | 5 | 8 | 1 | 8 | 7 | 9 | 10 | 6 | 54 |
| gpt-5-mini L2 | 6 | 3 | 1 | 8 | 8 | 10 | 10 | 5 | 51 |
| gpt-5-mini L3 | 5 | 5 | 1 | 9 | 8 | 10 | 10 | 4 | 52 |
