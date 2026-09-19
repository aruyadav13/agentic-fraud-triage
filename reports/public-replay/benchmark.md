# Benchmark

Backend: **replay**. Replay measures deterministic rules through the agent protocol, NOT an LLM

| System / population | N | Precision | Recall (all fraud) | Coverage | Review rate |
|---|---:|---:|---:|---:|---:|
| baseline_all_test | 20000 | 0.787 | 0.649 | 1.000 | 0.000 |
| baseline_same_cohort | 100 | 0.667 | 0.800 | 1.000 | 0.000 |
| agent_same_cohort | 100 | 0.100 | 0.400 | 0.750 | 0.250 |

Recall counts abstained fraud as not automatically detected. No credit is assigned to human review.
Undefined metrics are N/A; see JSON for confusion counts, cohort IDs, provenance and assumptions.

Estimated cost/investigation: 0.75 USD.
Baseline token + human-review cost/investigation: 0 USD (forced binary decisions).
Costs are scenario estimates excluding compute/training, not actual bills.

## Limitations

- Public data are simulated, not real bank transactions
- Temporal history uses prior unlabeled events, including earlier test rows
- No human outcomes are simulated or credited as correct
- No confidence calibration or production fraud guarantees
- Agent metrics concern sampled flags, not all transactions
