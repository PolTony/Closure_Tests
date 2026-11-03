# Results and recorded executions

| Filename pattern or folder | Contents |
| --- | --- |
| `results_*` | Task and occupation summaries, case traces, and model-agent results |
| `compiler_repair_logs.json` | Reference compiler's archive-learning and refinement records |
| `dynamics_*.csv` | Allocation and performance across periods and publication arrangements |
| `dynamics_labor_*.csv` | Workforce stocks, demand, promotion, and experience measures |
| `dynamics_knowledge_*.csv` | Knowledge shares, exposure, and competence at promotion |
| `bumping_*.csv` | Comparisons of professional and subordinate work |
| `robustness_summary.csv`, `archive_boundary.csv`, `structural_challenge.csv` | Learner sensitivity and representation tests |
| `closure_formation*.csv`, `validation_ladder.csv` | Institutional comparisons and their evidentiary limits |
| `llm_transcripts_*.jsonl` | Recorded model requests and responses used for replay |
| `llm_provenance*` | Served-model identifiers and request summaries |
| `sourced/` | Results under externally sourced occupational attributes |
| `run_logs/`, `RUN_MANIFEST.json` | Printed output and provenance from recorded executions |

The appendix's evidence-file table identifies the files behind individual results. Filenames distinguish experimental conditions; similarly named files should not be combined without checking their settings and denominators.

Recorded executions retain their original hashes. Changes to source files or commands require a new run before the integrity checker can certify the resulting checkout. Model replay reuses supplied responses and executes generated programs; it does not contact a provider. The appendix documents the limits of the available multi-turn histories.
