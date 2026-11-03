# Source code

| Files | Purpose |
| --- | --- |
| `domain_cases.py`, `model_config.py` | Synthetic cases, rule interpretation, and model parameters |
| `compiler_agent.py` | Written-rule compilation and formula learning from resolved cases |
| `negotiated_order_lab.py` | Task execution, professional review, and client allocation |
| `labor_market.py`, `profession.py` | Staffing, promotion, and experience-based knowledge acquisition |
| `llm_agents.py`, `occupation_compiler.py` | Language-model providers, program generation, execution, and replay |
| `run_*.py` | Experiment runners and result summaries |
| `propose_corpus.py` | Construction of the expanded task corpus |
| `make_figures.py`, `build_docx.py` | Figure generation and validation of supplied Word documents |
| `reproduce.py`, `juri_paths.py` | Experiment orchestration and workspace resolution |
| `juri_lab_resources/` | Resources included in the installed Python package |

Run scripts from the repository root using `python3 src/<script>.py`. Inputs and outputs remain in their top-level folders. The experiment manifest under `reproduce/` specifies the commands and dependencies for the full set of comparisons.
