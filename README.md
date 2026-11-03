# Closure Under Threat

A computational laboratory for studying automation, professional jurisdiction, and the reproduction of expertise.

Start with the [research proposal](submission/PROPOSAL_STAGE1.docx) and [technical appendix](submission/TECHNICAL_APPENDIX.docx). The appendix explains the experiments, assumptions, findings, and evidence files.

## Repository contents

| Folder | Contents |
| --- | --- |
| [submission/](submission/) | Proposal, technical appendix, and seven manuscript figures |
| [src/](src/) | Simulation models, learning algorithms, language-model interfaces, and experiment runners |
| [data_inputs/](data_inputs/) | Task specifications, occupational attributes, codebooks, and empirical reference material |
| [data_outputs/](data_outputs/) | Numerical results, case traces, recorded model responses, and run records |
| [compiled_tools/](compiled_tools/) | Generated standalone task programs for the three compiler generations |
| [reproduce/](reproduce/) | Experiment manifest, document checks, and installed-package check |
| [tests/](tests/) | Regression tests for the model and supporting infrastructure |
| [docs/](docs/) | Prospective validation protocol |

`pyproject.toml` specifies the package and optional dependencies. Package resources are maintained under `src/juri_lab_resources/` through links to the corresponding repository folders.

## What the laboratory represents

The reference model contains 18 occupations, 54 tasks, and 1,080 synthetic evaluation cases across six workflow families. Cases distinguish written rules, explicit exceptions, conventions inferred from resolved cases, and private precedents. Clients choose among direct automation, supervised automation, and professional execution under stated prices, liability, and sign-off rules.

The principal experiments use a deterministic simulation. Its archive learner fits formulas within a declared candidate set. The professional-learning specification represents knowledge acquisition through experience thresholds. Separate language-model experiments examine document reading, program generation, execution, and review. These components are described separately in the appendix.

The results support exploratory comparisons within the modeled workplace. They do not estimate occupational employment effects or independently validate an explanation of licensing.

## Running the code

Python 3.9 or later is required. The principal simulation uses the standard library. Install optional figure dependencies when needed:

```bash
python3 -m pip install -e '.[figures]'
```

From the repository root:

```bash
python3 src/run_live_benchmark.py
python3 src/run_dynamics.py --help
python3 src/run_llm_arm.py --help
python3 src/reproduce.py --list
```

Results are written relative to the repository root, independently of the location of the source files. Set `JURI_LAB_HOME` to use a separate workspace.

The complete experiment manifest can be executed with:

```bash
python3 src/reproduce.py --all --parallel 4
python3 src/reproduce.py --check
```

The default reproduction uses recorded model responses and makes no paid API requests. Live model calls require the optional `llm` dependency, provider credentials, and explicit live-call options. Generated programs execute in separate processes.

Stored run records describe the archived executions. A new execution is required to produce provenance hashes matching changed source files or experiment commands; archived records are not a claim that the current checkout has been rerun.

The regression tests can be run separately:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For an installed package outside the checkout, create a workspace with `juri-lab init /path/to/workspace`, set `JURI_LAB_HOME` to that directory, and invoke runners with `python3 -m`, such as `python3 -m run_dynamics --help`.

## Interpreting the evidence

The [technical appendix](submission/TECHNICAL_APPENDIX.docx) connects each comparison to its assumptions and outputs. The [input guide](data_inputs/README.md) describes the source material; the [output guide](data_outputs/README.md) explains result files and model transcripts. The [validation protocol](docs/VALIDATION_PROTOCOL.md) distinguishes exploratory comparisons from prospective independent validation.
