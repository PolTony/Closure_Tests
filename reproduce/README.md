# Experiment and artifact utilities

- `manifest.json` lists experiment commands, dependencies, required inputs, and declared outputs.
- `check_docs.py` checks the supplied Word documents and documentation links without rewriting manuscript content.
- `wheel_install_test.sh` builds and installs the package in a temporary environment and runs an installed benchmark.

From the repository root, `python3 src/reproduce.py --list` lists the experiments without executing them. Use `--jobs` for named jobs, `--all` for the complete manifest, and `--check` to compare a checkout against its recorded execution hashes. The installed-package script executes a simulation and is separate from a packaging-only inspection.
