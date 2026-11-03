# Regression tests

Tests cover information restrictions, generated-program execution, allocation, professional knowledge, JSON validation, model-response provenance, experiment dependencies, and packaging.

Run from the repository root with `PYTHONPATH=src python3 -m unittest discover -s tests -v`. Some tests construct synthetic cases and exercise model calculations; they are separate from the full reproduction manifest.
