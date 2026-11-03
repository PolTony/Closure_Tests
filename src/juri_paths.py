"""Project root resolution for checkout and installed (wheel) use.

Every script reads inputs and writes outputs relative to ROOT:
  * JURI_LAB_HOME, if set, names a workspace directory;
  * otherwise, in a source checkout, ROOT is the repository containing src/ and data_inputs/;
  * otherwise (installed wheel), ROOT is the current working directory, which should be a workspace created with
    `juri-lab init DIR` (copies the packaged inputs and the shipped transcripts out of the juri_lab_resources package).
Importing this module has no side effects on the filesystem.
"""
import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT_ROOT = os.path.dirname(_HERE)
IS_CHECKOUT = os.path.isdir(os.path.join(_CHECKOUT_ROOT, "data_inputs")) and os.path.isfile(os.path.join(_CHECKOUT_ROOT, "pyproject.toml"))


def _resolve() -> str:
    env = os.environ.get("JURI_LAB_HOME")
    if env:
        return os.path.abspath(env)
    if IS_CHECKOUT:
        return _CHECKOUT_ROOT
    return os.getcwd()


ROOT = _resolve()

# workspace members copied by `juri-lab init`: packaged inputs and the shipped live transcripts needed for replay
RESOURCE_ITEMS = ("data_inputs", "reproduce", "data_outputs/llm_transcripts_primary.jsonl",
                  "data_outputs/llm_transcripts_agentic.jsonl", "data_outputs/llm_transcripts_rulebooks.jsonl",
                  "submission", "docs", "tests", "README.md", "pyproject.toml")


def resource_root():
    """The packaged resources (a Traversable); in a checkout this is the checkout itself."""
    if IS_CHECKOUT:
        from pathlib import Path
        return Path(_CHECKOUT_ROOT)
    from importlib.resources import files
    return files("juri_lab_resources")


def module_path(name: str) -> str:
    """Filesystem path of a top-level lab module (checkout file or installed module)."""
    if os.path.dirname(name):
        workspace_script = os.path.join(ROOT, name)
        if os.path.isfile(workspace_script):
            return workspace_script
    p = os.path.join(_HERE, name if name.endswith(".py") else name + ".py")
    if os.path.exists(p):
        return p
    import importlib.util
    spec = importlib.util.find_spec(os.path.splitext(os.path.basename(name))[0])
    if spec is None or not spec.origin:
        raise FileNotFoundError(name)
    return spec.origin


def _copy_traversable(src, dest: str) -> int:
    if src.is_dir():
        os.makedirs(dest, exist_ok=True)
        return sum(_copy_traversable(c, os.path.join(dest, c.name)) for c in src.iterdir()
                   if c.name not in ("__pycache__", "__init__.py") and not c.name.startswith((".", "~$")))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with src.open("rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return 1


def init_workspace(dest: str) -> int:
    """Copy packaged inputs into `dest`; refuses to overwrite existing files. Returns the number of files copied."""
    dest = os.path.abspath(dest)
    res = resource_root()
    n = 0
    for item in RESOURCE_ITEMS:
        src = res
        for part in item.split("/"):
            src = src / part
        target = os.path.join(dest, *item.split("/"))
        if os.path.exists(target):
            raise FileExistsError(f"{target} exists; choose an empty workspace")
        n += _copy_traversable(src, target)
    os.makedirs(os.path.join(dest, "data_outputs"), exist_ok=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(prog="juri-lab", description="JURI Lab workspace helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init", help="create a workspace with the packaged inputs and shipped transcripts")
    pi.add_argument("dir")
    sub.add_parser("where", help="print the resolved project root")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        n = init_workspace(a.dir)
        print(f"copied {n} files into {os.path.abspath(a.dir)}; run lab scripts from there or set JURI_LAB_HOME")
    else:
        print(ROOT, "(checkout)" if IS_CHECKOUT and not os.environ.get("JURI_LAB_HOME") else "")


if __name__ == "__main__":
    main(sys.argv[1:])
