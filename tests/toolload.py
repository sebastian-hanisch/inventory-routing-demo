"""Test-Hilfsmodul: ein Werkzeug aus tools/ als Modul laden (tools/ ist kein Paket); Umgebungsvariablen, die der Import setzt, werden zurückgenommen."""
import importlib.util
import os
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"


def load_tool(name):
    before = dict(os.environ)
    spec = importlib.util.spec_from_file_location(f"irp_tool_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for k in set(os.environ) - set(before):
        del os.environ[k]
    return mod
