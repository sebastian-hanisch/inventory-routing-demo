"""Projektwurzel und tests/ auf den Importpfad, damit `pytest tests/` auch ohne `python -m` die irp_-Module und die Test-Hilfsmodule (irp_checks, irp_frozen)
findet. Die CI installiert immer das neueste NumPy (Ströme von default_rng dürfen sich ändern) und das neueste OR-Tools: Referenzwerte laufen deshalb über
EINGEFRORENE INSTANZEN (tests/data/irp_reference.json), Tests von make_instance prüfen nur Invarianten, und von CP-SAT wird nur der Zielwert geprüft."""
import pathlib
import sys

sys.dont_write_bytecode = True
TESTS = str(pathlib.Path(__file__).resolve().parent)
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
for p in (ROOT, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
