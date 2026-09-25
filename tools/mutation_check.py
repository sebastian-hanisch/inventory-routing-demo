"""Fehler-Einbau-Test: baut einzelne Fehler in die Module ein und prüft, ob die Tests (ohne AppTests, die sind zu langsam für viele Mutanten) sie finden.

Aufruf (im Projektordner): ./venv/Scripts/python.exe tools/mutation_check.py [Teilstring des Dateinamens] [--jobs N]
Jeder Mutant ersetzt genau eine Stelle; Überlebende sind entweder gleichwertig (kein sichtbarer Unterschied) oder eine Lücke der Tests. Jeder Mutant läuft in einer eigenen
temporären Kopie (deshalb parallel möglich, Standard 6 Jobs); PYTHONDONTWRITEBYTECODE=1, damit veralteter Bytecode keine Überlebenden vortäuscht; Quelltexte als LF (Windows-Python
schreibt sonst CRLF und die Zeichenketten in tools/mutants.py finden nichts).

Selbstprüfung (Baseline): VOR den Mutanten läuft eine UNVERÄNDERTE Kopie gegen die Tests, für ein Kernmodul und für ein Randmodul. Besteht sie nicht, bricht das Werkzeug ab: sonst wäre jeder
"gefundene" Mutant vorgetäuscht (in wellenfreigabe-demo fehlte der Kopie zuerst das README, das test_claims.py liest, und alle Läufe waren ungültig). Die Kopie enthält deshalb das ganze
Projekt: alle *.py, README.md, data/, tests/ (mit tests/data) und tools/. IRP_MUTATION_RUN=1 schützt die Meta-Tests (Mutantenliste, Quelltext-Texte des README), die sich sonst in der
mutierten Kopie selbst "finden" würden.

Die Mutanten sind maschinell erzeugt (tools/gen_mutants.py: Vergleichsoperatoren, Plus/Minus, and/or, True/False, min/max, Zahlen +1 bzw. +10 %), stichprobenartig über alle Module verteilt
(feste Zufallszahlen) und stehen als Literal in tools/mutants.py. Die Tests der Kernmodule laufen auf EINGEFRORENEN Instanzen (tests/data): kein Mutant hängt vom Zufallsgenerator ab.

Reihenfolge: die Tests des Moduls laufen zuerst, damit ein gefundener Mutant schnell scheitert; die langsamen Modelltests (test_checks.py, test_frozen_reference.py, test_oracle.py) laufen nur
für die Kerndateien. Bewusst NICHT als Mutanten geführt (gleichwertig, kein sichtbarer Unterschied): siehe tools/mutants.py (EQUIVALENT_NOTES)."""
import concurrent.futures as cf
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")                # die Mutanten enthalten Emoji und Umlaute

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mutants import MUTANTS  # noqa: E402

PY = sys.executable
TIMEOUT = 420
RERUN_MODE = False                    # von --rerun gesetzt: Mutanten, deren Stelle inzwischen entfernt wurde, gelten als entfallen statt als Fehler der Liste
CORE = ("irp_model.py", "irp_routing.py", "irp_policy.py", "irp_oracle.py")
APP_TESTS = ("tests/test_app.py",)
SLOW_CORE_TESTS = ("tests/test_checks.py", "tests/test_frozen_reference.py", "tests/test_oracle.py")
# Testdateien, die ein Modul zuerst prüfen (schnelles Scheitern); der Rest folgt in Dateireihenfolge
FIRST = {
    "irp_model.py": ["test_model_units.py", "test_frozen_reference.py"], "irp_routing.py": ["test_model_units.py", "test_frozen_reference.py"],
    "irp_policy.py": ["test_model_units.py", "test_frozen_reference.py"], "irp_oracle.py": ["test_oracle.py", "test_checks.py"],
    "irp_live.py": ["test_live.py"], "irp_results.py": ["test_results.py"], "irp_stories.py": ["test_stories.py"], "irp_presets.py": ["test_presets.py"],
    "irp_visualization.py": ["test_visualization.py"], "irp_pdf_export.py": ["test_pdf_export.py"], "irp_ui_panel.py": ["test_ui_panel.py"], "irp_constants.py": ["test_presets.py", "test_results.py"],
    "irp_format.py": ["test_ui_panel.py"], "tools/build_results.py": ["test_tools.py"], "tools/analyze.py": ["test_tools.py"], "tools/oracle_run.py": ["test_tools.py"],
    "tools/confirm_default.py": ["test_tools.py"],
}


def build_args(name):
    tests = sorted(p.as_posix() for p in (ROOT / "tests").glob("test_*.py"))
    rel = [str(pathlib.PurePosixPath("tests") / pathlib.PurePosixPath(t).name) for t in tests]
    skip = set(APP_TESTS)
    if name not in CORE:
        skip |= set(SLOW_CORE_TESTS)
    first = [f"tests/{f}" for f in FIRST.get(name, []) if f"tests/{f}" not in skip]
    rest = [t for t in rel if t not in skip and t not in first]
    return first + rest


def run_one(n, name, old, new, tmp_root):
    """Ein Mutant in eigener Kopie; Rückgabe (n, name, old, new, Status)."""
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"irp_mut{n}_", dir=tmp_root))
    try:
        for f in ROOT.glob("*.py"):
            (work / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        (work / "tools").mkdir()
        for f in (ROOT / "tools").glob("*.py"):
            (work / "tools" / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        shutil.copy(ROOT / "README.md", work / "README.md")                        # test_claims.py liest das README und prüft die Dateistruktur-Tabelle gegen die echten Dateien
        for extra in ("requirements.txt", "requirements-dev.txt", ".gitignore"):
            shutil.copy(ROOT / extra, work / extra)
        shutil.copytree(ROOT / ".github", work / ".github")
        shutil.copytree(ROOT / "data", work / "data")                              # test_results.py liest die Ergebnisdatei
        shutil.copytree(ROOT / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        for f in (work / "tests").rglob("*.py"):
            f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        path = work / name
        original = path.read_bytes().decode("utf-8")
        if original.count(old) != 1:
            return n, name, old, new, ("ENTFALLEN" if original.count(old) == 0 and RERUN_MODE else f"FEHLER:{original.count(old)}")     # ENTFALLEN: die Stelle wurde mit totem Code entfernt
        path.write_bytes(original.replace(old, new).encode("utf-8"))
        args = [PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"] + build_args(name)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["IRP_MUTATION_RUN"] = "1"                      # tests/test_tools.py und test_claims.py prüfen die Mutantenliste bzw. Quelltexte selbst und würden sonst jeden Mutanten "finden"
        try:
            r = subprocess.run(args, cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT)
            return n, name, old, new, "UEBERLEBT" if r.returncode == 0 else "gefunden"
        except subprocess.TimeoutExpired:
            return n, name, old, new, "gefunden(Zeitueberschreitung)"     # Endlosschleife gilt als gefunden
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = 6
    for i, a in enumerate(sys.argv[1:]):
        if a == "--jobs":
            jobs = int(sys.argv[i + 2])
            args = [x for x in args if x != sys.argv[i + 2]]
    only = args[0] if args else ""
    rerun = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--rerun":                                    # nur die Mutanten, die im Protokoll eines früheren Laufs überlebt haben (nach dem Schließen von Testlücken)
            log = pathlib.Path(sys.argv[i + 2])
            global RERUN_MODE
            RERUN_MODE = True
            rerun = {int(m.group(1)) for m in re.finditer(r"^\[\s*(\d+)\] UEBERLEBT", log.read_text(encoding="utf-8"), flags=re.M)}
            args = [x for x in args if x != sys.argv[i + 2]]
            only = args[0] if args else ""
    tmp_root = tempfile.mkdtemp(prefix="irp_mut_")
    # Selbstprüfung des Werkzeugs: ein Mutant, der nichts ändert, MUSS überleben. Sonst scheitern die Tests schon in der Kopie (fehlende Datei, Umgebung),
    # und jeder "gefundene" Mutant wäre vorgetäuscht. Ein Kernmodul und ein Randmodul.
    for probe, marker in (("irp_policy.py", "import numpy as np"), ("irp_results.py", "import json")):
        status = run_one(0, probe, marker, marker, tmp_root)[4]
        if status != "UEBERLEBT":
            print(f"ABBRUCH: unveränderte Kopie besteht die Tests nicht ({probe}: {status}) - Ergebnisse wären wertlos")
            shutil.rmtree(tmp_root, ignore_errors=True)
            return 2
    print("Selbstprüfung: unveränderte Kopie besteht alle Tests (Werkzeug funktioniert)", flush=True)
    todo = [(n, *m) for n, m in enumerate(MUTANTS, 1) if (not only or only in m[0]) and (rerun is None or n in rerun)]
    survivors, errors, killed = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, n, name, old, new, tmp_root) for n, name, old, new in todo]
        for fut in cf.as_completed(futures):
            n, name, old, new, status = fut.result()
            if status == "ENTFALLEN":
                print(f"[{n:3d}] ENTFALLEN (Code entfernt)  {name}: {old[:60]!r}", flush=True)
            elif status.startswith("FEHLER"):
                errors.append((n, name, old[:60], status))
                print(f"[{n:3d}] FEHLER (Stelle nicht eindeutig: {status})  {name}: {old[:60]!r}", flush=True)
            elif status == "UEBERLEBT":
                survivors.append((n, name, old[:70], new[:70]))
                print(f"[{n:3d}] UEBERLEBT  {name}: {old[:80]!r} -> {new[:80]!r}", flush=True)
            else:
                killed += 1
                print(f"[{n:3d}] {status}  {name}", flush=True)
    print(f"\n{killed} gefunden, {len(survivors)} überlebt, {len(errors)} Fehler in der Mutantenliste (von {len(todo)})")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 1 if (survivors or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
