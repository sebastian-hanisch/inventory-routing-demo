"""Werkzeuge im Ordner tools/: Reduktion der Messreihe zur Ergebnisdatei (build_results), Auswertung der Rohdaten (analyze), die Konfigurations- und Regelliste des Sweeps (sweep), die
Auswertung der Orakel-Läufe (oracle_run), Bau der eingefrorenen Referenz (freeze_reference), Bestätigung der Standardregel (confirm_default), Abstimmung der Presets, Vollständigkeit der
Mutantenliste. Der Sweep selbst (126 s auf 12 Prozessen) und die Orakel-Läufe laufen nie in den Tests."""
import copy
import json
import math
import os
import pathlib

import numpy as np
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_oracle as OR
import irp_results as R
from toolload import load_tool

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ROOT.parent / "bestand-planung" / "messreihe_inventory_routing"
DATA = R.load_results()
BR = load_tool("build_results")
AN = load_tool("analyze")
SW = load_tool("sweep")
OQ = load_tool("oracle_run")
FR = load_tool("freeze_reference")
CD = load_tool("confirm_default")
TP = load_tool("tune_presets")
REF = FZ.reference()


# ---------------------------------------------------------------------------------------------------
# build_results
# ---------------------------------------------------------------------------------------------------
def test_rnd_rounds_recursively_and_keeps_special_values():
    assert BR.rnd(1.23456789) == 1.2346 and BR.rnd(5) == 5 and BR.rnd(None) is None and BR.rnd(True) is True and BR.rnd("x") == "x" and BR.rnd(0.00004) == 0.0
    assert BR.rnd([1.23456, [2.98766, None]]) == [1.2346, [2.9877, None]] and BR.rnd((0.00004, 1)) == [0.0, 1]
    assert BR.rnd({"a": 0.123456, "b": {"c": [1.55557]}}) == {"a": 0.1235, "b": {"c": [1.5556]}} and BR.rnd(1.23456789, 2) == 1.23
    assert BR.DIGITS == 4 and BR._pick({"a": 1, "b": 2}, ("b", "z")) == {"b": 2}


def test_dumps_is_deterministic_compact_and_sorted():
    a = BR.dumps({"b": [1, 2], "a": {"z": 1.5, "y": "ä"}})
    assert a == '{"a":{"y":"ä","z":1.5},"b":[1,2]}\n' and BR.dumps({"a": 1, "b": 2}) == BR.dumps({"b": 2, "a": 1})


def test_the_base_configuration_of_the_tool_equals_the_app_constants():
    assert BR.BASE_CFG == C.BASE_CFG and BR.GAIN_KEYS == ("mean", "se", "median", "q1", "q3", "share_loss")


def synthetic_cell():
    gain = dict(mean=1.234567, se=0.1, median=1.2, q1=0.5, q3=2.0, share_loss=0.05, extra=99)
    return dict(tau=0.6, JR=1000.5, Q_over_qL=3.0, qL=100.0, short_R=10.0, short_P=5.0, cv_gain=20.0, cv_chosen=["P_H8_g0.25", "R"], best_insample="P_H8_g0.25",
                default=dict(gain), best_insample_gain=dict(gain), P={"P_H3_g0.50": dict(gain)}, E={"E_L1": dict(gain)}, routes_R=1.0, routes_P=1.0, visits_R=1.0, visits_P=1.0, opt_visits_P=1.0,
                stockout_R=1.0, stockout_P=1.0, deferred_R=0.0, util_R=0.5, util_P=0.6, km_R=10.0, km_P=9.0, short_cmp=dict(more=0.1, less=0.9, equal=0.0),
                routes_E=1.5, visits_E=2.5, km_E=11.0, util_E=0.55, short_E=0.0, stockout_E=0.0, unwanted=5)


def test_build_cell_keeps_only_what_the_app_shows_and_checks_the_penalty():
    out = BR.build_cell("q150", {"Q": 150.0}, 5.0, synthetic_cell())
    assert out["name"] == "q150" and out["cfg"] == dict(C.BASE_CFG, Q=150.0, penalty=5.0) and "unwanted" not in out
    assert set(out["default"]) == set(BR.GAIN_KEYS) and "extra" not in out["P"]["P_H3_g0.50"] and set(out["E"]["E_L1"]) == set(BR.GAIN_KEYS)
    assert out["cv_chosen"] == ["P_H8_g0.25", "R"] and out["short_cmp"]["more"] == 0.1 and set(out) >= set(BR.CELL_KEYS)
    assert BR.build_cell("pen100", {"pfac": 1.0}, 100.0, synthetic_cell())["cfg"]["penalty"] == 100.0
    with pytest.raises(AssertionError):
        BR.build_cell("pen100", {"pfac": 1.0}, 5.0, synthetic_cell())


def test_build_assembles_all_parts():
    meta = {"n_inst": 200, "days": 120, "policies": {"R": [0.0, 0.5, 0.0]}, "configs": {"base": {}, "q150": {"Q": 150.0}}, "penalty": {"base": 5.0, "q150": 5.0}}
    sweep = {"configs": {"base": synthetic_cell(), "q150": synthetic_cell()}, "spearman_gain_vs_Q_over_qL": {"rho": 0.7, "n": 23}, "base_distribution": {"min": 1.0}, "n_negative_pairs": [1, 2], "E_negative_pairs": [3, 4]}
    orow = {"seed": 0, "status": "OPTIMAL", "time": 0.5, "J_oracle": 100.0, "oracle_routes": 3, "oracle_visits": 10, "oracle_short": 0.0, "R": 120.0, "R_routes": 5, "R_short": 0.0, "P_H3_g0.50": 110.0,
            "P_H3_g0.50_routes": 4, "P_H3_g0.50_short": 0.0, "P_H8_g0.25": 111.0, "P_H8_g0.25_routes": 4, "P_H8_g0.25_short": 0.0, "E_L1": 150.0, "E_L1_routes": 6, "E_L1_short": 0.0, "junk": 1}
    oracle = {"agg": {k: 1.0 for k in BR.ORACLE_AGG_KEYS}, "rows": [orow]}
    out = BR.build(meta, sweep, oracle, copy.deepcopy(oracle))
    assert set(out) == {"_meta", "cells", "oracle"} and [c["name"] for c in out["cells"]] == ["base", "q150"] and out["_meta"]["default"] == {"H": 3, "gamma": 0.5}
    assert out["_meta"]["spearman"] == {"n": 23, "rho": 0.7} and out["_meta"]["n_negative_pairs"] == [1, 2] and out["_meta"]["E_negative_pairs"] == [3, 4] and out["_meta"]["policies"] == ["R"]
    assert set(out["oracle"]) == {"Q150", "Q250"} and "junk" not in out["oracle"]["Q150"]["rows"][0] and set(out["oracle"]["Q150"]["rows"][0]) == set(BR.ORACLE_ROW_KEYS)
    assert out["cells"][0]["default"]["mean"] == 1.2346 and out["oracle"]["Q250"]["agg"]["n"] == 1.0


@pytest.mark.skipif(not (SOURCES / "sweep_meta.json").exists() or not (ROOT / "tools" / "_out" / "sweep_data.json").exists(), reason="Quellen der Messreihe nicht vorhanden (nur lokal)")
def test_rebuilding_from_the_sources_reproduces_the_committed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(BR, "OUT", tmp_path / "irp_results.json")
    assert BR.main([]) == 0
    assert (tmp_path / "irp_results.json").read_bytes() == R.DATA_PATH.read_bytes()


def test_committed_results_are_sorted_compact_and_small():
    raw = R.DATA_PATH.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "\n" not in raw[:-1] and BR.dumps(json.loads(raw)) == raw and len(raw) < 200_000


# ---------------------------------------------------------------------------------------------------
# analyze auf künstlichen Rohdaten
# ---------------------------------------------------------------------------------------------------
METRICS = ["routing", "short", "dI", "delivered", "served", "n_routes", "n_visits", "opt_visits", "stockout_days", "deferred", "route_days", "util_sum"]


def synthetic_sweep(n=20, seed=3):
    rng = np.random.default_rng(seed)
    policies = {"R": [0.0, 0.5, 0.0], "P_H3_g0.50": [3.0, 0.5, 0.0], "P_H1_g0.10": [1.0, 0.1, 0.0], "P_H12_g1.00": [12.0, 1.0, 0.0], "E_L1": [0.0, 0.5, 1.0], "E_L2": [0.0, 0.5, 2.0], "E_L3": [0.0, 0.5, 3.0]}
    configs = {"base": {}, "q100": {"Q": 100.0}, "q600": {"Q": 600.0}}
    raw = {}
    for c in configs:
        base = rng.uniform(900, 1100, n)
        for pol, gain in (("R", 0.0), ("P_H3_g0.50", 0.15), ("P_H1_g0.10", 0.05), ("P_H12_g1.00", -0.1), ("E_L1", -0.05), ("E_L2", -0.1), ("E_L3", -0.2)):
            arr = np.zeros((12, n))
            arr[0] = base * (1 - gain) + rng.normal(0, 5, n)                        # routing
            arr[1] = rng.uniform(0, 4, n)                                           # short
            arr[2] = rng.normal(0, 3, n)                                            # dI
            arr[3] = rng.uniform(1500, 1600, n)                                     # delivered
            arr[5] = rng.integers(100, 140, n)                                      # n_routes
            arr[6] = rng.integers(200, 260, n)                                      # n_visits
            arr[7] = 0 if pol == "R" else rng.integers(50, 100, n)                  # opt_visits
            arr[10] = rng.integers(80, 100, n)
            arr[11] = arr[5] * 0.6
            raw[f"{c}|{pol}"] = arr
    meta = {"metrics": METRICS, "policies": policies, "configs": configs, "n_inst": n, "days": 120, "penalty": {c: 5.0 for c in configs}}
    return meta, raw


def test_pct_gain_is_the_paired_difference_in_percent_of_the_mean_reactive_cost():
    jr = np.array([100.0, 200.0, 300.0, 400.0])
    jp = np.array([90.0, 150.0, 330.0, 300.0])
    g = AN.pct_gain(jr, jp)
    diff = jr - jp
    assert g["mean"] == pytest.approx(100 * diff.mean() / jr.mean()) and g["se"] == pytest.approx(100 * diff.std(ddof=1) / math.sqrt(4) / jr.mean())
    per = 100 * diff / jr
    assert g["median"] == pytest.approx(np.median(per)) and g["q1"] == pytest.approx(np.percentile(per, 25)) and g["q3"] == pytest.approx(np.percentile(per, 75)) and g["share_loss"] == 0.25


def test_ranks_orders_from_zero():
    assert list(AN.ranks(np.array([3.0, 1.0, 2.0]))) == [2.0, 0.0, 1.0]


def test_analyze_computes_tau_gains_cross_validation_and_summaries():
    meta, raw = synthetic_sweep()
    data, report = AN.analyze(meta, raw)
    assert set(data) == {"configs", "spearman_gain_vs_Q_over_qL", "base_distribution", "n_negative_pairs", "E_negative_pairs"} and list(data["configs"]) == ["base", "q100", "q600"]
    c = data["configs"]["base"]
    tau = raw["base|R"][0].sum() / raw["base|R"][3].sum()
    jr = raw["base|R"][0] + 5.0 * raw["base|R"][1] - tau * raw["base|R"][2]
    jp = raw["base|P_H3_g0.50"][0] + 5.0 * raw["base|P_H3_g0.50"][1] - tau * raw["base|P_H3_g0.50"][2]
    assert c["tau"] == pytest.approx(tau) and c["JR"] == pytest.approx(jr.mean()) and c["default"] == c["P"]["P_H3_g0.50"] and c["default"]["mean"] == pytest.approx(100 * (jr - jp).mean() / jr.mean())
    assert c["Q_over_qL"] == pytest.approx(300.0 / (raw["base|R"][3].sum() / raw["base|R"][6].sum())) and data["configs"]["q100"]["Q_over_qL"] == pytest.approx(100.0 / (raw["q100|R"][3].sum() / raw["q100|R"][6].sum()))
    assert set(c["P"]) == {"P_H3_g0.50", "P_H1_g0.10", "P_H12_g1.00"} and set(c["E"]) == {"E_L1", "E_L2", "E_L3"} and c["best_insample"] == max(c["P"], key=lambda k: c["P"][k]["mean"])
    assert c["opt_visits_P"] == pytest.approx(raw["base|P_H3_g0.50"][7].mean()) and c["routes_R"] == pytest.approx(raw["base|R"][5].mean()) and c["km_P"] == pytest.approx(raw["base|P_H3_g0.50"][0].mean())
    assert c["util_R"] == pytest.approx(raw["base|R"][11].sum() / raw["base|R"][5].sum()) and c["short_R"] == pytest.approx(raw["base|R"][1].mean()) and c["deferred_R"] == 0.0
    sr, sp = raw["base|R"][1], raw["base|P_H3_g0.50"][1]
    assert c["short_cmp"] == {"more": float((sp > sr + 1e-9).mean()), "less": float((sp < sr - 1e-9).mean()), "equal": float((abs(sp - sr) <= 1e-9).mean())}
    assert abs(sum(c["short_cmp"].values()) - 1.0) < 1e-9
    assert 0.0 < c["cv_gain"] < 30.0 and len(c["cv_chosen"]) == 2 and all(k in c["P"] for k in c["cv_chosen"])
    assert data["spearman_gain_vs_Q_over_qL"]["n"] == 3 and -1.0 <= data["spearman_gain_vs_Q_over_qL"]["rho"] <= 1.0
    assert data["n_negative_pairs"] == [3, 9] and data["E_negative_pairs"] == [9, 9]                              # P_H12_g1.00 ist überall negativ; alle drei E negativ
    assert set(data["base_distribution"]) == {"min", "q1", "median", "q3", "p90", "max", "loss_share"} and data["base_distribution"]["min"] <= data["base_distribution"]["median"] <= data["base_distribution"]["max"]
    assert any(line.startswith("base") for line in report) and any("Rangkorrelation" in line for line in report) and any("insgesamt 3 von 9" in line for line in report)


def test_analyze_cross_validation_falls_back_to_reactive_when_no_rule_wins():
    meta, raw = synthetic_sweep()
    for key in list(raw):
        if key.startswith("base|P_"):
            raw[key] = raw["base|R"] * 1.5                                           # jede proaktive Regel ist auf allen Instanzen teurer
    data, _ = AN.analyze(meta, raw)
    assert data["configs"]["base"]["cv_chosen"] == ["R", "R"] and data["configs"]["base"]["cv_gain"] == 0.0


def test_analyze_main_reads_and_writes_the_files(tmp_path):
    meta, raw = synthetic_sweep()
    (tmp_path / "sweep_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(tmp_path / "raw_sweep.npz", **raw)
    assert AN.main([str(tmp_path)]) == 0
    data = json.loads((tmp_path / "sweep_data.json").read_text(encoding="utf-8"))
    assert list(data["configs"]) == ["base", "q100", "q600"] and (tmp_path / "sweep_report.txt").read_text(encoding="utf-8").count("\n") > 5


@pytest.mark.skipif(not (SOURCES / "raw_sweep.npz").exists(), reason="Rohdaten der Messreihe nicht vorhanden (nur lokal)")
def test_analyze_reproduces_the_aggregates_of_the_measurement():
    meta = json.loads((SOURCES / "sweep_meta.json").read_text(encoding="utf-8"))
    data, _ = AN.analyze(meta, np.load(SOURCES / "raw_sweep.npz"))
    orig = json.loads((SOURCES / "sweep_data.json").read_text(encoding="utf-8"))
    new = json.loads(json.dumps(data, default=float))
    extra = {"short_cmp", "routes_E", "visits_E", "km_E", "util_E", "short_E", "stockout_E"}                       # neu gegenüber der Messreihe (Aggregate, die dort nur im Bericht standen)
    strip = lambda d: {c: {k: v for k, v in row.items() if k not in extra} for c, row in d["configs"].items()}
    assert strip(new) == strip(orig) and all(new[k] == orig[k] for k in ("spearman_gain_vs_Q_over_qL", "base_distribution", "n_negative_pairs", "E_negative_pairs"))


# ---------------------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------------------
def test_sweep_configurations_are_exactly_the_cells_of_the_results_file():
    assert list(SW.CONFIGS) == [c["name"] for c in R.cells(DATA)] and len(SW.CONFIGS) == 27
    for cell in R.cells(DATA):
        overrides = SW.CONFIGS[cell["name"]]
        want = dict(C.BASE_CFG, **overrides)
        assert {k: cell["cfg"][k] for k in C.BASE_CFG} == want, cell["name"]
    assert SW.N_INST == C.MEASURED_N == 200 and SW.D_DAYS == C.DAYS == 120 and SW.METRICS == METRICS and SW.CHUNK == 20


def test_sweep_policy_grid_is_the_live_grid():
    assert len(SW.POLICIES) == 32 and list(SW.POLICIES) == DATA["_meta"]["policies"] or sorted(SW.POLICIES) == sorted(DATA["_meta"]["policies"])
    assert SW.POLICIES["R"] == (0.0, 0.5, 0.0) and SW.POLICIES["P_H3_g0.50"] == (3.0, 0.5, 0.0) and SW.POLICIES["E_L2"] == (0.0, 0.5, 2.0)
    grid = {(h, g) for h in C.HORIZON_OPTIONS for g in C.GAMMA_OPTIONS}
    assert {(int(v[0]), v[1]) for k, v in SW.POLICIES.items() if k.startswith("P_")} == grid and {v[2] for k, v in SW.POLICIES.items() if k.startswith("E_")} == {float(x) for x in C.EARLY_OPTIONS}
    assert {R.p_key(h, g) for h, g in grid} == {k for k in SW.POLICIES if k.startswith("P_")}


def test_sweep_chunk_on_frozen_reference_instances():
    cname, lo, hi, out, pen = SW.run_chunk(("base", 0, 2))
    assert (cname, lo, hi, pen) == ("base", 0, 2, 5.0) and set(out) == set(SW.POLICIES) and all(a.shape == (12, 2) for a in out.values())
    if np.__version__ == REF["numpy"]:                                                                           # der Zufallsstrom kann sich zwischen NumPy-Versionen ändern
        for j, s in enumerate((0, 1)):
            c = FZ.case("base", s)
            for pol, values in c["expected"].items():
                assert list(out[pol][:, j]) == pytest.approx(values, rel=1e-9, abs=1e-9)


class FakePool:
    """Ersatz für multiprocessing.Pool: rechnet nacheinander im selben Prozess."""
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, tasks):
        return map(fn, tasks)


def test_sweep_main_writes_raw_data_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(SW, "Pool", FakePool)
    monkeypatch.setattr(SW, "CONFIGS", {"base": {}, "q100": {"Q": 100.0}})
    monkeypatch.setattr(SW, "N_INST", 2)
    monkeypatch.setattr(SW, "CHUNK", 1)
    monkeypatch.setattr(SW, "D_DAYS", 10)
    assert SW.main([str(tmp_path)]) == 0
    raw = np.load(tmp_path / "raw_sweep.npz")
    meta = json.loads((tmp_path / "sweep_meta.json").read_text(encoding="utf-8"))
    assert len(raw.files) == 2 * 32 and raw["base|R"].shape == (12, 2) and meta["metrics"] == METRICS and list(meta["configs"]) == ["base", "q100"] and meta["n_inst"] == 2 and meta["days"] == 10
    assert meta["penalty"] == {"base": 5.0, "q100": 5.0} and meta["policies"]["P_H3_g0.50"] == [3.0, 0.5, 0.0] and np.all(raw["base|R"][7] == 0) and np.all(raw["q100|R"][0] >= 0)


# ---------------------------------------------------------------------------------------------------
# oracle_run
# ---------------------------------------------------------------------------------------------------
def synthetic_rows():
    rows = []
    for k, (jo, r, p3, p8, e1) in enumerate([(100.0, 130.0, 110.0, 115.0, 170.0), (120.0, 150.0, 130.0, 125.0, 190.0), (140.0, 140.0, 150.0, 160.0, 200.0)]):
        row = {"seed": k, "status": "OPTIMAL", "time": 1.0, "J_oracle": jo, "oracle_routes": 3, "oracle_visits": 10, "R": r, "P_H3_g0.50": p3, "P_H8_g0.25": p8, "E_L1": e1}
        for name, routes in (("R", 5), ("P_H3_g0.50", 4), ("P_H8_g0.25", 4), ("E_L1", 6)):
            row[name + "_routes"], row[name + "_short"], row[name + "_beats_oracle_with_shortage"] = routes, 0.0, False
        rows.append(row)
    return rows


def test_oracle_aggregate_computes_gaps_gains_and_closed_share():
    rows = synthetic_rows()
    agg, lines = OQ.aggregate(rows, 150, 120.0)
    assert agg["n"] == 3 and agg["n_optimal"] == 3 and agg["tau"] == 0.6 and agg["J_oracle"] == pytest.approx(120.0) and agg["R"] == pytest.approx(140.0)
    gap = 100 * (np.array([130.0, 150.0, 140.0]) - np.array([100.0, 120.0, 140.0])) / np.array([100.0, 120.0, 140.0])
    g = agg["R_gap_to_oracle"]
    assert g["mean"] == pytest.approx(gap.mean()) and g["se"] == pytest.approx(gap.std(ddof=1) / math.sqrt(3)) and g["median"] == pytest.approx(np.median(gap)) and g["max"] == pytest.approx(gap.max()) and g["min"] == 0.0
    gain = 100 * (np.array([130.0, 150.0, 140.0]) - np.array([110.0, 130.0, 150.0])) / np.array([130.0, 150.0, 140.0])
    assert agg["P_H3_g0.50_gain_over_R"]["mean"] == pytest.approx(gain.mean())
    closed = (np.array([130.0, 150.0]) - np.array([110.0, 130.0])) / (np.array([130.0, 150.0]) - np.array([100.0, 120.0]))     # Zeile 3: R = Optimum, nicht definiert
    assert agg["P_H3_g0.50_share_of_R_gap_closed"] == pytest.approx(closed.mean()) and agg["oracle_gain_over_R"]["mean"] == pytest.approx((100 * (np.array([130.0, 150.0, 140.0]) - np.array([100.0, 120.0, 140.0])) / np.array([130.0, 150.0, 140.0])).mean())
    assert any(line.startswith("Optimum: mittlere Touren 3.00, Besuche 10.00; R: Touren 5.00") for line in lines) and any("Exakter Maszstab: 3 Instanzen" in line for line in lines)


def test_oracle_solve_row_on_a_small_instance_is_consistent():
    row = OQ.solve_row(0, 150, 60.0)
    assert row["status"] == "OPTIMAL" and row["seed"] == 0 and abs(row["J_oracle"] - row["J_oracle_model"]) < 1e-6 and row["oracle_short"] == 0.0 and row["bound_J"] == pytest.approx(row["J_oracle_model"])
    for name in OQ.POL:
        assert row[name + "_routes"] >= 1 and (row[name] >= row["J_oracle"] - 1e-6 or row[name + "_short"] > 0)
    assert row["oracle_routes"] <= 6 and set(OQ.POL) == {"R", "P_H3_g0.50", "P_H8_g0.25", "E_L1"} and OQ.TAU == 0.6


def test_oracle_main_writes_data_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(OQ, "solve_row", lambda s, wagon, tl: synthetic_rows()[s] if s < 3 else {"seed": s, "status": "NONE"})
    assert OQ.main(["3", "10", "250", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "oracle_data_Q250.json").read_text(encoding="utf-8"))
    assert len(data["rows"]) == 3 and data["agg"]["n"] == 3 and "Q=250" in (tmp_path / "oracle_report_Q250.txt").read_text(encoding="utf-8")


def test_committed_oracle_rows_pass_the_same_consistency_checks():
    for wagon in (150, 250):
        rows = R.oracle_rows(DATA, wagon)
        agg, _ = OQ.aggregate([dict(r, **{f"{n}_beats_oracle_with_shortage": False for n in OQ.POL}) for r in rows], wagon, 120.0)
        assert agg["R_gap_to_oracle"]["mean"] == pytest.approx(R.oracle_agg(DATA, wagon)["R_gap_to_oracle"]["mean"], abs=2e-3)


# ---------------------------------------------------------------------------------------------------
# freeze_reference, confirm_default, tune_presets
# ---------------------------------------------------------------------------------------------------
def test_freeze_helpers_round_trip_bit_exactly():
    a = np.array([[0.1, -0.0, 1e-300, 123456789.123456789], [np.pi, np.e, 2.0 ** -52, 1.0]])
    b = FR.dec(FR.enc(a))
    assert b.shape == a.shape and a.tobytes() == b.tobytes() and FZ.dec(FR.enc(a)).tobytes() == a.tobytes()
    inst = OR.make_oracle_instance(1)
    back = FR.inst_from_json(json.loads(json.dumps(FR.inst_to_json(inst))))
    assert FR.same_instance(inst, back) and FZ.inst_from_json(FR.inst_to_json(inst)).dist.tobytes() == inst.dist.tobytes()
    back.cons[0, 1] = np.nextafter(back.cons[0, 1], 1e9)
    assert not FR.same_instance(inst, back)
    back2 = FR.inst_from_json(FR.inst_to_json(inst))
    back2.Q += 1
    assert not FR.same_instance(inst, back2) and (back2.N, back2.D, back2.F) == (inst.N, inst.D, inst.F)


def test_freeze_cases_are_present_in_the_committed_reference():
    assert [(c["config"], c["seed"]) for c in REF["cases"]] == [(n, s) for n, seeds in FR.CASES for s in seeds] and len(REF["cases"]) == 14
    oracle = FZ.oracle_frozen()
    assert [(c["wagon"], c["seed"]) for c in oracle["cases"]] == [(w, s) for w in FR.ORACLE_WAGONS for s in FR.ORACLE_SEEDS] and oracle["tau"] == OR.TAU
    for c in REF["cases"]:
        assert FR.same_instance(FR.inst_from_json(c["instance"]), FZ.inst_from_json(c["instance"]))


@pytest.mark.skipif(not (SOURCES / "raw_sweep.npz").exists() or np.__version__ != REF["numpy"], reason="Quellen der Messreihe und dieselbe NumPy-Version nötig (nur lokal)")
def test_refreezing_reproduces_the_committed_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "ROOT", tmp_path)
    assert FR.main([str(SOURCES)]) == 0
    assert (tmp_path / "tests" / "data" / "irp_reference.json").read_bytes() == (ROOT / "tests" / "data" / "irp_reference.json").read_bytes()
    assert (tmp_path / "tests" / "data" / "irp_oracle_frozen.json").read_bytes() == (ROOT / "tests" / "data" / "irp_oracle_frozen.json").read_bytes()


def test_confirm_default_evaluates_the_standard_rule_on_fresh_seeds():
    res = CD.confirm({}, seeds=range(200, 230))
    assert res["n"] == 30 and res["tau"] > 0 and res["q1"] <= res["median"] <= res["q3"] and res["min"] <= res["q1"] and res["q3"] <= res["max"]
    assert 12.0 < res["mean"] < 22.0 and 0.0 < res["se"] < 2.0 and res["share_loss"] == 0.0                 # Basisfall: die Standardregel spart 17 %, die Streuung ist klein
    same = CD.confirm({}, seeds=range(200, 230), tau=res["tau"])
    assert same["mean"] == res["mean"]
    zero_tau = CD.confirm({}, seeds=range(200, 230), tau=0.0)
    assert zero_tau["mean"] != res["mean"] and abs(zero_tau["mean"] - res["mean"]) < 3.0
    full = CD.confirm({"Q": 100.0}, seeds=range(200, 230))
    assert abs(full["mean"]) < 2.0 and CD.CONFIGS["flotte_F1_Q300"] == {"F": 1, "Q": 300.0} and CD.SEEDS == range(200, 300)
    pen = CD.confirm({"pfac": 1.0}, seeds=range(200, 210))
    assert pen["n"] == 10 and pen["mean"] != res["mean"]


def test_confirm_default_main_prints_the_comparison(capsys):
    assert CD.main(["q100"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("q100: Sweep (Seeds 0-199) +0.40 +- 0.05 %") and "frisch (Seeds 200-299" in out and "SE der Differenz" in out


def test_tune_presets_evaluates_all_presets_on_one_instance():
    res = TP.evaluate(C.SEED_DEFAULT)
    assert list(res) == list(C.PRESETS) and all(isinstance(ok, bool) and isinstance(crit, list) and isinstance(gain, float) for ok, crit, gain in res.values())
    assert res["Standard"][2] > 0 and res["Wagen fast voll"][2] < 5 and res["Zu großzügig"][2] < 0 and res["Großer Wagen"][2] > res["Standard"][2] > 0
    single = TP.evaluate(C.SEED_DEFAULT, {"Standard": C.PRESETS["Standard"]})
    assert list(single) == ["Standard"] and single["Standard"][2] == res["Standard"][2]


def test_check_full_is_importable():
    mod = load_tool("check_full")
    assert callable(mod.main) and mod.MEASURED_WITH_NUMPY == REF["numpy"]


@pytest.mark.skipif(os.environ.get("IRP_MUTATION_RUN") == "1", reason="im Fehler-Einbau-Lauf ist die Stelle des Mutanten absichtlich verändert")
def test_mutation_list_is_well_formed():
    mut = load_tool("mutation_check")
    assert len(mut.MUTANTS) >= 300 and len(set(mut.MUTANTS)) == len(mut.MUTANTS)
    for name, old, new in mut.MUTANTS:
        assert old != new, (name, old)
        path = ROOT / name
        assert path.exists(), name
        assert path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8").count(old) == 1, (name, old[:60])
    assert {m[0] for m in mut.MUTANTS} <= set(mut.FIRST)                                  # zu jedem mutierten Modul gibt es eine Testreihenfolge
    assert {m[0] for m in mut.MUTANTS} >= set(mut.CORE)


def test_analyze_adds_the_mechanism_of_early_delivery():
    meta, raw = synthetic_sweep()
    data, _ = AN.analyze(meta, raw)
    c = data["configs"]["base"]
    e = raw["base|E_L2"]
    assert c["routes_E"] == pytest.approx(e[5].mean()) and c["visits_E"] == pytest.approx(e[6].mean()) and c["km_E"] == pytest.approx(e[0].mean()) and c["short_E"] == pytest.approx(e[1].mean())
    assert c["util_E"] == pytest.approx(e[11].sum() / e[5].sum()) and c["stockout_E"] == pytest.approx(e[8].mean()) and AN.EARLY_POLICY == "E_L2"
