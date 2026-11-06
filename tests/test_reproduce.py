import hashlib
import json
from pathlib import Path
import sys
import pytest

from benchmarks import reproduce
from benchmarks.applications.run import arguments, configuration
from benchmarks.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/reproduced/b200-20261010"


def test_full_gpu_plan_preserves_isolation_and_matched_methods(tmp_path):
    steps = reproduce.plan("b200", tmp_path, tmp_path / "trace.npy")
    ladder = [(name, cmd) for name, cmd in steps if name.startswith("well-")]
    assert len(ladder) == 14
    assert {int(cmd[cmd.index("--ranks") + 1]) for _, cmd in ladder} == set(reproduce.WELL_RANKS)
    for _, cmd in ladder:
        assert cmd[cmd.index("--ranks") + 2] == "--trace"
    assert [name for name, _ in steps[:3]] == ["tests", "symmetric", "asymmetric"]
    smoke = reproduce.plan("b200", tmp_path, tmp_path / "trace.npy", smoke=True)
    assert len([name for name, _ in smoke if name.startswith("well-")]) == 2


@pytest.mark.parametrize("tag", ["skipped", "failure", "error"])
def test_zero_exit_is_not_enough_for_test_validation(tmp_path, tag):
    xml = tmp_path / "tests.xml"
    xml.write_text(f'<testsuite><testcase name="unvalidated"><{tag}/></testcase></testsuite>')
    with pytest.raises(RuntimeError, match="did not all pass"):
        reproduce.require_test_pass(xml)


def test_failed_process_stops_sequence_and_preserves_log(tmp_path):
    output = tmp_path / "run"
    steps = [("broken", [sys.executable, "-c", "print('evidence'); raise SystemExit(3)"]),
             ("must-not-run", [sys.executable, "-c", "raise AssertionError()"])]
    with pytest.raises(RuntimeError, match="exit 3"):
        reproduce.execute(steps, output, "cpu", False)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed"
    assert len(report["steps"]) == 1
    assert report["steps"][0]["returncode"] == 3
    assert "evidence" in (output / "broken.log").read_text()
    assert not (output / "must-not-run.log").exists()
    with pytest.raises(FileExistsError):
        reproduce.execute([], output, "cpu", False)


def test_recovered_artifacts_are_exact_verified_copies():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == entry["sha256"]


def test_offline_tables_match_frozen_measurements(tmp_path):
    summarize(EVIDENCE, tmp_path / "tables")
    import csv
    for method, expected in [("tiled", 113.3988785404709), ("inplace", 80.04348397845408)]:
        with (tmp_path / "tables" / f"well-{method}.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 7
        assert all(row["all_correctness_passed"] == "True" for row in rows)
        assert float(rows[-1]["median_transition_speedup"]) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("defect", ["failed_cross", "missing_sample", "bad_median", "nonfinite"])
def test_incomplete_or_failed_ladder_is_rejected(tmp_path, defect):
    data = json.loads((EVIDENCE / "well-tiled/ladder_r2048.json").read_text())
    if defect == "failed_cross":
        data["cases"][0]["cross_method_probe_comparison"]["passed"] = False
    elif defect == "missing_sample":
        data["cases"][0]["methods"]["inverse"]["event_seconds_samples"].pop()
    elif defect == "bad_median":
        data["cases"][0]["methods"]["inverse"]["event_seconds_median"] *= 2
    else:
        data["cases"][0]["methods"]["inverse"]["event_seconds_samples"][0] = float("nan")
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        reproduce.require_ladder_pass(path, 2048, "tiled")


def test_applications_explicit_inputs_and_protocol():
    bo = arguments(["bo"])
    assert bo.dataset == "gb1" and bo.backend == "extension"
    ng = arguments(["ng"])
    config = configuration(ng)
    assert config["dataset"] == "protein" and config["seeds"] == [42]
    assert config["chart_modes"] == ["inverse", "sandwich"]
    smoke = arguments(["bo", "--smoke", "--backend", "torch"])
    assert smoke.dataset == "synthetic" and smoke.ranks == [8]
    with pytest.raises(SystemExit):
        arguments(["bo", "--ranks", "128", "--initial", "96"])
    with pytest.raises(SystemExit):
        arguments(["bo", "--dataset", "gb1", "--smoke"])
    with pytest.raises(SystemExit):
        arguments(["ng", "--batch-size", "128"])
    with pytest.raises(SystemExit):
        arguments(["bo", "--ng-seeds", "1"])


def test_preserved_application_source_hashes():
    directory = ROOT / "benchmarks/applications"
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, entry in provenance["files"].items():
        assert hashlib.sha256((directory / "_corrected" / name).read_bytes()).hexdigest() == entry["ported_sha256"]

def test_missing_gb1_fails_before_creating_results(tmp_path):
    import subprocess
    destination = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.applications.run", "bo",
         "--data-dir", str(tmp_path), "--output-dir", str(destination)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Required real dataset missing" in result.stderr
    assert not destination.exists()
