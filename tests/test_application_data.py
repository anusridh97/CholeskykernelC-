import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.applications import prepare_gb1 as gb1
from benchmarks.applications.campaign import plan, require_completed


def rows():
    return [dict(zip("ABCDE", ["Variants", "HD", "Count input", "Count selected", "Fitness"])),
            {"A": "VDGV", "B": "0", "E": "1"},
            {"A": "ADGV", "B": "1", "E": "0"},
            {"A": "WWWW", "B": "4", "E": "8.5"}]


def test_measured_encoding_keeps_row_order_zeros_and_raw_targets():
    arrays = gb1.encode(rows())
    X, y = arrays["gb1_X.npy"], arrays["gb1_y.npy"]
    assert X.shape == (3, 80) and X.dtype == np.float32
    np.testing.assert_array_equal(X.sum(axis=1), [4, 4, 4])
    np.testing.assert_array_equal(y, [1, 0, 8.5])
    assert arrays["gb1_variants.npy"].tolist() == [b"VDGV", b"ADGV", b"WWWW"]
    # RBF geometry must encode two differing one-hot entries per changed site.
    assert np.square(X[0] - X[1]).sum() == 2
    assert np.square(X[0] - X[2]).sum() == 8
    for site, aa in enumerate("VDGV"):
        assert X[0, site * 20 + gb1.ALPHABET.index(aa)] == 1


@pytest.mark.parametrize("defect", ["duplicate", "missing_fitness", "nan", "negative", "amino_acid", "wrong_hd"])
def test_bad_measurements_are_rejected(defect):
    data = rows()
    if defect == "duplicate":
        data.append(data[1])
    elif defect == "missing_fitness":
        data[-1]["E"] = ""
    elif defect == "nan":
        data[-1]["E"] = "nan"
    elif defect == "negative":
        data[-1]["E"] = "-1"
    elif defect == "amino_acid":
        data[-1]["A"] = "WWWX"
    else:
        data[-1]["B"] = "0"
    with pytest.raises(ValueError):
        gb1.encode(data)


def test_source_checksum_failure_does_not_create_dataset(tmp_path):
    source = tmp_path / "changed.xlsx"
    source.write_bytes(b"untrusted data")
    destination = tmp_path / "data"
    with pytest.raises(ValueError, match="source checksum"):
        gb1.prepare(destination, source)
    assert not destination.exists()


def test_existing_user_inputs_are_not_overwritten(tmp_path):
    path = tmp_path / "gb1_X.npy"
    path.write_bytes(b"original user data")
    with pytest.raises(FileExistsError):
        gb1.prepare(tmp_path)
    assert path.read_bytes() == b"original user data"


def test_manifest_cannot_self_certify_modified_arrays(tmp_path):
    manifest = {"dataset": gb1.DATASET, "source": {"sha256": gb1.RAW_SHA256},
                "output_sha256": {"gb1_X.npy": "a" * 64}}
    (tmp_path / gb1.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen recipe"):
        gb1.verify(tmp_path)
    manifest["output_sha256"] = gb1.ARRAY_HASHES
    (tmp_path / gb1.MANIFEST).write_text(json.dumps(manifest))
    (tmp_path / "gb1_X.npy").write_bytes(b"changed array")
    with pytest.raises(ValueError, match="checksum mismatch"):
        gb1.verify(tmp_path)


def test_campaign_explicit_dimensions_and_no_silent_rank_clipping(tmp_path):
    jobs = plan(tmp_path, tmp_path / "out")
    assert len(jobs) == 7
    for name, cmd in jobs[:2]:
        rank = int(cmd[cmd.index("--ranks") + 1])
        initial = int(cmd[cmd.index("--initial") + 1])
        assert (rank, initial) in {(64, 96), (128, 128)}
        assert cmd[cmd.index("--rounds") + 1] == "15"
        assert cmd[cmd.index("--bo-seeds") + 1] == "5"
    for name, cmd in jobs[2:]:
        i = cmd.index("--hidden-dims")
        assert cmd[i + 1:i + 4] == ["512", "512", "128"]
        assert cmd[cmd.index("--epochs") + 1] == "10"
    one = plan(tmp_path, tmp_path / "out", experiments=("ng",), ng_ranks=(64,))
    assert len(one) == 1


def test_campaign_rejects_incomplete_child_even_if_process_returned_zero(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"status": "passed", "runs": [{"status": "running"}]}))
    with pytest.raises(RuntimeError, match="incomplete"):
        require_completed(path)
