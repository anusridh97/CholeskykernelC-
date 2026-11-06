"""Prepare the measured Wu et al. GB1 landscape from its original supplement.

This freezes a new, explicit encoding; it does not recover the missing encoding
used for the submitted paper's Table 4. Only NumPy and the standard library are
required. The original supplement and generated arrays are checksum verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from benchmarks.reproduce import ROOT, sha256

DATASET = "wu2016-gb1-measured-onehot-v1"
URL = "https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx"
RAW_SHA256 = "ac9a16295cd24110ad3ef4bcc57741f023ff9f3f5c3d9f5fdc0a94b996dca9b7"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
COUNT = 149361
MANIFEST = "gb1_manifest.json"
# Set from the independently inspected, pinned supplement; NPY format 1.0.
ARRAY_HASHES = {
    "gb1_X.npy": "3dbe1eca7051328e70d32fca14ce43be0a20367cf915aef72c2db4db67703a2c",
    "gb1_y.npy": "aff322c2b7d852e55e830986307d964249aaddf87d2866e124db1ebca2a86c21",
    "gb1_variants.npy": "11eda7f8d825f227242e5085b0581e92f75007ed6ea9cee66b2aae5903c42ff1",
}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_supplement(path):
    """Read only the fixed supplement's worksheet; never evaluate Excel formulas."""
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(node.itertext()) for node in ET.fromstring(
            archive.read("xl/sharedStrings.xml"))]
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != NS + "row":
                    continue
                values = {}
                for cell in row:
                    if cell.find(NS + "f") is not None:
                        raise ValueError("Unexpected formula in GB1 source")
                    node = cell.find(NS + "v")
                    value = node.text if node is not None else ""
                    if cell.get("t") == "s":
                        value = strings[int(value)]
                    column = cell.get("r").rstrip("0123456789")
                    values[column] = value
                yield values
                row.clear()


def encode(rows):
    import numpy as np
    iterator = iter(rows)
    header = next(iterator)
    if header != dict(zip("ABCDE", ["Variants", "HD", "Count input", "Count selected", "Fitness"])):
        raise ValueError("GB1 supplement header changed")
    variants, fitness = [], []
    seen = set()
    for row in iterator:
        variant = row["A"]
        if len(variant) != 4 or any(letter not in ALPHABET for letter in variant):
            raise ValueError(f"Invalid GB1 variant: {variant}")
        if variant in seen:
            raise ValueError(f"Duplicate GB1 variant: {variant}")
        value = float(row["E"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("GB1 requires finite, nonnegative measured fitness")
        if int(row["B"]) != sum(a != b for a, b in zip(variant, "VDGV")):
            raise ValueError("GB1 variant does not match its Hamming distance")
        seen.add(variant)
        variants.append(variant)
        fitness.append(value)
    if not variants:
        raise ValueError("No measured GB1 variants")
    codes = np.array([[ALPHABET.index(a) for a in v] for v in variants])
    X = np.zeros((len(variants), 80), dtype="<f4")
    X[np.arange(len(variants))[:, None], codes + 20 * np.arange(4)] = 1
    return {"gb1_X.npy": X, "gb1_y.npy": np.asarray(fitness, dtype="<f4"),
            "gb1_variants.npy": np.asarray(variants, dtype="S4")}


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("dataset") != DATASET or manifest.get("source", {}).get("sha256") != RAW_SHA256:
        raise ValueError("Unrecognized GB1 data manifest")
    if manifest.get("output_sha256") != ARRAY_HASHES:
        raise ValueError("GB1 manifest array hashes do not match the frozen recipe")
    for name, expected in ARRAY_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"GB1 array checksum mismatch: {name}")
    return manifest


def prepare(directory, source=None):
    import numpy as np
    directory = Path(directory)
    names = ["gb1_X.npy", "gb1_y.npy", "gb1_variants.npy", MANIFEST]
    if any((directory / name).exists() for name in names):
        if not all((directory / name).is_file() for name in names):
            raise FileExistsError("Partial GB1 inputs exist; use a fresh --data-dir")
        verify(directory)
        print(f"Verified existing {DATASET} in {directory}")
        return
    with tempfile.TemporaryDirectory(prefix="gb1-") as scratch:
        raw = Path(source) if source else Path(scratch) / "supplement.xlsx"
        if source is None:
            print(f"Downloading original GB1 supplement: {URL}", flush=True)
            request = urllib.request.Request(URL, headers={"User-Agent": "Cholesky-reproduction/1"})
            with urllib.request.urlopen(request, timeout=60) as response, raw.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        if sha256(raw) != RAW_SHA256:
            raise ValueError("GB1 source checksum mismatch; refusing changed input")
        arrays = encode(read_supplement(raw))
        if len(arrays["gb1_y.npy"]) != COUNT:
            raise ValueError("Wrong measured GB1 row count")
        wt = np.flatnonzero(arrays["gb1_variants.npy"] == b"VDGV")
        if len(wt) != 1 or arrays["gb1_y.npy"][wt[0]] != 1:
            raise ValueError("Wild-type GB1 fitness must be exactly 1")
        hashes = {}
        for name, array in arrays.items():
            path = Path(scratch) / name
            with path.open("wb") as out:
                np.lib.format.write_array(out, array, version=(1, 0), allow_pickle=False)
            hashes[name] = sha256(path)
        if hashes != ARRAY_HASHES:
            raise ValueError("Generated arrays differ from the frozen encoding")
        manifest = {
            "dataset": DATASET,
            "scope": "Explicit reconstruction; original Table 4 array equivalence unverified",
            "source": {"url": URL, "sha256": RAW_SHA256,
                       "citation": "Wu et al. (2016), eLife 5:e16965, Supplementary file 1",
                       "doi": "https://doi.org/10.7554/eLife.16965.024",
                       "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"},
            "encoding": {"positions": [39, 40, 41, 54], "alphabet": ALPHABET,
                         "features": "site-major one-hot, 4 sites x 20 amino acids",
                         "row_order": "original worksheet order; no sorting or shuffling",
                         "target": "published measured fitness, raw scale, float32",
                         "filter": "all measured rows including zero fitness; no imputed variants",
                         "normalization": "none", "npy_format": "1.0"},
            "rows": COUNT, "features": 80,
            "output_sha256": hashes,
            "preparation_source_sha256": sha256(Path(__file__)),
        }
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves user-supplied arrays and concurrent output.
        for name in names[:-1]:
            with (directory / name).open("xb") as out:
                out.write((Path(scratch) / name).read_bytes())
        with (directory / MANIFEST).open("x") as out:
            out.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(f"Prepared {COUNT} measured variants, 80 features in {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source", type=Path, help="Use a downloaded original XLSX; checksum still required")
    parser.add_argument("--verify", action="store_true", help="Verify prepared arrays without downloading")
    args = parser.parse_args(argv)
    if args.verify:
        verify(args.data_dir)
        print("GB1 source identity and all prepared array hashes verified")
    else:
        prepare(args.data_dir, args.source)


if __name__ == "__main__":
    main()
