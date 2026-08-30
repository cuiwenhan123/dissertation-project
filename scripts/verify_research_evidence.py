from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research_evidence" / "chapter4_main_study"
MODELS = ("transformer", "cnn")
DEGRADATIONS = ("blur", "lowlight", "jpeg")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def expected_condition_names() -> set[str]:
    names = {f"{model}__clean__s0.json.gz" for model in MODELS}
    names.update(
        f"{model}__{degradation}__s{severity}.json.gz"
        for model in MODELS
        for degradation in DEGRADATIONS
        for severity in range(1, 6)
    )
    return names


def verify_checksums() -> None:
    for line in (EVIDENCE / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = EVIDENCE / relative
        if not path.is_file() or digest(path) != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def verify_conditions(manifest: dict[str, Any]) -> None:
    condition_dir = EVIDENCE / "conditions"
    actual = {path.name for path in condition_dir.glob("*.json.gz")}
    expected = expected_condition_names()
    if actual != expected:
        raise AssertionError(f"condition archive set differs: {sorted(actual ^ expected)}")

    selected = manifest["selectedImages"]
    if len(selected) != 500 or len(set(selected)) != 500:
        raise AssertionError("manifest does not contain 500 unique selected images")
    for path in sorted(condition_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            samples = json.load(handle)["samples"]
        names = [sample["name"] for sample in samples]
        if names != selected:
            raise AssertionError(f"image pairing differs in {path.name}")


def close(actual: float, expected: float, tolerance: float = 0.0005) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"reported value differs: {actual} != {expected}")


def verify_appendix_values(results: dict[str, Any], transitions: dict[str, Any]) -> None:
    rows = results["rows"]
    if len(rows) != 36:
        raise AssertionError("Appendix C requires 36 displayed curve rows")
    if results["imageCount"] != 500 or results["taskCount"] != 16000:
        raise AssertionError("main-study image or task count differs")

    indexed = {
        (row["model"], row["degradation"], int(row["severity"])): row
        for row in rows
    }
    expected_map = {
        ("transformer", "blur", 0): 0.498,
        ("transformer", "blur", 5): 0.148,
        ("transformer", "lowlight", 5): 0.041,
        ("transformer", "jpeg", 5): 0.313,
        ("cnn", "blur", 0): 0.431,
        ("cnn", "blur", 5): 0.144,
        ("cnn", "lowlight", 5): 0.045,
        ("cnn", "jpeg", 5): 0.224,
    }
    for key, expected in expected_map.items():
        close(indexed[key]["map"], expected)

    if transitions["objectCount"] != 1978 or len(transitions["combinations"]) != 6:
        raise AssertionError("Appendix C object-transition design differs")


def main() -> None:
    verify_checksums()
    manifest = load_json(EVIDENCE / "manifest.json")
    results = load_json(EVIDENCE / "results.json")
    transitions = load_json(EVIDENCE / "analysis" / "object_failure_transitions.json")
    verify_conditions(manifest)
    verify_appendix_values(results, transitions)
    print("Research evidence verified: 32 conditions, 500 images, Appendix C values aligned.")


if __name__ == "__main__":
    main()
