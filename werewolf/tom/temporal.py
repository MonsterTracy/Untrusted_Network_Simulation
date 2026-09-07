"""Versioned non-trainable temporal bytes; inference never recomputes trigonometry."""

import math
import platform
from pathlib import Path

import numpy as np
import torch

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes, verify_artifact
from werewolf.canonical_collection.public_history import PUBLIC_PHASES

TEMPORAL_VERSION = "classic7_temporal_code_v1"
PHASE_DIGEST = "27e974cdecc7927b04ccd2f5f2e4d13351a2edf382c88f12854633dd0be77490"
PHASE_FORMULA = "0.02*(-1)^popcount(r&c);r=1..5;c=0..127"
DAY_FORMULA = "0.02*sqrt(2)*sin_cos(day/10000^(2k/128));k=0..63"


def publish_temporal(destination, max_observed_day, parent_digest):
    if type(max_observed_day) is not int or max_observed_day < 0:
        raise ValueError("maximum day must be nonnegative")
    phase = np.array([[.02 * (-1)**((r & c).bit_count()) for c in range(128)] for r in range(1, 6)], dtype="<f4")
    day = np.array([[.02 * math.sqrt(2) * f(d / 10000**(2*k/128))
                     for k in range(64) for f in (math.sin, math.cos)]
                    for d in range(max_observed_day + 1)], dtype="<f4")
    if sha256_bytes(phase.tobytes()) != PHASE_DIGEST:
        raise ValueError("canonical phase byte construction mismatch")
    day_metadata = {"artifact_type": "canonical_day_code_table", "schema_version": TEMPORAL_VERSION,
        "parent_digest": parent_digest, "temporal_code_version": TEMPORAL_VERSION,
        "formula": DAY_FORMULA, "amplitude": .02, "shape": [max_observed_day + 1, 128],
        "max_observed_day": max_observed_day, "dtype": "<f4", "layout": "C",
        "sha256": sha256_bytes(day.tobytes()),
        "generation_provenance": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}}
    return publish_artifact(destination, manifest_name="phase_codebook.manifest.json", manifest_fields={
        "artifact_type": "deterministic_temporal_code", "schema_version": TEMPORAL_VERSION,
        "parent_digest": parent_digest, "temporal_code_version": TEMPORAL_VERSION,
        "phase_order": list(PUBLIC_PHASES), "row_indices": [1, 2, 3, 4, 5],
        "phase_formula": PHASE_FORMULA, "day_formula": DAY_FORMULA,
        "amplitude": .02, "subspace_dimension": 128, "dtype": "<f4", "layout": "C",
        "max_observed_day": max_observed_day, "phase_shape": [5, 128],
        "day_shape": [max_observed_day + 1, 128], "phase_digest": PHASE_DIGEST,
        "day_digest": sha256_bytes(day.tobytes()),
        "generation_provenance": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }, files={"phase_codebook.bin": phase.tobytes(), "canonical_day_code_table.bin": day.tobytes(),
              "canonical_day_code_table.manifest.json": canonical_json_bytes(day_metadata)})


class TemporalCodeProvider:
    @classmethod
    def implicit(cls, path, parent_digest):
        return cls(path, parent_digest, False)

    @classmethod
    def explicit(cls, path, parent_digest):
        return cls(path, parent_digest, True)

    def __init__(self, path, parent_digest, explicit):
        artifact = verify_artifact(path, expected_artifact_type="deterministic_temporal_code",
            expected_schema_version=TEMPORAL_VERSION, manifest_name="phase_codebook.manifest.json")
        m = artifact.manifest
        expected = {"parent_digest": parent_digest, "temporal_code_version": TEMPORAL_VERSION,
                    "phase_order": list(PUBLIC_PHASES), "row_indices": [1, 2, 3, 4, 5],
                    "phase_formula": PHASE_FORMULA, "day_formula": DAY_FORMULA,
                    "amplitude": .02, "subspace_dimension": 128, "dtype": "<f4", "layout": "C",
                    "phase_shape": [5, 128], "phase_digest": PHASE_DIGEST}
        fields = set(expected) | {"artifact_type", "schema_version", "max_observed_day", "day_shape", "day_digest", "generation_provenance", "file_table", "manifest_digest"}
        if set(m) != fields or any(m.get(k) != v for k, v in expected.items()):
            raise ValueError("temporal metadata/parent mismatch")
        maximum = m["max_observed_day"]
        if type(maximum) is not int or maximum < 0 or m["day_shape"] != [maximum + 1, 128]:
            raise ValueError("temporal day shape mismatch")
        if set(m["file_table"]) != {"phase_codebook.bin", "canonical_day_code_table.bin", "canonical_day_code_table.manifest.json"}:
            raise ValueError("temporal file set mismatch")
        phase_bytes = (Path(path) / "phase_codebook.bin").read_bytes()
        day_bytes = (Path(path) / "canonical_day_code_table.bin").read_bytes()
        day_meta_bytes = (Path(path) / "canonical_day_code_table.manifest.json").read_bytes()
        expected_day_metadata = {"artifact_type": "canonical_day_code_table", "schema_version": TEMPORAL_VERSION,
            "parent_digest": parent_digest, "temporal_code_version": TEMPORAL_VERSION, "formula": DAY_FORMULA,
            "amplitude": .02, "shape": [maximum + 1, 128], "max_observed_day": maximum,
            "dtype": "<f4", "layout": "C", "sha256": m["day_digest"],
            "generation_provenance": m["generation_provenance"]}
        if day_meta_bytes != canonical_json_bytes(expected_day_metadata):
            raise ValueError("canonical day table metadata mismatch")
        if sha256_bytes(phase_bytes) != PHASE_DIGEST or sha256_bytes(day_bytes) != m["day_digest"]:
            raise ValueError("temporal byte digest mismatch")
        self.phase = np.frombuffer(phase_bytes, dtype="<f4").reshape(5, 128)
        self.day = np.frombuffer(day_bytes, dtype="<f4").reshape(maximum + 1, 128)
        if not np.isfinite(self.day).all():
            raise ValueError("day table must be finite")
        self.explicit_enabled = explicit
        self.artifact_digest = artifact.manifest_digest

    def code(self, days, phases):
        if days.dtype != torch.long or phases.dtype != torch.long or days.shape != phases.shape:
            raise ValueError("day/phase tensors must have matching integer shapes")
        if torch.any(days < 0) or torch.any(days >= len(self.day)) or torch.any(phases < 0) or torch.any(phases >= 5):
            raise ValueError("unknown or out-of-range public temporal state")
        if not self.explicit_enabled:
            return torch.zeros(*days.shape, 256, device=days.device)
        day = torch.tensor(self.day.copy(), device=days.device)[days]
        phase = torch.tensor(self.phase.copy(), device=phases.device)[phases]
        return torch.cat((day, phase), -1)
