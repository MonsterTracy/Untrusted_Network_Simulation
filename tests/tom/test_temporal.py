import math

import numpy as np
import pytest
import torch


def test_temporal_known_bytes_energy_and_fail_closed(tmp_path):
    from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

    publish_temporal(tmp_path / "temporal", 3, "a" * 64)
    assert {p.name for p in (tmp_path / "temporal").iterdir()} == {
        "phase_codebook.manifest.json", "phase_codebook.bin",
        "canonical_day_code_table.manifest.json", "canonical_day_code_table.bin"}
    implicit = TemporalCodeProvider.implicit(tmp_path / "temporal", "a" * 64)
    explicit = TemporalCodeProvider.explicit(tmp_path / "temporal", "a" * 64)
    days = torch.tensor([[0, 3]])
    phases = torch.tensor([[0, 4]])
    code = explicit.code(days, phases)
    assert code.shape == (1, 2, 256)
    assert torch.allclose(code.square().mean(-1).sqrt(), torch.full((1, 2), .02))
    assert code[0, 0, 0] == 0
    assert code[0, 0, 1].item() == np.float32(.02 * math.sqrt(2))
    assert torch.equal(implicit.code(days, phases), torch.zeros_like(code))
    assert np.allclose(explicit.phase @ explicit.phase.T, np.eye(5) * .02**2 * 128, atol=1e-8)
    for d, p in [(4, 0), (-1, 0), (0, 5)]:
        with pytest.raises(ValueError):
            explicit.code(torch.tensor([[d]]), torch.tensor([[p]]))
    with pytest.raises(ValueError):
        TemporalCodeProvider.explicit(tmp_path / "temporal", "b" * 64)
    (tmp_path / "temporal" / "phase_codebook.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        TemporalCodeProvider.explicit(tmp_path / "temporal", "a" * 64)


@pytest.mark.parametrize("violation", ["version", "learned_field", "phase_order", "day_metadata"])
def test_temporal_rejects_rehashed_noncanonical_semantic_metadata(tmp_path, violation):
    import json
    from werewolf.artifact_io import canonical_json_bytes, publish_artifact
    from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider
    artifact = publish_temporal(tmp_path / "source", 0, "a" * 64)
    fields = {k: v for k, v in artifact.manifest.items() if k not in {"manifest_digest", "file_table"}}
    files = {p: (artifact.path / p).read_bytes() for p in artifact.manifest["file_table"]}
    if violation == "version":
        fields["schema_version"] = "other"
    elif violation == "learned_field":
        fields["learned_scale"] = True
    elif violation == "phase_order":
        fields["phase_order"].reverse()
    else:
        name = "canonical_day_code_table.manifest.json"
        metadata = json.loads(files[name])
        metadata["parent_digest"] = "b" * 64
        files[name] = canonical_json_bytes(metadata)
    publish_artifact(tmp_path / "bad", manifest_name="phase_codebook.manifest.json", manifest_fields=fields, files=files)
    with pytest.raises(ValueError):
        TemporalCodeProvider.explicit(tmp_path / "bad", "a" * 64)
