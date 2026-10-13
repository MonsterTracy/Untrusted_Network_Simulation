"""Synthetic numeric checks plus opt-in regression against the supplied real export.

Set UNS_TOM_SINGLE_GAME_REVIEW to the extracted review root for real-data tests.
No fixture is inferred/recollected, and original inputs are never rewritten.
"""
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scripts import visualize_tom_single_game as viz


def uniform():
    values = np.full((7, 7), 1 / 6)
    np.fill_diagonal(values, 0)
    return values


@pytest.mark.parametrize("damage", ["shape", "nan", "negative", "above_one", "self", "sum", "string", "bool"])
def test_matrix_rejects_invalid_raw_values(damage):
    values = uniform().tolist()
    if damage == "shape":
        values.pop()
    else:
        values[0][1] = {"nan": float("nan"), "negative": -.1, "above_one": 1.1,
                        "sum": .8, "string": "0.1", "bool": True}.get(damage, 1 / 6)
        if damage == "self":
            values[0][0] = .01
    with pytest.raises(ValueError):
        viz.validate_matrix(values)


def test_matrix_readonly_and_export_tolerance():
    values = uniform().tolist()
    before = deepcopy(values)
    values[0][1] += 2e-6
    result = viz.validate_matrix(values)  # Export v1 permits a row error up to 1e-5.
    assert result[0, 1] == values[0][1] and not result.flags.writeable
    assert before[1:] == values[1:]


def test_tv_uses_all_targets_and_dead_observer_outputs_without_renormalizing():
    before = uniform()
    after = before.copy()
    after[0, 1] -= .1
    after[0, 2] += .1
    alive = np.ones((2, 7), dtype=bool)
    alive[1, 0] = False
    alive[1, 2] = False  # Changed/dead target is still part of the raw distribution.
    result = viz.tv_statistics(np.stack([before, after]), alive)[0]
    assert result["tv_by_observer"]["player1"] == pytest.approx(.1)
    assert result["all_observers_mean"] == pytest.approx(.1 / 7)
    assert result["common_alive_count"] == 5
    assert result["common_alive_mean"] == 0
    assert "player1" not in result["common_alive_observers"]
    assert after[0, 2] == pytest.approx(1 / 6 + .1)


def test_identical_tv_empty_common_panel_and_single_pre():
    q = np.stack([uniform(), uniform()])
    result = viz.tv_statistics(q, np.zeros((2, 7), dtype=bool))[0]
    assert result["all_observers_mean"] == 0
    assert result["common_alive_mean"] is None
    assert result["common_alive_observers"] == []
    assert viz.tv_statistics(q[:1], np.ones((1, 7), dtype=bool)) == []


def test_output_must_be_new_and_disjoint(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for output in (inputs / "plots", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            viz.check_output_path(inputs, output)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        viz.check_output_path(inputs, occupied)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        viz.check_output_path(inputs, linked)
    assert sentinel.read_text() == "unchanged"


@pytest.fixture
def real_root():
    name = os.environ.get("UNS_TOM_SINGLE_GAME_REVIEW")
    if not name:
        pytest.skip("real export not bundled into the repo; set UNS_TOM_SINGLE_GAME_REVIEW")
    return Path(name)


@pytest.fixture
def copied_root(real_root, tmp_path):
    result = tmp_path / "review"
    shutil.copytree(real_root, result)
    return result


def rehash_export(root):
    folder = root / viz.EXPORT_FOLDER
    manifest = json.loads((folder / "manifest.json").read_bytes())
    for name in manifest["file_table"]:
        payload = (folder / name).read_bytes()
        manifest["file_table"][name] = {"byte_size": len(payload), "sha256": viz.sha256_bytes(payload)}
    manifest["manifest_digest"] = viz.sha256_bytes(viz.canonical_json_bytes(
        {k: v for k, v in manifest.items() if k != "manifest_digest"}))
    (folder / "manifest.json").write_bytes(viz.canonical_json_bytes(manifest))


def test_real_schema_provenance_and_pre_alignment(real_root):
    data = viz.load_review(real_root)
    assert data.manifest["manifest_digest"] == "32e22a7825d5b2bb54f1127bbe6f37888020ae385906d84fba4609f44c5e119d"
    assert data.bundle_manifest["seed"] == 4464547787855486986
    assert data.summary["gameplay_source_revision"] == "57fd2d716eff8ac83477a17298e160c04b7e98b5"
    assert data.q.shape == (17, 7, 7) and data.q.size == 833
    assert len(data.events) == 56
    assert not data.q.flags.writeable and not data.alive.flags.writeable
    assert sum(~data.alive.reshape(-1)) == 22
    assert [r["day"] for r in data.records] == [1] * 9 + [2] * 5 + [3] * 3
    intervals = viz.event_intervals(data)
    used = [event for row in intervals for event in row["event_ids"]]
    assert len(used) == len(set(used)) == 52
    assert used == [e["event_id"] for e in data.events[:52]]
    assert len(viz.tv_statistics(data.q, data.alive)) == 16


@pytest.mark.parametrize("damage", [
    "seat_order", "prefix_digest", "speaker", "alive", "boundary_duplicate",
    "step_order", "coverage", "q_self", "q_nan", "fields",
])
def test_rehashed_export_still_rejects_semantic_corruption(copied_root, damage):
    path = copied_root / viz.EXPORT_FOLDER / "all_tom_matrices.jsonl"
    rows = viz.read_jsonl(path.read_bytes())
    if damage == "seat_order":
        rows[0]["observer_ids"].reverse()
    elif damage == "prefix_digest":
        rows[0]["prefix_digest"] = "0" * 64
    elif damage == "speaker":
        rows[0]["current_speaker"] = "player1"
    elif damage == "alive":
        rows[0]["alive_observer_ids"].pop()
    elif damage == "boundary_duplicate":
        rows[1]["boundary_id"] = rows[0]["boundary_id"]
    elif damage == "step_order":
        rows.reverse()
    elif damage == "coverage":
        rows.pop()
    elif damage == "q_self":
        rows[0]["q"][0][0] = .01
    elif damage == "q_nan":
        # Noncanonical JSON is rejected independently of updated file hashes.
        path.write_bytes(path.read_bytes().replace(b"0.0", b"NaN", 1))
    elif damage == "fields":
        rows[0]["unexpected"] = True
    if damage != "q_nan":
        path.write_bytes(viz.canonical_jsonl_bytes(rows))
    rehash_export(copied_root)
    with pytest.raises(ValueError):
        viz.load_review(copied_root)


@pytest.mark.parametrize("damage", ["value", "alive", "duplicate", "columns"])
def test_rehashed_csv_disagreement_rejected(copied_root, damage):
    path = copied_root / viz.EXPORT_FOLDER / "tom_cells.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if damage == "value":
        rows[1]["value"] = "0.5"
    elif damage == "alive":
        rows[0]["observer_alive"] = "False"
    elif damage == "duplicate":
        rows[1] = rows[0].copy()
    else:
        fields = list(reversed(fields))
    path.write_bytes(viz.csv_bytes(rows, fields))
    rehash_export(copied_root)
    with pytest.raises(ValueError, match="CSV"):
        viz.load_review(copied_root)


def test_public_snapshot_hash_and_lineage_reject_tampering(copied_root):
    path = copied_root / "canonical_public/public_event_stream.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest"):
        viz.load_review(copied_root)


def test_rehashed_public_manifest_does_not_break_source_binding(copied_root):
    path = copied_root / "canonical_public/manifest.json"
    value = json.loads(path.read_bytes())
    value["source_revision"] = "0" * 40
    value["manifest_digest"] = viz.sha256_bytes(viz.canonical_json_bytes(
        {k: v for k, v in value.items() if k != "manifest_digest"}))
    path.write_bytes(viz.canonical_json_bytes(value))
    with pytest.raises(ValueError, match="source mismatch"):
        viz.load_review(copied_root)


def test_rehashed_model_pin_mismatch_rejected(copied_root):
    path = copied_root / viz.EXPORT_FOLDER / "summary.json"
    value = json.loads(path.read_bytes())
    value["fit_digest"] = "0" * 64
    path.write_bytes(viz.canonical_json_bytes(value))
    rehash_export(copied_root)
    with pytest.raises(ValueError, match="fit_digest"):
        viz.load_review(copied_root)


def test_real_render_all_formats_preserves_inputs_and_is_reproducible(real_root, tmp_path, monkeypatch):
    runtime_modules_before = {name for name in sys.modules
                              if name == "torch" or name.startswith("torch.") or name == "run_random"}
    from scripts import qwen3_gameplay_predictor
    monkeypatch.setattr(qwen3_gameplay_predictor, "Qwen3GameplayPredictorClient",
                        lambda **_: pytest.fail("no predictor construction allowed"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl-cache"))
    before = viz.input_hashes(real_root)
    data = viz.load_review(real_root)
    original = data.q.tobytes()
    outputs = []
    for name in ("a", "b"):
        result = viz.generate(data, tmp_path / name, dpi=72, columns=5, font_size=8,
                              comparison_pairs=((9, 10), (14, 15)))
        output = Path(result["output_directory"])
        outputs.append(output)
        assert len(list(output.glob("*.png"))) == 7
        assert len(list(output.glob("*.pdf"))) == 7
        assert len(list(output.glob("*.svg"))) == 7
        assert all(b"<image" not in p.read_bytes() for p in output.glob("*.svg"))
        assert all(b"/Subtype /Image" not in p.read_bytes() for p in output.glob("*.pdf"))
        viz.verify_artifact(output, expected_artifact_type="tom_single_game_visualization",
                            expected_schema_version=viz.ANALYSIS_VERSION)
        report = json.loads((output / "analysis.json").read_bytes())
        assert report["model_provenance"]["gameplay_source_revision"] == "57fd2d716eff8ac83477a17298e160c04b7e98b5"
        assert report["validation"]["full_canonical_bundle_revalidated"] is False
        assert report["validation"]["router_panel_available"] is False
        assert report["parameters"]["figure_width_inches"] == 7
        assert report["parameters"]["comparison_pairs"] == [[9, 10], [14, 15]]
    for path in outputs[0].iterdir():
        assert path.read_bytes() == (outputs[1] / path.name).read_bytes(), path.name
    assert data.q.tobytes() == original
    assert viz.input_hashes(real_root) == before
    with pytest.raises(FileExistsError):
        viz.generate(data, outputs[0], dpi=72)
    runtime_modules_after = {name for name in sys.modules
                             if name == "torch" or name.startswith("torch.") or name == "run_random"}
    assert runtime_modules_after == runtime_modules_before


def test_cli_help_does_not_import_inference_runtime():
    code = (
        "import sys; from scripts.visualize_tom_single_game import main; "
        "assert 'torch' not in sys.modules and 'run_random' not in sys.modules; "
        "main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0 and "--input-dir" in result.stdout


@pytest.fixture
def paper_figures(real_root):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    data = viz.load_review(real_root)
    original = data.q.tobytes()
    stats = viz.tv_statistics(data.q, data.alive)
    pairs = ((9, 10), (14, 15))
    with matplotlib.rc_context({"font.size": 9, "font.family": "DejaVu Sans"}):
        figures = {
            "comparison": viz._comparisons(data, pairs, plt, 7),
            "sequence": viz._sequence(data, 5, plt, 7),
            "tv": viz._variation(data, stats, plt, 7),
            "events": viz._event_alignment(data, stats, plt, 7),
            "key_events": viz._key_events(data, stats, pairs, plt, 7),
            "trajectories": viz._trajectories(data, plt, 7),
        }
        yield data, stats, figures
        assert data.q.tobytes() == original
        for figure in figures.values():
            plt.close(figure)


def test_comparison_meshes_preserve_exact_raw_values_and_shared_scale(paper_figures):
    data, _, figures = paper_figures
    from matplotlib.collections import QuadMesh
    for ax, ordinal in zip(figures["comparison"].axes[:4], (9, 10, 14, 15)):
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        np.testing.assert_array_equal(mesh.get_array(), data.q[ordinal - 1])
        assert mesh.get_clim() == (0, 1)
        assert [t.get_text().rstrip("*") for t in ax.get_xticklabels()] == [f"P{i}" for i in range(1, 8)]
    figures["comparison"].canvas.draw()
    bar = next(c for c in figures["comparison"].axes[-1].collections if isinstance(c, QuadMesh))
    np.testing.assert_array_equal(bar.get_edgecolors(), bar.get_facecolors())
    assert not bar.get_rasterized()


def test_sequence_keeps_all_pres_and_explicit_day_headers(paper_figures):
    _, _, figures = paper_figures
    figure = figures["sequence"]
    headers = [t.get_text() for ax in figure.axes for t in ax.texts if t.get_text().startswith("Day")]
    assert headers == ["Day 1 · 9 PREs", "Day 2 · 5 PREs", "Day 3 · 3 PREs"]
    titles = [ax.get_title() for ax in figure.axes if ax.get_title().startswith("PRE")]
    assert [int(t.split()[1]) for t in titles] == list(range(1, 18))


def test_tv_intervals_panels_and_cross_day_marks(paper_figures):
    _, stats, figures = paper_figures
    top, panel, mean = figures["tv"].axes[:3]
    assert [t.get_text() for t in mean.get_xticklabels()] == [f"{r['from_pre']}→{r['to_pre']}" for r in stats]
    assert [t.get_text() for t in panel.texts] == ["7"] * 8 + ["5"] * 5 + ["3"] * 3
    assert [t.get_text() for t in mean.texts] == ["Day 1→2", "Day 2→3"]
    assert mean.get_xticklabels()[8].get_fontweight() == "bold"
    assert mean.get_xticklabels()[13].get_fontweight() == "bold"
    np.testing.assert_array_equal(mean.lines[0].get_ydata(), [r["all_observers_mean"] for r in stats])
    np.testing.assert_array_equal(mean.lines[1].get_ydata(), [r["common_alive_mean"] for r in stats])


def test_public_timeline_marks_empty_results_tail_and_interval_extent(paper_figures):
    _, stats, figures = paper_figures
    top, mean = figures["events"].axes
    assert [t.xy[0] for t in top.texts if t.get_text() == "none"] == [19, 1]
    np.testing.assert_array_equal(mean.lines[0].get_xdata(), [3, 5])
    np.testing.assert_array_equal(mean.lines[0].get_ydata(), [stats[0]["all_observers_mean"]] * 2)
    assert mean.patches[0].get_x() == 51.5
    assert any("events 52–55" in t.get_text() for t in figures["events"].texts)
    assert "new events 24–31" in figures["key_events"].axes[0].get_title()
    assert "new events 40–47" in figures["key_events"].axes[1].get_title()


def test_target_death_hollow_points_and_observer_death_dashes_keep_raw_mass(paper_figures):
    data, _, figures = paper_figures
    ax = figures["trajectories"].axes[4]  # Observer P5, target P1 dies before PRE10.
    hollow = ax.collections[1]
    np.testing.assert_array_equal(hollow.get_offsets()[:, 0], np.arange(10, 18))
    np.testing.assert_array_equal(hollow.get_offsets()[:, 1], data.q[9:, 4, 0])
    assert len(hollow.get_facecolors()) == 0
    dead_observer = figures["trajectories"].axes[0]
    assert dead_observer.lines[1].get_linestyle() == "--"
    np.testing.assert_array_equal(dead_observer.lines[1].get_ydata()[8:], data.q[8:, 0, 1])


def test_paper_width_and_minimum_native_font_size(paper_figures):
    from matplotlib.text import Text
    for name, figure in paper_figures[2].items():
        assert figure.get_size_inches()[0] == 7, name
        figure.canvas.draw()
        assert all(t.get_fontsize() >= 7.5 for t in figure.findobj(Text) if t.get_visible() and t.get_text()), name


def test_paper_text_stays_inside_canvas_and_headers_do_not_overlap(paper_figures):
    for name, figure in paper_figures[2].items():
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        text = list(figure.texts)
        headers = list(figure.texts)
        for ax in figure.axes:
            text.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
            headers.extend([ax.title, *(t for t in ax.texts if t.get_text().startswith("Day"))])
            if ax.axison:
                for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                    ticks = [tick.label1 for tick in axis.get_major_ticks()
                             if min(limits) <= tick.get_loc() <= max(limits)]
                    text.extend(ticks)
                    if name == "sequence":
                        headers.extend(ticks)
            if ax.get_legend():
                text.extend(ax.get_legend().get_texts())
        boxes = [t.get_window_extent(renderer) for t in text if t.get_visible() and t.get_text()]
        assert all(b.x0 >= -1 and b.y0 >= -1 and b.x1 <= figure.bbox.x1 + 1
                   and b.y1 <= figure.bbox.y1 + 1 for b in boxes), name
        boxes = [t.get_window_extent(renderer) for t in headers if t.get_visible() and t.get_text()]
        assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), name


@pytest.mark.parametrize("pairs", [((9, 11), (14, 15)), ((9, 10),), ((9, 10), (9, 10)), ((0, 1), (14, 15))])
def test_comparison_pairs_reject_nonadjacent_duplicate_or_out_of_range(real_root, tmp_path, pairs):
    with pytest.raises(ValueError, match="comparison pairs"):
        viz.generate(viz.load_review(real_root), tmp_path / "not-published", comparison_pairs=pairs)
    assert not (tmp_path / "not-published").exists()
