"""Read-only analysis of tom_one_game_all_pre_export_v1; no inference/gameplay.

Run with python -m scripts.visualize_tom_single_game --help.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from importlib.metadata import version
import io
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from scripts.qwen3_gameplay_predictor import FIT_DIGEST, SEAL_DIGEST, SOURCE_REVISION
from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes,
    verify_artifact,
)
from werewolf.artifact_io.canonical import _load_canonical_json
from werewolf.canonical_collection.game_bundle import _history_from_records, _prefix_from_record
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS

EXPORT_VERSION = "tom_one_game_all_pre_export_v1"
ANALYSIS_VERSION = "tom_single_game_visualization_v1"
ROW_SUM_ATOL = 1e-5  # Exact convention in the supplied export script.
EXPORT_FOLDER = "tom-single-game-matrices-demo-v1"
CSV_COLUMNS = (
    "game_id", "boundary_id", "step_index", "day", "phase", "current_speaker",
    "observer", "target", "observer_alive", "target_alive", "value",
)
METADATA_FIELDS = (
    "game_id", "boundary_id", "prefix_digest", "step_index", "day", "phase",
    "current_speaker", "alive_observer_ids",
)
EVENT_TYPES = (
    "phase_change", "turn_start", "public_speech", "vote_result",
    "exile_result", "death_announcement",
)
NOTICE = (
    "Predicted observer-conditioned suspicion mass; not wolf-role probability. "
    "All seven seats retained. Public alive status does not establish non-wolf "
    "or Router eligibility. Temporal correspondence is descriptive, not causal."
)
PHASE_LABELS = {"night": "N", "discussion": "S", "pk_discussion": "PK",
                "vote": "V", "pk_vote": "PV"}
TARGET_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#6B4C3B", "#444444")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(payload):
    rows = [_load_canonical_json(line) for line in payload.splitlines()]
    require(bool(rows) and canonical_jsonl_bytes(rows) == payload, "noncanonical/empty JSONL")
    return rows


def validate_matrix(values):
    require(isinstance(values, list) and len(values) == 7
            and all(isinstance(row, list) and len(row) == 7 for row in values), "Q shape must be 7x7")
    require(all(type(v) is float and math.isfinite(v) and 0 <= v <= 1
                for row in values for v in row), "Q cells must be finite float probabilities")
    require(all(values[i][i] == 0.0 for i in range(7)), "Q self target must be exactly zero")
    require(all(abs(math.fsum(row) - 1) <= ROW_SUM_ATOL for row in values), "Q row sum mismatch")
    result = np.array(values, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Review:
    root: Path
    manifest: dict
    summary: dict
    bundle_manifest: dict
    records: tuple
    prefixes: tuple
    events: tuple
    q: np.ndarray
    alive: np.ndarray
    input_files: dict


def _review_files(root, export_dir=None, canonical_dir=None):
    export = Path(export_dir).resolve(strict=True) if export_dir is not None else root / EXPORT_FOLDER
    canonical = Path(canonical_dir).resolve(strict=True) if canonical_dir is not None else root / "canonical_public"
    files = {f"export/{name}": export / name for name in (
        "manifest.json", "summary.json", "all_tom_matrices.jsonl", "tom_cells.csv")}
    files["canonical/manifest.json"] = canonical / "manifest.json"
    for name in ("authoritative_pre_prefixes.jsonl", "public_event_stream.jsonl"):
        candidates = [canonical / "public" / name, canonical / name]
        present = [p for p in candidates if os.path.lexists(p)]
        require(len(present) == 1, f"missing/ambiguous canonical public layout: {name}")
        files[f"canonical/{name}"] = present[0]
    if (root / "export_all_tom_from_canonical.py").is_file():
        files["export_script"] = root / "export_all_tom_from_canonical.py"
    return files


def input_hashes(root, *, export_dir=None, canonical_dir=None):
    result = {}
    for path in _review_files(root, export_dir, canonical_dir).values():
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        require(not path.is_symlink() and path.is_file(), f"input must be a regular file: {name}")
        payload = path.read_bytes()
        result[name] = {"byte_size": len(payload), "sha256": sha256_bytes(payload)}
    return result


def load_review(input_dir, *, export_dir=None, canonical_dir=None):
    root = Path(input_dir).expanduser().resolve(strict=True)
    files = _review_files(root, export_dir, canonical_dir)
    hashes = input_hashes(root, export_dir=export_dir, canonical_dir=canonical_dir)
    artifact = verify_artifact(files["export/manifest.json"].parent,
        expected_artifact_type="tom_one_game_all_pre_matrices", expected_schema_version=EXPORT_VERSION)
    require(set(artifact.manifest["file_table"]) == {
        "summary.json", "all_tom_matrices.jsonl", "tom_cells.csv"}, "export inventory mismatch")
    summary = _load_canonical_json(files["export/summary.json"].read_bytes())
    records = read_jsonl(files["export/all_tom_matrices.jsonl"].read_bytes())
    bundle = _load_canonical_json(files["canonical/manifest.json"].read_bytes())
    require(bundle["artifact_type"] == "canonical_game_bundle"
            and bundle["schema_version"] == "classic7-canonical-game-bundle-v1", "canonical manifest schema")
    require(bundle["manifest_digest"] == sha256_bytes(canonical_json_bytes(
        {k: v for k, v in bundle.items() if k != "manifest_digest"})), "canonical manifest digest")
    require(bundle["canonical_eligibility"] is True, "source manifest lacks canonical eligibility")
    public_payloads = {}
    for name in ("authoritative_pre_prefixes.jsonl", "public_event_stream.jsonl"):
        payload = files[f"canonical/{name}"].read_bytes()
        entry = bundle["file_table"][f"public/{name}"]
        require(entry == {"byte_size": len(payload), "sha256": sha256_bytes(payload)},
                f"canonical public file digest mismatch: {name}")
        public_payloads[name] = payload
    raw_prefixes = read_jsonl(public_payloads["authoritative_pre_prefixes.jsonl"])
    prefixes = tuple(validate_authoritative_pre_prefix(_prefix_from_record(p)) for p in raw_prefixes)
    events = tuple(read_jsonl(public_payloads["public_event_stream.jsonl"]))
    require(_history_from_records(events).to_records() == list(events), "public event round-trip")
    n = len(records)
    require(len(prefixes) == n and n > 0, "matrix/PRE coverage mismatch")
    require(summary["schema_version"] == EXPORT_VERSION
            and summary["scope"] == "all_authoritative_public_speech_pre_boundaries", "summary scope/schema")
    for name, expected in (("num_pre", n), ("num_matrices", n), ("num_cells", n * 49)):
        require(type(summary[name]) is int and summary[name] == expected, f"summary {name} mismatch")
    require(summary["shape"] == [7, 7] and artifact.manifest["num_pre"] == n, "manifest shape/count")
    for name, expected in (("fit_digest", FIT_DIGEST), ("seal_digest", SEAL_DIGEST)):
        require(summary[name] == artifact.manifest[name] == expected, f"frozen {name} mismatch")
    require(summary["predictor_source_revision"] == SOURCE_REVISION, "frozen predictor source mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", bundle["source_revision"]) is not None
            and summary["gameplay_source_revision"] == bundle["source_revision"], "gameplay source mismatch")
    require(summary["source_bundle_manifest_digest"] ==
            artifact.manifest["source_bundle_manifest_digest"] == bundle["manifest_digest"],
            "export/canonical manifest lineage mismatch")
    require(summary["game_id"] == artifact.manifest["game_id"] == bundle["game_id"], "game identity mismatch")
    require(type(bundle["seed"]) is int and bundle["seed"] >= 0
            and bundle["game_id"].endswith(f"-seed-{bundle['seed']}"), "canonical seed/game identity")
    steps = [r["step_index"] for r in records]
    require(steps == sorted(set(steps)), "PRE step order must be strictly increasing")
    require(len({r["boundary_id"] for r in records}) == n, "duplicate boundary")
    matrices = []
    for record, prefix in zip(records, prefixes):
        require(set(record) == {*METADATA_FIELDS, "observer_ids", "target_ids", "q"}, "matrix record fields")
        expected = prefix.to_record()
        require(canonical_json_bytes({k: record[k] for k in METADATA_FIELDS}) ==
                canonical_json_bytes({k: expected[k] for k in METADATA_FIELDS}), "matrix/PRE identity mismatch")
        require(prefix.game_id == summary["game_id"], "PRE game identity")
        require(record["observer_ids"] == record["target_ids"] == list(PLAYER_IDS), "seat mapping mismatch")
        require(record["phase"] in ("discussion", "pk_discussion"), "not a public speech PRE")
        history = prefix.public_event_history.to_records()
        require(history == list(events[:len(history)]), "PRE is not the exact public event prefix")
        alive = set(PLAYER_IDS)
        for event in history:
            alive.difference_update(event.get("dead_players", ()))
            alive.difference_update(event.get("exiled_players", ()))
        require(tuple(p for p in PLAYER_IDS if p in alive) == prefix.alive_observer_ids, "public alive mismatch")
        matrices.append(validate_matrix(record["q"]))
    endpoints = [p.public_event_history.events[-1].event_index for p in prefixes]
    require(endpoints == sorted(set(endpoints)), "PRE public endpoints must strictly increase")
    with files["export/tom_cells.csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(CSV_COLUMNS), "CSV columns mismatch")
        cells = list(reader)
    require(len(cells) == n * 49, "CSV cell count mismatch")
    index = 0
    for record in records:
        for i, observer in enumerate(PLAYER_IDS):
            for j, target in enumerate(PLAYER_IDS):
                cell = cells[index]
                expected = {name: str(record[name]) for name in CSV_COLUMNS[:6]}
                expected.update(observer=observer, target=target,
                    observer_alive=str(observer in record["alive_observer_ids"]),
                    target_alive=str(target in record["alive_observer_ids"]))
                require(set(cell) == set(CSV_COLUMNS)
                        and all(cell[name] == value for name, value in expected.items())
                        and float(cell["value"]) == record["q"][i][j], "CSV/JSONL consistency mismatch")
                index += 1
    q = np.stack(matrices)
    alive = np.array([[p in r["alive_observer_ids"] for p in PLAYER_IDS] for r in records], dtype=bool)
    q.setflags(write=False)
    alive.setflags(write=False)
    require(input_hashes(root, export_dir=export_dir, canonical_dir=canonical_dir) == hashes, "inputs changed during validation")
    return Review(root, artifact.manifest, summary, bundle, tuple(records), prefixes, events, q, alive, hashes)


def tv_statistics(q, alive):
    q, alive = np.asarray(q), np.asarray(alive)
    require(q.ndim == 3 and q.shape[1:] == (7, 7) and alive.shape == q.shape[:2]
            and alive.dtype == bool, "TV dimensions/mask")
    distances = .5 * np.abs(np.diff(q, axis=0)).sum(axis=2)
    result = []
    for index, values in enumerate(distances):
        common = alive[index] & alive[index + 1]
        result.append({
            "from_pre": index + 1, "to_pre": index + 2,
            "tv_by_observer": dict(zip(PLAYER_IDS, map(float, values))),
            "all_observers_mean": float(values.mean()),
            "common_alive_observers": [p for p, included in zip(PLAYER_IDS, common) if included],
            "common_alive_count": int(common.sum()),
            "common_alive_mean": float(values[common].mean()) if common.any() else None,
        })
    return result


def event_intervals(review):
    result, previous_end = [], -1
    for index, prefix in enumerate(review.prefixes):
        end = prefix.public_event_history.events[-1].event_index
        result.append({"to_pre": index + 1, "boundary_id": prefix.boundary_id,
            "start_event_index": previous_end + 1, "end_event_index": end,
            "event_ids": [e["event_id"] for e in review.events[previous_end + 1:end + 1]],
            "event_type_counts": dict(Counter(e["event_type"] for e in review.events[previous_end + 1:end + 1]))})
        previous_end = end
    return result


def check_output_path(input_dir, output_dir):
    root = Path(input_dir).expanduser().resolve(strict=True)
    raw = Path(output_dir).expanduser().absolute()
    require(not raw.is_symlink(), "output must not be a symlink")
    destination = raw.resolve()
    require(not destination.is_relative_to(root) and not root.is_relative_to(destination),
            "output must be outside and disjoint from the input tree")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    return destination


def analysis_source():
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    return {"analysis_head": git("rev-parse", "HEAD"), "analysis_branch": git("branch", "--show-current"),
        "analysis_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "analysis_script_tracked": bool(git("ls-files", "--", "scripts/visualize_tom_single_game.py")),
        "tracked_changes_present": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _title(record, index):
    return (f"PRE {index} | Day {record['day']} | {record['phase']}\n"
            f"Speaker {record['current_speaker']} | step {record['step_index']}")


def _matrix(ax, values, alive, speaker, *, annotate=False, compact=False):
    from matplotlib.patches import Rectangle
    edges = np.arange(8) - .5
    image = ax.pcolormesh(edges, edges, values, vmin=0, vmax=1, cmap="viridis",
                         shading="flat", rasterized=False)
    ax.set(xlim=(-.5, 6.5), ylim=(6.5, -.5), aspect="equal")
    labels = [f"P{i + 1}{'*' if not living and not compact else ''}" for i, living in enumerate(alive)]
    ax.set_xticks(range(7), labels)
    ax.set_yticks(range(7), labels)
    for i in range(7):
        if not alive[i]:
            for xy, w, h in (((-.5, i - .5), 7, 1), ((i - .5, -.5), 1, 7)):
                ax.add_patch(Rectangle(xy, w, h, fill=False, hatch="//", edgecolor="0.7", linewidth=0))
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, facecolor="#e8e8e8", edgecolor="white", linewidth=.3))
        ax.text(i, i, "×", ha="center", va="center", fontsize=7.5, color="#444444")
        if annotate:
            for j in range(7):
                if i != j:
                    ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center",
                            fontsize=7.5, color="white" if values[i,j] < .5 else "black")
    ax.get_yticklabels()[PLAYER_IDS.index(speaker)].set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(labelsize=7.5, pad=2)
    return image


def _colorbar(figure, image, **kwargs):
    colorbar = figure.colorbar(image, **kwargs)
    colorbar.solids.set_rasterized(False)  # Matplotlib rasterizes continuous colorbars by default.
    colorbar.solids.set_edgecolor("face")  # Cover vector-cell seams without rasterizing.
    return colorbar


def _heatmap(review, index, plt, width):
    figure, ax = plt.subplots(figsize=(width, width * .92), layout="constrained")
    image = _matrix(ax, review.q[index], review.alive[index], review.records[index]["current_speaker"], annotate=True)
    ax.set(xlabel="Suspected target", ylabel="Observer", title=_title(review.records[index], index + 1))
    _colorbar(figure, image, ax=ax, label="Predicted suspicion mass")
    figure.supxlabel("× self=0 | * / hatch: dead seat | bold observer: current speaker", fontsize=8)
    return figure


def _comparisons(review, pairs, plt, width):
    figure, axes = plt.subplots(2, 2, figsize=(width, width * .95), layout="constrained")
    for row, pair in enumerate(pairs):
        for col, ordinal in enumerate(pair):
            index, ax = ordinal - 1, axes[row, col]
            image = _matrix(ax, review.q[index], review.alive[index],
                            review.records[index]["current_speaker"], annotate=True)
            r = review.records[index]
            ax.set_title(f"PRE {ordinal} · Day {r['day']} · {PHASE_LABELS[r['phase']]}\n"
                         f"Speaker {r['current_speaker'].replace('player', 'P')}", fontsize=9)
    _colorbar(figure, image, ax=axes, label="Predicted suspicion mass", shrink=.8, pad=.02)
    figure.suptitle("Before / after adjacent PRE intervals", fontsize=10)
    figure.supxlabel("Target seat | × self=0; */hatch: dead; bold row: speaker\n"
                       "Rows: observers. S=speech, PK=PK speech. Not wolf-role probability.", fontsize=8)
    return figure


def _sequence(review, columns, plt, width):
    days = list(dict.fromkeys(r["day"] for r in review.records))
    groups = [[i for i, r in enumerate(review.records) if r["day"] == day] for day in days]
    heights = [math.ceil(len(group) / columns) for group in groups]
    figure = plt.figure(figsize=(width, 1.35 * sum(heights) + .65 * len(days) + 1.1))
    ratios = [value for height in heights for value in (.65, height * 1.35)]
    outer = figure.add_gridspec(2 * len(days), 1, height_ratios=ratios,
                               left=.055, right=.985, top=.94, bottom=.165, hspace=.1)
    image = None
    for day_index, (day, group, height) in enumerate(zip(days, groups, heights)):
        header = figure.add_subplot(outer[day_index * 2])
        header.axis("off")
        header.text(0, .5, f"Day {day} · {len(group)} PREs", va="center", fontweight="bold", fontsize=9)
        grid = outer[day_index * 2 + 1].subgridspec(height, columns, hspace=.45, wspace=.23)
        for position, index in enumerate(group):
            ax = figure.add_subplot(grid[position // columns, position % columns])
            r = review.records[index]
            image = _matrix(ax, review.q[index], review.alive[index], r["current_speaker"], compact=True)
            ax.set_title(f"PRE {index + 1} · {r['current_speaker'].replace('player', 'P')}"
                         + (" · PK" if r["phase"] == "pk_discussion" else ""), fontsize=8)
    cax = figure.add_axes((.30, .095, .42, .014))
    _colorbar(figure, image, cax=cax, label="Predicted suspicion mass", orientation="horizontal")
    figure.suptitle("Complete ToM sequence · canonical PRE order", fontsize=10, y=.995)
    figure.text(.5, .009, "Rows: observers; columns: targets P1–P7. × self=0; hatch: dead; bold row: speaker.\n"
                "PK=PK speech. Raw predicted suspicion mass, not wolf-role probability.", ha="center", fontsize=7.5)
    return figure


def _trajectories(review, plt, width):
    figure, axes = plt.subplots(4, 2, figsize=(width, 8.5), sharex=True, sharey=True, layout="constrained")
    x = np.arange(1, len(review.records) + 1)
    for observer, ax in enumerate(axes.flat[:7]):
        living = review.alive[:, observer]
        audit = ~living
        if len(audit) > 1:
            audit[:-1] |= ~living[1:]  # Include the last alive endpoint of a dashed transition.
        for target, color in enumerate(TARGET_COLORS):
            if target == observer:
                continue  # Structural zero is retained in data, stated in the caption.
            values = review.q[:, observer, target]
            ax.plot(x, np.where(living, values, np.nan), color=color, linewidth=1.3)
            ax.plot(x, np.where(audit, values, np.nan), color=color, linewidth=1.3, linestyle="--")
            target_alive = review.alive[:, target]
            ax.scatter(x[target_alive], values[target_alive], color=color, s=11, zorder=3)
            ax.scatter(x[~target_alive], values[~target_alive], facecolors="none", edgecolors=color, s=17, zorder=3)
        dead = np.flatnonzero(~living)
        if len(dead):
            ax.axvspan(x[dead[0]] - .5, len(x) + .5, color="0.85", alpha=.4)
        for index in range(1, len(review.records)):
            if review.records[index]["day"] != review.records[index - 1]["day"]:
                ax.axvline(index + .5, color="0.5", linestyle=":", linewidth=.7)
        ax.set(title=f"Observer P{observer + 1}", ylim=(0, 1), ylabel="Suspicion mass")
        ax.set_xticks(x[::2])
        ax.set_xlim(.5, len(x) + .5)
        ax.grid(alpha=.2)
    legend = axes.flat[7]
    legend.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, label=f"Target P{i + 1}") for i, color in enumerate(TARGET_COLORS)]
    handles += [Line2D([], [], color="0.3", linestyle="--", label="Observer dead / death transition"),
                Line2D([], [], color="0.3", marker="o", linestyle="none", markerfacecolor="none", label="Target dead (raw mass retained)")]
    legend.legend(handles=handles, loc="center", fontsize=7.5)
    axes[-1, 0].set_xlabel("PRE ordinal (canonical step order)")
    figure.suptitle("Observer-specific predicted suspicion trajectories", fontsize=10)
    figure.supxlabel("Shading/dashes: observer death; hollow points: target death; dotted lines: day changes.\n"
                       "Self=0 omitted from curves. Raw mass retained; alive does not imply Router eligibility.", fontsize=7.5)
    return figure


def _tv_ymax(statistics):
    values = [r[key] for r in statistics for key in ("all_observers_mean", "common_alive_mean")
              if r[key] is not None]
    return min(1, max(.1, 1.3 * max(values, default=0)))


def _variation(review, statistics, plt, width):
    figure, axes = plt.subplots(3, 1, figsize=(width, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2, .28, 1.8]}, layout="constrained")
    if not statistics:
        for ax in axes:
            ax.text(.5, .5, "No adjacent PRE pair", ha="center", transform=ax.transAxes)
        return figure
    x = np.arange(len(statistics))
    values = np.array([list(r["tv_by_observer"].values()) for r in statistics]).T
    image = axes[0].pcolormesh(np.arange(len(x) + 1) - .5, np.arange(8) - .5,
                              values, vmin=0, vmax=1, cmap="magma", shading="flat", rasterized=False)
    for ax in axes:
        ax.set_xlim(-.5, len(x) - .5)
    axes[0].set(ylim=(6.5, -.5), yticks=range(7), yticklabels=[f"P{i}" for i in range(1, 8)],
                ylabel="Observer", title="Individual raw seven-dimensional TV")
    _colorbar(figure, image, ax=axes[0], label="TV")
    axes[1].set(ylim=(0, 1), yticks=[.5], yticklabels=["n"])
    for position, r in enumerate(statistics):
        axes[1].text(position, .5, str(r["common_alive_count"]), ha="center", va="center", fontsize=8)
    axes[1].tick_params(axis="x", bottom=False)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[2].plot(x, [r["all_observers_mean"] for r in statistics], marker="o", markerfacecolor="none",
                 markersize=5, linewidth=1.2, label="All seven observer rows")
    axes[2].plot(x, [np.nan if r["common_alive_mean"] is None else r["common_alive_mean"] for r in statistics],
                 marker="s", markersize=3, linewidth=1.2, label="Common-alive observer rows")
    labels = [f"{r['from_pre']}→{r['to_pre']}" for r in statistics]
    axes[2].set_xticks(x, labels, rotation=45, ha="right", fontsize=7.5)
    for position, r in enumerate(statistics):
        before, after = (review.records[r[key] - 1]["day"] for key in ("from_pre", "to_pre"))
        if before != after:
            for ax in axes:
                ax.axvspan(position - .48, position + .48, color="#E69F00", alpha=.14, zorder=0)
            axes[2].text(position, .90, f"Day {before}→{after}", ha="center", fontsize=8,
                         transform=axes[2].get_xaxis_transform())
            axes[2].get_xticklabels()[position].set_fontweight("bold")
    axes[2].set(xlabel="Compared PRE interval", ylabel="Mean TV", ylim=(0, _tv_ymax(statistics)))
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(alpha=.2)
    figure.suptitle("Adjacent-PRE changes in predicted suspicion distributions", fontsize=10)
    figure.supxlabel("n: observers alive at both endpoints. Shading: cross-day intervals.\n"
                       "TV uses all seven targets without renormalization; temporal association only.", fontsize=7.5)
    return figure


def _event_timeline(ax, review, first, last):
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#6B4C3B")
    for level, kind in enumerate(EVENT_TYPES):
        selected = [e for e in review.events if e["event_type"] == kind and first <= e["event_index"] <= last]
        ax.scatter([e["event_index"] for e in selected], [level] * len(selected), s=16, color=colors[level])
        for e in selected:
            label = (PHASE_LABELS[e["phase"]] if kind == "phase_change" else
                     e.get("speaker") or ",".join(e.get("dead_players", e.get("exiled_players", ()))))
            if kind in ("death_announcement", "exile_result") and not label:
                label = "none"
            if label:
                ax.annotate(label.replace("player", "P"), (e["event_index"], level),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)
            if kind == "phase_change":
                ax.axvline(e["event_index"], color="0.75", linewidth=.6, linestyle=":", zorder=0)
    for ordinal, prefix in enumerate(review.prefixes, 1):
        end = prefix.public_event_history.events[-1].event_index
        if first <= end <= last:
            ax.scatter([end], [-1], marker="v", color="black", s=18)
            ax.annotate(str(ordinal), (end, -1), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set(yticks=[-1, *range(len(EVENT_TYPES))],
           yticklabels=["PRE", "Phase", "Turn", "Speech", "Vote", "Exile", "Death"],
           ylim=(-1.7, 6), xlim=(first - .8, last + .8))
    ax.tick_params(labelsize=8)


def _event_alignment(review, statistics, plt, width):
    figure, axes = plt.subplots(2, 1, figsize=(width, 6.3), sharex=True, layout="constrained")
    endpoints = [p.public_event_history.events[-1].event_index for p in review.prefixes]
    last = review.events[-1]["event_index"]
    _event_timeline(axes[0], review, review.events[0]["event_index"], last)
    for position, r in enumerate(statistics):
        start, end = endpoints[position:position + 2]
        for key, color, marker, line_width in (("all_observers_mean", "#0072B2", "o", 2),
                                             ("common_alive_mean", "#E69F00", "s", 1)):
            value = r[key]
            if value is not None:
                axes[1].plot([start, end], [value, value], color=color, linewidth=line_width,
                             label=("All seven rows" if key == "all_observers_mean" else "Common-alive rows") if position == 0 else None)
                axes[1].plot([(start + end) / 2], [value], marker=marker, color=color, markersize=3,
                             markerfacecolor="none" if marker == "o" else color)
    tail = last > endpoints[-1]
    if tail:
        for ax in axes:
            ax.axvspan(endpoints[-1] + .5, last + .8, color="0.85", alpha=.5, zorder=0)
    axes[1].set(xlabel="Canonical public event index", ylabel="Interval mean TV", ylim=(0, _tv_ymax(statistics)))
    if statistics:
        axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    figure.suptitle("Complete public timeline and PRE-interval TV", fontsize=10)
    footer = "Horizontal TV segments span compared PRE endpoints; no event-level causal attribution."
    if tail:
        footer += f"\nGray tail: events {endpoints[-1] + 1}–{last}, without a subsequent ToM observation."
    figure.supxlabel(footer + "\nS=speech; PK=PK speech; V=vote; PV=PK vote; N=night; none=empty result.", fontsize=7.5)
    return figure


def _key_events(review, statistics, pairs, plt, width):
    figure, axes = plt.subplots(2, 1, figsize=(width, 5.8), layout="constrained")
    endpoints = [p.public_event_history.events[-1].event_index for p in review.prefixes]
    for ax, (before, after) in zip(axes, pairs):
        start, end = endpoints[before - 1], endpoints[after - 1]
        _event_timeline(ax, review, start, end)
        ax.axvspan(start, end, color="#E69F00", alpha=.1, zorder=0)
        r = statistics[before - 1]
        common = "missing" if r["common_alive_mean"] is None else f"{r['common_alive_mean']:.3f}"
        ax.set_title(f"PRE {before}→{after} · Day {review.records[before - 1]['day']}→{review.records[after - 1]['day']}\n"
                     f"TV_all={r['all_observers_mean']:.3f}; TV_common={common}; n={r['common_alive_count']} | new events {start + 1}–{end}", fontsize=9)
        ax.set_xticks(range(start, end + 1))
        ax.set_xlabel("Canonical public event index")
    figure.suptitle("Key PRE intervals · public events and distribution change", fontsize=10)
    figure.supxlabel("TV compares both PREs across the entire shaded interval; temporal correspondence only.\n"
                       "n: common-alive observers. S=speech; PK=PK speech; V=vote; PV=PK vote; N=night.", fontsize=7.5)
    return figure


def csv_bytes(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def generate(review, output_dir, *, formats=("png", "pdf", "svg"), pre_indices=(1,),
             dpi=300, columns=5, font_size=9, figure_width=7, comparison_pairs=()):
    destination = check_output_path(review.root, output_dir)
    require(formats and len(set(formats)) == len(formats) and set(formats) <= {"png", "pdf", "svg"}, "invalid formats")
    require(pre_indices and len(set(pre_indices)) == len(pre_indices)
            and all(type(i) is int and 1 <= i <= len(review.records) for i in pre_indices), "invalid PRE indices")
    require(72 <= dpi <= 1200 and 1 <= columns <= 8 and 5 <= font_size <= 20, "invalid plotting parameters")
    require(6 <= figure_width <= 12, "figure width must be 6–12 inches")
    require(not comparison_pairs or (len(comparison_pairs) == 2 and len(set(comparison_pairs)) == 2
            and all(len(pair) == 2 and all(type(i) is int for i in pair)
                    and 1 <= pair[0] < pair[1] <= len(review.records) and pair[1] == pair[0] + 1
                    for pair in comparison_pairs)), "comparison pairs require two distinct adjacent PRE pairs")
    source = analysis_source()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    statistics = tv_statistics(review.q, review.alive)
    intervals = event_intervals(review)
    files = {}
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": font_size,
            "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": ANALYSIS_VERSION, "hatch.linewidth": .4}):
        figures = [(f"heatmap_pre_{i:02d}", lambda i=i: _heatmap(review, i - 1, plt, figure_width)) for i in pre_indices]
        figures += [
            ("matrix_sequence", lambda: _sequence(review, columns, plt, figure_width)),
            ("observer_trajectories", lambda: _trajectories(review, plt, figure_width)),
            ("total_variation", lambda: _variation(review, statistics, plt, figure_width)),
            ("public_event_alignment", lambda: _event_alignment(review, statistics, plt, figure_width)),
        ]
        if comparison_pairs:
            figures += [("heatmap_comparisons", lambda: _comparisons(review, comparison_pairs, plt, figure_width)),
                        ("key_interval_events", lambda: _key_events(review, statistics, comparison_pairs, plt, figure_width))]
        for name, build in figures:
            figure = build()
            try:
                for fmt in formats:
                    metadata = ({"CreationDate": None, "ModDate": None, "Creator": ANALYSIS_VERSION} if fmt == "pdf"
                                else {"Date": None, "Creator": ANALYSIS_VERSION} if fmt == "svg"
                                else {"Software": ANALYSIS_VERSION})
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format=fmt, dpi=dpi, metadata=metadata)
                    files[f"{name}.{fmt}"] = buffer.getvalue()
            finally:
                plt.close(figure)
    tv_rows = []
    for r in statistics:
        for i, observer in enumerate(PLAYER_IDS):
            tv_rows.append({"from_pre": r["from_pre"], "to_pre": r["to_pre"], "observer": observer,
                "tv": r["tv_by_observer"][observer], "alive_before": bool(review.alive[r["from_pre"] - 1, i]),
                "alive_after": bool(review.alive[r["to_pre"] - 1, i]),
                "in_common_alive_panel": observer in r["common_alive_observers"]})
    files["total_variation.csv"] = csv_bytes(tv_rows, (
        "from_pre", "to_pre", "observer", "tv", "alive_before", "alive_after", "in_common_alive_panel"))
    files["event_intervals.json"] = canonical_json_bytes(intervals)
    report = {"schema_version": ANALYSIS_VERSION, "input_files": review.input_files,
        "input_directory": str(review.root), "source": source,
        "model_provenance": {k: review.summary[k] for k in (
            "gameplay_source_revision", "predictor_source_revision", "fit_digest", "seal_digest")},
        "export_manifest_digest": review.manifest["manifest_digest"],
        "source_bundle_manifest_digest": review.bundle_manifest["manifest_digest"],
        "game_id": review.summary["game_id"], "seed": review.bundle_manifest["seed"],
        "population": {"num_pre": len(review.records), "num_cells": int(review.q.size),
            "structural_self_cells": len(review.records) * 7,
            "alive_observer_rows": int(review.alive.sum()),
            "dead_observer_rows": int((~review.alive).sum()),
            "public_events": len(review.events)},
        "row_sum_max_error": float(np.max(np.abs(review.q.sum(axis=2) - 1))),
        "validation": {"export_artifact_verified": True, "public_snapshot_verified": True,
            "full_canonical_bundle_revalidated": False, "frozen_model_weights_revalidated": False,
            "csv_matches_jsonl": True, "role_data_available": False, "router_panel_available": False},
        "interpretation": NOTICE, "parameters": {"formats": list(formats), "pre_indices": list(pre_indices),
            "dpi": dpi, "columns": columns, "font_size": font_size, "figure_width_inches": figure_width,
            "comparison_pairs": [list(pair) for pair in comparison_pairs], "row_sum_atol": ROW_SUM_ATOL},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": version("matplotlib")},
        "tv": statistics, "plots": sorted(name for name in files if name.rsplit(".", 1)[-1] in formats)}
    files["analysis.json"] = canonical_json_bytes(report)
    files["README.md"] = (
        "# Single-game ToM visualization\n\n" + NOTICE + "\n\n"
        "Diagonal crosses mark structural self zeros; hatching marks dead seats without changing Q. "
        "Dashed/shaded trajectories include raw outputs after observer death; hollow markers denote "
        "dead targets. TV compares complete "
        "seven-dimensional rows. Empty common-alive panels are null, never zero.\n\n"
        "PRE is before its speaker's speech. Adjacent PRE intervals can include several events, "
        "phase changes and deaths. TV points/segments represent intervals, not effects of events "
        "at their arrival PRE. No role truth, outcome scoring, significance or causal effect "
        "is inferred. Only supplied public snapshot hashes/identities were checked, not private "
        "full-bundle replay or model weight files.\n"
    ).encode("utf-8")
    require(input_hashes(review.root) == review.input_files, "inputs changed during plotting")
    require(sha256_bytes(Path(__file__).read_bytes()) == source["analysis_script_sha256"], "analysis source changed")
    artifact = publish_artifact(destination, manifest_fields={"artifact_type": "tom_single_game_visualization",
        "schema_version": ANALYSIS_VERSION, "game_id": review.summary["game_id"],
        "export_manifest_digest": review.manifest["manifest_digest"], "analysis_source": source}, files=files)
    return {"output_directory": str(artifact.path), "manifest_digest": artifact.manifest_digest,
            "files": sorted(files), "num_pre": len(review.records), "num_cells": int(review.q.size)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Review root containing export and canonical_public folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the input tree")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--pre-indices", nargs="+", type=int, default=[1], help="One-based PRE ordinals for individual heatmaps")
    parser.add_argument("--comparison-pairs", nargs="+", default=[], metavar="FROM:TO",
                        help="Two adjacent PRE pairs for a 2x2 matrix comparison and key event figure")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--font-size", type=float, default=9)
    parser.add_argument("--figure-width", type=float, default=7, help="Final placement width in inches (default: double-column 7 inches)")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", pair) for pair in args.comparison_pairs):
        parser.error("comparison pairs must be FROM:TO")
    pairs = tuple(tuple(map(int, pair.split(":"))) for pair in args.comparison_pairs)
    check_output_path(args.input_dir, args.output_dir)
    result = generate(load_review(args.input_dir), args.output_dir, formats=tuple(args.formats),
        pre_indices=tuple(args.pre_indices), dpi=args.dpi, columns=args.columns, font_size=args.font_size,
        figure_width=args.figure_width, comparison_pairs=pairs)
    print(canonical_json_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
