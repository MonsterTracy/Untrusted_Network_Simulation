"""Discover verified offline ToM exports and print Chinese three-line tables.

No gameplay, inference, model loading or pagination. Role truth is display-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

from scripts import visualize_tom_single_game as viz
from werewolf.canonical_collection.game_bundle import (
    _backend_call_from_record, _history_from_records, _private_replay_from_record,
)
from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose, BackendCallStatus

EXPORT_TYPE = "tom_one_game_all_pre_matrices"
CANONICAL_TYPE = "canonical_game_bundle"
ROLE_NAMES = {"Werewolf": "狼人", "Villager": "村民", "Seer": "预言家", "Witch": "女巫"}
PHASE_NAMES = {"discussion": "普通发言", "pk_discussion": "PK发言", "vote": "普通投票", "pk_vote": "PK投票"}
TOM_MARKER = "\n\nPRIVATE ToM INFORMATION\n"


@dataclass
class Game:
    game_id: str
    exports: tuple[Path, ...]
    canonicals: tuple[Path, ...]
    status: str = "INVALID"
    reason: str = ""
    review: viz.Review | None = None
    roles: dict | None = None
    online_tom: str = "无法确定（离线矩阵不证明在线使用）"
    evidence_files: int = 0


def _present_bound_files(root, manifest):
    """Verify every supplied canonical payload; absent private files stay absent."""
    result = {}
    for name, expected in manifest["file_table"].items():
        rel = Path(name)
        viz.require(bool(rel.parts) and not rel.is_absolute() and ".." not in rel.parts, "unsafe canonical file path")
        candidates = [root / rel]
        if rel.parts[0] == "public":
            candidates.append(root / rel.name)  # Documented flattened public snapshot.
        present = [p for p in candidates if os.path.lexists(p)]
        viz.require(len(present) <= 1, f"ambiguous canonical payload: {name}")
        if not present:
            continue
        p = present[0]
        viz.require(p.is_file() and not p.is_symlink(), f"not a regular canonical payload: {name}")
        payload = p.read_bytes()
        viz.require(expected == {"byte_size": len(payload), "sha256": viz.sha256_bytes(payload)},
                    f"canonical payload digest mismatch: {name}")
        result[name] = payload
    return result


def _extra_evidence(review, roots):
    files = {}
    for root in roots:
        for name, payload in _present_bound_files(root, review.bundle_manifest).items():
            viz.require(name not in files or files[name] == payload, f"canonical copy conflict: {name}")
            files[name] = payload
    roles = None
    if "private/private_replay_evidence.json" in files:
        record = viz._load_canonical_json(files["private/private_replay_evidence.json"])
        evidence = _private_replay_from_record(record)
        viz.require(evidence.game_id == review.summary["game_id"] and evidence.seed == review.bundle_manifest["seed"],
                    "private replay game/seed mismatch")
        viz.require(evidence.expected_public_event_digest == _history_from_records(review.events).digest,
                    "private replay public-history mismatch")
        viz.require(viz.sha256_bytes(evidence.runtime_configuration.canonical_bytes)
                    == review.bundle_manifest["runtime_configuration_digest"], "runtime configuration mismatch")
        roles = dict(evidence.role_assignment)
    online = "无法确定（未提供完整在线ToM使用证据；离线矩阵不证明在线使用）"
    if "private/backend_call_evidence.jsonl" in files:
        calls = [_backend_call_from_record(r) for r in viz.read_jsonl(files["private/backend_call_evidence.jsonl"])]
        viz.require(len({c.call_id for c in calls}) == len(calls), "duplicate backend call identity")
        prefixes = {p.boundary_id: p for p in review.prefixes}
        confirmed = []
        for call in calls:
            payload = call.private_payload.to_value()
            # Match the existing production prompt section in request fields only, never responses.
            request = {"args": payload.get("args", []), "kwargs": payload.get("kwargs", {})}
            def contains_marker(value):
                if isinstance(value, str):
                    return TOM_MARKER in value
                if isinstance(value, dict):
                    return any(contains_marker(v) for v in value.values())
                if isinstance(value, list):
                    return any(contains_marker(v) for v in value)
                return False
            if not contains_marker(request):
                continue
            prefix = prefixes.get(call.boundary_id)
            if (roles and prefix and call.observer_id == prefix.current_speaker
                    and roles[call.observer_id] == "Werewolf" and call.status is BackendCallStatus.SUCCESS
                    and call.purpose is BackendCallPurpose.SPEECH_GENERATION):
                confirmed.append(call.call_id)
        if confirmed:
            online = f"已确认在线ToM段落进入成功公开发言生成请求：{len(confirmed)}笔（不等同于在线模型权重复核）"
        else:
            online = "无法确定（所提供请求未形成此ToM注入接口的阳性证据；不判定为未使用）"
    return roles, online, len(files)


def discover(roots):
    """Count export game identities, not directories or bare canonical bundles."""
    exports, canonicals, issues = defaultdict(list), defaultdict(list), []
    seen_paths, protected_dirs = set(), set()
    for name in roots:
        raw = Path(name).expanduser()
        try:
            if not raw.is_dir():
                raise FileNotFoundError(raw)
            root = raw.resolve(strict=True)
        except OSError as error:
            issues.append(f"搜索目录不存在或不可读：{raw}：{error}（不会假设服务器目录已挂载）")
            continue
        def onerror(error):
            issues.append(f"目录读取失败：{error}")
        for current, directories, names in os.walk(root, followlinks=False, onerror=onerror):
            directories[:] = sorted(d for d in directories if d not in {".git", ".codegraph", "__pycache__"})
            if "manifest.json" not in names:
                continue
            path = Path(current) / "manifest.json"
            protected_dirs.add(path.parent.resolve())
            if path.is_symlink():
                issues.append(f"拒绝符号链接manifest：{path}")
                continue
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                m = json.loads(path.read_bytes())
                if not isinstance(m, dict):
                    raise ValueError("manifest must be an object")
                kind = m.get("artifact_type")
                if kind not in {EXPORT_TYPE, CANONICAL_TYPE}:
                    continue
                gid = m.get("game_id")
                viz.require(isinstance(gid, str) and bool(gid), "missing game_id")
                (exports if kind == EXPORT_TYPE else canonicals)[gid].append((path.parent, m))
            except (OSError, ValueError, TypeError) as error:
                issues.append(f"manifest无法识别：{path}：{error}")
    games = []
    for gid, copies in sorted(exports.items()):
        linked = canonicals.get(gid, [])
        game = Game(gid, tuple(p for p, _ in copies), tuple(p for p, _ in linked))
        games.append(game)
        try:
            digests = {m.get("manifest_digest") for _, m in copies}
            canonical_digests = {m.get("manifest_digest") for _, m in linked}
            if len(digests) != 1 or len(canonical_digests) > 1:
                game.status = "CONFLICT"
                raise ValueError("相同game_id存在不同产物内容或canonical来源；禁止静默选择")
            viz.require(bool(linked), "missing matching canonical public evidence")
            loaded = []
            for export, manifest in copies:
                candidates = [(p, m) for p, m in linked if m.get("manifest_digest") == manifest.get("source_bundle_manifest_digest")]
                viz.require(bool(candidates), "export/canonical source digest mismatch")
                for canonical, _ in candidates:
                    loaded.append(viz.load_review(export.parent, export_dir=export, canonical_dir=canonical))
            game.review = loaded[0]
            game.roles, game.online_tom, game.evidence_files = _extra_evidence(game.review, game.canonicals)
            game.status = "VERIFIED_PUBLIC"
            game.reason = "矩阵/CSV/manifest/PRE/provenance通过；未执行完整canonical重放或在线模型权重复核"
        except (OSError, ValueError, TypeError, KeyError) as error:
            game.review = None
            game.reason = str(error)
    return games, issues, protected_dirs


def safe_text(text):
    return "".join("?" if unicodedata.category(c).startswith("C") else c for c in str(text))


def display_width(text, ambiguous_width=1):
    total = 0
    for c in text:
        if unicodedata.combining(c) or unicodedata.category(c) in {"Mn", "Me", "Cf"}:
            continue
        kind = unicodedata.east_asian_width(c)
        total += 2 if kind in {"W", "F"} else ambiguous_width if kind == "A" else 1
    return total


def pad(text, width, ambiguous_width=1, *, center=False):
    gap = width - display_width(text, ambiguous_width)
    viz.require(gap >= 0, "table cell exceeds display width")
    left = gap // 2 if center else 0
    return " " * left + text + " " * (gap - left)


def wrap_display(text, width, ambiguous_width=1):
    lines, current = [], ""
    for c in text:
        if display_width(current + c, ambiguous_width) > width:
            lines.append(current)
            current = ""
        current += c
    return lines + [current]


def _rule(char, width, ambiguous_width):
    size = display_width(char, ambiguous_width)
    return char * (width // size) + "-" * (width % size)


def table_width(precision=3, ambiguous_width=1):
    row = display_width("观察者", ambiguous_width) + 1
    column = max(6, precision + 2 + display_width("↑", ambiguous_width)) + 1
    return row + 7 * column


def matrix_lines(game, game_number, index, *, precision=3, tolerance=1e-8, ambiguous_width=1):
    r, q = game.review.records[index], game.review.q
    width = table_width(precision, ambiguous_width)
    row_width = display_width("观察者", ambiguous_width) + 1
    column_width = (width - row_width) // 7
    ordinal_in_day = sum(record["day"] == r["day"] for record in game.review.records[:index + 1])
    labels = [f"P{k + 1}" + ("狼" if game.roles and game.roles[p] == "Werewolf" else "")
              + ("亡" if p not in r["alive_observer_ids"] else "") for k, p in enumerate(viz.PLAYER_IDS)]
    title = [f"游戏{game_number}｜第{r['day']}天｜当天第{ordinal_in_day}个发言前时点",
             f"全局PRE {index + 1}｜{PHASE_NAMES[r['phase']]}｜当前发言玩家{r['current_speaker'].replace('player', 'P')}",
             "存活：" + " ".join(p.replace("player", "P") for p in r["alive_observer_ids"])]
    lines = [line for item in title for line in wrap_display(item, width, ambiguous_width)]
    lines += [_rule("━", width, ambiguous_width), pad("观察者", row_width, ambiguous_width)
              + "".join(pad(label, column_width, ambiguous_width, center=True) for label in labels),
              _rule("─", width, ambiguous_width)]
    for i, observer in enumerate(viz.PLAYER_IDS):
        cells = []
        for k, target in enumerate(viz.PLAYER_IDS):
            value = f"{q[index, i, k]:.{precision}f}"
            if i != k and game.roles and game.roles[target] == "Werewolf":
                delta = q[index, i, k] - q[index - 1, i, k] if index else None
                value += "·" if delta is None else "→" if abs(delta) <= tolerance else "↑" if delta > 0 else "↓"
            cells.append(pad(value, column_width, ambiguous_width, center=True))
        label = f"P{i + 1}" + ("亡" if observer not in r["alive_observer_ids"] else "")
        lines.append(pad(label, row_width, ambiguous_width) + "".join(cells))
    lines.append(_rule("━", width, ambiguous_width))
    return [pad(line, width, ambiguous_width) for line in lines]


def vote_rounds(events):
    result = []
    for position, event in enumerate(events):
        if event["event_type"] != "vote_result":
            continue
        counts = Counter(v["target"] for v in event["votes"] if v["target"] is not None)
        exiles = None
        for following in events[position + 1:]:
            if following["event_type"] == "vote_result" or (following["day"], following["phase"]) != (event["day"], event["phase"]):
                break
            if following["event_type"] == "exile_result":
                exiles = following["exiled_players"]
                break
        result.append({"event": event, "counts": counts,
                       "abstain": sum(v["target"] is None for v in event["votes"]), "exiles": exiles})
    return result


def game_description(game, number):
    review = game.review
    lines = [f"游戏{number}：{safe_text(game.game_id)}", f"校验状态：{game.status}；{game.reason}"]
    lines += [f"ToM导出来源：{p}" for p in game.exports]
    lines += [f"canonical来源：{p}" for p in game.canonicals]
    if review is None:
        return "\n".join([*lines, "天数/PRE/矩阵数量未核验，不作为真实数量报告。"])
    days = {e["day"] for e in review.events if e["day"] > 0}
    lines += [f"公共记录覆盖{len(days)}天；实际PRE={len(review.records)}；矩阵={len(review.q)}；单元={review.q.size}",
              f"已校验canonical文件：{game.evidence_files}/{len(review.bundle_manifest['file_table'])}（允许公开快照缺少私密文件）",
              f"gameplay source：{review.summary['gameplay_source_revision']}",
              f"predictor source：{review.summary['predictor_source_revision']}",
              f"export manifest：{review.manifest['manifest_digest']}",
              f"canonical manifest：{review.bundle_manifest['manifest_digest']}",
              f"fit：{review.summary['fit_digest']}；seal：{review.summary['seal_digest']}",
              f"在线ToM：{game.online_tom}"]
    if game.roles is None:
        lines.append("角色信息未提供；不猜测狼人，不展示狼人列标记或箭头。")
    else:
        lines.append("真实角色（仅离线展示）：" + "；".join(f"P{i + 1}={ROLE_NAMES[game.roles[p]]}" for i, p in enumerate(viz.PLAYER_IDS)))
    lines.append("已记录公开发言顺序：")
    speeches = defaultdict(list)
    for e in review.events:
        if e["event_type"] == "public_speech":
            speeches[(e["day"], e["phase"])].append(e["speaker"].replace("player", "P"))
    lines += [f"  第{day}天 {PHASE_NAMES[phase]}：{'→'.join(players)}" for (day, phase), players in speeches.items()]
    lines.append("实际投票结算（不是发言中的投票意图）：")
    for ordinal, row in enumerate(vote_rounds(review.events), 1):
        e = row["event"]
        counts = "，".join(f"{p.replace('player', 'P')}={row['counts'][p]}票" for p in viz.PLAYER_IDS if p in row["counts"]) or "无目标票"
        exile = "未提供" if row["exiles"] is None else "、".join(p.replace("player", "P") for p in row["exiles"]) or "无人放逐"
        lines.append(f"  轮{ordinal} 第{e['day']}天 {PHASE_NAMES[e['phase']]}：{counts}；弃票={row['abstain']}；放逐={exile}（event {e['event_index']}）")
    return "\n".join(lines)


def select_games(games, *, game=None, selected=None, game_id=None, all_games=False):
    if game_id is not None:
        indices = [i for i, g in enumerate(games) if g.game_id == game_id]
        viz.require(len(indices) == 1, f"game_id未找到：{game_id}")
    elif all_games:
        indices = list(range(len(games)))
    else:
        numbers = [game] if game is not None else selected
        viz.require(numbers and len(set(numbers)) == len(numbers), "请选择不重复的游戏序号")
        viz.require(all(1 <= n <= len(games) for n in numbers), "游戏序号超出列表范围")
        indices = [n - 1 for n in numbers]
    viz.require(bool(indices), "没有可展示的ToM游戏")
    for i in indices:
        viz.require(games[i].status == "VERIFIED_PUBLIC", f"游戏{i + 1}禁止输出：{games[i].status}：{games[i].reason}")
    return indices


def render_game(game, number, *, indices=None, per_row=2, precision=3, tolerance=1e-8, ambiguous_width=1):
    selected = list(range(1, len(game.review.records) + 1)) if indices is None else indices
    viz.require(selected and len(set(selected)) == len(selected) and all(1 <= n <= len(game.review.records) for n in selected),
                f"游戏{number}的PRE序号无效或重复")
    lines = [game_description(game, number),
             f"展示：每行{per_row}矩阵；精度={precision}；箭头容差={tolerance:g}；箭头对比上一实际PRE（含未选中的PRE）。",
             "数值为预测嫌疑质量，不是狼人身份概率；亡=已死亡；自身结构零不带箭头；不删除或重新归一化。", ""]
    width = table_width(precision, ambiguous_width)
    selected = sorted(selected)
    for start in range(0, len(selected), per_row):
        tables = [matrix_lines(game, number, ordinal - 1, precision=precision, tolerance=tolerance,
                              ambiguous_width=ambiguous_width) for ordinal in selected[start:start + per_row]]
        for row in range(max(map(len, tables))):
            lines.append("    ".join(t[row] if row < len(t) else " " * width for t in tables))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", action="extend", required=True, type=Path, help="One or more search roots; repeatable")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--list", action="store_true", help="List discovered identities and verification status (default)")
    choice.add_argument("--game", type=int, help="Select one 1-based list index")
    choice.add_argument("--games", nargs="+", type=int, help="Select multiple 1-based list indices")
    choice.add_argument("--game-id", help="Select exact canonical game_id")
    choice.add_argument("--all", action="store_true", help="Display all discovered games; refuses any invalid/conflicting game")
    parser.add_argument("--per-row", choices=(1, 2, 3), type=int, default=2)
    parser.add_argument("--pre-indices", nargs="+", type=int, help="1-based PRE ordinals, sorted into real chronology")
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--delta-tolerance", type=float, default=1e-8, help="Display-only arrow tolerance on unrounded values")
    parser.add_argument("--ambiguous-width", choices=(1, 2), type=int, default=1, help="Terminal width of East-Asian ambiguous characters, including arrows")
    parser.add_argument("--terminal-width", type=int, help="Explicit column width; otherwise check actual stdout TTY only")
    parser.add_argument("--output-dir", type=Path, help="New output directory, never overwritten; UTF-8 TXT")
    args = parser.parse_args(argv)
    if not (0 <= args.precision <= 8 and math.isfinite(args.delta_tolerance) and args.delta_tolerance >= 0):
        parser.error("precision must be 0–8 and delta tolerance finite/nonnegative")
    if args.terminal_width is not None and args.terminal_width <= 0:
        parser.error("terminal width must be positive")
    games, issues, input_dirs = discover(args.root)
    for issue in issues:
        print(f"警告：{issue}", file=sys.stderr)
    listing = f"可读搜索范围内发现{len(games)}个唯一ToM游戏标识；校验通过{sum(g.status == 'VERIFIED_PUBLIC' for g in games)}局。canonical-only游戏不计入。\n"
    listing += "\n\n".join(game_description(game, i + 1) for i, game in enumerate(games))
    is_listing = not (args.game is not None or args.games is not None or args.game_id is not None or args.all)
    try:
        if is_listing:
            text = listing
        else:
            selected = select_games(games, game=args.game, selected=args.games, game_id=args.game_id, all_games=args.all)
            width = args.terminal_width
            if width is None and sys.stdout.isatty():
                width = os.get_terminal_size(sys.stdout.fileno()).columns
            columns = min(args.per_row, max(len(args.pre_indices) if args.pre_indices else len(games[i].review.records) for i in selected))
            needed = columns * table_width(args.precision, args.ambiguous_width) + (columns - 1) * 4
            viz.require(width is None or width >= needed,
                        f"布局需要{needed}列，当前{width}列；请扩大终端或改用--per-row {2 if args.per_row == 3 else 1}。不会截断第七列。")
            text = f"所需表格列宽：{needed}；东亚模糊字符宽度：{args.ambiguous_width}\n"
            text += f"展示脚本SHA256：{viz.sha256_bytes(Path(__file__).read_bytes())}\n"
            text += f"数据读取脚本SHA256：{viz.sha256_bytes(Path(viz.__file__).read_bytes())}\n\n"
            text += "\n\n".join(render_game(games[i], i + 1, indices=args.pre_indices, per_row=args.per_row,
                precision=args.precision, tolerance=args.delta_tolerance, ambiguous_width=args.ambiguous_width) for i in selected)
        text = "\n".join(safe_text(line) for line in text.split("\n"))
        if args.output_dir is not None:
            # Search roots may be broad /data directories; protect actual input trees.
            viz.require(bool(input_dirs), "没有已识别输入，不创建输出目录")
            for root in input_dirs:
                viz.check_output_path(root, args.output_dir)
            destination = args.output_dir.expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "tom-game-matrices.txt").write_text(text + "\n", encoding="utf-8")
            print(f"UTF-8 TXT：{destination / 'tom-game-matrices.txt'}", file=sys.stderr)
        print(text)
        return 1 if is_listing and (issues or any(g.status != "VERIFIED_PUBLIC" for g in games)) else 0
    except (ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
