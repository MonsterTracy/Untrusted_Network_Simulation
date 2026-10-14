"""Train-only weighted equivalence classes and smoothed event probabilities.

Prediction accepts ObserverFeatures only. Raw reports, Q targets and Phase2
objects are not accepted. TrainingRow.y is an explicit offline target.
"""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
import re
from numbers import Real

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.features import ObserverFeatures

VERSION = "observer_dtrs_equivalence_estimator_v1"
STATES = tuple((p, n) for p in ("discussion", "pk_discussion") for n in (1, 2))


def condition(features):
    if not isinstance(features, ObserverFeatures):
        raise TypeError("predict requires legal ObserverFeatures, never supervised records")
    f = features
    values = (f.s, f.m, f.r)
    if (f.phase not in ("discussion", "pk_discussion") or type(f.nW) is not int or f.nW not in (1, 2)
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   or not 0 <= v <= 1 + 1e-6 for v in values)
            or f.s > 1 or f.m > 1
            or not math.isclose(f.s + f.m, f.r, rel_tol=0, abs_tol=1e-12)
            or (f.nW == 1 and (f.m != 0 or f.dW is not None))
            or (f.nW == 2 and (isinstance(f.dW, bool) or not isinstance(f.dW, Real)
                or not math.isfinite(f.dW) or not math.isclose(f.dW, abs(f.s - f.m), rel_tol=0, abs_tol=1e-12)))):
        raise ValueError("invalid legal condition attributes")
    return f.phase, f.nW


@dataclass(frozen=True)
class TrainingRow:
    sample_id: str
    game_id: str
    features: ObserverFeatures
    y: int

    def __post_init__(self):
        condition(self.features)
        if (type(self.y) is not int or self.y not in (0, 1)
                or any(not isinstance(v, str) or not v for v in (self.sample_id, self.game_id))):
            raise ValueError("invalid explicit offline training row")


@dataclass(frozen=True)
class CellStats:
    rows: int
    games: int
    positive_games: int
    negative_games: int
    mass: float
    positive_mass: float
    squared_game_mass: float

    @property
    def kish_games(self):
        return self.mass ** 2 / self.squared_game_mass if self.squared_game_mass else 0.0

    def to_record(self):
        return {**asdict(self), "kish_games": self.kish_games}


def cell_stats(rows, weights):
    by_game = defaultdict(list)
    for row in rows:
        by_game[row.game_id].append(weights[row.sample_id])
    mass = [math.fsum(v) for v in by_game.values()]
    return CellStats(len(rows), len(by_game), len({r.game_id for r in rows if r.y}),
        len({r.game_id for r in rows if not r.y}), math.fsum(mass),
        math.fsum(weights[r.sample_id] * r.y for r in rows), math.fsum(v * v for v in mass))


def _node(stats, parent, smoothing):
    probability = (stats.positive_mass + smoothing * parent) / (stats.mass + smoothing)
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("smoothed model probability must be finite and strictly inside (0,1)")
    return {"stats": stats.to_record(), "probability": probability}


def train_edges(rows, weights, bins, spec):
    """Left-inverse weighted quantiles, then X-only deterministic support merges."""
    if not rows or bins == 1:
        return ()
    ordered = sorted(rows, key=lambda r: (r.features.r, r.sample_id))
    values = sorted({r.features.r for r in rows})
    by_value = defaultdict(list)
    for row in ordered:
        by_value[row.features.r].append(weights[row.sample_id])
    masses = [math.fsum(by_value[v]) for v in values]
    total = math.fsum(masses)
    cuts = set()
    for j in range(1, bins):
        left, right = 0, len(values) - 1
        while left < right:
            mid = (left + right) // 2
            if math.fsum(masses[:mid + 1]) >= total * j / bins:
                right = mid
            else:
                left = mid + 1
        cuts.add(values[left])
    edges = sorted(cuts - {values[-1]})
    while edges:
        groups = [[] for _ in range(len(edges) + 1)]
        for row in rows:
            groups[bisect_left(edges, row.features.r)].append(row)
        support = [cell_stats(group, weights) for group in groups]
        bad = next((j for j, s in enumerate(support) if
            s.games < spec["min_distinct_games"] or
            s.kish_games + spec["support_float_tolerance"] < spec["min_kish_game_mass_effective_support"]), None)
        if bad is None:
            break
        if bad == 0:
            delete = 0
        elif bad == len(groups) - 1:
            delete = bad - 1
        else:
            delete = bad - 1 if support[bad - 1].mass <= support[bad + 1].mass else bad
        edges.pop(delete)
    return tuple(edges)


@dataclass(frozen=True)
class FittedModel:
    """Immutable canonical bytes prevent caller mutation of fitted tables/identity."""
    payload: bytes

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def digest(self):
        return sha256_bytes(self.payload)

    def predict_detail(self, features):
        phase, nW = condition(features)
        m = self.record
        p, level = m["root"]["probability"], "global"
        wolf = m["wolf_nodes"].get(str(nW))
        if wolf:
            p, level = wolf["probability"], "nW"
        state = m["state_nodes"].get(f"{phase}/{nW}")
        if state:
            p, level = state["probability"], "phase_nW"
            if state["leaves"]:
                index = bisect_left(state["edges"], features.r)
                p, level = state["leaves"][index]["probability"], "r_bin"
        return {"probability": p, "level": level, "model_digest": self.digest}

    def predict(self, features):
        return self.predict_detail(features)["probability"]


def fit_model(rows, *, protocol, model_id, bins, smoothing, source, weighting="game_equal"):
    """Fit this explicit complete training partition; accepts no validation/test data."""
    rows = tuple(sorted(rows, key=lambda r: (r.game_id, r.sample_id)))
    p = protocol.record
    if (not rows or any(not isinstance(r, TrainingRow) for r in rows)
            or len({r.sample_id for r in rows}) != len(rows)
            or model_id not in p["primary_models"] or type(bins) is not int
            or bins not in p["histogram"]["requested_bin_counts"]
            or (model_id == "OBS-EQ0" and bins != 1)
            or isinstance(smoothing, bool) or smoothing not in p["smoothing"]["lambda_grid_game_equivalent_mass"]
            or weighting not in ("game_equal", "record_equal")):
        raise ValueError("invalid frozen EQ fitting request")
    _validate_source(source, protocol)
    actual = defaultdict(int)
    for row in rows:
        actual[row.game_id] += 1
    if actual != source["game_sizes"]:
        raise ValueError("fit requires all rows of exactly its declared whole training games")
    weights = {r.sample_id: (1 / actual[r.game_id] if weighting == "game_equal" else len(actual) / len(rows))
               for r in rows}
    stats = cell_stats(rows, weights)
    root = _node(stats, .5, 1)
    wolf_nodes, states = {}, {}
    for nW in (1, 2):
        group = [r for r in rows if r.features.nW == nW]
        if group:
            wolf_nodes[str(nW)] = _node(cell_stats(group, weights), root["probability"], smoothing)
    for phase, nW in STATES:
        group = [r for r in rows if condition(r.features) == (phase, nW)]
        if not group:
            continue
        node = _node(cell_stats(group, weights), wolf_nodes[str(nW)]["probability"], smoothing)
        edges = train_edges(group, weights, bins, p["histogram"]) if model_id == "OBS-EQ1" else ()
        node.update(edges=edges, leaves=[])
        if edges:
            for index in range(len(edges) + 1):
                leaf = [r for r in group if bisect_left(edges, r.features.r) == index]
                node["leaves"].append(_node(cell_stats(leaf, weights), node["probability"], smoothing))
        states[f"{phase}/{nW}"] = node
    training_fingerprint = sha256_bytes(canonical_json_bytes([
        {"sample_id": r.sample_id, "game_id": r.game_id, "conditions": r.features.to_record(), "y": r.y}
        for r in rows]))
    model = FittedModel(canonical_json_bytes({"schema_version": VERSION, "protocol_digest": protocol.digest,
        "model_id": model_id, "parameters": {"requested_bins": bins, "lambda": smoothing, "weighting": weighting},
        "source": source, "training_fingerprint": training_fingerprint,
        "root": root, "wolf_nodes": wolf_nodes, "state_nodes": states}))
    return validate_model(model, protocol)


def _validate_source(source, protocol):
    required = {"scope", "protocol_digest", "dataset_digest", "split_manifest_digest",
                "tom_fold", "outer_fold", "inner_fold", "game_sizes", "origins", "implementation_digest"}
    if (set(source) != required or source["scope"] not in ("controlled_fixture", "formal_training_partition")
            or source["protocol_digest"] != protocol.digest
            or not isinstance(source["game_sizes"], dict)
            or any(not isinstance(g, str) or not g or type(n) is not int or n < 1
                   for g, n in source["game_sizes"].items())
            or not isinstance(source["origins"], dict)
            or not isinstance(source["implementation_digest"], str)
            or re.fullmatch(r"[0-9a-f]{64}", source["implementation_digest"]) is None):
        raise ValueError("fit requires explicit valid source identity")
    if source["scope"] == "formal_training_partition":
        if (source["dataset_digest"] != protocol.record["dataset_digest"]
                or not isinstance(source["split_manifest_digest"], str)
                or re.fullmatch(r"[0-9a-f]{64}", source["split_manifest_digest"]) is None
                or any(type(source[k]) is not int or source[k] not in range(5) for k in ("tom_fold", "outer_fold"))
                or (source["inner_fold"] is not None and
                    (type(source["inner_fold"]) is not int or source["inner_fold"] not in range(3)))
                or len(source["game_sizes"]) != (240 if source["inner_fold"] is None else 160)):
            raise ValueError("formal fit partition identity mismatch")
    elif any(source[k] is not None for k in ("dataset_digest", "split_manifest_digest", "tom_fold", "outer_fold", "inner_fold")):
        raise ValueError("controlled fixtures must not claim formal dataset/fold identities")


def validate_model(model, protocol):
    """Check probabilities, sufficient statistics, schema and frozen parameters.

    This verifies a fitted artifact, not reconstruction of original training Y;
    that separately requires its recorded training partition/fingerprint.
    """
    m = model.record
    if (canonical_json_bytes(m) != model.payload or set(m) != {"schema_version", "protocol_digest",
        "model_id", "parameters", "source", "training_fingerprint", "root", "wolf_nodes", "state_nodes"}
            or m["schema_version"] != VERSION or m["protocol_digest"] != protocol.digest):
        raise ValueError("invalid estimator schema/protocol")
    parameters = m["parameters"]
    p = protocol.record
    if (set(parameters) != {"requested_bins", "lambda", "weighting"}
            or parameters["weighting"] not in ("game_equal", "record_equal")
            or m["model_id"] not in p["primary_models"]
            or type(parameters["requested_bins"]) is not int
            or parameters["requested_bins"] not in p["histogram"]["requested_bin_counts"]
            or (m["model_id"] == "OBS-EQ0" and parameters["requested_bins"] != 1)
            or isinstance(parameters["lambda"], bool)
            or parameters["lambda"] not in p["smoothing"]["lambda_grid_game_equivalent_mass"]):
        raise ValueError("invalid estimator parameters")
    _validate_source(m["source"], protocol)
    if (not m["source"]["game_sizes"] or not isinstance(m["training_fingerprint"], str)
            or re.fullmatch(r"[0-9a-f]{64}", m["training_fingerprint"]) is None):
        raise ValueError("invalid estimator training identity")
    def node(record, parent, strength, extra=()):
        if set(record) != {"stats", "probability", *extra}:
            raise ValueError("invalid estimator node schema")
        raw = record["stats"]
        if set(raw) != {"rows", "games", "positive_games", "negative_games", "mass", "positive_mass", "squared_game_mass", "kish_games"}:
            raise ValueError("invalid estimator sufficient statistics schema")
        if (any(type(raw[k]) is not int or raw[k] < 0 for k in ("rows", "games", "positive_games", "negative_games"))
                or not 0 < raw["games"] <= raw["rows"]
                or max(raw["positive_games"], raw["negative_games"]) > raw["games"]
                or raw["positive_games"] + raw["negative_games"] < raw["games"]
                or any(isinstance(raw[k], bool) or not isinstance(raw[k], Real) or not math.isfinite(raw[k])
                       for k in ("mass", "positive_mass", "squared_game_mass", "kish_games"))
                or raw["mass"] <= 0 or not 0 <= raw["positive_mass"] <= raw["mass"] + 1e-12
                or raw["squared_game_mass"] <= 0):
            raise ValueError("invalid estimator support/counts")
        stats = CellStats(**{k: v for k, v in raw.items() if k != "kish_games"})
        if (not math.isclose(stats.kish_games, raw["kish_games"], abs_tol=1e-12)
                or stats.kish_games > stats.games + 1e-9
                or record["probability"] != _node(stats, parent, strength)["probability"]):
            raise ValueError("invalid estimator shrinkage or Kish calculation")
    node(m["root"], .5, 1)
    sizes = m["source"]["game_sizes"]
    if m["root"]["stats"]["rows"] != sum(sizes.values()) or m["root"]["stats"]["games"] != len(sizes):
        raise ValueError("training population support mismatch")
    def conserved(parent, children):
        for key in ("rows", "mass", "positive_mass"):
            if not math.isclose(parent["stats"][key], math.fsum(c["stats"][key] for c in children), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("estimator child masses do not partition parent")
    if not set(m["wolf_nodes"]) <= {"1", "2"}:
        raise ValueError("invalid wolf stratum")
    for wolf in m["wolf_nodes"].values():
        node(wolf, m["root"]["probability"], parameters["lambda"])
    conserved(m["root"], m["wolf_nodes"].values())
    if not set(m["state_nodes"]) <= {f"{phase}/{n}" for phase, n in STATES}:
        raise ValueError("invalid phase/wolf stratum")
    for key, state in m["state_nodes"].items():
        node(state, m["wolf_nodes"][key.split("/")[1]]["probability"], parameters["lambda"], ("edges", "leaves"))
        edges = state["edges"]
        if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                or not 0 <= v <= 1 + 1e-6 for v in edges)
                or edges != sorted(set(edges)) or len(edges) >= parameters["requested_bins"]
                or bool(edges) != bool(state["leaves"])
                or (edges and len(state["leaves"]) != len(edges) + 1)):
            raise ValueError("invalid fitted bin partition")
        for leaf in state["leaves"]:
            node(leaf, state["probability"], parameters["lambda"])
            if (leaf["stats"]["games"] < p["histogram"]["min_distinct_games"]
                    or leaf["stats"]["kish_games"] + p["histogram"]["support_float_tolerance"] < p["histogram"]["min_kish_game_mass_effective_support"]):
                raise ValueError("unsupported fitted leaf")
        if edges:
            conserved(state, state["leaves"])
    for nW, wolf in m["wolf_nodes"].items():
        conserved(wolf, [state for key, state in m["state_nodes"].items() if key.endswith('/' + nW)])
    return model
