"""把一致的报告事件概率划分为两层可引用的认知方向。"""

from dataclasses import asdict, dataclass
import json
from math import fsum, isclose

from werewolf.artifact_io import open_artifact_envelope, read_artifact_file
from werewolf.observer_dtrs.decision import Region


ARTIFACT_TYPE = "observer_dtrs_report_state_risk"
MAIN_CONFIGURATION = "main"
EVENTS = ("speaker_hit", "team_hit")
NEG, BND, POS = Region.NEG.value, Region.BND.value, Region.POS.value
REGIONS = (NEG, BND, POS)
JOINT_REGIONS = ("NN", "NB", "NP", "BB", "BP", "PP")


def _is_probability(value):
    return type(value) in (int, float) and 0 <= value <= 1


def event_probabilities(record, wolf_count):
    """接受第一轮的联合概率记录并核对事件包含关系和单狼结构零。"""
    q0, q1, q2, ps, pT = (record[key] for key in ("q0", "q1", "q2", "ps", "pT"))
    if (not all(map(_is_probability, (q0, q1, q2, ps, pT)))
            or not isclose(fsum((q0, q1, q2)), 1, rel_tol=0, abs_tol=1e-12)
            or ps != q2 or pT != q1 + q2 or wolf_count not in (1, 2)
            or (wolf_count == 1 and q1 != 0)):
        raise ValueError("报告状态概率不满足联合概率或单狼结构约束")
    return ps, pT


def _region(p, beta, alpha):
    if p <= beta:
        return NEG
    if p >= alpha:
        return POS
    return BND


@dataclass(frozen=True, slots=True)
class Thresholds:
    beta_s: float
    alpha_s: float
    beta_T: float
    alpha_T: float

    def __post_init__(self):
        pairs = ((self.beta_s, self.alpha_s), (self.beta_T, self.alpha_T))
        if any(not (_is_probability(beta) and _is_probability(alpha) and beta < .5 < alpha)
               for beta, alpha in pairs):
            raise ValueError("阈值须满足 0 ≤ beta < 0.5 < alpha ≤ 1")
        if self.beta_s < self.beta_T or self.alpha_s < self.alpha_T:
            raise ValueError("自身 NEG 与 POS 阈值均须大于或等于狼队的对应阈值")

    @classmethod
    def shared(cls, beta, alpha):
        return cls(beta, alpha, beta, alpha)

    @property
    def is_shared(self):
        return self.beta_s == self.beta_T and self.alpha_s == self.alpha_T

    def regions(self, ps, pT):
        return _region(ps, self.beta_s, self.alpha_s), _region(pT, self.beta_T, self.alpha_T)

    def to_record(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrozenSharedRule:
    thresholds: Thresholds
    rule_manifest_digest: str
    probability_manifest_digest: str


def load_shared_rule(path, *, probability_manifest_digest):
    """读取主预算下发布的共享规则，并绑定同一份概率模型产物。"""
    artifact = open_artifact_envelope(path, expected_artifact_type=ARTIFACT_TYPE,
                                     expected_schema_version=ARTIFACT_TYPE)
    if artifact.manifest["probability_manifest_digest"] != probability_manifest_digest:
        raise ValueError("冻结规则与概率模型 Artifact 不匹配")
    rules = json.loads(read_artifact_file(artifact, "rules.json"))
    selected = rules[MAIN_CONFIGURATION]["shared"]
    if not selected["feasible"]:
        raise ValueError("主预算下没有可发布的 Shared-risk 阈值")
    thresholds = Thresholds(**selected["thresholds"])
    if not thresholds.is_shared:
        raise ValueError("在线主规则要求共享阈值")
    return FrozenSharedRule(thresholds, artifact.manifest_digest, probability_manifest_digest)


def decide_pre(panel, rule):
    """使用 predict_pre 的当前 PRE 输出，每次独立确定观察者认知方向。"""
    if not isinstance(rule, FrozenSharedRule) or not rule.thresholds.is_shared:
        raise TypeError("在线决策需要已发布的 Shared-risk 规则")
    result = {key: panel[key] for key in ("game_id", "boundary_id", "prefix_digest", "step_index",
                                         "acting_wolf", "alive_wolf_ids")}
    wolf_count = len(result["alive_wolf_ids"])
    observers = {}
    for seat, record in panel["observers"].items():
        ps, pT = event_probabilities(record, wolf_count)
        ds, dT = rule.thresholds.regions(ps, pT)
        decisions = tuple(zip(EVENTS, (ds, dT), strict=True))
        observers[seat] = {"observer_id": seat, "ps": ps, "pT": pT, "ds": ds, "dT": dT,
                           "permitted_premises": {event: region == POS for event, region in decisions if region != BND},
                           "unresolved_premises": [event for event, region in decisions if region == BND]}
    return {**result, "rule_manifest_digest": rule.rule_manifest_digest,
            "probability_manifest_digest": rule.probability_manifest_digest, "observers": observers}
