"""Fsynced, append-only Online Pilot-T stage ledger and fail-closed recovery.

The ledger is a running journal, not a canonical study artifact. A game cannot
be replayed after process restart without a full simulator checkpoint, so an
unfinished stage is marked interrupted and the campaign continues with new
game IDs. Randomized assignments are never regenerated.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_online_plan import (
    PLAN_VERSION, SELECTION_RULE, Phase2OnlineTerminalPilotPlanV1,
    _uniform_index,
)
from werewolf.phase2_treatment import VERSION as TREATMENT_VERSION


LEDGER_VERSION = "phase2_online_terminal_ledger_v1"


class OnlinePilotLedgerError(ValueError):
    """Journal provenance, hash chain, or exactly-once stage is invalid."""


class OnlinePilotAssignmentLedgerV1:
    def __init__(self, path: Path, *, plan: Phase2OnlineTerminalPilotPlanV1,
                 source_commit: str):
        if (not isinstance(path, Path) or not isinstance(plan, Phase2OnlineTerminalPilotPlanV1)
                or not isinstance(source_commit, str) or len(source_commit) != 40
                or any(c not in "0123456789abcdef" for c in source_commit)):
            raise OnlinePilotLedgerError("ledger needs a frozen plan and source commit")
        self.path = path
        self.plan = plan
        self.source_commit = source_commit
        if path.is_symlink():
            raise OnlinePilotLedgerError("ledger path must not be a symlink")
        if path.exists():
            self.snapshot()  # validate the complete chain before any future call
        else:
            self._append("START", {"schema_version": LEDGER_VERSION,
                                   "pilot_id": plan.pilot_id,
                                   "plan_digest": plan.digest(),
                                   "campaign_purpose": plan.campaign_purpose,
                                   "target_assignment_count": plan.target_assignment_count,
                                   "source_commit": source_commit})

    def _read_locked(self, descriptor: int) -> list[dict]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
        if data and not data.endswith(b"\n"):
            raise OnlinePilotLedgerError("ledger has an incomplete final line")
        try:
            rows = [json.loads(line) for line in data.splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OnlinePilotLedgerError("ledger JSON is invalid") from error
        previous = None
        for number, row in enumerate(rows):
            try:
                valid = (isinstance(row, dict)
                         and row.get("sequence") == number + 1
                         and row.get("previous_digest") == previous
                         and row.get("digest") == sha256_bytes(canonical_json_bytes({
                             key: value for key, value in row.items() if key != "digest"})))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise OnlinePilotLedgerError("ledger hash chain differs")
            previous = row["digest"]
        if rows:
            header = rows[0]
            expected = {"schema_version": LEDGER_VERSION,
                        "pilot_id": self.plan.pilot_id,
                        "plan_digest": self.plan.digest(),
                        "campaign_purpose": self.plan.campaign_purpose,
                        "target_assignment_count": self.plan.target_assignment_count,
                        "source_commit": self.source_commit}
            if header["kind"] != "START" or header["payload"] != expected:
                raise OnlinePilotLedgerError("ledger belongs to another campaign/source")
            for row in rows[1:]:
                if row["kind"] == "ASSIGNMENT":
                    self._validate_assignment(row["payload"])
        return rows

    def _validate_assignment(self, payload: dict) -> None:
        """Recompute both draws from the frozen PRE identity on every recovery."""
        try:
            assigned = payload["assignment"]
            selection = assigned["selection"]
            opportunity = assigned["opportunity"]
            identity = opportunity["identity"]
            treatment = assigned["treatment"]
            pool = selection["candidate_pool"]
            game_id, boundary_id, prefix = (
                identity["game_id"], identity["boundary_id"], identity["prefix_digest"])
            candidate_index, candidate_key = _uniform_index([
                PLAN_VERSION, "candidate", self.plan.digest(),
                self.plan.assignment_seed, game_id, boundary_id, prefix, pool], len(pool))
            arm_index, arm_key = _uniform_index([
                PLAN_VERSION, "treatment", self.plan.digest(),
                self.plan.assignment_seed, game_id, boundary_id, prefix], 2)
            arm = "PUSH" if arm_index == 0 else "REDIRECT"
            selected = pool[candidate_index]
            selected_digest = sha256_bytes(canonical_json_bytes([
                self.plan.digest(), game_id, boundary_id, prefix, pool,
                selected, candidate_key, SELECTION_RULE]))
            opportunity_digest = sha256_bytes(canonical_json_bytes(opportunity))
            treatment_id = sha256_bytes(canonical_json_bytes([
                TREATMENT_VERSION, opportunity_digest, arm,
                "randomized_pilot", 0.5, arm_key]))
            if (payload["pilot_id"] != self.plan.pilot_id
                    or payload["source_commit"] != self.source_commit
                    or payload["game_id"] != game_id
                    or payload["assignment_id"] != sha256_bytes(canonical_json_bytes(assigned))
                    or len(pool) < 2 or pool != opportunity["public_legal"]["J"]
                    or selected != identity["candidate_j"]
                    or selected != selection["candidate_j"]
                    or selection["rule"] != SELECTION_RULE
                    or selection["plan_digest"] != self.plan.digest()
                    or selection["candidate_selection_seed"] != self.plan.assignment_seed
                    or selection["candidate_selection_key"] != candidate_key
                    or selection["candidate_selection_probability"] != 1 / len(pool)
                    or selection["selection_digest"] != selected_digest
                    or assigned["assignment_seed"] != self.plan.assignment_seed
                    or assigned["assignment_key"] != arm_key
                    or assigned["assigned_action"] != arm
                    or assigned["assignment_probability"] != 0.5
                    or not {"PUSH", "REDIRECT"}.issubset(assigned["legal_action_set"])
                    or treatment["requested_action"] != arm
                    or treatment["randomization_key"] != arm_key
                    or treatment["assignment_probability"] != 0.5
                    or treatment["opportunity_digest"] != opportunity_digest
                    or treatment["treatment_id"] != treatment_id):
                raise OnlinePilotLedgerError("assignment randomization/provenance differs")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, OnlinePilotLedgerError):
                raise
            raise OnlinePilotLedgerError("assignment record is malformed") from error

    @staticmethod
    def _stages(rows: list[dict]) -> dict[str, dict]:
        stages: dict[str, dict] = {}
        for row in rows[1:]:
            kind, payload = row["kind"], row["payload"]
            game_id = payload.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                raise OnlinePilotLedgerError("ledger stage lacks game identity")
            stage = stages.setdefault(game_id, {})
            if kind not in ("GAME_STARTED", "ASSIGNMENT", "BACKEND_CALL", "EXECUTION",
                            "CONSEQUENCE", "GAME_RESULT", "INTERRUPTED"):
                raise OnlinePilotLedgerError("unknown ledger stage")
            if kind in stage and kind != "BACKEND_CALL":
                raise OnlinePilotLedgerError("duplicate ledger stage")
            if kind != "GAME_STARTED" and "GAME_STARTED" not in stage:
                raise OnlinePilotLedgerError("stage precedes game start")
            if kind in ("BACKEND_CALL", "EXECUTION", "CONSEQUENCE", "INTERRUPTED") and "ASSIGNMENT" not in stage:
                raise OnlinePilotLedgerError("stage precedes assignment")
            if kind == "BACKEND_CALL":
                if "EXECUTION" in stage:
                    raise OnlinePilotLedgerError("backend call follows execution")
                if payload.get("assignment_id") != stage["ASSIGNMENT"]["assignment_id"]:
                    raise OnlinePilotLedgerError("backend call assignment binding differs")
                prior = stage.setdefault("BACKEND_CALL", [])
                if payload.get("call", {}).get("sequence") != len(prior) + 1:
                    raise OnlinePilotLedgerError("backend call sequence differs")
                prior.append(payload)
                continue
            if kind == "CONSEQUENCE" and "EXECUTION" not in stage:
                raise OnlinePilotLedgerError("consequence precedes execution")
            if kind == "CONSEQUENCE" and stage["EXECUTION"]["execution"]["success"] is not True:
                raise OnlinePilotLedgerError("failed execution cannot have consequence")
            if kind == "INTERRUPTED" and "CONSEQUENCE" in stage:
                raise OnlinePilotLedgerError("completed consequence cannot be interrupted")
            if kind == "GAME_RESULT" and payload.get("winner") not in ("Werewolf", "Villager"):
                raise OnlinePilotLedgerError("invalid final game audit")
            if "INTERRUPTED" in stage and kind != "INTERRUPTED":
                raise OnlinePilotLedgerError("stage follows interrupted game")
            stage[kind] = payload
        if sum("ASSIGNMENT" in value for value in stages.values()) > rows[0]["payload"]["target_assignment_count"]:
            raise OnlinePilotLedgerError("assignment target exceeded")
        return stages

    def snapshot(self) -> dict:
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            rows = self._read_locked(descriptor)
            if not rows:
                raise OnlinePilotLedgerError("ledger has no header")
            try:
                stages = self._stages(rows)
            except (KeyError, IndexError, TypeError, AttributeError) as error:
                raise OnlinePilotLedgerError("ledger stages are malformed") from error
            return {"events": tuple(rows), "games": stages,
                    "games_attempted": len(stages),
                    "assignment_count": sum("ASSIGNMENT" in value for value in stages.values())}
        finally:
            os.close(descriptor)

    def _append(self, kind: str, payload: dict) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            rows = self._read_locked(descriptor)
            if rows:
                stages = self._stages(rows)
                game_id = payload.get("game_id")
                stage = stages.get(game_id, {})
                if kind in stage:
                    if kind == "BACKEND_CALL":
                        existing = next((item for item in stage[kind]
                                         if item["call"]["sequence"] ==
                                         payload["call"]["sequence"]), None)
                    else:
                        existing = stage[kind]
                    if existing is not None:
                        if existing == payload:
                            os.fsync(descriptor)
                            return False
                        raise OnlinePilotLedgerError("conflicting existing ledger stage")
                if kind == "GAME_STARTED":
                    limit = self.plan.max_games_attempted
                    if limit is not None and len(stages) >= limit:
                        raise OnlinePilotLedgerError("max_games_attempted reached: INCOMPLETE")
                if kind == "ASSIGNMENT" and sum("ASSIGNMENT" in s for s in stages.values()) >= self.plan.target_assignment_count:
                    raise OnlinePilotLedgerError("assignment target already reached")
                candidate = rows + [{"kind": kind, "payload": payload}]
                # Validate ordering before writing; full hash fields are added below.
                self._stages(candidate)
            elif kind != "START":
                raise OnlinePilotLedgerError("first ledger stage must be START")
            row = {"sequence": len(rows) + 1, "kind": kind, "payload": payload,
                   "previous_digest": rows[-1]["digest"] if rows else None}
            row["digest"] = sha256_bytes(canonical_json_bytes(row))
            os.lseek(descriptor, 0, os.SEEK_END)
            encoded = canonical_json_bytes(row) + b"\n"
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise OnlinePilotLedgerError("ledger write made no progress")
                written += count
            os.fsync(descriptor)
            if not rows:
                parent = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            return True
        finally:
            os.close(descriptor)

    def start_game(self, game_id: str) -> None:
        if game_id in self.snapshot()["games"]:
            raise OnlinePilotLedgerError("resume cannot replay a previously started game")
        self._append("GAME_STARTED", {"game_id": game_id})

    def persist_assignment(self, assignment) -> bool:
        row = assignment.to_record()
        return self._append("ASSIGNMENT", {"game_id": assignment.opportunity.legal_context.game_id,
                                           "pilot_id": self.plan.pilot_id,
                                           "source_commit": self.source_commit,
                                           "assignment_id": assignment.digest(),
                                           "assignment": row})

    def persist_execution(self, record) -> bool:
        return self._append("EXECUTION", {"game_id": record.game_id,
                                          "assignment_id": record.assignment.digest(),
                                          "execution": record.to_record()["execution"],
                                          "backend_calls": [call.to_record()
                                                            for call in record.backend_calls]})

    def persist_backend_call(self, call) -> bool:
        if call.pilot_id != self.plan.pilot_id or not call.assignment_id:
            raise OnlinePilotLedgerError("backend call lacks pilot/assignment binding")
        return self._append("BACKEND_CALL", {
            "game_id": call.game_id, "assignment_id": call.assignment_id,
            "call": call.to_record()})

    def persist_consequence(self, record) -> bool:
        if record.day_outcome is None:
            raise OnlinePilotLedgerError("cannot persist an absent consequence")
        return self._append("CONSEQUENCE", {"game_id": record.game_id,
                                            "assignment_id": record.assignment.digest(),
                                            "day_consequence": record.to_record()["day_consequence"],
                                            "final_game_result": record.final_game_result,
                                            "theta_audit_label": record.theta_audit_label})

    def persist_game_result(self, game_id: str, winner: str) -> bool:
        return self._append("GAME_RESULT", {"game_id": game_id, "winner": winner})

    def mark_interrupted_on_resume(self) -> tuple[str, ...]:
        """No PRE checkpoint exists: retain old assignments and refuse game replay."""
        interrupted = []
        for game_id, stage in self.snapshot()["games"].items():
            if "ASSIGNMENT" not in stage or "INTERRUPTED" in stage:
                continue
            needs_outcome = ("EXECUTION" in stage and
                             stage["EXECUTION"]["execution"]["success"] is True)
            if "EXECUTION" not in stage or (needs_outcome and "CONSEQUENCE" not in stage):
                self._append("INTERRUPTED", {
                    "game_id": game_id,
                    "reason": ("ASSIGNMENT_WITHOUT_EXECUTION" if "EXECUTION" not in stage
                               else "EXECUTION_WITHOUT_CONSEQUENCE")})
                interrupted.append(game_id)
        return tuple(interrupted)

    def status(self) -> str:
        snap = self.snapshot()
        if snap["assignment_count"] == self.plan.target_assignment_count:
            return "READY_TO_SEAL"
        if (self.plan.max_games_attempted is not None
                and snap["games_attempted"] >= self.plan.max_games_attempted):
            return "INCOMPLETE"
        return "RUNNING"

    def sealable(self) -> bool:
        snapshot = self.snapshot()
        if snapshot["assignment_count"] != self.plan.target_assignment_count:
            return False
        for stage in snapshot["games"].values():
            if "ASSIGNMENT" not in stage:
                continue
            if "EXECUTION" not in stage and "INTERRUPTED" not in stage:
                return False
            if ("EXECUTION" in stage and stage["EXECUTION"]["execution"]["success"] is True
                    and "CONSEQUENCE" not in stage and "INTERRUPTED" not in stage):
                return False
        return True
