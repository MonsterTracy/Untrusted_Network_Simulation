"""OFFLINE native evidence consumer. No I/O, replay, Actor, Parser or commit.

The caller must obtain VerifiedCanonicalGameBundle from the original authorized
reader. Python types/digests do not establish external custody. This consumer
checks in-memory linkage, not that an arbitrary caller truly ran that reader.
No native schema carries the research assignment token; attribution stays UNKNOWN.
"""
from dataclasses import dataclass
from enum import Enum
import json

from werewolf.artifact_io import VerifiedArtifact, canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import (
    validate_attempt_claim, validate_attempt_terminal, TerminalOutcome,
)
from werewolf.canonical_collection.game_bundle import (
    VerifiedCanonicalGameBundle, _prefix_from_record, _replay_result,
    _BUNDLE_DOMAIN_MANIFEST_FIELDS, _BUNDLE_FILE_PATHS,
    CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE, CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
)
from werewolf.canonical_collection.pre import construct_speaker_pre_belief_handoff
from werewolf.canonical_collection.public_history import validate_public_event_history
from werewolf.canonical_collection.speech import validate_v1_annotation_binding, V1AnnotationStatus
from werewolf.canonical_collection.trajectory_evidence import construct_submitted_gameplay_action
from werewolf.observer_dtrs.action_evidence_ledger import digest, validate_ledger
from werewolf.phase2_offline import _resolved_exiles
from scripts.speech_realization import expected_action
from scripts.speech_planning import generate_candidates

VERSION = 'research_native_action_link_v1'


class Status(str, Enum):
    OBSERVED = 'OBSERVED'
    UNKNOWN = 'UNKNOWN'
    CONFLICT = 'CONFLICT'


@dataclass(frozen=True)
class Finding:
    status: Status
    reason: str
    evidence_json: bytes

    def to_record(self):
        return {'status': self.status.value, 'reason': self.reason,
                'evidence': json.loads(self.evidence_json)}


def _finding(status, reason, **evidence):
    return Finding(status, reason, canonical_json_bytes(evidence))


@dataclass(frozen=True)
class NativeActionLink:
    sources_json: bytes
    submission: Finding
    assignment_link: Finding
    semantics: Finding
    day: Finding
    completion: Finding

    def to_record(self):
        return {'version': VERSION, 'scope': 'OFFLINE_EVIDENCE_LINKAGE_NOT_EXECUTION_AUTHORITY',
                'sources': json.loads(self.sources_json),
                **{k: getattr(self, k).to_record() for k in
                   ('submission', 'assignment_link', 'semantics', 'day', 'completion')},
                'submission_permission': False, 'F1_permission': False,
                'external_exactly_once_proven': False, 'replay_executed': False,
                'external_custody_verified_by_linker': False}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _payload(bundle, path, raw):
    descriptor = bundle.manifest['file_table'][path]
    _require(type(descriptor['byte_size']) is int and descriptor['byte_size'] == len(raw)
             and descriptor['sha256'] == sha256_bytes(raw), 'NATIVE_PAYLOAD_DESCRIPTOR_MISMATCH:'+path)


def _submission(bundle, pre, op):
    actions = bundle.submitted_gameplay_actions
    _require(len({a.action_id for a in actions}) == len(actions), 'DUPLICATE_NATIVE_ACTION_ID')
    matches = [a for a in actions if a.boundary_id == op['boundary_id']]
    if not matches:
        return _finding(Status.UNKNOWN, 'NATIVE_ACTION_MISSING'), None, None
    _require(len(matches) == 1, 'DUPLICATE_NATIVE_PRE_ACTION')
    action = matches[0]
    h = action.speaker_pre_belief_handoff
    _require(h is not None, 'NATIVE_HANDOFF_MISSING_OR_CONFLICTING')
    # Reuse native constructors, including exact roundtrip and handoff digest.
    expected_handoff = construct_speaker_pre_belief_handoff(pre,
        observation_id=h.observation_id, observation_digest=h.observation_digest,
        observer_id=h.observer_id, observation_status='success', suspicion_support=h.suspicion_support)
    _require(h == expected_handoff, 'NATIVE_HANDOFF_BINDING_MISMATCH')
    rebuilt = construct_submitted_gameplay_action(action_id=action.action_id,
        step_index=action.step_index, actor_id=action.actor_id, action_type=action.action_type,
        action_payload=action.action_payload, resulting_public_event_ids=action.resulting_public_event_ids,
        boundary_id=action.boundary_id, speaker_pre_belief_handoff=h, fallback_used=action.fallback_used)
    _require(action == rebuilt and action.action_type == 'public_speech'
             and type(action.step_index) is int and action.step_index == pre.step_index
             and action.actor_id == pre.current_speaker, 'NATIVE_ACTION_PRE_MISMATCH')
    events = [e for e in bundle.public_event_stream.events if e.event_id in action.resulting_public_event_ids]
    _require(len(action.resulting_public_event_ids) == 1 and len(events) == 1, 'NATIVE_ACTION_EVENT_MISSING_OR_DUPLICATE')
    event = events[0]
    _require(event.event_type == 'public_speech' and event.event_index == len(pre.public_event_history.events)
             and event.speaker == pre.current_speaker and event.raw_text == action.action_payload.to_value()
             and event.temporal_state == pre.public_temporal_state, 'NATIVE_EVENT_PRE_TEXT_MISMATCH')
    evidence = dict(action_id=action.action_id, action_digest=action.action_digest,
                    event_id=event.event_id, event_index=event.event_index,
                    public_text_digest=sha256_bytes(event.raw_text.encode('utf-8')),
                    event_record_digest=digest(event.to_record()),
                    handoff_digest=h.handoff_digest, observation_id=h.observation_id,
                    observation_digest=h.observation_digest, fallback_used=action.fallback_used)
    return _finding(Status.OBSERVED, 'NATIVE_SUBMISSION_OBSERVED_NOT_ASSIGNMENT_ATTRIBUTION', **evidence), action, event


def _semantics(bundle, event, expected):
    if event is None:
        return _finding(Status.UNKNOWN, 'SUBMISSION_NOT_OBSERVED')
    matches = [a for a in bundle.speech_annotations_v1 if a.event_id == event.event_id]
    if not matches:
        return _finding(Status.UNKNOWN, 'V1_ANNOTATION_MISSING')
    _require(len(matches) == 1, 'DUPLICATE_V1_ANNOTATION')
    a = validate_v1_annotation_binding(matches[0], bundle.public_event_stream)
    actual = [x.to_record() for x in a.actions]
    category = ('PARSE_ERROR' if a.status is V1AnnotationStatus.ERROR else
                'EMPTY_ACTIONS' if not actual else 'MULTI_ACTION' if len(actual) > 1 else
                'EXACT_SINGLE' if actual == [expected] else 'SINGLE_MISMATCH')
    return _finding(Status.OBSERVED, 'V1_SEMANTICS_OBSERVED_NOT_ACTION_EFFECT',
                    classification=category, actual_full_v1=actual, expected_single_v1=expected,
                    annotation_digest=a.annotation_digest, annotation_status=a.status.value,
                    exact_single_compliance=category == 'EXACT_SINGLE')


def _day(bundle, op):
    events = tuple(e for e in bundle.public_event_stream.events if e.temporal_state.day == op['day'])
    if not any(e.event_type == 'vote_result' for e in events):
        if any(e.event_type == 'exile_result' for e in events):
            return _finding(Status.CONFLICT, 'EXILE_WITHOUT_BALLOT')
        return _finding(Status.UNKNOWN, 'DAY_BALLOT_OR_RESOLUTION_MISSING')
    try:
        result = _resolved_exiles(events)
    except ValueError as error:
        # Native resolver remains authoritative. Only a genuinely open frontier
        # with structurally incomplete vote/PK is treated as missing, never zero.
        frontier = bundle.public_event_stream.current_state
        incomplete = str(error) in ('incomplete or duplicated day resolution', 'ordinary tie lacks valid PK resolution')
        normal = [e for e in events if e.event_type == 'vote_result' and e.temporal_state.phase.value == 'vote']
        pk = [e for e in events if e.event_type == 'vote_result' and e.temporal_state.phase.value == 'pk_vote']
        exiles = [e for e in events if e.event_type == 'exile_result']
        open_day = frontier.day == op['day'] and frontier.phase.value in ('vote', 'pk_discussion', 'pk_vote')
        if incomplete and open_day and len(normal) == 1 and len(pk) <= 1 and len(exiles) < 1 + len(pk):
            return _finding(Status.UNKNOWN, 'DAY_NOT_CLOSED', native_reason=str(error))
        if incomplete and open_day and len(normal) == 1 and not pk and len(exiles) == 1:
            # The resolver rejected this ordinary tie because PK is not complete.
            return _finding(Status.UNKNOWN, 'PK_NOT_CLOSED', native_reason=str(error))
        return _finding(Status.CONFLICT, 'NATIVE_DAY_RESOLUTION_CONFLICT', native_reason=str(error))
    if op['day'] not in result:
        return _finding(Status.UNKNOWN, 'DAY_RESOLUTION_MISSING')
    final = result[op['day']]
    _require(final.event_index > op['step_index'], 'DAY_ENDPOINT_NOT_AFTER_PRE')
    return _finding(Status.OBSERVED, 'FINAL_PUBLIC_DAY_RESOLUTION', day=op['day'],
                    event_id=final.event_id, final_phase=final.temporal_state.phase.value,
                    exiled_seats=list(final.affected_players), observed_no_exile=not final.affected_players,
                    day_events_digest=digest([e.to_record() for e in events]))


def _completion(bundle, plan, claim, terminal):
    if terminal is None:
        return _finding(Status.UNKNOWN, 'ATTEMPT_TERMINAL_MISSING')
    validate_attempt_terminal(terminal, plan, claim)
    _require(terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS
             and terminal.canonical_game_bundle_id == bundle.game_id
             and terminal.canonical_game_bundle_digest == bundle.manifest_digest, 'TERMINAL_BUNDLE_CONFLICT')
    identity = bundle.manifest['replay_executor_identity']
    _require(bundle.replay_result == _replay_result(bundle.public_event_stream.digest, identity)
             and bundle.manifest['replay_result_digest'] == bundle.replay_result.replay_result_digest,
             'NATIVE_REPLAY_IDENTITY_CONFLICT')
    # Native replay results legitimately include partial trajectories. They do
    # not persist whether FullGameReplayExecutor was used or attest a winner.
    # Never read private replay/roles/reports to manufacture that missing proof.
    return _finding(Status.UNKNOWN, 'CANONICAL_SUCCESS_NOT_FULL_GAME_TERMINAL_PROOF',
                    canonical_attempt_closed=True, terminal_digest=terminal.record_digest,
                    replay_result_digest=bundle.replay_result.replay_result_digest,
                    replay_executor_identity=identity,
                    scope='ATTEMPT_CLOSURE_LINK_OBSERVED_FULL_GAME_COMPLETION_UNVERIFIED')


def link_native_evidence(records, *, ledger_id, assignment_ref, plan, claim, bundle=None,
                         expected_bundle_digest=None, attempt_terminal=None):
    """Consume original native reader objects; return five independent findings.

    No paths/readers/callbacks are invoked. expected_bundle_digest must come from
    caller-trusted custody; its authenticity cannot be established by this pure
    function. Native replay records do not attest full-game termination, so
    canonical-success terminals alone NEVER produce OBSERVED game completion.
    No private replay/role/report payload is accessed, even for completion.
    """
    sources = {'ledger_id': ledger_id, 'assignment_ref': assignment_ref,
               'research_input_digest': digest(records), 'native_bundle_digest': None,
               'expected_bundle_digest': expected_bundle_digest,
               'restricted_belief_or_replay_payloads_read': False}
    unknown = _finding(Status.UNKNOWN, 'EVIDENCE_NOT_AVAILABLE')
    def result(sub=unknown, attr=unknown, sem=unknown, day=unknown, end=unknown):
        return NativeActionLink(canonical_json_bytes(sources), sub, attr, sem, day, end)
    try:
        view = validate_ledger(records, ledger_id=ledger_id)
        sources['research_head'] = view.head
        sources['research_record_refs'] = [dict(sequence=r['sequence'], kind=r['kind'],
                                                record_digest=r['record_digest']) for r in records]
        sources['attempt_terminal_input_digest'] = (digest(attempt_terminal.to_record())
                                                     if attempt_terminal is not None else None)
        assignments = [r for r in records if r['record_digest'] == assignment_ref and r['kind'] == 'ASSIGNMENT']
        _require(len(assignments) == 1, 'RESEARCH_ASSIGNMENT_MISSING_OR_DUPLICATE')
        assignment = assignments[0]['payload']
        opportunity = next(r for r in records if r['record_digest'] == assignment['opportunity_ref'])
        op = opportunity['payload']
        if plan is None or claim is None:
            return result(*([_finding(Status.UNKNOWN, 'NATIVE_ATTEMPT_PROVENANCE_MISSING')]*5))
        sources.update(opportunity_ref=opportunity['record_digest'], game_id=op['game_id'],
                       game_attempt_id=op['game_attempt_id'], boundary_id=op['boundary_id'],
                       prefix_digest=op['prefix_digest'], plan_digest=plan.plan_digest, claim_digest=claim.record_digest)
        sources['plan_input_digest'] = digest(plan.to_record())
        sources['claim_input_digest'] = digest(claim.to_record())
        validate_attempt_claim(claim, plan)
        _require(claim.attempt_id == op['game_attempt_id'], 'ASSIGNMENT_CLAIM_ATTEMPT_CONFLICT')
        if bundle is None:
            return result(*([_finding(Status.UNKNOWN, 'NATIVE_BUNDLE_NOT_PROVIDED')]*5))
        _require(type(bundle) is VerifiedCanonicalGameBundle, 'ORIGINAL_NATIVE_READER_OBJECT_REQUIRED')
        _require(type(bundle.artifact) is VerifiedArtifact, 'ORIGINAL_VERIFIED_ARTIFACT_OBJECT_REQUIRED')
        m = bundle.manifest
        _require(set(m) == _BUNDLE_DOMAIN_MANIFEST_FIELDS | {'file_table', 'manifest_digest'}
                 and set(m['file_table']) == _BUNDLE_FILE_PATHS
                 and m['artifact_type'] == CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE
                 and m['schema_version'] == CANONICAL_GAME_BUNDLE_SCHEMA_VERSION
                 and m['canonical_eligibility'] is True
                 and type(m['ordinal']) is int and type(m['seed']) is int, 'NATIVE_MANIFEST_SCHEMA_CONFLICT')
        _require(digest({k:v for k,v in m.items() if k != 'manifest_digest'}) == m['manifest_digest'] == bundle.manifest_digest,
                 'BUNDLE_MANIFEST_DIGEST_CONFLICT')
        sources['native_bundle_digest'] = bundle.manifest_digest
        if expected_bundle_digest is None:
            return result(*([_finding(Status.UNKNOWN, 'CALLER_TRUSTED_BUNDLE_ANCHOR_MISSING')]*5))
        _require(expected_bundle_digest == bundle.manifest_digest, 'BUNDLE_EXTERNAL_ANCHOR_CONFLICT')
        _require(all(m[k] == value for k,value in {
            'game_id':op['game_id'], 'attempt_id':claim.attempt_id, 'collection_id':plan.collection_id,
            'collection_plan_digest':plan.plan_digest,
            'claim_record_digest':claim.record_digest, 'ordinal':claim.ordinal, 'seed':claim.seed,
            'source_revision':plan.source_revision, 'runtime_provenance_digest':plan.runtime_provenance_digest,
        }.items()) and bundle.game_id == op['game_id'], 'NATIVE_GAME_ATTEMPT_SOURCE_CONFLICT')
        history = validate_public_event_history(bundle.public_event_stream)
        sources['public_history_digest'] = history.digest
        payloads = dict((path, canonical_jsonl_bytes(values)) for path, values in (
            ('public/public_event_stream.jsonl', history.to_records()),
            ('public/authoritative_pre_prefixes.jsonl', [p.to_record() for p in bundle.authoritative_pre_prefixes]),
            ('public/speech_annotations_v1.jsonl', [a.to_record() for a in bundle.speech_annotations_v1]),
            ('private/submitted_gameplay_actions.jsonl', [a.to_record() for a in bundle.submitted_gameplay_actions])))
        sources['native_payloads'] = {path: dict(byte_size=len(raw), sha256=sha256_bytes(raw))
                                      for path,raw in payloads.items()}
        for path, raw in payloads.items():
            _payload(bundle, path, raw)
        # Daily/game findings are evaluated independently of action availability.
        try:
            day = _day(bundle, op)
        except (TypeError, ValueError, KeyError) as error:
            day = _finding(Status.CONFLICT, str(error))
        try:
            end = _completion(bundle, plan, claim, attempt_terminal)
        except (TypeError, ValueError, KeyError) as error:
            end = _finding(Status.CONFLICT, str(error))
        try:
            prefixes = [p for p in bundle.authoritative_pre_prefixes if p.boundary_id == op['boundary_id']]
            if not prefixes:
                return result(sub=_finding(Status.UNKNOWN, 'NATIVE_PRE_MISSING'),
                              attr=_finding(Status.UNKNOWN, 'NATIVE_PRE_MISSING'), day=day, end=end)
            _require(len(prefixes) == 1, 'DUPLICATE_NATIVE_BOUNDARY')
            pre = _prefix_from_record(prefixes[0].to_record())
            _require(pre.game_id == op['game_id'] and pre.prefix_digest == op['prefix_digest']
                     and type(op['step_index']) is int and pre.step_index == op['step_index']
                     and pre.current_speaker == op['actor'] and pre.public_temporal_state.day == op['day']
                     and pre.public_temporal_state.phase.value == op['phase']
                     and pre.public_event_digest == op['public_history_digest']
                     and digest(pre.to_record()) == op['capture_digest'], 'ASSIGNMENT_NATIVE_PRE_CONFLICT')
            _require(pre.public_event_history.to_records() == history.to_records()[:len(pre.public_event_history.events)],
                     'NATIVE_PRE_NOT_HISTORY_PREFIX')
            sub, action, event = _submission(bundle, pre, op)
            try:
                candidate = next(c for c in op['candidates'] if digest(c) == assignment['candidate_id'])
                _require(any(c.public_payload() == candidate['plan'] for c in generate_candidates(pre)),
                         'DECLARED_PLAN_NOT_LEGAL_AT_NATIVE_PRE')
                expected = expected_action(candidate['plan'], pre.current_speaker).to_record()
                _require(candidate['expected_v1'] == expected, 'DECLARED_PLAN_V1_CONFLICT')
            except (TypeError, ValueError, KeyError, StopIteration) as error:
                conflict = _finding(Status.CONFLICT, str(error))
                return result(sub, conflict, conflict, day, end)
            try:
                sem = _semantics(bundle, event, expected)
            except (TypeError, ValueError, KeyError) as error:
                sem = _finding(Status.CONFLICT, str(error))
            attr = _finding(Status.UNKNOWN, 'NATIVE_SCHEMA_HAS_NO_ASSIGNMENT_TOKEN',
                            native_submission_status=sub.status.value, native_attempt_token=None,
                            unverified_native_fields=['assignment_token', 'exact_post_step_history_boundary', 'executor_cessation'])
            # Research COMMITTED payloads are declarations, never a native token source.
            try:
                if action is not None:
                    for r in records:
                        if (r['kind'] == 'EXECUTION' and r['payload'].get('assignment_ref') == assignment_ref
                                and r['payload']['status'] == 'VERIFIED'):
                            _require(r['payload']['speech_digest'] == sha256_bytes(event.raw_text.encode('utf-8'))
                                     and sem.status is Status.OBSERVED
                                     and r['payload']['actions'] == json.loads(sem.evidence_json)['actual_full_v1'],
                                     'RESEARCH_VERIFIED_EXECUTION_NATIVE_CONFLICT')
                for r in records:
                    p = r['payload']
                    if action is not None and r['kind'] == 'COMMIT_WITNESS' and p.get('assignment_ref') == assignment_ref:
                        if p['status'] == 'COMMITTED':
                            x = p['external']
                            _require(x['action_id'] == action.action_id and x['event_id'] == event.event_id
                                     and sem.status is Status.OBSERVED
                                     and x['annotation_digest'] == json.loads(sem.evidence_json)['annotation_digest'],
                                     'RESEARCH_COMMIT_CLAIM_NATIVE_CONFLICT')
                        elif p['status'] == 'NOT_COMMITTED':
                            raise ValueError('RESEARCH_NOT_COMMITTED_VS_NATIVE_SUBMISSION')
            except (TypeError, ValueError, KeyError) as error:
                attr = _finding(Status.CONFLICT, str(error))
            return result(sub, attr, sem, day, end)
        except (TypeError, ValueError, KeyError, StopIteration) as error:
            conflict = _finding(Status.CONFLICT, str(error))
            return result(conflict, conflict, day=day, end=end)
    except (TypeError, ValueError, KeyError, StopIteration) as error:
        conflict = _finding(Status.CONFLICT, str(error))
        return result(*([conflict]*5))
