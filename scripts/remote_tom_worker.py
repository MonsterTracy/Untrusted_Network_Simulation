"""Copied adapter: run with -m scripts.remote_tom_worker in detached frozen cwd."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys

from scripts.remote_tom_protocol import (
    PLANNER_BASELINE_COMMIT, REMOTE_TOM_PROTOCOL_VERSION, WorkerIdentity,
    RemoteToMError, canonical_json_bytes, sha256_bytes, decode_json_line,
    encode_json_line, validate_request,
)

ADAPTER_FILES = ('scripts/remote_tom_protocol.py', 'scripts/remote_tom_worker.py')


def adapter_digest(root):
    """SHA256 of filename NUL byte-length ':' raw bytes, in ADAPTER_FILES order."""
    digest = hashlib.sha256()
    for name in ADAPTER_FILES:
        content = (Path(root) / name).read_bytes()
        digest.update(name.encode('utf-8') + b'\0' + str(len(content)).encode('ascii') + b':' + content)
    return digest.hexdigest()


class Worker:
    """Only shared validation and existing planner dispatch; dependencies aid tests."""
    def __init__(self, identity, consumer, planner, derive_order, queue_rule_version, capacity):
        self.identity = identity
        self.consumer = consumer
        self.planner = planner
        self.derive_order = derive_order
        self.queue_rule_version = queue_rule_version
        self.capacity = capacity

    def handle(self, request):
        validate_request(request, self.identity)
        if (request['candidate_order_version'] != self.planner.CANDIDATE_ORDER_VERSION
                or request['public_queue_rule_version'] != self.queue_rule_version):
            raise RemoteToMError('public rule version mismatch')
        # Official pinned canonical bundle reader; no second PRE interpretation.
        from werewolf.canonical_collection.game_bundle import _prefix_from_record
        from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
        parent = validate_authoritative_pre_prefix(_prefix_from_record(request['parent_pre']))
        if canonical_json_bytes(parent.to_record()) != canonical_json_bytes(request['parent_pre']):
            raise RemoteToMError('noncanonical PRE wire record')
        actual = (parent.game_id, parent.boundary_id, parent.prefix_digest, parent.current_speaker,
                  parent.public_temporal_state.day, parent.public_temporal_state.phase.value)
        if actual != tuple(request[k] for k in ('game_id', 'boundary_id', 'parent_digest', 'speaker', 'day', 'phase')):
            raise RemoteToMError('PRE opportunity mismatch')
        wolves = tuple(request['alive_wolves'])
        if (any(p not in parent.alive_observer_ids for p in wolves)
                or parent.current_speaker not in wolves
                or set(wolves) == set(parent.alive_observer_ids)):
            raise RemoteToMError('invalid living objective population')
        order = self.derive_order(parent)
        if parent.current_speaker == order[-1]:
            raise RemoteToMError('unsupported last speaker')
        from werewolf.tom.final_capacity import validate_final_pre
        plan = validate_final_pre(parent, self.capacity)
        if plan.token_count + 3 > self.capacity['max_seq_len']:
            raise RemoteToMError('counterfactual exceeds frozen capacity')
        candidates = self.planner.generate_candidates(parent)
        if [plan.public_payload() for plan in candidates] != request['candidates']:
            raise RemoteToMError('canonical candidate tuple mismatch')
        rows = self.planner.evaluate_candidates(parent, candidates, self.consumer, alive_wolves=wolves)
        selected = self.planner.select_minimum_suspicion(candidates, rows)
        index = candidates.index(selected)
        return dict(protocol_version=REMOTE_TOM_PROTOCOL_VERSION, request_id=request['request_id'],
            request_digest=request['request_digest'], status='ok', selected_candidate_index=index,
            selected_plan=candidates[index].public_payload(), worker_identity=asdict(self.identity))


def bootstrap(experiment_path, condition):
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError('worker cwd must be its frozen checkout')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != PLANNER_BASELINE_COMMIT:
        raise RuntimeError('worker requires the fixed detached planner baseline')
    if subprocess.run(['git', '-C', str(root), 'symbolic-ref', '-q', 'HEAD'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('worker checkout must be detached')
    protected = ('werewolf', 'run_random.py', 'scripts/counterfactual_tom.py',
                 'scripts/speech_planning.py', 'scripts/suspicion_objectives.py')
    if git('diff', 'HEAD', '--', *protected):
        raise RuntimeError('frozen planner sources were modified')
    # All imports resolve normally inside THIS checkout. No path manipulation.
    from werewolf.tom.final_experiment import open_final_experiment
    from werewolf.development_publication import open_publication
    from scripts import counterfactual_tom as cf, speech_planning as planner
    experiment = open_final_experiment(experiment_path)
    publication = open_publication(experiment.manifest['publication_path'])
    if (publication.manifest_digest != experiment.manifest['publication_digest']
            or publication.public_view.publication_id != experiment.manifest['publication_id']
            or sorted(publication.public_view.game_ids) != experiment.manifest['game_ids']):
        raise RuntimeError('final development publication binding mismatch')
    consumer = cf.CounterfactualToMConsumer(experiment, condition)
    runtime = experiment.manifest['runtime']  # validated by the original consumer loader
    provenance = consumer.provenance
    identity = WorkerIdentity(PLANNER_BASELINE_COMMIT, adapter_digest(root),
        sha256_bytes(canonical_json_bytes(runtime)), runtime['implementation_digest'],
        experiment.digest, provenance.parent_final_seal_digest, provenance.checkpoint_digest, condition)
    return Worker(identity, consumer, planner, cf.derive_public_phase_speaker_order, cf.QUEUE_RULE_VERSION,
                  experiment.manifest['protocol_inputs']['capacity'])


def serve(worker, stdin, stdout, stderr):
    for line in stdin:
        request = None
        try:
            request = decode_json_line(line)
            # Frozen imports/planner diagnostics must never corrupt protocol stdout.
            with redirect_stdout(stderr):
                response = worker.handle(request)
        except Exception as error:
            response = dict(protocol_version=REMOTE_TOM_PROTOCOL_VERSION, status='error',
                error_code='REQUEST_FAILED', error_message='request validation or planning failed',
                worker_identity=asdict(worker.identity))
            # Echo an ID only after complete shared request binding validation.
            try:
                validate_request(request, worker.identity)
                response['request_id'] = request['request_id']
            except Exception:
                pass
            print(type(error).__name__, file=stderr)
        stdout.write(encode_json_line(response))
        stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--condition', choices=('implicit', 'explicit_day_phase'), required=True)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            worker = bootstrap(args.experiment, args.condition)
    except Exception as error:
        print(f'worker bootstrap failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(encode_json_line({'worker_identity': asdict(worker.identity)}).strip(), file=sys.stderr)
    serve(worker, sys.stdin, sys.stdout, sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
