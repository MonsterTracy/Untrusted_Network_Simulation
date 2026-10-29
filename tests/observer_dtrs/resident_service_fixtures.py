"""Explicit CONTROLLED_FAULT_FIXTURE entry; never a production service option."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import signal
import struct
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser();p.add_argument('--fd', type=int, required=True)
    p.add_argument('--mode', required=True);p.add_argument('--deny-root', default=None)
    args = p.parse_args()
    if args.deny_root:
        root = Path(args.deny_root)
        def guard(event, values):
            if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(values[0])).absolute()
                if path.name in ('primary.json', 'roles.json', 'optimizer.bin', 'rng.bin', 'loss.jsonl'):
                    raise PermissionError('restricted fixture path')
                if any(path.is_relative_to(root/n) for n in ('lifecycle', 'parents', 'predictor')):
                    raise PermissionError('original fixture lifecycle')
        sys.addaudithook(guard)
    if args.mode == 'cold_hang':
        time.sleep(60)
    from werewolf.observer_dtrs import final_tom_service as service
    from werewolf.observer_dtrs.final_tom_wire import Channel
    if args.mode == 'guarded':
        import importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = ('werewolf.development_publication', 'werewolf.tom.dataset', 'werewolf.tom.model',
                    'werewolf.tom.backbone_model', 'werewolf.tom.qwen3_final', 'werewolf.tom.training',
                    'werewolf.tom.experiment', 'werewolf.observer_dtrs.offline',
                    'werewolf.observer_dtrs.final_tom_audit', 'werewolf.observer_dtrs.final_tom_capsule_build',
                    'werewolf.observer_dtrs.dataset_offline', 'werewolf.observer_dtrs.training_audit',
                    'werewolf.observer_dtrs.shadow', 'werewolf.envs', 'werewolf.agents')
                if fullname.startswith(forbidden): raise ImportError('restricted service dependency')
        sys.meta_path.insert(0, Block())
    if args.mode == 'ignore_term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
    def install_predictor_fault():
        import torch
        import numpy as np
        from werewolf.observer_dtrs import final_tom_capsule as cap
        original = cap.CapsulePredictor
        class FixturePredictor(original):
            def predict(self, prefix):
                if args.mode == 'crash' or (args.mode == 'crash_on_game' and prefix.game_id == 'controlled-game'): os._exit(19)
                if args.mode in ('hang', 'ignore_term'):
                    while True: time.sleep(60)
                if args.mode in ('delay', 'late_success', 'service_timeout'): time.sleep(.2)
                py, nps, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                result = super().predict(prefix)
                assert random.getstate() == py and np.array_equal(np.random.get_state()[1], nps[1])
                assert np.random.get_state()[0] == nps[0] and np.random.get_state()[2:] == nps[2:]
                assert torch.equal(torch.get_rng_state(), tr)
                return result
        cap.CapsulePredictor = FixturePredictor
    write, read = Channel.write, Channel.read
    ready = False
    def altered_write(channel, message, deadline):
        nonlocal ready
        message = deepcopy(message)
        if message['kind'] == 'BOOTED':
            result = write(channel, message, deadline)
            # Match production: model imports/loading belong to READY, not cold
            # startup. Install faults before serve() imports CapsulePredictor.
            if args.mode == 'ready_hang': time.sleep(60)
            install_predictor_fault()
            return result
        if message['kind'] == 'READY':
            ready = True
            if args.mode == 'ready_source': message['payload']['session']['capsule']['capsule_digest'] = 'e'*64
        if message['kind'] == 'RESULT':
            if args.mode == 'partial':
                channel.sock.send(struct.pack('!I', 1000)+b'{');return
            if args.mode == 'oversized':
                channel.sock.send(struct.pack('!I', channel.limits.max_frame_bytes+1));return
            if args.mode in ('wrong_token', 'wrong_epoch', 'wrong_instance'):
                message[{'wrong_token':'token', 'wrong_epoch':'epoch', 'wrong_instance':'instance'}[args.mode]] = 'e'*64
            if args.mode == 'wrong_source': message['payload']['session']['capsule']['capsule_digest'] = 'e'*64
            if args.mode == 'wrong_pre' and message['payload']['prediction']:
                message['payload']['prediction']['game_id'] = 'another-game'
            if args.mode == 'bad_Q' and message['payload']['prediction']:
                rows = message['payload']['prediction']['probabilities']
                rows = [list(row) for row in rows];rows[0][1] = .99
                message['payload']['prediction']['probabilities'] = rows
            if args.mode == 'late_success' and message['payload']['status'] in ('CANCELLED', 'SERVICE_TIMEOUT'):
                late = deepcopy(message);late['payload']['status'] = 'SUCCEEDED'
                write(channel, late, deadline)
        if message['kind'] == 'CLOSED' and args.mode == 'close_hang': time.sleep(60)
        return write(channel, message, deadline)
    def altered_read(channel, deadline):
        if ready and args.mode == 'write_hang': time.sleep(60)
        return read(channel, deadline)
    Channel.write, Channel.read = altered_write, altered_read
    return service.serve(args.fd)


if __name__ == '__main__':
    try: code = main()
    except Exception: code = 2
    os._exit(code)
