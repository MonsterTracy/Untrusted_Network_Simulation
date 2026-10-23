"""Spawn targets with no model/torch/Agent dependency; artificial PREs only."""
import json
import os
from pathlib import Path
import sys
import time


def roundtrip(lease, records, output):
    from werewolf.canonical_collection.game_bundle import _prefix_from_record
    rows=[]
    try:
        for i,record in enumerate(records):
            status,result=lease.predict(_prefix_from_record(record),request_id=str(i),timeout_s=5.)
            rows.append({'admission':status,'status':result.status,'matrix':result.matrix.values})
        forbidden=('torch','werewolf.tom.dataset','werewolf.tom.qwen3_final','werewolf.observer_dtrs.shadow')
        assert not any(n in sys.modules for n in forbidden)
        Path(output).write_text(json.dumps({'rows':rows,'restricted_imports':False}))
    finally:
        lease.close()


def disappear_after_dispatch(lease, record):
    from werewolf.observer_dtrs.final_tom_broker import _message
    from werewolf.observer_dtrs.final_tom_wire import Channel
    Channel(lease.sock,lease.limits).write(_message('QUERY',lease.binding,'lost',
        {'pre':record,'deadline_monotonic':time.monotonic()+3.}),time.monotonic()+1.)
    time.sleep(.08)
    os._exit(23)
