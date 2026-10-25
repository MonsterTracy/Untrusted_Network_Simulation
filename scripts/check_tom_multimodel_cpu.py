"""Offline tokenizer parity only. Missing assets are NOT_RUN, never inferred/downloaded."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.tom_llm.model_spec import ModelSpec, check_cpu_parity
from werewolf.tom_llm.common import strict_json
from werewolf.tom_llm.task import validate_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, help='public-only task JSONL; no labels/reference objects')
    for key in ('qwen3','yi','mistral'):
        parser.add_argument('--'+key+'-assets')
    args = parser.parse_args()
    tasks = [strict_json(line) for line in Path(args.requests).read_bytes().splitlines()]
    for task in tasks:
        validate_task(task)
    results = {}
    for key in ('qwen3','yi','mistral'):
        directory = getattr(args,key+'_assets')
        results[key] = (check_cpu_parity(tasks,ModelSpec(key),directory) if directory and Path(directory).is_dir() else {
            'status':'NOT_RUN','reason':'explicit local official tokenizer assets required',
            'model_spec_digest':ModelSpec(key).digest,'gpu_admission':False})
    print(json.dumps({'engineering_only':True,'models':results,'physical_admission':False},ensure_ascii=False,sort_keys=True))


if __name__ == '__main__':
    main()
