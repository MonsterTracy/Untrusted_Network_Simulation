from copy import deepcopy
from pathlib import Path
import fcntl
import json

import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from werewolf.tom_llm.selection import select_games, validate_selection, validate_folds
from werewolf.tom_llm.common import digest
from werewolf.tom_llm.study import make_request, prepare_mock_study, context_preflight, open_study
from werewolf.tom_llm.inference import MockBackend, Reply, TransportFailure, InfrastructureFailure
from werewolf.tom_llm.inference import run_mock_study, seal_mock_study, open_mock_predictions


def folds():
    games = [f"game-{i:04d}" for i in range(1500)]
    return [{"fold": f, "held_out_game_ids": games[f*300:(f+1)*300],
             "training_game_ids": [g for i, g in enumerate(games) if i//300 != f]} for f in range(5)]


def uniform(request):
    seats = request["response_format"]["json_schema"]["schema"]["properties"]["probabilities"]["required"]
    return Reply(json.dumps({"probabilities": {s:1/6 for s in seats}}), "stop",
                 {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})


def prepared(tmp_path, *, observer=0):
    root = tmp_path / "mock"
    request = make_request(_prefix(), observer, fold=0)
    prepare_mock_study(root, [request], tokenizer_identity="synthetic:counter-v1", count_tokens=lambda _:100)
    return root, request


def test_fixed_thirty_whole_games_per_fold_and_order_invariance():
    partition = folds()
    subset = select_games(partition, scope="subset150")
    assert len(subset["game_ids"]) == 150
    assert all(len(r["game_ids"]) == 30 for r in subset["folds"])
    assert validate_selection(subset, partition) == subset
    reversed_partition = deepcopy(partition)
    for row in reversed_partition:
        row["held_out_game_ids"].reverse()
        row["training_game_ids"].reverse()
    second = select_games(reversed_partition, scope="subset150")
    assert second["game_ids"] == subset["game_ids"]
    assert second["parent_folds_digest"] != subset["parent_folds_digest"]
    assert len(select_games(partition, scope="full1500")["game_ids"]) == 1500
    subset["game_ids"][0] = "foreign"
    with pytest.raises(ValueError):
        validate_selection(subset, partition)


@pytest.mark.parametrize("change", ["overlap", "leak", "omit", "fold", "too_small"])
def test_fold_and_selection_corruption_rejected(change):
    partition = folds()
    if change == "overlap":
        partition[1]["held_out_game_ids"][0] = partition[0]["held_out_game_ids"][0]
    elif change == "leak":
        partition[0]["training_game_ids"].append(partition[0]["held_out_game_ids"][0])
    elif change == "omit":
        partition[0]["training_game_ids"].pop()
    elif change == "fold":
        partition[0]["fold"] = True
    else:
        partition[0]["held_out_game_ids"].pop()
    with pytest.raises(ValueError):
        select_games(partition, scope="subset150")


def test_context_preflight_never_truncates_or_dispatches(tmp_path):
    request = make_request(_prefix(), 0, fold=0)
    assert context_preflight([request], lambda _:7680)["maximum_input_tokens"] == 7680
    with pytest.raises(ValueError, match="no truncation"):
        context_preflight([request], lambda _:7681)
    with pytest.raises(ValueError, match="mock tokenizer"):
        prepare_mock_study(tmp_path, [request], tokenizer_identity="real", count_tokens=lambda _:100)


def test_durable_mock_success_seal_and_no_terminal_regeneration(tmp_path):
    root, request = prepared(tmp_path)
    backend = MockBackend(uniform)
    run_mock_study(root, backend)
    run_mock_study(root, backend)
    assert len(backend.calls) == 1
    first = seal_mock_study(root)
    assert seal_mock_study(root).manifest_digest == first.manifest_digest
    seal, rows = open_mock_predictions(root)
    assert seal.manifest["engineering_only"] is True
    assert rows[0]["prediction"]["status"] == "valid"
    assert rows[0]["fold"] == 0 and rows[0]["request_digest"] != rows[0]["prompt_digest"]
    with pytest.raises(ValueError, match="closes"):
        run_mock_study(root, backend)


def test_invalid_response_fallback_has_no_semantic_retry(tmp_path):
    root, _ = prepared(tmp_path)
    backend = MockBackend(lambda _:Reply("bad JSON", "stop", {}))
    run_mock_study(root, backend)
    seal_mock_study(root)
    _, rows = open_mock_predictions(root)
    assert len(backend.calls) == 1 and rows[0]["prediction"]["status"] == "fallback_uniform"


def test_exact_transport_retry_is_audited(tmp_path):
    root, _ = prepared(tmp_path)
    calls = []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise TransportFailure()
        return uniform(request)
    run_mock_study(root, MockBackend(respond))
    assert len(calls) == 2 and calls[0] == calls[1]
    seal_mock_study(root)
    _, rows = open_mock_predictions(root)
    assert len(rows[0]["attempt_digests"]) == 2


def test_exhausted_transport_never_fabricates_uniform(tmp_path):
    root, _ = prepared(tmp_path)
    def broken(_):
        raise TransportFailure()
    backend = MockBackend(broken)
    for _ in range(2):
        with pytest.raises(InfrastructureFailure, match="budget exhausted"):
            run_mock_study(root, backend)
    assert len(backend.calls) == 2
    assert not list(root.rglob("result.json"))
    with pytest.raises((ValueError, OSError)):
        seal_mock_study(root)


def test_durable_response_crash_recovers_without_new_generation(tmp_path):
    root, _ = prepared(tmp_path)
    backend = MockBackend(uniform)
    def crash():
        raise RuntimeError("after durable response")
    with pytest.raises(RuntimeError):
        run_mock_study(root, backend, after_response=crash)
    run_mock_study(root, backend)
    assert len(backend.calls) == 1
    seal_mock_study(root)


def test_unknown_inflight_outcome_fails_closed(tmp_path):
    root, _ = prepared(tmp_path)
    def broken(_):
        raise InfrastructureFailure("OOM")
    backend = MockBackend(broken)
    with pytest.raises(InfrastructureFailure):
        run_mock_study(root, backend)
    with pytest.raises(InfrastructureFailure, match="in-flight"):
        run_mock_study(root, backend)
    assert len(backend.calls) == 1 and not list(root.rglob("result.json"))


def test_live_objects_and_concurrent_writers_are_rejected(tmp_path):
    root, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="only local"):
        run_mock_study(root, object())
    with (root/"inference.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_mock_study(root, MockBackend(uniform))


def test_modified_source_or_result_cannot_resume(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path)
    from werewolf.tom_llm import study
    original = study.development_source_identity()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "development_source_identity", lambda: {**original, "implementation_digest":"0"*64})
        with pytest.raises(ValueError, match="drift"):
            open_study(root)
    run_mock_study(root, MockBackend(uniform))
    target = next(root.rglob("result.json"))
    target.write_text(target.read_text().replace('"status":"valid"', '"status":"fallback_uniform"'))
    with pytest.raises(ValueError):
        seal_mock_study(root)


def test_selection_artifact_is_immutable(tmp_path):
    from werewolf.tom_llm.selection import publish_selection
    partition=folds()
    artifact=publish_selection(tmp_path/"selection",partition,scope="subset150")
    assert artifact.manifest["scope"]=="subset150"
    assert artifact.manifest["selection_digest"]==select_games(partition,scope="subset150")["selection_digest"]
    with pytest.raises(ValueError,match="already exists"):
        publish_selection(tmp_path/"selection",partition,scope="subset150")


def test_multiple_observer_requests_never_include_previous_outputs(tmp_path):
    root=tmp_path/"two-observers"
    requests=[make_request(_prefix(),observer,fold=0) for observer in (0,1)]
    prepare_mock_study(root,requests,tokenizer_identity="synthetic:counter-v1",count_tokens=lambda _:100)
    backend=MockBackend(uniform)
    run_mock_study(root,backend)
    assert len(backend.calls)==2
    assert all([m["role"] for m in call["messages"]]==["system","user"] for call in backend.calls)
    assert backend.calls[0]["messages"][0]==backend.calls[1]["messages"][0]


def test_missing_unsealed_result_cannot_be_consumed(tmp_path):
    root,_=prepared(tmp_path)
    with pytest.raises((ValueError,OSError)):
        open_mock_predictions(root)


def test_reply_usage_and_failure_are_bound_in_sealed_evidence(tmp_path):
    root,_=prepared(tmp_path)
    run_mock_study(root,MockBackend(uniform)); seal_mock_study(root)
    _,rows=open_mock_predictions(root)
    assert rows[0]["usage"]=={"input_tokens":100,"output_tokens":50,"total_tokens":150}
    assert len(rows[0]["attempt_digests"])==1
    assert len(rows[0]["dispatch_elapsed_seconds"])==1
    assert rows[0]["dispatch_elapsed_seconds"][0]>=0
    saved=rows[0]["dispatch_elapsed_seconds"]
    _,again=open_mock_predictions(root)
    assert again[0]["dispatch_elapsed_seconds"]==saved
