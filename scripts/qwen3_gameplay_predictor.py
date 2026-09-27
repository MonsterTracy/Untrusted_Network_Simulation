"""One sealed Qwen3 PRE predictor across a clean-checkout JSONL process boundary."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import select
import subprocess
import sys


SOURCE_REVISION = "f199a9e6c64168a412de91ff8ca04d0c795c57e2"
FIT_DIGEST = "87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248"
TERMINAL_DIGEST = "b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af"
MODEL_DIGEST = "fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9"
SEAL_DIGEST = "fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718"
_TIMEOUT_SECONDS = 600


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _decode(line):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate predictor protocol key")
            result[key] = value
        return result
    value = json.loads(line, object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if _wire(value) != line:
        raise ValueError("noncanonical predictor protocol line")
    return value


def _clean_checkout(checkout):
    if Path.cwd().resolve() != checkout:
        raise RuntimeError("predictor worker cwd differs from pinned checkout")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args],
                                       env=env, text=True, stderr=subprocess.PIPE).strip()
    if (Path(git("rev-parse", "--show-toplevel")).resolve() != checkout
            or git("rev-parse", "HEAD") != SOURCE_REVISION
            or git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")):
        raise RuntimeError("Qwen3 predictor requires the clean pinned source checkout")


def _serve(checkout, fit_path):
    checkout = Path(checkout).resolve()
    stdout = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        _clean_checkout(checkout)
        import run_random
        from werewolf.tom import qwen3_final
        from werewolf.artifact_io import verify_artifact
        from werewolf.canonical_collection.game_bundle import _prefix_from_record
        from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix

        if (Path(run_random.__file__).resolve() != checkout / "run_random.py"
                or Path(qwen3_final.__file__).resolve() != checkout / "werewolf/tom/qwen3_final.py"):
            raise RuntimeError("Qwen3 predictor imports do not originate in pinned checkout")
        fit = qwen3_final.open_fit(Path(fit_path).resolve())
        if fit.digest != FIT_DIGEST:
            raise RuntimeError("Qwen3 final fit identity mismatch")
        predictor = qwen3_final.SealedQwen3Predictor(fit)
        if (predictor.seal.manifest_digest != SEAL_DIGEST
                or predictor.seal.manifest["terminal_digest"] != TERMINAL_DIGEST):
            raise RuntimeError("Qwen3 final seal or terminal identity mismatch")
        terminal = verify_artifact(fit.runs_path / "terminal",
                                   expected_artifact_type="qwen3_final_terminal",
                                   expected_schema_version=qwen3_final.TERMINAL_VERSION)
        if terminal.manifest_digest != TERMINAL_DIGEST or terminal.manifest["model_digest"] != MODEL_DIGEST:
            raise RuntimeError("Qwen3 final model identity mismatch")
    stdout.write(_wire({"status": "ready", "source_revision": SOURCE_REVISION,
                        "fit_digest": FIT_DIGEST, "seal_digest": SEAL_DIGEST}))
    stdout.flush()
    for line in sys.stdin.buffer:
        request = _decode(line)
        if (set(request) != {"request_id", "fit_digest", "seal_digest", "prefix"}
                or type(request["request_id"]) is not int or request["request_id"] < 0
                or request["fit_digest"] != FIT_DIGEST or request["seal_digest"] != SEAL_DIGEST):
            raise ValueError("Qwen3 predictor request identity mismatch")
        prefix_record = request["prefix"]
        prefix = validate_authoritative_pre_prefix(_prefix_from_record(prefix_record))
        if prefix.to_record() != prefix_record:
            raise ValueError("Qwen3 predictor PRE record failed canonical round-trip")
        with redirect_stdout(sys.stderr):
            probabilities = predictor.log_probabilities(prefix).exp().tolist()
        stdout.write(_wire({"request_id": request["request_id"],
                            "prefix_digest": prefix.prefix_digest,
                            "fit_digest": FIT_DIGEST, "seal_digest": SEAL_DIGEST,
                            "probabilities": probabilities}))
        stdout.flush()


class Qwen3GameplayPredictorClient:
    fit_digest = FIT_DIGEST
    seal_digest = SEAL_DIGEST

    def __init__(self, *, checkout, fit_path, python_executable=sys.executable):
        checkout = Path(checkout).resolve()
        fit_path = Path(fit_path).resolve()
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["PYTHONPATH"] = str(checkout)
        env["PYTHONNOUSERSITE"] = "1"
        self._process = subprocess.Popen(
            [str(python_executable), str(Path(__file__).resolve()), "--worker",
             "--checkout", str(checkout), "--fit", str(fit_path)],
            cwd=checkout, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._next_request = 0
        try:
            if self._read(timeout_seconds=None) != {"status": "ready", "source_revision": SOURCE_REVISION,
                                "fit_digest": FIT_DIGEST, "seal_digest": SEAL_DIGEST}:
                raise RuntimeError("Qwen3 predictor startup identity mismatch")
        except Exception:
            self.close()
            raise

    def _read(self, *, timeout_seconds):
        if not select.select([self._process.stdout], [], [], timeout_seconds)[0]:
            raise RuntimeError("Qwen3 predictor response timed out")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen3 predictor terminated without a response")
        return _decode(line)

    def predict(self, prefix):
        request_id = self._next_request
        self._next_request += 1
        request = {"request_id": request_id, "fit_digest": FIT_DIGEST,
                   "seal_digest": SEAL_DIGEST, "prefix": prefix.to_record()}
        try:
            self._process.stdin.write(_wire(request))
            self._process.stdin.flush()
            response = self._read(timeout_seconds=_TIMEOUT_SECONDS)
            if (set(response) != {"request_id", "prefix_digest", "fit_digest",
                                     "seal_digest", "probabilities"}
                    or response["request_id"] != request_id
                    or response["prefix_digest"] != prefix.prefix_digest
                    or response["fit_digest"] != FIT_DIGEST
                    or response["seal_digest"] != SEAL_DIGEST):
                raise ValueError("Qwen3 predictor response identity mismatch")
            return response["probabilities"]
        except Exception:
            self.close()
            raise

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        self._process.stdin.close()
        self._process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_errors):
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--fit", required=True)
    args = parser.parse_args()
    try:
        _serve(args.checkout, args.fit)
    except Exception as error:
        print(f"Qwen3 predictor failed: {error}", file=sys.stderr)
        sys.exit(1)
