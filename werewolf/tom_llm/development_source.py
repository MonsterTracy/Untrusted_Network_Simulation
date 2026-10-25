"""Mainline development identity; never a legacy Candidate/admission identity."""

import os
from pathlib import Path
import subprocess

from werewolf.artifact_io import sha256_bytes
from werewolf.tom_llm.common import digest, require

SOURCE_VERSION = "classic7_llm_mainline_development_source_v1"
INTEGRATION_BASE = "24950337cd307d4bc900cf41b5193907cde505ae"


def development_source_identity():
    """Bind actual HEAD/tree and exact working bytes, including uncommitted code.

    Only Mock engineering artifacts may use this contract. Committing, changing
    a runtime file or resuming under another source is an identity change, not
    an automatic upgrade. The old d839+overlay contract remains separate.
    """
    root = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], env=env,
                                       stderr=subprocess.PIPE).decode().strip()

    require(Path(git("rev-parse", "--show-toplevel")).resolve() == root, "source root differs from checkout")
    head = git("rev-parse", "HEAD")
    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", INTEGRATION_BASE, head],
                            env=env, capture_output=True)
    require(result.returncode == 0, "Mainline source must descend from the explicit S4 integration base")
    assets = root / "werewolf/tom_llm/assets"
    require({p.name for p in assets.iterdir()} == {"system.txt", "chat_template.jinja"},
            "development source accepts only the two reviewed prompt assets")
    paths = sorted((root / "werewolf").rglob("*.py")) + sorted(assets.iterdir())
    paths += [root / "configs/paper/tom-llm-reasoner-v1.json",
              root / "configs/paper/tom-llm-wire-server-fixed-v2.json",
              root / "setup.py", root / "scripts/paper_backbone_contrasts.py"]
    files = {}
    for path in paths:
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                "source inventory cannot traverse symlinks")
        files[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *sorted(files))
    return {"schema_version": SOURCE_VERSION, "integration_base_commit": INTEGRATION_BASE,
            "source_revision": head, "commit_tree": git("rev-parse", "HEAD^{tree}"),
            "implementation_digest": digest(files), "file_digests": files,
            "source_inventory_clean": not bool(status), "source_status_digest": digest(status),
            "engineering_only": True, "qualification": "MOCK_ONLY_NOT_CANDIDATE"}
