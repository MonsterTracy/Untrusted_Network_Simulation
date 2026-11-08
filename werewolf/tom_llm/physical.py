"""运行开始的轻量资产检查；不读取大权重内容或发布资格 Receipt。"""

from pathlib import Path

from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.model_spec import ROOT
from werewolf.artifact_io import sha256_bytes

ASSET_PATH = ROOT/'configs/paper/tom-llm-physical-assets-v1.json'


def verify_model(spec, directory):
    """沿用历史已核验资产：登记版本、分片覆盖及 stat，不冒充新的完整 SHA 核验。"""
    root = Path(directory).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)),'model directory missing/symlink')
    expected = strict_json(ASSET_PATH.read_bytes())[spec.key]
    require(expected['revision']==spec.record()['revision']
            and (root/'HF_REVISION').read_text()==expected['revision']+'\n','registered model revision mismatch')
    files = expected['files']
    index = strict_json((root/'model.safetensors.index.json').read_bytes())
    require(set(index['weight_map'].values())=={n for n in files if n.endswith('.safetensors')},
            'model shard/index mismatch')
    stats = {}
    for name,wanted in files.items():
        path = root/name
        require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),
                'model asset missing/symlink')
        stat = path.stat()
        require(stat.st_size==wanted['bytes'],'model asset size mismatch')
        if not name.endswith('.safetensors') and wanted.get('sha256'):
            require(sha256_bytes(path.read_bytes())==wanted['sha256'],'small model/tokenizer asset identity drift')
        stats[name] = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
    return {'directory':str(root),'revision':expected['revision'],'files':stats,
            'verification':'revision_index_and_stat','full_weight_sha_verified':False,
            'registered_asset_spec_digest':digest(expected),'weights_loaded':False}
