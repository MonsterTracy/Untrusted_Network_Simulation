"""Mocked Git and temporary synthetic repositories; real repository history is untouched."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from werewolf.tom import paper_study as m

ROOT=Path(m.__file__).resolve().parents[2]
REV='a'*40


def mock_git(monkeypatch,*,head=REV,status=''):
    calls=[]
    def check_output(command,**kwargs):
        args=command[3:]; calls.append(args)
        assert command[:3]==['git','-C',str(ROOT)]
        assert not any(k.startswith('GIT_') for k in kwargs['env'])
        if args==['rev-parse','--show-toplevel']: return str(ROOT)+'\n'
        if args==['rev-parse','HEAD']: return head+'\n'
        if args[0]=='status': return status
        if args[0]=='ls-files': return ''
        pytest.fail(str(args))
    monkeypatch.setattr(m,'subprocess',SimpleNamespace(check_output=check_output,
        PIPE=subprocess.PIPE, CalledProcessError=subprocess.CalledProcessError))
    return calls


@pytest.mark.parametrize('status',[' M tracked.py','M  staged.py','?? untracked.py',' M submodule'])
def test_source_attestation_dirty_worktree_rejected(monkeypatch,status):
    mock_git(monkeypatch,status=status)
    with pytest.raises(ValueError,match='clean'): m.attest_source()


def test_source_attestation_unavailable(monkeypatch):
    mock_git(monkeypatch)
    def missing(*args,**kwargs): raise FileNotFoundError('git')
    monkeypatch.setattr(m.subprocess,'check_output',missing)
    with pytest.raises(ValueError,match='unavailable'): m.attest_source()
    mock_git(monkeypatch,head='config-string')
    with pytest.raises(ValueError,match='full commit'): m.attest_source()


def test_clean_matching_head_preserves_implementation_digest(monkeypatch):
    from werewolf.tom.experiment import runtime_provenance
    calls=mock_git(monkeypatch)
    monkeypatch.setenv('GIT_DIR','/incorrect')
    result=m.attest_source()
    runtime=runtime_provenance(SimpleNamespace(source_revision=REV,device='cpu',torch_num_threads=1))
    assert result=={'source_revision':REV,'implementation_digest':runtime['implementation_digest']}
    assert ['status','--porcelain=v1','--untracked-files=all','--ignore-submodules=none'] in calls


def test_contract_preparation_and_reopen(tmp_path,monkeypatch):
    mock_git(monkeypatch)
    config=ROOT/'configs/paper/tom-study-v1.json'
    artifact=m.prepare_contract(config,tmp_path/'contract')
    assert m.open_contract(artifact.path).manifest_digest==artifact.manifest_digest
    value=artifact.manifest['contract']
    assert value['terminal_lineages']==20
    assert set(value['effects'])=={'Delta_uniform','Delta_ToM','Delta_temp','Delta_stress'}
    assert value['effects']['Delta_ToM']['temporal_conditions']=='both_required'
    assert value['bootstrap']['replicates']==10000
    mock_git(monkeypatch,status='?? new.py')
    with pytest.raises(ValueError,match='clean'): m.open_contract(artifact.path)


def test_contract_refuses_changed_design_and_dirty_preparation(tmp_path,monkeypatch):
    config=tmp_path/'config.json'
    changed=json.loads(json.dumps(m.CONTRACT)); changed['fold_count']=4
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='frozen design'): m.prepare_contract(config,tmp_path/'bad')
    mock_git(monkeypatch,status=' M changed.py')
    with pytest.raises(ValueError,match='clean'):
        m.prepare_contract(ROOT/'configs/paper/tom-study-v1.json',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_formal_cli_contract_path(tmp_path,monkeypatch,capsys):
    from werewolf.cli import main
    mock_git(monkeypatch)
    profile=tmp_path/'storage.json'; profile.write_text(json.dumps({'artifact_root':str(tmp_path)}))
    assert main(['--storage-profile',str(profile),'prepare-paper-study-contract',
        '--config',str(ROOT/'configs/paper/tom-study-v1.json'),
        '--destination','contract'])==0
    assert len(capsys.readouterr().out.strip())==64


@pytest.fixture
def synthetic_git(tmp_path,monkeypatch):
    import os
    repo=tmp_path/'repo'; repo.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],env=env,stderr=subprocess.PIPE,text=True).strip()
    git('init')
    git('config','user.name','Synthetic Fixture')
    git('config','user.email','fixture@example.invalid')
    git('config','commit.gpgsign','false')
    (repo/'werewolf/tom').mkdir(parents=True)
    (repo/'werewolf/tom/paper_study.py').write_text('# synthetic source\n')
    (repo/'run_random.py').write_text('# synthetic entry\n')
    (repo/'.gitignore').write_text('ignored.txt\n')
    config=repo/'study.json'; config.write_text(json.dumps(m.CONTRACT))
    git('add','study.json','.gitignore','run_random.py','werewolf/tom/paper_study.py')
    git('commit','-m','Synthetic initial source')
    monkeypatch.setattr(m,'__file__',str(repo/'werewolf/tom/paper_study.py'))
    return repo,config,git


def test_policy_only_config_and_actual_head_capture(synthetic_git,tmp_path):
    from werewolf.artifact_io import ArtifactConflictError
    repo,config,git=synthetic_git
    value=json.loads(config.read_text())
    assert value['source_binding']=='clean_git_head_at_study_prepare_v1'
    assert 'source_revision' not in value
    assert git('rev-parse','HEAD') not in config.read_text()
    artifact=m.prepare_contract(config,tmp_path/'contract')
    assert artifact.manifest['source']['source_revision']==git('rev-parse','HEAD')
    before=(artifact.path/'manifest.json').read_bytes()
    assert m.open_contract(artifact.path).manifest_digest==artifact.manifest_digest
    git('commit','--allow-empty','-m','Synthetic later commit')
    with pytest.raises(ValueError,match='current HEAD'): m.open_contract(artifact.path)
    assert (artifact.path/'manifest.json').read_bytes()==before
    # Re-preparation cannot overwrite the already published immutable artifact.
    with pytest.raises(ArtifactConflictError): m.prepare_contract(config,artifact.path)
    assert (artifact.path/'manifest.json').read_bytes()==before


@pytest.mark.parametrize('change',['tracked','staged','untracked'])
def test_real_git_dirty_rejected_before_publication(synthetic_git,tmp_path,change):
    repo,config,git=synthetic_git
    if change=='untracked': (repo/'new.txt').write_text('untracked')
    else:
        (repo/'run_random.py').write_text('# changed\n')
        if change=='staged': git('add','run_random.py')
    with pytest.raises(ValueError,match='clean'): m.prepare_contract(config,tmp_path/'contract')
    assert not (tmp_path/'contract').exists()


def test_real_git_ignored_file_is_clean(synthetic_git,tmp_path):
    repo,config,git=synthetic_git
    (repo/'ignored.txt').write_text('ignored scratch file')
    assert git('status','--porcelain=v1','--untracked-files=all')==''
    artifact=m.prepare_contract(config,tmp_path/'contract')
    assert m.open_contract(artifact.path).manifest_digest==artifact.manifest_digest


def test_git_failure_does_not_publish(synthetic_git,tmp_path,monkeypatch):
    repo,config,git=synthetic_git
    def unavailable(*args,**kwargs): raise FileNotFoundError('git unavailable')
    monkeypatch.setattr(m,'subprocess',SimpleNamespace(check_output=unavailable,
        PIPE=subprocess.PIPE,CalledProcessError=subprocess.CalledProcessError))
    with pytest.raises(ValueError,match='unavailable'): m.prepare_contract(config,tmp_path/'contract')
    assert not (tmp_path/'contract').exists()


def test_rehashed_implementation_digest_mismatch_rejected(synthetic_git,tmp_path):
    from werewolf.artifact_io import publish_artifact
    repo,config,git=synthetic_git
    artifact=m.prepare_contract(config,tmp_path/'contract')
    fields={k:v for k,v in artifact.manifest.items() if k not in ('manifest_digest','file_table')}
    fields['source']={**fields['source'],'implementation_digest':'0'*64}
    corrupted=publish_artifact(tmp_path/'corrupted',manifest_fields=fields,files={})
    with pytest.raises(ValueError,match='implementation digest mismatch'): m.open_contract(corrupted.path)


def test_preparation_cli_has_no_revision_override():
    from werewolf.cli import build_parser
    parser=build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['prepare-paper-study-contract','--config','study.json',
                           '--destination','out','--source-revision',REV])
