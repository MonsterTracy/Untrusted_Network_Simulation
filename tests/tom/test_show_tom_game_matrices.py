"""Real export regression plus explicitly synthetic multi-game/private evidence tests."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import show_tom_game_matrices as show
from scripts import visualize_tom_single_game as viz
from werewolf.canonical_collection.pre import construct_authoritative_pre_prefix
from werewolf.canonical_collection.trajectory_evidence import (
    BackendCallPurpose, BackendCallStatus, construct_backend_call_evidence,
    construct_private_replay_evidence,
)


@pytest.fixture
def real_root():
    value = os.environ.get('UNS_TOM_SINGLE_GAME_REVIEW')
    if not value:
        pytest.skip('set UNS_TOM_SINGLE_GAME_REVIEW for actual export regression')
    return Path(value)


@pytest.fixture
def copied_root(real_root, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(real_root, root)
    return root


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(viz.canonical_json_bytes(value))


def rebind(root):
    """Rehash modified test copies only; never alter authoritative originals."""
    canonical = root / 'canonical_public'
    m = json.loads((canonical / 'manifest.json').read_bytes())
    for name in m['file_table']:
        p = canonical / name
        if name.startswith('public/') and not p.exists():
            p = canonical / Path(name).name
        if p.is_file():
            b = p.read_bytes()
            m['file_table'][name] = {'byte_size': len(b), 'sha256': viz.sha256_bytes(b)}
    m['manifest_digest'] = viz.sha256_bytes(viz.canonical_json_bytes({k:v for k,v in m.items() if k!='manifest_digest'}))
    write_json(canonical / 'manifest.json', m)
    export = root / viz.EXPORT_FOLDER
    summary = json.loads((export / 'summary.json').read_bytes())
    summary['source_bundle_manifest_digest'] = m['manifest_digest']
    write_json(export / 'summary.json', summary)
    manifest = json.loads((export / 'manifest.json').read_bytes())
    manifest['source_bundle_manifest_digest'] = m['manifest_digest']
    for name in manifest['file_table']:
        b = (export / name).read_bytes()
        manifest['file_table'][name] = {'byte_size':len(b), 'sha256':viz.sha256_bytes(b)}
    manifest['manifest_digest'] = viz.sha256_bytes(viz.canonical_json_bytes({k:v for k,v in manifest.items() if k!='manifest_digest'}))
    write_json(export / 'manifest.json', manifest)


def synthetic_second(root):
    """Distinct test identity with copied numeric rows, not a new simulated game."""
    data = viz.load_review(root)
    gid = 'synthetic-test-only-game-000001-seed-1'
    prefixes = []
    for p in data.prefixes:
        links = {v['observer_id']:v['observation_id'] for v in p.to_record()['belief_observation_links']}
        prefixes.append(construct_authoritative_pre_prefix(game_id=gid,boundary_id=p.boundary_id,
            step_index=p.step_index,report_trigger_id=p.report_trigger_id,current_speaker=p.current_speaker,
            alive_observer_ids=p.alive_observer_ids,public_event_history=p.public_event_history,
            v1_annotations=p.v1_annotations,belief_observation_ids_by_observer=links).to_record())
    canonical = root/'canonical_public'
    (canonical/'authoritative_pre_prefixes.jsonl').write_bytes(viz.canonical_jsonl_bytes(prefixes))
    m=json.loads((canonical/'manifest.json').read_bytes())
    m.update(game_id=gid,artifact_identity=gid,seed=1,collection_id='synthetic-test-only',ordinal=1)
    write_json(canonical/'manifest.json',m)
    export=root/viz.EXPORT_FOLDER
    rows=deepcopy(list(data.records))
    for r,p in zip(rows,prefixes): r.update(game_id=gid,prefix_digest=p['prefix_digest'])
    (export/'all_tom_matrices.jsonl').write_bytes(viz.canonical_jsonl_bytes(rows))
    with (export/'tom_cells.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;cells=list(reader)
    for r,p in zip(rows,prefixes):
        for c in cells:
            if c['boundary_id']==r['boundary_id']:c['game_id']=gid
    (export/'tom_cells.csv').write_bytes(viz.csv_bytes(cells,fields))
    for name in ('summary.json','manifest.json'):
        v=json.loads((export/name).read_bytes());v['game_id']=gid;write_json(export/name,v)
    rebind(root)
    return gid


def add_private(root, *, with_call=False, request=True, success=True, purpose=BackendCallPurpose.SPEECH_GENERATION):
    data=viz.load_review(root)
    roles=dict(zip(viz.PLAYER_IDS,('Werewolf','Werewolf','Villager','Villager','Villager','Seer','Witch')))
    e=construct_private_replay_evidence(game_id=data.summary['game_id'],seed=data.bundle_manifest['seed'],
        role_assignment=roles,runtime_configuration={},initial_runtime_state={},replay_inputs={},
        expected_public_event_digest=show._history_from_records(data.events).digest)
    write_json(root/'canonical_public/private/private_replay_evidence.json',e.to_record())
    m=json.loads((root/'canonical_public/manifest.json').read_bytes())
    m['runtime_configuration_digest']=viz.sha256_bytes(e.runtime_configuration.canonical_bytes)
    write_json(root/'canonical_public/manifest.json',m)
    if with_call:
        payload={'args':[show.TOM_MARKER+'synthetic-test-only ToM information']} if request else {'response':show.TOM_MARKER}
        call=construct_backend_call_evidence(call_id='test-call',operation_id='test-op',purpose=purpose,
            status=BackendCallStatus.SUCCESS if success else BackendCallStatus.ERROR,
            boundary_id=data.prefixes[0].boundary_id,observer_id=data.prefixes[0].current_speaker,attempt_index=1,
            backend_identity='test-backend',model_identity='test-model',parser_identity='test-parser',
            prompt_identity='test-prompt',retry_policy_identity='test-retry',call_budget_identity='test-budget',private_payload=payload)
        (root/'canonical_public/private/backend_call_evidence.jsonl').write_bytes(viz.canonical_jsonl_bytes([call.to_record()]))
    rebind(root)


def test_real_discovery_missing_roles_online_unknown_and_vote_counts(real_root):
    games,issues,_=show.discover([real_root])
    assert not issues and len(games)==1
    g=games[0]
    assert g.status=='VERIFIED_PUBLIC' and g.review.q.shape==(17,7,7)
    assert g.roles is None and '无法确定' in g.online_tom and g.evidence_files==2
    rounds=show.vote_rounds(g.review.events)
    assert [dict(r['counts']) for r in rounds]==[{'player1':2,'player3':2},{'player3':6,'player1':1},{'player4':3,'player7':1},{'player7':2,'player5':1}]
    assert [r['abstain'] for r in rounds]==[3,0,1,0]
    assert [r['exiles'] for r in rounds]==[[],['player3'],['player4'],['player7']]
    text=show.game_description(g,1)
    assert '角色信息未提供' in text and 'P2→P3→P4→P5→P6→P7→P1' in text


def test_duplicate_roots_and_identical_copies_are_one_game(real_root,copied_root):
    games,issues,_=show.discover([real_root,real_root,real_root.parent,copied_root])
    assert not issues and len(games)==1 and len(games[0].exports)==2
    assert games[0].status=='VERIFIED_PUBLIC'


def test_canonical_only_is_not_a_tom_game(real_root,tmp_path):
    root=tmp_path/'canonical-only';shutil.copytree(real_root/'canonical_public',root)
    games,issues,protected=show.discover([root])
    assert not games and not issues and root in protected


def test_renamed_export_and_full_public_subdirectory_layout(copied_root):
    (copied_root/viz.EXPORT_FOLDER).rename(copied_root/'arbitrary-export-name')
    canonical=copied_root/'canonical_public';(canonical/'public').mkdir()
    for name in ('authoritative_pre_prefixes.jsonl','public_event_stream.jsonl'):
        (canonical/name).rename(canonical/'public'/name)
    canonical.rename(copied_root/'arbitrary-canonical-name')
    games,issues,_=show.discover([copied_root])
    assert not issues and len(games)==1 and games[0].status=='VERIFIED_PUBLIC'


def test_two_games_and_selectors(copied_root,tmp_path,capsys):
    second=tmp_path/'second';shutil.copytree(copied_root,second);gid=synthetic_second(second)
    games,issues,_=show.discover([tmp_path])
    assert len(games)==2 and not issues and all(g.status=='VERIFIED_PUBLIC' for g in games)
    assert show.select_games(games,selected=[1,2])==[0,1]
    assert games[show.select_games(games,game_id=gid)[0]].game_id==gid
    assert show.main(['--root',str(tmp_path),'--games','1','2','--pre-indices','1','--per-row','2'])==0
    output=capsys.readouterr().out
    assert '游戏1：' in output and '游戏2：' in output
    for kwargs in ({'game':0},{'selected':[1,1]},{'game_id':'missing'}):
        with pytest.raises(ValueError):show.select_games(games,**kwargs)


def test_different_export_content_same_game_id_is_conflict(real_root,copied_root):
    p=copied_root/viz.EXPORT_FOLDER/'summary.json';m=json.loads(p.read_bytes());m['interpretation']='synthetic conflict';write_json(p,m);rebind(copied_root)
    games,_,_=show.discover([real_root,copied_root])
    assert len(games)==1 and games[0].status=='CONFLICT'
    with pytest.raises(ValueError,match='禁止输出'):show.select_games(games,game=1)


def test_canonical_source_conflict_does_not_silently_pick_valid_copy(real_root,copied_root):
    p=copied_root/'canonical_public/manifest.json';m=json.loads(p.read_bytes());m['source_revision']='0'*40;write_json(p,m);rebind(copied_root)
    games,_,_=show.discover([real_root,copied_root])
    assert len(games)==1 and games[0].status=='CONFLICT'


@pytest.mark.parametrize('damage',['csv','missing_csv','missing_canonical','schema','canonical_payload','unreadable_manifest'])
def test_damaged_inputs_not_printed(copied_root,tmp_path,capsys,damage):
    if damage=='csv':
        p=copied_root/viz.EXPORT_FOLDER/'tom_cells.csv';p.write_bytes(p.read_bytes()+b'bad\n')
    elif damage=='missing_csv':(copied_root/viz.EXPORT_FOLDER/'tom_cells.csv').unlink()
    elif damage=='missing_canonical':shutil.rmtree(copied_root/'canonical_public')
    elif damage=='schema':
        p=copied_root/viz.EXPORT_FOLDER/'manifest.json';m=json.loads(p.read_bytes());m['schema_version']='unsupported';write_json(p,m)
    elif damage=='canonical_payload':
        p=copied_root/'canonical_public/public_event_stream.jsonl';p.write_bytes(p.read_bytes()+b'\n')
    else:(copied_root/viz.EXPORT_FOLDER/'manifest.json').write_text('{bad')
    out=tmp_path/'should-not-exist'
    assert show.main(['--root',str(copied_root),'--all','--output-dir',str(out)])==2
    assert not out.exists() and '━━━━━━━━' not in capsys.readouterr().out


def test_optional_role_record_and_online_request_are_bound(copied_root):
    add_private(copied_root,with_call=True)
    g=show.discover([copied_root])[0][0]
    assert g.status=='VERIFIED_PUBLIC' and g.roles['player2']=='Werewolf'
    assert '已确认' in g.online_tom
    p=copied_root/'canonical_public/private/private_replay_evidence.json';p.write_bytes(p.read_bytes()+b'\n')
    assert show.discover([copied_root])[0][0].status=='INVALID'


def test_non_speech_gameplay_request_is_not_online_speech_tom_evidence(copied_root):
    add_private(copied_root, with_call=True, purpose=BackendCallPurpose.GAMEPLAY_ACTION)
    g = show.discover([copied_root])[0][0]
    assert g.status == 'VERIFIED_PUBLIC' and '无法确定' in g.online_tom


@pytest.mark.parametrize('is_request,success',[(False,True),(True,False)])
def test_response_marker_or_failed_call_is_not_online_use(copied_root,is_request,success):
    add_private(copied_root,with_call=True,request=is_request,success=success)
    g=show.discover([copied_root])[0][0]
    assert g.status=='VERIFIED_PUBLIC' and '无法确定' in g.online_tom


def numeric_game():
    q=np.full((3,7,7),1/6)
    for a in q:np.fill_diagonal(a,0)
    q[1:,0,1]+=1e-4;q[1:,0,2]-=1e-4
    q[2,0,1]-=5e-9;q[2,0,2]+=5e-9
    q.setflags(write=False)
    records=[{'day':1,'phase':'discussion','current_speaker':'player1','alive_observer_ids':list(viz.PLAYER_IDS)} for _ in range(3)]
    records[2]['alive_observer_ids'].remove('player2')
    roles=dict(zip(viz.PLAYER_IDS,('Villager','Werewolf','Werewolf','Villager','Villager','Seer','Witch')))
    return show.Game('synthetic-unit',(),(),status='VERIFIED_PUBLIC',roles=roles,
                     review=SimpleNamespace(q=q,records=records))


@pytest.mark.parametrize('ambiguous_width',[1,2])
def test_three_rules_seven_columns_arrows_unrounded_and_dead_seats(ambiguous_width):
    game=numeric_game();before=game.review.q.tobytes()
    panels=[show.matrix_lines(game,1,i,ambiguous_width=ambiguous_width) for i in range(3)]
    width=show.table_width(3,ambiguous_width)
    assert all(show.display_width(line,ambiguous_width)==width for panel in panels for line in panel)
    assert all(sum(line.startswith('━') or line.startswith('─') for line in panel)==3 for panel in panels)
    assert all('P7' in panel[4] for panel in panels)
    assert 'P2狼亡' in panels[2][4] and panels[2][7].startswith('P2亡')
    assert '0.167·' in panels[0][6] and '0.167↑' in panels[1][6] and '0.167↓' in panels[1][6]
    assert '0.167→' in panels[2][6]
    assert '0.000↑' not in '\n'.join(panels[1]) and '0.000·' not in '\n'.join(panels[0])
    assert game.review.q.tobytes()==before


@pytest.mark.parametrize('per_row',[2,3])
def test_real_parallel_layout_complete_width_and_last_column(real_root,per_row):
    g=show.discover([real_root])[0][0]
    before=g.review.q.tobytes()
    text=show.render_game(g,1,indices=list(range(1,per_row+1)),per_row=per_row)
    rules=[line for line in text.splitlines() if line.startswith('━') or line.startswith('─')]
    expected=per_row*show.table_width()+4*(per_row-1)
    assert len(rules)==3 and all(show.display_width(line)==expected for line in rules)
    header=next(line for line in text.splitlines() if line.startswith('观察者'))
    assert header.count('P7')==per_row and show.display_width(header)==expected
    assert g.review.q.tobytes()==before


def test_selected_pre_compares_previous_actual_not_previous_displayed(real_root):
    g=show.discover([real_root])[0][0]
    g.roles=dict(zip(viz.PLAYER_IDS,('Villager','Villager','Werewolf','Villager','Seer','Werewolf','Witch')))  # Renderer-only mapping.
    text=show.render_game(g,1,indices=[1,9],per_row=1)
    current=g.review.q[8,0,5];prior=g.review.q[7,0,5]
    assert current < prior and current > g.review.q[0,0,5]
    panel=text.split('全局PRE 9')[1]
    assert f'{current:.3f}↓' in panel and f'{current:.3f}↑' not in panel


def test_output_cannot_enter_canonical_only_artifact(real_root,tmp_path,capsys):
    bare=tmp_path/'bare';shutil.copytree(real_root/'canonical_public',bare)
    destination=bare/'output'
    assert show.main(['--root',str(real_root),str(bare),'--game','1','--pre-indices','1',
                      '--output-dir',str(destination)])==2
    assert not destination.exists()


def test_invalid_optional_file_path_is_reported(copied_root):
    p=copied_root/'canonical_public/manifest.json';m=json.loads(p.read_bytes())
    m['file_table']['']={'byte_size':0,'sha256':viz.sha256_bytes(b'')};write_json(p,m);rebind(copied_root)
    assert show.discover([copied_root])[0][0].status=='INVALID'


def test_terminal_width_refuses_without_writing(real_root,tmp_path,capsys):
    out=tmp_path/'no-output'
    assert show.main(['--root',str(real_root),'--game','1','--pre-indices','1','2','3',
                      '--per-row','3','--terminal-width','175','--output-dir',str(out)])==2
    messages=capsys.readouterr()
    assert '需要176列' in messages.err and not out.exists() and not messages.out


def test_actual_tty_width_wins_over_columns_environment(real_root,monkeypatch,capsys):
    monkeypatch.setenv('COLUMNS','999')
    monkeypatch.setattr(sys.stdout,'isatty',lambda:True)
    monkeypatch.setattr(sys.stdout,'fileno',lambda:999)
    monkeypatch.setattr(os,'get_terminal_size',lambda fd:os.terminal_size((115,24)))
    assert show.main(['--root',str(real_root),'--game','1','--pre-indices','1','2'])==2
    assert '当前115列' in capsys.readouterr().err


def test_output_utf8_all_selector_and_no_overwrite_or_input_nesting(real_root,tmp_path,capsys):
    out=tmp_path/'txt'
    assert show.main(['--root',str(real_root),'--all','--pre-indices','9','10','--output-dir',str(out)])==0
    text=(out/'tom-game-matrices.txt').read_text(encoding='utf-8')
    assert '全局PRE 9' in text and '全局PRE 10' in text and 'P1亡' in text
    assert show.main(['--root',str(real_root),'--all','--output-dir',str(out)])==2
    assert (out/'tom-game-matrices.txt').read_text()==text
    assert show.main(['--root',str(real_root),'--list','--output-dir',str(real_root/'canonical_public/nested')])==2
    assert not (real_root/'canonical_public/nested').exists()


def test_missing_root_is_explicit_and_does_not_claim_server_mount(tmp_path,capsys):
    assert show.main(['--root',str(tmp_path/'missing'),'--list'])==1
    assert '不会假设服务器目录已挂载' in capsys.readouterr().err


def test_bad_selectors_or_pres_fail_before_output(real_root,tmp_path,capsys):
    for values in (['--game','2'],['--games','1','1'],['--game-id','missing'],['--game','1','--pre-indices','18']):
        assert show.main(['--root',str(real_root),*values,'--output-dir',str(tmp_path/'absent')])==2
        assert not (tmp_path/'absent').exists()


def test_cli_has_no_predictor_gameplay_or_plotting_import():
    code="import sys; from scripts.show_tom_game_matrices import main; assert not {'torch','run_random','matplotlib.pyplot'} & set(sys.modules); main(['--help'])"
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert p.returncode==0 and '--games' in p.stdout and '--ambiguous-width' in p.stdout
