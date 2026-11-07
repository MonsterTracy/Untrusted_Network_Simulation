"""Read-only sealed Qwen3 OOF adapter. Never constructs or calls a model."""

from dataclasses import dataclass
from pathlib import Path

from scripts import paper_backbone_contrasts as historical
from werewolf.artifact_io import verify_artifact, canonical_jsonl_bytes
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.population import select_primary_population
from werewolf.canonical_collection.public_history import PUBLIC_PHASES
from werewolf.tom_llm.common import digest, identity, require, strict_json, valid_distribution
from werewolf.tom_llm.selection import validate_folds


@dataclass(frozen=True)
class Reference:
    seal_digest: str
    publication_digest: str
    fold_manifest_digest: str
    population_metadata: dict
    folds: list
    rows: tuple
    primary_rows: tuple
    alive_keys: frozenset
    game_ids: tuple
    bootstrap_indices: object
    bootstrap_digest: str
    prediction_digests: dict
    provenance: dict | None = None

    def selected(self, game_ids):
        games = tuple(sorted(game_ids))
        require(games and len(set(games)) == len(games) and set(games) <= set(self.game_ids),
                "reference selection has duplicate/missing/foreign games")
        rows = tuple(row for row in self.primary_rows if row["game_id"] in set(games))
        require({row["game_id"] for row in rows} == set(games), "game has zero effective Primary rows")
        return rows


def open_reference(publication_path, evaluation_root, *, expected_seal_digest):
    """External seal required; historical source gates and re-inference are not used."""
    root = Path(evaluation_root).resolve(strict=True)
    report, prepared, training_root, train_seal, primary = historical._inputs(root, expected_seal_digest)
    games, fold_games, indices, bootstrap_digest = historical._bootstrap(report, prepared.manifest["folds"])
    require(validate_folds(prepared.manifest["folds"]) == games, "reference game/fold mismatch")
    publication = open_publication(publication_path)
    inputs = prepared.manifest["protocol_inputs"]
    require(publication.manifest_digest == inputs["publication_digest"]
            and publication.public_view.fold_manifest.manifest_digest == inputs["fold_manifest_digest"],
            "reference publication/fold binding mismatch")
    population = select_primary_population(publication)
    require(population.metadata == inputs["primary_identity"] and population.rows == primary,
            "reference Primary/selector mismatch")
    actual_folds = {fold.fold_index: set(fold.game_ids) for fold in publication.public_view.fold_manifest.folds}
    require(actual_folds == {i: set(ids) for i, ids in enumerate(fold_games)}, "reference publication fold mismatch")
    require(tuple(sorted(publication.public_view.game_ids)) == tuple(games), "reference publication game coverage mismatch")
    descriptors = {(r["architecture"], r["fold"]): r for r in report.manifest["predictions"]}
    terminals = {(r["architecture"], r["fold"]): r for r in train_seal.manifest["terminals"]}
    require(len(descriptors) == len(report.manifest["predictions"])
            and len(terminals) == len(train_seal.manifest["terminals"]), "duplicate lineage descriptor")
    expected_lineages = {(a, f) for a in historical.ARCHITECTURES for f in range(5)}
    require(set(descriptors) == set(terminals) == expected_lineages
            and all(t["temporal_condition"] == "explicit_day_phase" for t in train_seal.manifest["terminals"]),
            "reference lineage inventory mismatch")
    all_rows, primary_rows, alive_keys, prediction_digests = [], [], set(), {}
    checkpoint_digests = {}
    for fold, ids in enumerate(fold_games):
        key = "qwen3", fold
        require(key in descriptors and key in terminals, "missing Qwen3 fold descriptor")
        descriptor = descriptors[key]
        terminal = verify_artifact(training_root / "runs/qwen3" / str(fold) / "terminal",
            expected_artifact_type="backbone_terminal", expected_schema_version=historical.TERMINAL_VERSION)
        artifact = verify_artifact(root / "folds/qwen3" / str(fold),
            expected_artifact_type="backbone_oof_fold_predictions", expected_schema_version=historical.PREDICTION_VERSION)
        m = artifact.manifest
        require(terminal.manifest_digest == descriptor["terminal_digest"] == terminals[key]["terminal_digest"]
                and artifact.manifest_digest == descriptor["prediction_digest"]
                and m["architecture"] == "qwen3" and type(m["fold"]) is int and m["fold"] == fold
                and m["temporal_condition"] == "explicit_day_phase"
                and m["evaluation_digest"] == report.manifest["evaluation_digest"]
                and m["training_seal_digest"] == report.manifest["training_seal_digest"]
                and m["terminal_digest"] == terminal.manifest_digest
                and m["checkpoint_digest"] == terminal.manifest["checkpoint_ancestry"][-1]["digest"]
                and set(m["file_table"]) == {"predictions.jsonl"}, "reference fold/terminal lineage mismatch")
        raw = (artifact.path / "predictions.jsonl").read_bytes()
        rows = [strict_json(line) for line in raw.splitlines()]
        dataset = CanonicalToMDataset(publication.public_view, ids, ExperimentCapacity(1024))
        require(raw == canonical_jsonl_bytes(rows) and len(rows) == m["row_count"] == len(dataset) * 7,
                "reference prediction coverage/serialization mismatch")
        selected_ids = []
        for sample, batch in zip(dataset, (rows[i:i+7] for i in range(0, len(rows), 7)), strict=True):
            for observer, row in enumerate(batch):
                semantic = {"game_id": sample.game_id, "boundary_id": sample.boundary_id, "observer": observer,
                    "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                    "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                    "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": m["checkpoint_digest"], "temporal_condition": "explicit_day_phase"}
                require(set(row) == historical.PREDICTION_FIELDS and type(row["observer"]) is int
                        and type(row["label_observed"]) is bool and type(row["observer_alive"]) is bool
                        and all(type(v) in (int, float) for v in row["q"])
                        and all(row[k] == v for k, v in semantic.items()), "reference PRE/observer/q semantics mismatch")
                valid_distribution(row)
                entry = {**row, "fold": fold, "canonical_shift": 0,
                         "primary_eligible": primary[sample.game_id][sample.boundary_id][observer],
                         "reference_prediction_digest": artifact.manifest_digest,
                         "public_token_count": sample.token_count,
                         "public_phase": PUBLIC_PHASES[int(sample.public.phase_ids[sample.token_count - 1])]}
                all_rows.append(entry)
                if row["observer_alive"]:
                    alive_keys.add(identity(entry))
                if row["label_observed"] and entry["primary_eligible"]:
                    require(row["observer_alive"], "dead Primary observer")
                    primary_rows.append(entry)
                    selected_ids.append([sample.game_id, sample.boundary_id, observer])
        require(digest(selected_ids) == m["primary_row_identity_digest"]
                == report.manifest["primary_row_bindings"][str(fold)], "reference Primary row identity mismatch")
        prediction_digests[str(fold)] = artifact.manifest_digest
        checkpoint_digests[str(fold)] = m['checkpoint_digest']
    require(len({identity(r) for r in all_rows}) == len(all_rows), "duplicate reference row")
    return Reference(report.manifest_digest, publication.manifest_digest,
        publication.public_view.fold_manifest.manifest_digest, population.metadata, prepared.manifest["folds"],
        tuple(all_rows), tuple(primary_rows), frozenset(alive_keys), tuple(games), indices,
        bootstrap_digest, prediction_digests,
        {'architecture':'qwen3', 'temporal_condition':'explicit_day_phase',
         'training_source':report.manifest['training_source'], 'evaluation_source':report.manifest['evaluation_source'],
         'training_seal_digest':report.manifest['training_seal_digest'],
         'checkpoint_digests':checkpoint_digests, 'canonical_jsonl_verified':True})


def raw_set_labels(reference, publication_path, game_ids):
    """评价阶段从封存报告恢复原集合，均匀 q 不参与逆推。"""
    from werewolf.canonical_collection.public_history import PLAYER_IDS
    from werewolf.tom.dataset import belief_target
    publication = open_publication(publication_path)
    require(publication.manifest_digest == reference.publication_digest
            and publication.public_view.fold_manifest.manifest_digest == reference.fold_manifest_digest,
            'raw-set publication/Reference identity mismatch')
    games = set(game_ids)
    require(games and games <= set(reference.game_ids), 'raw-set evaluation games mismatch')
    observed = {}
    for game in publication.public_view.games:
        if game.game_id not in games:
            continue
        observations = {o.observation_id:o for o in game.belief_observations}
        for prefix in game.authoritative_pre_prefixes:
            for link in prefix.belief_observation_links:
                observation = observations.get(link.observation_id)
                if observation is None:
                    continue
                require(observation.game_id == prefix.game_id and observation.boundary_id == prefix.boundary_id
                        and observation.prefix_digest == prefix.prefix_digest
                        and observation.observer_id == link.observer_id, 'raw-set PRE/observer binding conflict')
                key = (prefix.game_id, prefix.boundary_id, prefix.prefix_digest, PLAYER_IDS.index(link.observer_id))
                require(key not in observed, 'duplicate raw-set observation')
                observed[key] = observation
    labels = []
    for row in reference.rows:
        if row['game_id'] not in games or not row['primary_eligible'] or not row['observer_alive']:
            continue
        observation = observed.get(identity(row))
        label = {k:row[k] for k in ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')}
        label.update(status='not_recoverable', suspected_players=None, observation_digest=None)
        if observation is not None:
            success = observation.status.value == 'success'
            require(success == row['label_observed'], 'raw-set/Reference observability conflict')
            q, valid = belief_target(row['observer'], observation.suspicion_support, observation.status.value)
            require(valid == row['label_observed'] and q.tolist() == row['q'], 'raw-set/Reference q conflict')
            label.update(status='observed' if success else 'unobservable',
                         suspected_players=list(observation.suspicion_support) if success else None,
                         observation_digest=observation.observation_digest)
        labels.append(label)
    return {'publication_digest':publication.manifest_digest, 'reference_seal_digest':reference.seal_digest,
            'labels':labels, 'labels_digest':digest(labels)}


def fixed_oof_contract():
    """历史三模型共同使用的唯一 OOF；Source 取自封存合同，不取当前 HEAD。"""
    return {
        'architecture':'qwen3', 'temporal_condition':'explicit_day_phase',
        'reference':{
            'scope':'historical_v1_explicit_day_phase',
            'seal_digest':'66bd044bd072a8673d4761ad043d0af3a208ae55b809ebffded2228b0371f925',
            'publication_digest':'396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd',
            'fold_manifest_digest':'754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69',
            'folds_digest':'40b3e80d77a0c44f0ca35a1656538257c08fee7f1fd02fad7f5d7b2baac3dc53',
            'bootstrap_digest':'5635d3225e7259c584cd9942dc07dae85f89e1acf90bf23cb636bf17aa547181',
            'prediction_digests':{
                '0':'e6bcb0887220e89d6b75245929d422c26506b5969398eb5395e56e52db93c186',
                '1':'07202a0016bd4d973b04cd6a18ceb7e5ea297cd8e061a9f10ed2c94bc36c7fca',
                '2':'c5d3a9e0f2f40be4371d7221779afd130a1c14da6c96fabdbbedd92ed3b0161c',
                '3':'f63263af146d2ed9887cb743938d7bef3659a2a78936cb6ddb9a5c94ab399eed',
                '4':'e16c3ad654b16d37103368278aa2cb9cd75d1789c5aa8fc1025564d4882601d2'}},
        'training_source':{'source_revision':'ebf98b7e3249f78d3f1fa1fd46c20affe0486913',
                           'implementation_digest':'dca9dbdf3121a6c891d1dab619fd1948f58ea3799f0a52a56ab5b3f549148da7'},
        'evaluation_source':{'source_revision':'46b0cb24421a2f98e063d67aedaedb8e31875403',
                             'implementation_digest':'a514805b2354066e709b2e32b52a2cfccd0e474d54a152df6f6092b101ac38ec'},
        'training_seal_digest':'10f37a27dc951ab89021653dc433e510f844350c82f71b0d7711dfff957a8f86',
        'checkpoint_digests':{
            '0':'3e89f066dd1778ffbe3a515365b5fb2ae9074098103ea16acec7073b2f81adbd',
            '1':'d6533b6eee7d15c90e24d9e7bc4e25e4b12923767fb38c8e856ee3febcd071bb',
            '2':'258fe3bdf57acd93bb8ddf69c801294174ca11c5d68ef50e02104a75b16d235f',
            '3':'514f02a4ad5c47209ed87ed9490602cc8b1b014069571395f14140ca5c0f097c',
            '4':'2eabf7a61a94e2e077161b5c81d583b87aacae9ec4ae6ef81f83d9dfbc14b34f'}}


def projection_source(reference, *, engineering_only):
    """Mock 保留真实的合成出处；正式数据必须与固定 OOF 合同逐项相同。"""
    from werewolf.tom_llm.multimodel_evaluation import reference_binding
    from werewolf.tom_llm.common import require_digest
    provenance = reference.provenance
    require(type(provenance) is dict and provenance.get('canonical_jsonl_verified') is True
            and provenance.get('architecture')=='qwen3'
            and provenance.get('temporal_condition')=='explicit_day_phase', 'unverified OOF projection source')
    checkpoints = provenance['checkpoint_digests']
    require(set(checkpoints)==set(reference.prediction_digests)=={str(f) for f in range(5)}, 'OOF fold inventory mismatch')
    for value in checkpoints.values(): require_digest(value)
    for row in reference.rows:
        fold = str(row['fold'])
        require(fold in checkpoints and row['checkpoint_digest']==checkpoints[fold]
                and row['reference_prediction_digest']==reference.prediction_digests[fold]
                and row['temporal_condition']=='explicit_day_phase', 'OOF row/checkpoint/source mismatch')
    actual = {k:v for k,v in provenance.items() if k!='canonical_jsonl_verified'}
    actual['reference'] = reference_binding(reference)
    fixed = fixed_oof_contract()
    if not engineering_only:
        require(actual==fixed, 'approved protocol requires the fixed historical OOF source')
    return {'status':'ENGINEERING_ONLY_NOT_FORMAL_OOF' if engineering_only else 'FIXED_HISTORICAL_OOF_VERIFIED',
            'actual':actual, 'actual_digest':digest(actual), 'fixed_contract_digest':digest(fixed)}
