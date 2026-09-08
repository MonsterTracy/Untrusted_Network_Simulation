# Explicit current configuration

The designated first formal development protocol is in
[`formal/development-experiment-v1`](formal/development-experiment-v1/README.md).
It is pending researcher-selected values and cannot run or count as frozen
while those fields remain null. It uses the existing `ExperimentConfig` schema.

There is no default experiment, collection mode or supervision scope. The
formal commands accept explicit files; credentials are environment variables.

A runtime YAML contains exactly backends, parser, agent_config and env_config.
Each named backend declares type=openai_compatible, base_url and api_key_env
(and optional default_model/supports_json_schema). Parser declares backend,
model and model_params. A playing profile declares profile_name, agent_type=gpt,
backend, model, model_params and sample_ratio. agent_config has all_candidates,
must_include and allow_cross_team_profiles. The only accepted gameplay prompt
contract is strict_classic7. env_config retains Classic7 7/2/3/1/1 counts,
n_guard=0 and n_hunter=0.

Construct a Collection Plan using
werewolf.canonical_collection.attempt_ledger.construct_collection_plan, then
serialize plan.to_record() as canonical JSON. In environment_provenance,
runtime_config_sha256 is SHA-256 of canonical JSON of normalize_runtime_config,
and configured_call_limit is the exact decimal limit passed to collect.
The plan also declares ordered seeds, target count, schema identities, source
revision and backend/parser/prompt/retry provenance. A resume must use the same
plan and configuration.

The prepare-experiment protocol JSON supplies every field of
werewolf.tom.experiment.ExperimentConfig, with no hidden budget defaults:
capacity, AdamW settings, constant scheduler, game batch size, complete rotation
cycles, initialization/RNG/schedule/bootstrap seeds, bootstrap replicates,
confidence, recovery cadence, device, thread count, deterministic algorithms
and source revision. max_seq_len must cover the publication maximum.

Backend instances are external transport dependencies only. Tests inject
deterministic adapters, never a second collection mode. API calls are not made
by configuration validation or the self-contained test suite.
