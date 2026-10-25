# ToM V2 Development input integration

This is an input/interface integration, not a new training protocol or experiment
qualification. V1 publications, historical model identities and Experiment A's
paired temporal contract remain distinct.

## Input admission

`werewolf.tom.development_input.open_development_input` dispatches only the exact
V1 Development or V2 Development publication type/schema pair. V1 still uses the
original reader. V2 requires a trusted Publication digest; initial preparation
also supplies the independently approved Split digest. A worker reopens using
its immutable parent's Publication digest, which binds the embedded Split
header, Development membership, fold manifest and public projection identities.
This is transitive identity validation, not discovery of a new trusted anchor.

Unknown/Test schemas, wrong digests, parent traversal, linked paths/manifests,
inventory drift and out-of-Development game IDs fail closed. The public reader
does not open role bytes; only the existing typed Primary selector may access
the Development-only sidecar. No raw Collection, full Split or Test payload is
needed by a training/evaluation worker. Canonical content validation is reused.

## Explicit preparation commands

The new commands are `prepare-experiment-v2`, `prepare-final-experiment-v2` and
`prepare-backbone-tom-study-v2`. Each requires `--publication`,
`--publication-digest`, `--split-digest` and `--destination`, plus `--protocol`
for the first two or `--config` for Backbone. Paths still pass through the
existing Storage Profile boundary. Existing commands and defaults are unchanged.

The V2 Backbone design uses `classic7_backbone_study_v2`. Only publication ID and
Development game count vary; the count must support five equal complete-game
folds. Numerical controls must equal the frozen reference design. The validated
schedule determines its equal per-fold step budget; historical V1 and paired
formal budgets remain exactly 25200, with cadence 1000. Recovery, terminal/seal,
source comparison and ownership checks are retained. V2 evaluation additionally
pins the Split, V2 publication and input-adapter source files.

## Preserved boundaries and unsupported combinations

- `classic7_qwen3_paired_temporal_study_v2` is Experiment A's existing temporal
  pairing schema, not the new V2 data schema. Its shared initialization bytes,
  selection-blind RNG, lineage, condition-specific prediction paths and CLI
  flags remain unchanged. V2 data plus paired temporal is not registered here
  and is explicitly rejected; this integration does not invent its protocol.
- The generic Final experiment can prepare and reopen V2 Development training
  inputs. This does not make a same-Collection V2 Test partition eligible for
  the historical independent-Final evaluation protocol. That evaluation and
  its independent-Collection/attempted-seed guards remain unchanged. No V2 Test
  evaluation or actual frozen-model qualification is authorized by this code.
- Frozen `qwen3_final` remains the original all-1500 Development contract.
  Historical artifacts must use their pinned source/runtime; the integrated
  code must not impersonate old execution, model, seal or Analysis identities.
- Tests use synthetic publications and CPU engineering fixtures. No real
  campaign, seed pool, labels, training result or model artifact is bundled.

Source: selected working-byte changes from Canonical V2, applied on integrated
Mainline after S1, rather than replacing its older Backbone/CLI modules.
