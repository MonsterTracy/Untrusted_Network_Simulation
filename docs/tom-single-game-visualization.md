# Single-game ToM visualization

This independent analysis CLI reads the supplied
tom_one_game_all_pre_export_v1 review folder. It neither computes predictions
nor invokes gameplay. Frozen ToM, mapper, Router and qualification source are
unchanged.

Install the optional analysis dependencies in a separate environment:

~~~sh
python -m pip install "numpy>=1.24,<3" "matplotlib>=3.8,<4"
python -m scripts.visualize_tom_single_game \
  --input-dir /absolute/path/to/tom-single-game-review \
  --output-dir /absolute/path/to/new-analysis-directory \
  --formats png pdf svg --pre-indices 9 10 14 15 \
  --comparison-pairs 9:10 14:15 --dpi 450 \
  --figure-width 7 --font-size 9 --columns 5
~~~

The input root contains tom-single-game-matrices-demo-v1/ (the four export files)
and canonical_public/ (manifest.json, authoritative_pre_prefixes.jsonl and
public_event_stream.jsonl). The original export script, when provided at the
root, is hashed but never executed. The CLI requires a new output directory
outside the input tree and publishes its files atomically without overwriting.

Validation checks the complete export artifact, supplied canonical public file
hashes, original manifest digest, typed PRE round-trips, exact public prefixes,
seat/alive mapping, frozen predictor pins, coverage, order and every CSV cell.
The original gameplay revision remains independent of the analysis HEAD.
The public-only snapshot cannot revalidate private full-bundle replay or final
winner evidence. Missing role truth means a non-wolf/Router observer panel is
unavailable; public aliveness alone is not that panel.

Plots preserve raw seven-seat distributions, including nonzero dead-target mass
and dead-observer outputs. Heatmaps use a common [0,1] scale. Self zeros have
crosses and dead seats have light hatching. Post-death observer trajectories
are dashed/shaded; hollow points denote a dead target, independently of the
observer's status. All raw mass remains plotted. Cell labels are rounded only
for display. Individual heatmap
indices are one-based ordinals in strictly increasing canonical step order;
day groups have separate labeled bands in the compact overview. Trajectory curves connect only
observed points; integer ticks denote PREs and dotted lines mark day changes.

The two comparison pairs must be distinct adjacent PRE pairs. They generate
heatmap_comparisons (a 2x2 before/after figure with a shared scale) and
key_interval_events. The pairs select displayed observations, never alter
matrices, the TV calculation or event ordering. For this game, 9→10 and 14→15
are cross-day intervals, each spanning eight new public events.

Total Variation is half the L1 difference of complete raw seven-dimensional
rows. Both all-observer and alive-in-both-PRE means are reported; an empty common
panel is null. The x-axis names compared PRE intervals, cross-day intervals
are shaded, and a separate n strip gives the common-alive panel count for every
interval. This is not a verified non-wolf panel. TV heatmaps use [0,1]; summary curves start at zero with a shared
data-derived upper axis limit for readability. Event intervals end at each
observed PRE; the first interval is initial history, not a measured change.
PRE precedes its speaker's speech. Adjacent PREs can span several events,
phase changes or deaths. The complete event figure draws horizontal TV segments
between the actual PRE endpoints, rather than assigning change to the arrival
event. Empty death/exile announcements say "none". Events after the last PRE
are shaded as having no subsequent ToM observation. Phase abbreviations are
S (speech), PK (PK speech), V (vote), PV (PK vote) and N (night).
Association is descriptive, never a causal effect.

Recommended manuscript figures are heatmap_comparisons, total_variation and
key_interval_events. Individual detailed heatmaps, matrix_sequence,
observer_trajectories and public_event_alignment are appendix material.
The default physical width is 7 inches (177.8 mm), intended for double-column
placement. At font-size 9, the smallest labels are 7.5 pt at this width.
Do not shrink these files into a single column; DPI cannot preserve physical
font size. Matrix columns default to five; increasing them can crowd labels.
The complete trajectory sheet is 7 by 8.5 inches and should be placed on a
dedicated appendix page. Match the actual journal template
before submission. PDF embeds fonts; SVG keeps editable text and needs the
same DejaVu Sans font for identical typography.

Heatmap cells use vector meshes, without embedded raster images in SVG/PDF.
Outputs include PNG/PDF/SVG figures, total_variation.csv, event_intervals.json,
analysis.json, README.md and an immutable manifest. Analysis records input
hashes, plotting/software versions, script hash, tracked status and original
model/gameplay lineage, physical width and selected comparison pairs.
Generated PDF/SVG timestamps are suppressed and SVG IDs
are deterministic.

Real-data regression is opt-in because private local review files are not
committed as test data:

~~~sh
UNS_TOM_SINGLE_GAME_REVIEW=/absolute/path/to/tom-single-game-review \
  python -m pytest tests/tom/test_single_game_visualization.py -q
~~~

Synthetic numeric/safety tests run without that variable. Real-data tests use
temporary copies for corruption checks and verify byte-identical repeated
renders; they never rewrite original inputs.

The integrated self-contained suite additionally builds five explicitly artificial
PREs with the existing public-history/PRE constructors and exercises the real
reader, plotting CLI and repeated PNG/PDF/SVG publication. These are schema and
interface tests, not model predictions or a replacement for the pinned 17-PRE
real-data regression. No real input is discovered when the variable is unset;
the corresponding historical tests remain opt-in and are reported as skipped.
Matplotlib is an optional analysis dependency; the existing gameplay defaults
and `uns` command set are unchanged. The self-contained rendering test requires
that optional dependency and reports a skip when it is absent; a rendering
acceptance run must use the documented analysis environment and execute it.
