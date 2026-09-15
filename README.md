# RIME: Rule-Based Instructions for Music Editing

## Ground-Truth Pipeline

The ground-truth stack is now split into four symbolic stages:

1. `scripts/build_analysis_manifest.py`
   Dataset metadata -> intrinsic clip analysis manifest
2. `scripts/generate_ground_truth_plans.py`
   Analysis manifest -> full set of permissible symbolic edit plans
3. `scripts/subsample_ground_truth_plans.py`
   Permissible plans -> policy-selected subset
4. `scripts/plan_coverage.py`
   Permissible plans -> coverage and distribution report

The planner does not use `request.intent`. It emits every recipe that is allowed
for a clip by the YAML rules, heuristics, and metadata constraints.

## Defaults

The scripts are wired to the MTG-Jamendo metadata already present in this repo.
If you run them with no arguments, they use these defaults:

```bash
python scripts/build_analysis_manifest.py
python scripts/generate_ground_truth_plans.py
python scripts/subsample_ground_truth_plans.py
python scripts/plan_coverage.py
```

Default outputs:

- `derived/ground_truth/mtg_jamendo_analysis_manifest.jsonl`
- `derived/ground_truth/permissible_plans.jsonl`
- `derived/ground_truth/subsampled_plans.jsonl`
- `derived/ground_truth/plan_coverage.json`

## 1. Build The Analysis Manifest

```bash
python scripts/build_analysis_manifest.py
python scripts/build_analysis_manifest.py --limit 100
python scripts/build_analysis_manifest.py --audio-root /path/to/mtg_audio
```

Important arguments:

- `--dataset-name`
  Default: `mtg_jamendo`
- `--metadata-dir`
  Default: `mtg-jamendo-dataset/data`
- `--dataset-config`
  Default: `configs/ground_truth/datasets/mtg_jamendo.yaml`
- `--audio-root`
  Optional root for actual audio files if you have them locally
- `--output-path`
  Default: `derived/ground_truth/mtg_jamendo_analysis_manifest.jsonl`

The manifest contains intrinsic metadata only. Current fields include:

- `clip_id`
- `audio_path`
- `analysis.dataset.*`
- `analysis.genres`
- `analysis.mood_themes`
- `analysis.instrument_tags`
- `analysis.target_candidates`
- `analysis.issues`
- `analysis.tempo_bpm`
- `analysis.key`
- `analysis.mode`

Issue labels are currently empty unless some upstream detector adds them. MTG
tags are trusted for instrument and genre metadata for now.

## 2. Generate All Permissible Plans

```bash
python scripts/generate_ground_truth_plans.py
python scripts/generate_ground_truth_plans.py --limit 100
python scripts/generate_ground_truth_plans.py --max-variants-per-recipe 8
```

Important arguments:

- `--analysis-path`
  Default: `derived/ground_truth/mtg_jamendo_analysis_manifest.jsonl`
- `--output-path`
  Default: `derived/ground_truth/permissible_plans.jsonl`
- `--config-dir`
  Default: `configs/ground_truth`
- `--max-variants-per-recipe`
  Optional cap per recipe per clip. Default: no cap

This stage computes the full cartesian product of:

- clips
- eligible recipes
- eligible target candidates
- enumerated parameter supports

and then removes invalid combinations through the YAML rules.

The output is symbolic. It does not import or execute the DSP libraries. Each
row includes:

- `clip_id`
- `audio_path`
- `plan_id`
- `recipe_id`
- `weight`
- `target_stem`
- `target_family`
- `genres`
- `mood_themes`
- `issues`
- `bindings`
- `graph_spec`
- `graph_description`
- `recipe_tags`
- `applied_policies`

## 3. Subsample Plans

```bash
python scripts/subsample_ground_truth_plans.py
python scripts/subsample_ground_truth_plans.py --policy feature_submodular --limit 200
python scripts/subsample_ground_truth_plans.py --policy random --limit 200 --seed 7
```

Available policies:

- `feature_submodular`
  Encode each plan's joint clip/target/recipe/graph text and run apricot
  feature-based submodular selection over the global candidate pool.
- `random`
  Random baseline with the same global limit.

## 4. Coverage

```bash
python scripts/plan_coverage.py
```

The coverage report includes:

- total clips and total plans
- plans per clip
- recipe usage
- operator usage
- target-family usage
- genre usage
- block-kind usage
- applied-policy usage
- dead recipes
- dead operators
- parameter-value histograms

## 5. Render Plans

```bash
python scripts/render_permissible_plans.py
```

Important arguments:

- `--plans-path`
  Default: `derived/ground_truth/subsampled_plans.json`
- `--output-root`
  Default: `derived/ground_truth/generated-audio`

## 6. Generate Prompts

```bash
python scripts/generate_ground_truth_prompts.py
```

Important arguments:

- `--plans-path`
  Default: `derived/ground_truth/subsampled_plans.jsonl`
- `--output-path`
  Default: `derived/ground_truth/subsampled_plan_prompts.jsonl`
- `--config-dir`
  Default: `configs/ground_truth`
- `--levels`
  Abstraction levels to emit, e.g. `0,1`. Levels they derive from are generated
  regardless. Default: every level in the ladder.
- `--max-attempts`
  Generation attempts per level before keeping a text that still fails its hard
  checks. Default: `3`

The number of rewrites per graph is set by the ladder in
[configs/ground_truth/abstraction_levels.yaml](configs/ground_truth/abstraction_levels.yaml),
which declares each abstraction level's rules, exemplars, and checks. The
descriptor vocabulary level 1 speaks in lives in
[configs/ground_truth/param_bands.yaml](configs/ground_truth/param_bands.yaml)
and is validated at load time against the supports in `distributions.yaml`.

## Current Config Layout

- [configs/ground_truth/datasets/mtg_jamendo.yaml](configs/ground_truth/datasets/mtg_jamendo.yaml)
  Dataset ingestion and tag-to-target mapping
- [configs/ground_truth/operators.yaml](configs/ground_truth/operators.yaml)
  Symbolic operator registry and optional runtime binding metadata
- [configs/ground_truth/distributions.yaml](configs/ground_truth/distributions.yaml)
  Parameter priors
- [configs/ground_truth/motifs.yaml](configs/ground_truth/motifs.yaml)
  Reusable signal-chain motifs
- [configs/ground_truth/recipes.yaml](configs/ground_truth/recipes.yaml)
  Clip-level permissible task patterns
- [configs/ground_truth/constraints.yaml](configs/ground_truth/constraints.yaml)
  Global chain-order and prior-shaping rules

## Delay Handling

Both delay modes are supported:

- tempo-synced delay
- free-time delay

Synced delay is the default prior. If `analysis.tempo_bpm` is missing, the
synced motif falls back to `120.0` BPM.

## Execution

The current generation path is symbolic by design. Plans are validated against
the YAML operator registry, not the live DSP imports. This keeps planning,
coverage, and subsampling decoupled from the runtime stack.

The next layer to add is a runtime binder / executor that reads the same
operator registry, resolves the `runtime.module` and `runtime.callable` entries,
and renders audio from these symbolic plans in parallel.

### Local Audit UI

```bash
python3 scripts/ground_truth_audit_ui.py
```

Open `http://127.0.0.1:8787`. The UI reads and writes the same YAML files under
`configs/ground_truth`, shows parsed recipes/motifs/constraints with line jumps
back into the raw YAML editor, runs the subset/planning/render/coverage scripts,
and lists source, rendered, and ad hoc audio for auditioning.

The ad hoc FX lane applies any non-separation, non-pitch operator from
`operators.yaml` to a selected local validation track with editable JSON params.
Outputs are written to `derived/validation/mtg_jamendo_10/ad_hoc`.
