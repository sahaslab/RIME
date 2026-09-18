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
5. `scripts/filter_ground_truth_prompts.py`
   Generated prompts -> per-row rejection verdicts

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

Model and run settings come from
[configs/ground_truth/summarization.yaml](configs/ground_truth/summarization.yaml),
so a prompt file is reproducible from config alone. Generation goes through
litellm to whichever provider that file names; the default is AWS Bedrock, which
needs `boto3` installed plus either a Bedrock API key in
`AWS_BEARER_TOKEN_BEDROCK` or credentials on the standard AWS chain (environment
variables, a `.env` file, a named profile, or an instance role).

Every flag below is an override: pass one and it wins, leave it out and the
value comes from `summarization.yaml`.

- `--plans-path` / `--output-path`
  Default: `summarization.paths.plans` / `summarization.paths.output`
- `--config-dir`
  Where the operators, distributions, abstraction and summarization configs
  live. This one is a real default, not a config value, because it is what
  locates `summarization.yaml`. Default: `configs/ground_truth`
- `--model`
  litellm model id. Default: `summarization.model.name`
- `--levels`
  Abstraction levels to emit, e.g. `0,1`. Levels they derive from are generated
  regardless. Default: `summarization.run.levels`, or every level in the ladder
  when that is empty.
- `--max-attempts`
  Generation attempts per level before keeping a text that still fails its hard
  checks. Default: `summarization.run.max_attempts`
- `--max-workers`
  Concurrent graph workers. Default: `summarization.run.max_workers`
- `--subsample-num` / `--seed`
  Random sample size for smoke tests and the seed that makes it repeatable.
  Default: `summarization.run.subsample_num` / `summarization.run.seed`

The number of rewrites per graph is set by the ladder in
[configs/ground_truth/abstraction_levels.yaml](configs/ground_truth/abstraction_levels.yaml),
which declares each abstraction level's rules, exemplars, and checks. The
descriptor vocabulary level 1 speaks in lives in
[configs/ground_truth/param_bands.yaml](configs/ground_truth/param_bands.yaml)
and is validated at load time against the supports in `distributions.yaml`.

## 7. Filter Generated Rows

```bash
python scripts/filter_ground_truth_prompts.py --reachability
python scripts/filter_ground_truth_prompts.py --dry-run --limit 5
python scripts/filter_ground_truth_prompts.py --limit 20
python scripts/filter_ground_truth_prompts.py --resume
```

Reads the prompt artifact and raises four flags per row, writing verdicts to a
new file. The input is never modified, because `generate_agent_input.py` and the
prompt lab both read `prompt_variants` and `prompt_levels` out of it.

| flag | output | question |
| --- | --- | --- |
| `al_rules` | binary | Does each level satisfy its own rules? |
| `al_consistency` | binary | Is AL1 consistent with AL0, AL2 with AL1? |
| `joint_params` | plausible / implausible / unclear | Are co-sampled parameters jointly coherent? |
| `stylistic` | appropriate / inappropriate / unclear | Does the edit suit the captioned music? |

Each row ends with a `disposition` of `accept`, `reject` or `review`. What an
`unclear` verdict does is configurable per criterion, and a judge call that
never returned parseable JSON is never silently accepted.

Two mechanisms, deliberately separated. Every rule in
[configs/ground_truth/rejection.yaml](configs/ground_truth/rejection.yaml)
carries a `trigger` in the same condition DSL as `constraints.yaml`, evaluated
exactly by [ground_truth/predicates.py](ground_truth/predicates.py). A row that
triggers nothing takes its default verdict without any model call at all. The
judge then adjudicates only the rules that fired, and is handed their
statements, their sources and the actual parameter values, so it never has to do
arithmetic. Rules therefore declare what they `suggests`, not a verdict: the
judge can overturn a trigger when the surrounding chain defuses it.

The judge runs a different model from the generator, at `temperature: 0.0`,
because a model grading prose it wrote itself measures authorship as much as
compliance. Note that any prompt file generated before the generator switched
models was written *by* the current judge, so that separation only holds for
rows regenerated since.

`al_rules` re-runs the mechanical checks rather than trusting the stored
`checks_passed`: generation keeps text that still fails after its attempt
budget. Because those checks read the current `param_bands.yaml`, the stage
refuses to run when a row's `abstraction_version` does not match the loaded
ladder, so a config edit since generation cannot masquerade as a rejection.

### Reachability

```bash
python scripts/filter_ground_truth_prompts.py --reachability
```

A rule whose trigger cannot fire under the current configs is dormant. Dormant
rules are kept on purpose -- they cost nothing, they record the intent, and they
arm themselves if a support widens -- and nothing declares its own dormancy, so
it cannot go stale. Two gates, operator first, because `operators.yaml` is what
may be *declared* while `motifs.yaml` and `recipes.yaml` are what can actually
be *placed*: five declared operators are referenced by neither.

| verdict | meaning |
| --- | --- |
| `live` | can fire |
| `vacuous` | always fires; the leaf is pinned inside the trigger region |
| `dormant_distribution` | support exists but never reaches the region; widening it re-arms the rule |
| `dormant_no_param` | the control does not exist, so no distribution change can arm it |
| `unreachable_operator` | no motif or recipe can place the operator |
| `data_dependent` | the trigger reads the corpus, not the configs |

Since the repo has no test suite, `--reachability` and `--dry-run` are the test
surface: both are deterministic, call no model, and are meant to be diffed.

### Full Manifest

```bash
python scripts/filter_ground_truth_prompts.py --generate-full-manifest
```

Writes every input record in full to `paths.full_manifest`, with its verdict
attached under a single `rejection` key. Input keys are copied through untouched
and nothing is renamed, so a training job can consume this one file without
joining anything. The compact sidecar at `paths.output` is still written either
way: it is what `--resume` reads back, and learning which rows are done should
not mean re-parsing every prompt record.

It is written under `--dry-run` too, since the manifest's shape is settled by the
deterministic half and should be checkable without spending a judge call.

### MERT Similarity

```bash
sbatch scripts/run_calculate_similarity.sbatch                     # embed on a GPU node
python scripts/filter_ground_truth_prompts.py --include-mert       # then join
```

Adds a fifth `similarity` flag: the cosine distance in MERT embedding space
between the original and the edited audio. This is the one criterion that needs
audio rather than symbols, and the only one that can catch an edit too subtle to
hear.

[scripts/calculate_ground_truth_similarity.py](scripts/calculate_ground_truth_similarity.py)
does the work and is runnable on its own; `--include-mert` makes the filter score
any pair not already in `similarity_path`. Run the sbatch first for anything
larger than a smoke test — the filter is an I/O-bound Bedrock job and embedding a
corpus inside it wastes a GPU allocation on waiting for HTTP.

Embedding reuses `fadtk.MERTModel` (`m-a-p/MERT-v1-95M`, 768-dim, 24 kHz),
already installed in `postmaster-clean`, subclassed to replace only the pooling.
`layer: all` averages the 13-layer stack then the time axis; `layer: 12`
reproduces an unmodified `fadtk.MERTModel()` **bit-exactly**, which is the
setting under which these numbers are comparable to the lab's existing FAD/KAD
analysis under `/dartfs/rc/lab/S/SinghN/rime/rime_analysis`.

Which "original" is used matters, and the render manifest carries three:

| `reference` | source | measures |
| --- | --- | --- |
| `baseline` | `baseline_path`, the no-FX remix | the effects chain alone, since separation artifacts cancel |
| `source` | `source_copy_path`, else `audio_path` | the edit *plus* Demucs separation loss |

Which one was used is recorded per row, because the two are not poolable. Poison
plans invert the pair — their `baseline_path` is the *degraded input*, so those
rows compare the source against the output and are marked `poison_repair`; do
not pool those either.

`similarity_threshold` is null by default, which keeps the criterion descriptive:
the cosine is recorded and the flag's verdict stays null, so nothing is rejected
on a cutoff nobody has chosen from data yet. Set a float and it becomes an
ordinary rejection reason. The raw value is always written, so re-thresholding
never re-runs MERT.

Calibrate before trusting a threshold. On real pairs the cosine sits in a very
narrow band near 1.0 — a gain-only edit measured 0.99889 against its baseline
where a reverb-plus-delay send measured 0.99836, so the ordering is right but the
whole signal spans about 5e-4, while the spread *between* clips is two orders of
magnitude wider. A single global cutoff is therefore unlikely to mean much;
per-clip normalisation, or a pooling that preserves frame-level differences, is
the direction to explore.

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
- [configs/ground_truth/abstraction_levels.yaml](configs/ground_truth/abstraction_levels.yaml)
  Abstraction ladder: each level's voice, rules, exemplars and checks
- [configs/ground_truth/param_bands.yaml](configs/ground_truth/param_bands.yaml)
  Descriptor vocabulary for parameters, operators and stems
- [configs/ground_truth/summarization.yaml](configs/ground_truth/summarization.yaml)
  Model and run settings for prompt generation
- [configs/ground_truth/rejection.yaml](configs/ground_truth/rejection.yaml)
  Rejection criteria: sourced rules, their triggers, the judge's settings, and
  the MERT similarity block

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

### Effect Lab UI

```bash
python3 scripts/effect_lab_ui.py --port 8789
```

Open `http://127.0.0.1:8789`. Upload a `.wav`, pick a recipe and then one effect
step inside it, and every parameter that step samples becomes a control clamped
to its prior's support in `distributions.yaml` — a slider for a continuous
distribution, a dropdown for a discrete one (weights shown), each opening at the
distribution's central value. Apply it and A/B the original against the result.

This answers the question reading the YAML cannot: what a given sampled value
actually sounds like. Effects are addressed recipe-first because
`distributions.yaml` is organized by workflow family, so a parameter's prior is
only discoverable through the step that samples it.

Three things it deliberately does not do:

- **One effect at a time.** Building whole chains is what `prompt_lab_ui.py`
  already does symbolically.
- **No separation.** 34 of 35 recipes open with a `separate` block, but demucs
  costs minutes and a checkpoint download, so the upload is used as the stem.
  The recipe's intended stem is shown; upload that stem yourself for fidelity.
- **No torch.** `graph/edit_graph.py` imports it at module scope and
  `planner.py`/`runtime.py` pull it in transitively, so this UI reimplements the
  two things it needs from them — the send-bus mix and the `soundfile` round
  trip. It runs with `fastapi uvicorn python-multipart soundfile numpy scipy
  pedalboard pyyaml` alone.

Steps on a `send_return` bus are mixed through an emulated bus by default. The
motifs pin those steps to `wet_level: 1.0, dry_level: 0.0` because
`add_send_return` supplies the dry path, so applying one raw gives a fully wet
signal with no reference; the bus trims are exposed as controls and a toggle
switches between emulated and raw. A `joint` prior (the compressor and shelf
settings) gets a mixture-component picker, since it models correlated
parameters and its sampled dict becomes the step's whole params map.

Parameters needing per-clip metadata are left at the operator's own default and
flagged — a tempo-synced delay falls back to the 120 BPM the recipe's `coalesce`
declares. Operators that cannot run on a bare array are listed but disabled with
the reason; `apply_harmony_effect` and `apply_autotune` need live `skey` models.
Peaks above full scale are reported rather than normalised, matching the
pipeline, which does not normalise either.

Two checks, neither needing audio or a browser:

```bash
python3 ground_truth/param_space.py      # every recipe effect builds a usable control set
python3 scripts/effect_lab_ui.py --selftest   # every available effect renders at its defaults
```
