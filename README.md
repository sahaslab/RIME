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
   Generated prompts -> per-row rejection verdicts (symbolic criteria)
6. `scripts/mert_similarity_report.py`
   Rendered audio -> MERT similarity and corpus percentile

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
python scripts/generate_ground_truth_prompts.py --resume
```

Model and run settings come from
[configs/ground_truth/summarization.yaml](configs/ground_truth/summarization.yaml),
so a prompt file is reproducible from config alone. Generation goes through
litellm to whichever provider that file names; the default is AWS Bedrock, which
needs `boto3` installed plus either a Bedrock API key in
`AWS_BEARER_TOKEN_BEDROCK` or credentials on the standard AWS chain (environment
variables, a `.env` file, a named profile, or an instance role).

Every flag below except `--resume` is an override: pass one and it wins, leave
it out and the value comes from `summarization.yaml`.

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
- `--resume`
  Append to the output file, skipping plans already written. Not an override:
  it describes one invocation rather than the run being reproduced, so it has
  no entry in `summarization.yaml`. Pass the same `--subsample-num` / `--seed`
  as the run being continued, or the resumed run draws a different sample and
  the plans it skips will not be the ones already on disk.

Each chain is written as soon as it finishes, so a run killed part-way keeps
every plan it had completed and `--resume` generates only the rest. Two
consequences worth knowing. Rows are in completion order rather than input
order, which is why each one carries its own `clip_id` and `plan_id`; sort on
those if a consumer needs a stable order. And a plan whose generation raises is
logged and skipped rather than written, so the failure never enters the prompt
file -- `filter_ground_truth_prompts.py` sniffs only the first row for format,
and a row with no levels to judge would otherwise be scored as a clean accept.
A resumed run retries those skipped plans.

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
new file. Symbolic criteria only — the embedding-space similarity is stage 8.

Each abstraction level gets **two independent statuses**, never combined:

| | |
| --- | --- |
| `mechanical` | `AbstractionLadder.check` re-run against the current config |
| `judged` | the judge's verdict on that level's `checks.rubric` |

`reject_policy` keys off `judged` alone. Folding a mechanical failure into
`al_rules` would have rejected 98.8% of the MusicCaps corpus for reasons about
the generator and the checker rather than about the plan, so the mechanical
status is reported and never rejects.

The checks are **re-run** rather than read from the row. The stored
`violations` were computed by whatever checker existed at generation time, and
before the `_value_renderings` fix that checker could not match a
full-precision float — so on any artifact generated earlier, half the recorded
AL0 failures are wrong. Re-running repairs them without regenerating.

Every judged level is judged regardless of its mechanical status. A `judged`
column that were null wherever the checks failed would carry no information,
and comparing the two is the point of keeping them apart.

**AL1 and AL2 are judged by default; AL0 is not.** AL0 is the verbatim
transcription of the graph, so its rubric — "every operator is named",
"operator order matches" — is close to what the mechanical checks already
settle, and it is a third of the calls for the least interesting level. The
abstraction proper is AL1 and AL2. Pass `--include_al_0` to judge it too.

Skipping AL0 costs none of the free signal: its mechanical status and its
leakage are still checked and reported, which matters because AL0 is where
almost all the leakage is. What it drops is AL0's rubric verdict and the
`graph->0` consistency pair; `0->1` survives, since that pair reads AL0's
*text*, not its verdict.

Budget accordingly: two calls per row by default, three with `--include_al_0`. The input is never modified, because `generate_agent_input.py` and the
prompt lab both read `prompt_variants` and `prompt_levels` out of it.

### Input

Agent input, from `scripts/generate_agent_input.py`, is the default. It is the
only artifact carrying **both** halves this stage needs: the prompt file has no
audio paths, and the render manifest has no prompts. So `--include-mert` needs
nothing else.

A bare prompt file from `generate_ground_truth_prompts.py` also works — the
format is detected, not configured — but then the similarity criterion needs a
`render_manifest` to find the audio. Either way the run logs which format it
read, and an input that is neither is refused rather than evaluated: a row with
no recognizable prompt has no levels to judge and no graph to trigger on, so
every criterion would return its default and the whole run would look like a
clean accept.

One caveat on the reference audio. Agent input's `input_audio` is the untouched
source, so a similarity computed from it measures the edit *plus* Demucs
separation loss. The no-FX baseline that isolates the effects chain exists only
in the render manifest, so supplying `--render-manifest` alongside agent input
upgrades the reference where a baseline was rendered. In poisoning mode agent
input sets `input_audio` to the *degraded* audio and moves the clean original to
`metadata.original_clean_audio`; the clean original is preferred there, because
comparing against the degraded version would measure repair fidelity rather than
edit strength.

| flag | output | question |
| --- | --- | --- |
| `al_rules` | binary | Does each level satisfy its own rules? |
| `implementation_leak` | `clean` / `leaked` | Does the text name a Python function or graph syntax? |
| `al_consistency` | binary + failure kinds | Does each level describe the one below it correctly? |
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

### Consistency Is Fidelity

`al_consistency` asks one question: **does this level describe the level below
it correctly?** Not whether it obeys the ladder's abstraction contract — that
is `al_rules`, judged against each level's own `checks.rubric`. A rewrite can be
abstracted perfectly and still be about a different edit, and that is what this
flag exists to catch.

The verdict is binary, and a failure also names its kind:

| failure | meaning |
| --- | --- |
| `wrong_interpretation` | the source is described, but wrongly — a value landing in the wrong band, an effect misnamed, the order inverted where order is preserved |
| `hallucination` | processing appears that the source does not contain |
| `omission` | something the level was obliged to carry is missing |

`wrong_interpretation` and `hallucination` reject: both mean the instruction
describes audio the graph will not produce, which is the one thing a training
pair cannot survive. `omission` does not reject by default — at high
abstraction a dropped processor is usually the difference between a terse
instruction and a complete one rather than a falsehood. `--omission-reject`
makes it fatal too, additively, so it means "and omission as well" whatever
`reject_policy.al_consistency_failures` already lists.

The verdict list and the failure list do different jobs, which is worth knowing
when reading a report. `al_consistency: [fail]` decides whether the judge's
sentence is printed; `al_consistency_failures` decides whether the row is
thrown away. So an omission-only row is `fail`, carries both its
`al_consistency_failures` and its `al_consistency_why`, and is still accepted —
a recorded failure nobody can read would not be a record of anything.

**What makes omission answerable.** AL2 is supposed to drop every parameter
value, so a judge asked "was anything left out" with no further context would
fail every correctly-written level. Each prompt therefore opens with a `WHAT
THIS LEVEL IS LICENSED TO DISCARD` block, built from the level's own
`constraints` glossed through `constraint_glosses`, closing with "Discarding any
of the above is correct and is never an omission." Omission then means only:
dropping something the level was obliged to keep.

The glosses are phrased in the same three words the judge answers in, so the
licence and the verdict cannot drift apart — `parameter_values: banded` says
that losing the numerals is licensed while landing in the wrong band is a
`wrong_interpretation`. Nothing is written pairwise, so a new abstraction level
is covered as soon as its constraint values appear in that table.

One boundary is fixed in the rules rather than left to the judge: misnaming an
effect the source does contain is `wrong_interpretation`, not `hallucination`.
Hallucination means added processing, not a wrong label on existing processing.

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

### Flat Report

```bash
python scripts/filter_ground_truth_prompts.py --report
python scripts/filter_ground_truth_prompts.py --report-path out.jsonl
```

One flat row per record at `paths.report` — the prompts, their ids, and one
column per flag:

```json
{"clip_id": "m7i4g_o-znQ", "plan_id": "hum_cleanup.001",
 "caption": "slide guitar blues country blues intricate acoustic guitar playing...",
 "graph_description": "hum_cleanup: audio -> apply_highpass_filter(cutoff_frequency_hz=55.63) -> final_audio",
 "prompt_variants": ["Apply a high-pass filter to the hum_cleanup stem...", "...", "..."],
 "al_rules": "pass", "al_consistency": "fail",
 "al_consistency_failures": ["hallucination"],
 "al_consistency_why": "0->1: AL1 adds a parallel compression bus the graph does not contain.",
 "joint_params": "plausible", "stylistic": "appropriate",
 "implementation_leak": "leaked", "implementation_leak_levels": ["0"],
 "implementation_leak_tokens": ["hum_cleanup"],
 "disposition": "reject", "reject_reasons": ["implementation_leak"]}
```

`caption` is what the stylistic criterion actually read, so a verdict about the
music can be checked against the description behind it. On the MusicCaps
artifact the raw caption is null and this is the mined aspect phrases.

`<flag>_why` carries the judge's one-sentence justification, and **only on a
verdict that failed** — a reason attached to a pass is noise in a table nobody
will read. Which verdicts count as failing comes from `reject_policy`, so adding
`unclear` to a criterion's reject list makes its reasons appear too. Failing is
not quite rejecting for `al_consistency`, whose failure kinds decide that
separately; an omission-only row explains itself and is still accepted. Every
criterion is instructed to give exactly one sentence, because "inappropriate"
with no reason is not something anyone can act on or check.

`al_rules` and `al_consistency` get their own sentence each rather than one
shared: they answer different questions and can disagree. The failing level is
named, so `al_rules_why` reads `1: The instruction never names the compressor`.
The nested sidecar keeps every rationale, passing ones included.

`prompt_variants` is carried verbatim because it is what was judged — it is
exactly `[e["text"] for e in prompt_levels]` ordered by level, so a verdict can
be read against its text without opening the input. `graph_description` is the
rendered graph the judge was shown in its EDIT GRAPH block, so the prompts, the
thing they describe and the verdict about them all sit in one row. The
structured `graph_spec` is five times the size and lives in the full manifest.

**Judged verdicts only.** The mechanical `checks.hard` results are deliberately
absent: they are a regex's answer to a different question, and on this corpus
they fail often enough to drown the verdicts this table exists to show. They are
still computed and still written to the sidecar. `implementation_leak` is here
despite also being deterministic, because it is a criterion in its own right
rather than part of the ladder's hard checks.

A flag with no verdict also writes a `<flag>_reason` column, so a null is never
indistinguishable from a criterion that did not run.

The nested sidecar at `paths.output` keeps the full audit trail — rationale,
cited rules, per-level rubric indices — and is what `--resume` reads back.

### Implementation Leakage

An instruction that names `separate_audio`, writes
`apply_reverb_effect(room_size=0.9)` or carries a `->` out of the graph is
describing the implementation rather than asking for a sound. It is not usable
as training text whatever else is right about it, so it gets its own flag —
deterministic, costing no judge call.

The identifiers are **derived**, not listed: every operator name, alias and
parameter from `operators.yaml`, plus the step labels and stem handles
(`target_stem`, `residual`, `hum_cleanup`) from each row's own `graph_spec`. A
new operator or a renamed motif step is covered with no edit. Only
underscore-bearing identifiers are matched — `drums` and `reverb` are what a
person would say; `apply_reverb_effect` and `room_size` are not.

Measured on 1000 rows of the MusicCaps artifact, **28% of rows leak**, almost
entirely at AL0, including texts that are the `graph_description` pasted
verbatim. The commonest tokens are `separate_audio`, `final_audio`,
`processed_stem`, `target_stem` and `mix_stems`.

This flag **does reject**, since such a row is unusable. Empty
`reject_policy.implementation_leak` in `rejection.yaml` to demote it to a
reported-only signal. `implementation_leak_tokens` names what leaked, which is
the actionable part.

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

### Running Without Captions

```bash
python scripts/filter_ground_truth_prompts.py --without-captions
```

Drops every criterion declaring `requires: [caption]`, for a corpus that has
none — MTG-Jamendo, or the mock manifest. This is not cosmetic: without it the
stylistic criterion reports `unclear` / `no_caption`, which routes through
`reject_policy.unclear.stylistic: review` and sends **every** row to the review
queue.

The flag reads each criterion's own `requires`, so a caption-dependent criterion
added later is covered without touching the flag, and the run logs which
criteria it dropped and why.

## 8. MERT Similarity

```bash
sbatch scripts/run_mert_similarity_report.sbatch                   # source vs render, one manifest
sbatch scripts/run_calculate_similarity.sbatch                     # reference-resolving variant
```

The cosine distance in MERT embedding space between the original and the edited
audio. This is the one criterion that needs audio rather than symbols, and the
only one that can catch an edit too subtle to hear.

**Deliberately separate from stage 7.** The judge is an I/O-bound job over text;
this is a GPU job over audio with a different runtime and a different failure
mode. Keeping them apart also keeps torch and fadtk off the filter's import
path. Join the two reports on `(clip_id, plan_id)` if you want one table.

Two scripts, differing only in which "original" they compare against:

| script | reference | when |
| --- | --- | --- |
| [mert_similarity_report.py](scripts/mert_similarity_report.py) | always the manifest's `audio_path` | one manifest, lean output, simplest thing that answers "how far did this move" |
| [calculate_ground_truth_similarity.py](scripts/calculate_ground_truth_similarity.py) | resolves per row, preferring the no-FX baseline | when the effects chain must be isolated from Demucs separation loss |

Both share the model, the embedding cache and the percentile maths, so they agree
by construction. Run them on `gpu_preempt` — measured at 0.220 s per embedding on
CPU, a 23k-row manifest is ~1.6 h of CPU or well under an hour on a GPU, and the
per-row flush plus `--resume` makes preemption cost one row.

Embedding reuses `fadtk.MERTModel` (`m-a-p/MERT-v1-95M`, 768-dim, 24 kHz),
already installed in `postmaster-clean`, subclassed to replace only the pooling.
`layer: all` averages the 13-layer stack then the time axis; `layer: 12`
reproduces an unmodified `fadtk.MERTModel()` **bit-exactly**, which is the
setting under which these numbers are comparable to the lab's existing FAD/KAD
analysis under `/dartfs/rc/lab/S/SinghN/rime/rime_analysis`.

Which "original" is used matters, and the render manifest carries three:

| `reference` | source | measures |
| --- | --- | --- |
| `baseline` | `baseline_path`, the no-FX remix — render manifest only | the effects chain alone, since separation artifacts cancel |
| `source` | agent input's `input_audio`, or `source_copy_path`/`audio_path` | the edit *plus* Demucs separation loss |
| `poisoned_input` | a poison row with no clean original recorded | repair fidelity, not edit strength — never pool these |

Which one was used is recorded per row, because the two are not poolable. Poison
plans invert the pair — their `baseline_path` is the *degraded input*, so those
rows compare the source against the output and are marked `poison_repair`; do
not pool those either.

`similarity_threshold` is null by default, which keeps the criterion descriptive:
the cosine is recorded and the flag's verdict stays null, so nothing is rejected
on a cutoff nobody has chosen from data yet. Set a float and it becomes an
ordinary rejection reason. The raw value is always written, so re-thresholding
never re-runs MERT.

Calibrate before trusting a raw threshold. On real pairs the cosine sits in a
very narrow band near 1.0 — a gain-only edit measured 0.99889 against its
baseline where a reverb-plus-delay send measured 0.99836, so the ordering is
right but the whole signal spans about 5e-4, while the spread *between* clips is
two orders of magnitude wider. That is what the percentile labels below are for:
a rank discriminates where the raw value does not.

### Percentile labels

Every scored row also carries its rank in the corpus:

| field | |
| --- | --- |
| `percentile` | 0–100, where 100 is the *most* similar to the original, i.e. the least audible edit |
| `percentile_bucket` | quartile, ascending by similarity: `most_audible`, `audible`, `subtle`, `least_audible` |
| `percentile_n` | the population the rank was taken against |

The buckets are named for what the number means rather than Q1–Q4, because a
high similarity is a *small* change: `least_audible` is the quartile at risk of
being imperceptible, which is the whole reason this criterion exists. On a
sample of twelve real pairs the ordering came out as you would hope — drum
distortion in `most_audible`, a 7 ms chorus at low mix in `least_audible`.

Labelling is a **second pass**, because a percentile describes a whole
population and cannot be computed while scoring is still adding to it. It runs
automatically at the end of a scoring run, and can be re-run on its own:

```bash
python scripts/calculate_ground_truth_similarity.py --label-percentiles
```

Two consequences worth knowing. A `--resume` that appends rows makes every
existing label stale, which is why `percentile_n` is recorded — the filter
compares it against what is actually in the file and warns rather than reporting
a silently wrong rank. And a sharded run should pass `--no-label-percentiles`,
then be labelled once at the end, or each shard ranks against its own slice
instead of the corpus. Rows with no similarity are left null and excluded from
the population, so the ranks do not depend on how much of the corpus has been
rendered.

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
to its prior in `distributions.yaml`. A continuous prior becomes a dropdown of
its own deciles, so every option is an equally likely tenth of that prior, with
a free numeric field alongside for deliberate overrides; a discrete prior keeps
its own weighted values. Each opens at the median. Apply it and A/B the original
against the result.

Deciles rather than sliders because these priors are fitted over supports far
wider than the mass they carry: the high-shelf `q` is fitted over 0.1–50 but 80%
of its draws land between 0.4 and 1.4, which is under 2% of a linear track. The
dropdown reports the shape a slider cannot, and each control also names its
p10–p90 range and reads its fit in words ("log-normal, centred near 81",
"2-mode prior, peaking near 0.63, 0.98").

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

The first needs only `pyyaml` and `scipy` (scipy for the `beta` and truncated
`normal` quantiles), so a `distributions.yaml` edit can be checked without the
audio stack.
