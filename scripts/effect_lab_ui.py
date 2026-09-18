"""Audition the parameter priors in distributions.yaml against your own audio.

Upload a .wav, pick a recipe and then one effect step inside it, and every
parameter that step samples becomes a control clamped to the prior's support:
a slider for a continuous distribution, a dropdown for a discrete one, each
opening at the distribution's central value. Apply it and A/B the result.

Scope, and why:

- One effect at a time. Reading a number in distributions.yaml tells you nothing
  about how it sounds; this closes that loop. Building whole chains is what
  scripts/prompt_lab_ui.py already does symbolically.
- Separation is skipped. 34 of 35 recipes open with a `separate` block, but
  demucs costs minutes and a checkpoint download, so your upload is treated as
  the stem. The recipe's intended stem is shown for context.
- Send/return steps are mixed through an emulated bus by default. The motifs pin
  those steps to wet_level 1.0 / dry_level 0.0 because add_send_return supplies
  the dry path, so applying one raw gives a fully wet signal and no reference.
  The toggle switches between the two.

No torch: graph/edit_graph.py imports it at module scope and planner.py and
runtime.py pull it in transitively, so this file deliberately reimplements the
two pieces it needs from them -- the send-bus mix and the soundfile round trip.

Usage:
    python scripts/effect_lab_ui.py --port 8789
    python scripts/effect_lab_ui.py --selftest
"""

import os
import sys
import uuid
import atexit
import shutil
import socket
import getpass
import inspect
import argparse
import tempfile
import importlib
import threading
import traceback
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import uvicorn
import soundfile as sf
from fastapi import Body, File, FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = Path("configs/ground_truth")
if str(DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO_ROOT))

from ground_truth import param_space
from ground_truth.operators import load_operator_registry

# Longest clip kept. Effects run per request and apply_deesser's envelope
# follower is a Python loop over samples, so a long upload makes the A/B sluggish
# for no benefit. Same idea as _ensure_ground_truth_audio (backend/server.py:1158).
DEFAULT_MAX_SECONDS = 30.0

# Signature parameters an effect callable may require and still be usable here:
# the audio itself and the sample rate are the two things we always have.
ALWAYS_SUPPLIED_PARAMS = {"audio", "sr", "sample_rate", "stem"}

APP_STATE: dict[str, Any] = {
    "repo_root": DEFAULT_REPO_ROOT,
    "config_dir": DEFAULT_REPO_ROOT / DEFAULT_CONFIG_DIR,
    "max_seconds": DEFAULT_MAX_SECONDS,
    "work_dir": None
}
UPLOADS: dict[str, dict[str, Any]] = {}
CALLABLE_CACHE: dict[str, Any] = {}
CALLABLE_LOCK = threading.Lock()

app = FastAPI()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host. Default: %(default)s")
    parser.add_argument("--port", type=int, default=8789, help="Bind port. Default: %(default)s")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root. Default: %(default)s")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Ground-truth config directory. Default: <repo-root>/%s" % DEFAULT_CONFIG_DIR
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Truncate uploads to this many seconds. Default: %(default)s"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Apply every available effect at its defaults to a generated clip, then exit."
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    APP_STATE["repo_root"] = repo_root
    APP_STATE["config_dir"] = (
        args.config_dir.expanduser().resolve() if args.config_dir is not None else repo_root / DEFAULT_CONFIG_DIR
    )
    APP_STATE["max_seconds"] = float(args.max_seconds)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.selftest:
        raise SystemExit(selftest())

    warm_quantiles()
    print_access_hint(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


def warm_quantiles() -> None:
    """Pay scipy's import cost at startup rather than on the first click.

    param_space imports scipy lazily, inside the one function that needs it, so
    that checking a distributions.yaml edit does not require the audio stack.
    That leaves the first request touching a `beta` or `normal` prior wearing a
    one-to-two second import, which in a UI reads as a stall on the first thing
    the user does.
    """
    try:
        param_space.fitted_quantile({"type": "beta", "alpha": 2.0, "beta": 2.0, "low": 1.0, "high": 10.0}, 0.5)
        param_space.fitted_quantile({"type": "normal", "mean": 0.0, "std": 1.0, "low": -3.0, "high": 3.0}, 0.5)
    except Exception as error:
        print("  warning: could not warm the quantile helpers (%s)" % format_error(error), flush=True)


def print_access_hint(host: str, port: int) -> None:
    """Print the SSH tunnel needed to reach this server from a laptop.

    Lifted from prompt_lab_ui.py:127 -- the UI is useless without a browser on
    it, and on a cluster the browser is never on the same machine.
    """
    node = socket.getfqdn()
    user = getpass.getuser()
    job_id = os.environ.get("SLURM_JOB_ID")
    submit_host = os.environ.get("SLURM_SUBMIT_HOST", "<login-host>")
    print("effect lab serving on %s:%d (node %s)" % (host, port, node), flush=True)
    if job_id:
        print(
            "  tunnel:  ssh -N -J %s@%s -L %d:127.0.0.1:%d %s@%s   # slurm job %s"
            % (user, submit_host, port, port, user, node, job_id),
            flush=True
        )
    else:
        print("  tunnel:  ssh -N -L %d:127.0.0.1:%d %s@%s" % (port, port, user, node), flush=True)
    print("  open:    http://127.0.0.1:%d" % port, flush=True)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("  warning: bound to %s, so anyone on this network can drive the UI." % host, flush=True)


# ----------------------------------------------------------------------
# State and config
# ----------------------------------------------------------------------
def config_dir_path() -> Path:
    return Path(APP_STATE["config_dir"]).resolve()


def work_dir() -> Path:
    if APP_STATE["work_dir"] is None:
        APP_STATE["work_dir"] = Path(tempfile.mkdtemp(prefix="effect_lab_"))
        atexit.register(shutil.rmtree, APP_STATE["work_dir"], True)
    return Path(APP_STATE["work_dir"])


def load_state() -> dict[str, Any]:
    """Configs, distribution leaves, recipes and registry, re-read from disk.

    Per request, like prompt_lab_ui.py:161, so a config edit shows up without a
    restart.
    """
    config_dir = config_dir_path()
    bundle = param_space.load_configs(config_dir)
    return {
        "leaves": param_space.distribution_leaves(bundle["distributions"]),
        "recipes": param_space.resolve_recipe_effects(bundle["recipes"], bundle["motifs"]),
        "registry": load_operator_registry(config_dir)
    }


def format_error(error: Exception) -> str:
    return "%s: %s" % (type(error).__name__, error)


def find_effect(state: Mapping[str, Any], recipe_id: str, effect_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for recipe in state["recipes"]:
        if recipe["id"] != recipe_id:
            continue
        for effect in recipe["effects"]:
            if effect["key"] == effect_key:
                return recipe, effect
        raise HTTPException(404, "Recipe '%s' has no effect '%s'." % (recipe_id, effect_key))
    raise HTTPException(404, "Unknown recipe '%s'." % recipe_id)


# ----------------------------------------------------------------------
# Operator resolution
# ----------------------------------------------------------------------
def resolve_effect_callable(registry: Any, operator_name: str) -> Any:
    """Import the DSP function an operator binds to.

    Same three lines as RuntimePlanCompiler._resolve_callable
    (ground_truth/runtime.py:147), without the logging wrapper or the graph.
    """
    with CALLABLE_LOCK:
        if operator_name in CALLABLE_CACHE:
            return CALLABLE_CACHE[operator_name]
        runtime_spec = registry.runtime_spec(operator_name)
        module = importlib.import_module(runtime_spec["module"])
        function = getattr(module, runtime_spec["callable"])
        CALLABLE_CACHE[operator_name] = function
        return function


def operator_availability(registry: Any, operator_name: str) -> tuple[bool, str | None]:
    """Whether an operator can run on a bare audio array, and why not.

    Decided from the callable's own signature rather than a denylist: anything
    with a required parameter we cannot supply needs the full pipeline. That is
    how apply_autotune and apply_harmony_effect are caught -- they want live
    hcqt/chromanet/crop_fn/device objects from skey (libraries/pitch.py:11).
    """
    try:
        function = resolve_effect_callable(registry, operator_name)
    except Exception as error:
        return False, "its DSP function will not import here (%s)" % format_error(error)
    signal_param = registry.resolve(operator_name).signal_param
    missing = [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        and name not in ALWAYS_SUPPLIED_PARAMS | {signal_param}
    ]
    if missing:
        return False, "needs %s, which only the full pipeline supplies" % ", ".join(missing)
    return True, None


# ----------------------------------------------------------------------
# Audio IO. Mirrors load_audio/save_audio (backend/server.py:298) without
# importing that module, which pulls in torch, fastmcp and demucs.
# ----------------------------------------------------------------------
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a file as channel-first float32 [channels, frames]."""
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return np.ascontiguousarray(audio.T), int(sample_rate)


def save_audio(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    """Write channel-first audio, passing no subtype just as save_audio does.

    libsndfile defaults a .wav to 16-bit PCM even for float input, so anything
    over full scale is clipped on write. Matching backend/server.py:314 rather
    than quietly gaining headroom keeps what you hear here the same as what the
    pipeline would render; audio_warnings() says so when it happens.
    """
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(array.T), sample_rate)
    return Path(path)


def apply_effect(
    audio: np.ndarray,
    sample_rate: int,
    function: Any,
    params: Mapping[str, Any],
    bus: Mapping[str, float] | None
) -> tuple[np.ndarray, list[str]]:
    """Run one effect, optionally through an emulated send/return bus.

    The bus arithmetic mirrors EditGraph.add_send_return
    (graph/edit_graph.py:265-339): optional dry trim, send trim, the fx, return
    trim, then a sum. Written out in numpy here so this UI needs no torch.
    """
    warnings: list[str] = []
    if bus is None:
        return np.asarray(function(audio, sample_rate, **params)), warnings

    dry = audio * float(bus.get("dry_level", 1.0))
    wet = np.asarray(function(audio * float(bus.get("send_level", 1.0)), sample_rate, **params))
    wet = wet * float(bus.get("return_level", 1.0))
    if wet.shape != dry.shape:
        # add_send_return sums a dry and wet path that an effect is assumed to
        # keep aligned; pedalboard does, but say so rather than broadcast-error.
        frames = min(dry.shape[-1], wet.shape[-1])
        warnings.append(
            "the effect returned %d frames against %d dry; both trimmed to %d for the bus mix"
            % (wet.shape[-1], dry.shape[-1], frames)
        )
        dry, wet = dry[..., :frames], wet[..., :frames]
    return dry + wet, warnings


def audio_warnings(params: Mapping[str, Any], sample_rate: int, processed: np.ndarray) -> list[str]:
    """Problems worth surfacing about one render, rather than silently fixing."""
    warnings: list[str] = []
    nyquist = sample_rate / 2.0
    for name, value in params.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and name.endswith("_hz"):
            if float(value) >= nyquist:
                warnings.append(
                    "%s is %.0f Hz, at or above this file's Nyquist of %.0f Hz; the filter is undefined there"
                    % (name, float(value), nyquist)
                )
    if not np.isfinite(processed).all():
        warnings.append("output contains non-finite samples")
    peak = float(np.abs(processed).max()) if processed.size else 0.0
    if peak > 1.0:
        warnings.append(
            "peak is %.2f, above full scale, so the .wav write clips it. Not normalised here, because "
            "save_audio (backend/server.py:314) does not normalise either -- the pipeline would clip too" % peak
        )
    return warnings


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    """Recipes, their effect steps, and which of those can actually run."""
    state = load_state()
    availability: dict[str, tuple[bool, str | None]] = {}
    recipes = []
    for recipe in state["recipes"]:
        effects = []
        for effect in recipe["effects"]:
            operator_name = effect["operator"]
            if operator_name not in availability:
                availability[operator_name] = operator_availability(state["registry"], operator_name)
            available, reason = availability[operator_name]
            effects.append(
                {
                    "key": effect["key"],
                    "block": effect["block"],
                    "block_kind": effect["block_kind"],
                    "step": effect["step"],
                    "operator": operator_name,
                    "motif": effect["motif"],
                    "group": effect["group"],
                    "available": available,
                    "unavailable_reason": reason
                }
            )
        recipes.append(
            {
                "id": recipe["id"],
                "description": recipe["description"],
                "tags": recipe["tags"],
                "target_hint": recipe["target_hint"],
                "effects": effects
            }
        )
    return {
        "config_dir": str(config_dir_path()),
        "max_seconds": APP_STATE["max_seconds"],
        "recipes": recipes
    }


@app.post("/api/effect")
def effect(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """The control schema for one effect step, resolved at the given values.

    Takes values so the same walk that builds the controls also recomputes the
    derived ones -- a tempo-synced delay time, a scaled gain -- and reselects a
    joint distribution's component. An empty `values` gives the defaults.
    """
    state = load_state()
    recipe_spec, effect_spec = find_effect(state, str(body.get("recipe", "")), str(body.get("effect", "")))
    try:
        schema = param_space.effect_schema(
            effect_spec, state["leaves"], recipe_spec["bindings"], body.get("values") or {}
        )
    except Exception as error:
        raise HTTPException(400, format_error(error)) from error
    available, reason = operator_availability(state["registry"], effect_spec["operator"])
    schema["available"] = available
    schema["unavailable_reason"] = reason
    schema["target_hint"] = recipe_spec["target_hint"]
    schema["params"] = _jsonable(schema["params"])
    return schema


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Stash an uploaded clip, truncated to --max-seconds."""
    upload_id = uuid.uuid4().hex
    directory = work_dir() / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    raw_path = directory / ("raw%s" % suffix)
    raw_path.write_bytes(await file.read())

    try:
        audio, sample_rate = load_audio(raw_path)
    except Exception as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(400, "Could not decode '%s'. %s" % (file.filename, format_error(error))) from error

    max_frames = int(APP_STATE["max_seconds"] * sample_rate)
    truncated = audio.shape[-1] > max_frames
    if truncated:
        audio = audio[..., :max_frames]
    original_path = save_audio(audio, sample_rate, directory / "original.wav")
    raw_path.unlink(missing_ok=True)

    UPLOADS[upload_id] = {
        "directory": directory,
        "original": original_path,
        "processed": None,
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "name": file.filename or "upload.wav",
        "version": 0
    }
    return {
        "upload_id": upload_id,
        "name": UPLOADS[upload_id]["name"],
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "seconds": audio.shape[-1] / float(sample_rate),
        "truncated": truncated,
        "max_seconds": APP_STATE["max_seconds"]
    }


@app.post("/api/apply")
def apply(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Resolve the submitted control values and render the effect."""
    upload_id = str(body.get("upload_id", ""))
    record = UPLOADS.get(upload_id)
    if record is None:
        raise HTTPException(404, "No upload '%s'. Upload a .wav first." % upload_id)

    state = load_state()
    recipe_spec, effect_spec = find_effect(state, str(body.get("recipe", "")), str(body.get("effect", "")))
    values = body.get("values") or {}
    emulate_bus = bool(body.get("emulate_bus", True))

    try:
        schema = param_space.effect_schema(effect_spec, state["leaves"], recipe_spec["bindings"], values)
        registry = state["registry"]
        registry.resolve(effect_spec["operator"]).validate_params(schema["params"])
        available, reason = operator_availability(registry, effect_spec["operator"])
        if not available:
            raise HTTPException(400, "%s cannot run here: %s" % (effect_spec["operator"], reason))
        function = resolve_effect_callable(registry, effect_spec["operator"])
        audio, sample_rate = load_audio(record["original"])
        bus = schema["bus"] if (emulate_bus and schema["bus"]) else None
        processed, warnings = apply_effect(audio, sample_rate, function, schema["params"], bus)
        warnings.extend(audio_warnings(schema["params"], sample_rate, processed))
        record["processed"] = save_audio(processed, sample_rate, record["directory"] / "processed.wav")
        record["version"] += 1
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(400, format_error(error)) from error

    return {
        "params": _jsonable(schema["params"]),
        "bus": schema["bus"],
        "bus_applied": bus is not None,
        "notes": schema["notes"],
        "warnings": warnings,
        "peak": float(np.abs(processed).max()) if processed.size else 0.0,
        "version": record["version"]
    }


@app.get("/api/audio/{upload_id}/{which}")
def audio(upload_id: str, which: str) -> FileResponse:
    record = UPLOADS.get(upload_id)
    if record is None:
        raise HTTPException(404, "No upload '%s'." % upload_id)
    if which not in {"original", "processed"}:
        raise HTTPException(404, "No audio '%s'." % which)
    path = record["original"] if which == "original" else record["processed"]
    if path is None or not Path(path).exists():
        raise HTTPException(404, "Nothing rendered yet.")
    return FileResponse(str(path), media_type="audio/wav")


def _jsonable(value: Any) -> Any:
    """Make numpy scalars from the DSP layer survive JSON encoding."""
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ----------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------
def selftest() -> int:
    """Apply every available effect at its defaults to a generated clip."""
    state = load_state()
    sample_rate = 44100
    seconds = 2.0
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 3000 * time)
    audio = np.ascontiguousarray(np.stack([tone, tone * 0.8]).astype(np.float32))

    failures: list[str] = []
    ran = skipped = 0
    for recipe in state["recipes"]:
        for effect_spec in recipe["effects"]:
            where = "%s/%s" % (recipe["id"], effect_spec["key"])
            available, reason = operator_availability(state["registry"], effect_spec["operator"])
            if not available:
                print("skip  %-58s %s" % (where, reason))
                skipped += 1
                continue
            try:
                schema = param_space.effect_schema(effect_spec, state["leaves"], recipe["bindings"])
                state["registry"].resolve(effect_spec["operator"]).validate_params(schema["params"])
                function = resolve_effect_callable(state["registry"], effect_spec["operator"])
                for bus in ([schema["bus"], None] if schema["bus"] else [None]):
                    processed, _ = apply_effect(audio, sample_rate, function, schema["params"], bus)
                    processed = np.asarray(processed)
                    if processed.shape != audio.shape:
                        raise AssertionError("shape %r != input %r" % (processed.shape, audio.shape))
                    if not np.isfinite(processed).all():
                        raise AssertionError("non-finite samples in output")
                ran += 1
                print("ok    %-58s peak %.3f" % (where, float(np.abs(processed).max())))
            except Exception as error:
                failures.append("%s: %s" % (where, format_error(error)))
                print("FAIL  %-58s %s" % (where, format_error(error)))

    print()
    print("applied %d, skipped %d, failed %d" % (ran, skipped, len(failures)))
    if failures:
        for failure in failures:
            print("  %s" % failure)
        return 1
    return 0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>effect lab</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 10px 16px; border-bottom: 1px solid #8884; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; }
  header .meta { opacity: 0.7; font-size: 12px; }
  main { display: grid; grid-template-columns: minmax(380px, 480px) 1fr; gap: 0; height: calc(100vh - 48px); }
  .col { overflow-y: auto; padding: 14px 16px; }
  .col + .col { border-left: 1px solid #8884; }
  fieldset { border: 1px solid #8884; border-radius: 6px; margin: 0 0 12px; padding: 10px 12px; }
  legend { padding: 0 4px; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }
  label { display: block; font-size: 11px; opacity: 0.75; margin-bottom: 2px; }
  input, select, button { font: inherit; padding: 3px 6px; border-radius: 4px; border: 1px solid #8886; background: transparent; color: inherit; width: 100%; }
  button { cursor: pointer; width: auto; }
  button.primary { border-color: currentColor; font-weight: 600; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
  .inline { display: flex; gap: 6px; align-items: center; font-size: 12px; opacity: 1; }
  .inline input[type=checkbox] { width: auto; }
  .hint { font-size: 10px; opacity: 0.55; margin-top: 2px; }
  .hint.src { font-size: 10px; opacity: 0.45; }
  .routing { font-size: 11px; opacity: 0.8; background: #8881; border-radius: 4px; padding: 7px 9px; margin: 0 0 8px; }
  .ctl { margin-bottom: 9px; }
  .ctl.pinned { opacity: 0.65; }
  .pickrow { display: flex; gap: 6px; align-items: center; }
  .pickrow select { flex: 1 1 auto; }
  .pickrow input[type=number] { flex: 0 0 104px; }
  .ro { padding: 3px 6px; border: 1px dashed #8886; border-radius: 4px; opacity: 0.8; }
  .layer { border: 1px solid #8884; border-radius: 6px; margin-bottom: 12px; overflow: hidden; }
  .layer > h2 { font-size: 12px; margin: 0; padding: 7px 10px; background: #8881; display: flex; justify-content: space-between; gap: 10px; }
  .layer > h2 span.tag { font-weight: 400; opacity: 0.7; }
  .layer .body { padding: 10px; }
  pre { margin: 0; padding: 10px; white-space: pre-wrap; word-break: break-word; }
  .player { margin-bottom: 10px; }
  .player audio { width: 100%; }
  .player .name { font-size: 11px; opacity: 0.7; margin-bottom: 3px; }
  ul { margin: 0; padding-left: 18px; }
  .warn { color: #b26a00; }
  .bad { color: #c62828; }
  .ok { color: #2e7d32; }
  .muted { opacity: 0.6; }
  .tags { font-size: 11px; opacity: 0.6; }
</style>
</head>
<body>
<header>
  <h1>effect lab</h1>
  <span class="meta" id="meta">loading…</span>
</header>
<main>
  <div class="col">
    <fieldset>
      <legend>audio</legend>
      <input type="file" id="file" accept=".wav,audio/wav,audio/x-wav" />
      <div class="hint" id="file-info">no file yet</div>
    </fieldset>

    <fieldset>
      <legend>recipe</legend>
      <select id="recipe"></select>
      <div class="hint" id="recipe-info"></div>
    </fieldset>

    <fieldset>
      <legend>effect</legend>
      <select id="effect"></select>
      <div class="hint" id="effect-info"></div>
    </fieldset>

    <fieldset id="bus-set" hidden>
      <legend>send bus</legend>
      <label class="inline"><input type="checkbox" id="emulate-bus" checked /> emulate the bus (dry + wet)</label>
      <div class="routing" id="bus-note"></div>
      <div id="bus-controls"></div>
    </fieldset>

    <fieldset>
      <legend>parameters</legend>
      <div id="controls"></div>
    </fieldset>

    <div class="row">
      <button class="primary" id="apply">apply effect</button>
      <button id="reset">reset to priors</button>
    </div>
  </div>

  <div class="col">
    <div class="layer">
      <h2>listen</h2>
      <div class="body">
        <div class="player"><div class="name">original</div><audio id="original" controls></audio></div>
        <div class="player"><div class="name">processed</div><audio id="processed" controls></audio></div>
      </div>
    </div>
    <div id="output"></div>
  </div>
</main>
<script>
const COMPONENT_ID = "__component__";
const state = { catalog: null, recipe: null, schema: null, values: {}, upload: null };
let refreshTimer = null;

const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

// Sliders and number boxes show a trimmed value, and that trimmed value is what
// gets applied: 6 significant digits is already finer than any of these priors
// are fitted to, and the raw float tail just makes the column unreadable.
function inputValue(value) {
  return typeof value === "number" && Number.isFinite(value) ? String(parseFloat(value.toPrecision(6))) : value;
}

// A value counts as on a decile only if it is that decile, to floating-point
// tolerance -- the number box can hold anything, including a neighbouring value.
function matchDecile(control, value) {
  return (control.options || []).find(
    (option) => Math.abs(option.value - value) <= Math.abs(value) * 1e-9 + 1e-12
  ) || null;
}

// Labels and hints read better rounded: four significant digits is plenty for
// choosing a decile, while the editable box keeps the fuller value.
function fmtShort(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? String(parseFloat(value.toPrecision(4)))
    : fmt(value);
}

function fmt(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(parseFloat(value.toPrecision(6)));
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

function layer(title, tag, body) {
  return `<div class="layer"><h2>${escapeHtml(title)}<span class="tag">${escapeHtml(tag || "")}</span></h2><div class="body">${body}</div></div>`;
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({ detail: "the server sent a non-JSON reply" }));
  return { ok: response.ok, payload };
}

// ---------------------------------------------------------------- catalog
async function loadCatalog() {
  const response = await fetch("/api/catalog");
  state.catalog = await response.json();
  el("meta").textContent = `${state.catalog.recipes.length} recipes · ${state.catalog.config_dir} · clips truncated to ${state.catalog.max_seconds}s`;
  el("recipe").innerHTML = state.catalog.recipes
    .map((recipe) => `<option value="${escapeHtml(recipe.id)}">${escapeHtml(recipe.id)}</option>`)
    .join("");
  await selectRecipe();
}

async function selectRecipe() {
  state.recipe = state.catalog.recipes.find((recipe) => recipe.id === el("recipe").value);
  const target = state.recipe.target_hint ? `targets ${escapeHtml(state.recipe.target_hint)}` : "no stem constraint";
  el("recipe-info").innerHTML =
    `${escapeHtml(state.recipe.description)}<div class="tags">${escapeHtml((state.recipe.tags || []).join(" · "))}</div>` +
    `<div class="tags">${target} — separation is skipped, so your upload is used as the stem</div>`;

  const groups = new Map();
  state.recipe.effects.forEach((effect) => {
    if (!groups.has(effect.group)) groups.set(effect.group, []);
    groups.get(effect.group).push(effect);
  });
  el("effect").innerHTML = [...groups.entries()]
    .map(([group, effects]) => {
      const label = group === "poison" ? "poison — degradation the task then corrects" : "edit";
      const options = effects
        .map((effect) => {
          const bus = effect.block_kind === "send_return" ? " [bus]" : "";
          const suffix = effect.available ? "" : " — unavailable";
          const label = `${effect.block} / ${effect.step} — ${effect.operator}${bus}`;
          const disabled = effect.available ? "" : " disabled";
          return `<option value="${escapeHtml(effect.key)}"${disabled}>${escapeHtml(label)}${suffix}</option>`;
        })
        .join("");
      return `<optgroup label="${escapeHtml(label)}">${options}</optgroup>`;
    })
    .join("");

  const first = state.recipe.effects.find((effect) => effect.available) || state.recipe.effects[0];
  el("effect").value = first.key;
  await loadEffect();
}

async function loadEffect() {
  const { ok, payload } = await postJson("/api/effect", { recipe: state.recipe.id, effect: el("effect").value, values: {} });
  if (!ok) {
    el("output").innerHTML = layer("error", "", `<pre class="bad">${escapeHtml(payload.detail)}</pre>`);
    return;
  }
  state.schema = payload;
  state.values = defaultsOf(payload);
  renderControls();
  renderResolved();
}

function defaultsOf(schema) {
  const values = {};
  [...schema.bus_controls, ...schema.controls].forEach((control) => {
    if (control.kind === "readonly") return;
    values[control.id] = control.kind === "dropdown" ? control.default_index : control.default;
  });
  return values;
}

// ---------------------------------------------------------------- controls
function controlHtml(control) {
  const units = control.units ? ` <span class="muted">${escapeHtml(control.units)}</span>` : "";
  const source = control.source
    ? `<div class="hint src">${escapeHtml(control.source)}${control.dist_type ? ` · ${escapeHtml(control.dist_type)}` : ""}</div>`
    : "";
  const note = control.note ? `<div class="hint">${escapeHtml(control.note)}</div>` : "";
  const head = `<label>${escapeHtml(control.label)}${units}</label>`;
  const pinned = control.bus_pinned ? " pinned" : "";

  if (control.kind === "quantile") {
    // Deciles rather than a track: each option is an equally likely tenth of
    // this prior, which says what the prior does and removes the question of
    // whether a 20 Hz..20 kHz support should be travelled linearly or in log.
    // The number box stays for deliberate overrides outside those ten points.
    const value = state.values[control.id] ?? control.default;
    const mass = control.mass_low === null || control.mass_low === undefined
      ? ""
      : ` \u00b7 most draws ${fmtShort(control.mass_low)} \u2026 ${fmtShort(control.mass_high)}`;
    const matched = matchDecile(control, value);
    const options = control.options
      .map((option) => `<option value="${option.value}"${option === matched ? " selected" : ""}>${escapeHtml(`${option.label}  \u00b7  ${fmtShort(option.value)}`)}</option>`)
      .join("");
    const custom = matched
      ? '<option value="" class="custom">off-prior\u2026</option>'
      : `<option value="" class="custom" selected>off-prior \u00b7 ${escapeHtml(fmtShort(value))}</option>`;
    return `<div class="ctl${pinned}">${head}
      <div class="pickrow">
        <select class="quant" data-id="${escapeHtml(control.id)}">${options}${custom}</select>
        <input type="number" class="num" data-id="${escapeHtml(control.id)}" min="${control.min}" max="${control.max}" step="any" value="${inputValue(value)}" />
      </div>
      <div class="hint">prior spans ${fmtShort(control.min)} \u2026 ${fmtShort(control.max)}${mass}</div>${source}${note}</div>`;
  }
  if (control.kind === "dropdown") {
    const index = state.values[control.id] ?? control.default_index;
    const options = control.options
      .map((option) => {
        const weight = `   w ${fmtShort(option.weight)}`;
        // A settings group carries its own name in the prior ("cut", "boost"),
        // which beats numbering the components for the reader.
        const name = option.label || `group ${option.index + 1}`;
        const text = control.id === COMPONENT_ID ? `${name}${weight}` : `${fmtShort(option.value)}${weight}`;
        return `<option value="${option.index}"${option.index === index ? " selected" : ""}>${escapeHtml(text)}</option>`;
      })
      .join("");
    return `<div class="ctl${pinned}">${head}<select class="pick" data-id="${escapeHtml(control.id)}">${options}</select>${source}${note}</div>`;
  }
  if (control.kind === "number") {
    const value = state.values[control.id] ?? control.default;
    return `<div class="ctl${pinned}">${head}<input type="number" class="num" data-id="${escapeHtml(control.id)}" step="any" value="${inputValue(value)}" />${source}${note}</div>`;
  }
  return `<div class="ctl${pinned}">${head}<div class="ro" data-ro="${escapeHtml(control.id)}">${escapeHtml(fmt(control.default))}</div>${source}${note}</div>`;
}

function renderControls() {
  const schema = state.schema;
  const busSet = el("bus-set");
  busSet.hidden = !schema.bus;
  if (schema.bus) {
    el("bus-note").textContent =
      `${schema.block} routes the stem through an aux bus: dry stays, a copy is sent through the effect and returned. ` +
      `That is why the blend params below are pinned. Untick to hear the effect raw instead.`;
    el("bus-controls").innerHTML = schema.bus_controls.map(controlHtml).join("");
  } else {
    el("bus-controls").innerHTML = "";
  }
  el("controls").innerHTML = schema.controls.length
    ? schema.controls.map(controlHtml).join("")
    : '<p class="muted">this step takes no parameters</p>';

  const unavailable = !schema.available;
  el("apply").disabled = unavailable;
  el("effect-info").innerHTML = [
    `operator <b>${escapeHtml(schema.operator)}</b>`,
    schema.motif ? `motif ${escapeHtml(schema.motif)}` : "",
    schema.block_kind === "send_return" ? "on a send/return bus" : "inline in the chain",
    unavailable ? `<div class="bad">unavailable: ${escapeHtml(schema.unavailable_reason || "")}</div>` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  wire(document);
}

function wire(root) {
  const byId = new Map();
  [...(state.schema.bus_controls || []), ...(state.schema.controls || [])].forEach((control) => byId.set(control.id, control));

  // The range and the number box hold the same value in different spaces on a
  // log control, so they are synced through the real value rather than by
  // copying one input's string into the other. The input being typed into is
  // left alone, or a half-typed "-" would be rewritten under the cursor.
  const commit = (id, value, source) => {
    if (!Number.isFinite(value)) return;
    state.values[id] = value;
    root.querySelectorAll(`input.num[data-id="${id}"]`).forEach((node) => {
      if (node !== source) node.value = inputValue(value);
    });
    // Keep the decile picker honest about a typed value: select the matching
    // decile, or fall back to the off-prior entry relabelled with what was
    // typed, so the picker never claims a decile the value is not on.
    root.querySelectorAll(`select.quant[data-id="${id}"]`).forEach((node) => {
      const matched = matchDecile(byId.get(id) || { options: [] }, value);
      const custom = node.querySelector("option.custom");
      if (custom) custom.textContent = matched ? "off-prior\u2026" : `off-prior \u00b7 ${fmtShort(value)}`;
      node.value = matched ? String(matched.value) : "";
    });
    scheduleRefresh(false);
  };

  root.querySelectorAll("select.quant").forEach((node) => {
    node.onchange = (event) => {
      if (event.target.value === "") return;
      commit(event.target.dataset.id, parseFloat(event.target.value), null);
    };
  });
  root.querySelectorAll("input.num").forEach((node) => {
    node.oninput = (event) => commit(event.target.dataset.id, parseFloat(event.target.value), event.target);
  });
  root.querySelectorAll("select.pick").forEach((node) => {
    node.onchange = (event) => {
      const id = event.target.dataset.id;
      const index = parseInt(event.target.value, 10);
      if (id !== COMPONENT_ID) {
        state.values[id] = index;
        scheduleRefresh(false);
        return;
      }
      // The component decides which parameters exist at all, so values held for
      // the previous component mean nothing here -- keeping them would have the
      // server clamp them into the new component's ranges and report those as
      // what will be applied, disagreeing with the controls on screen. Send only
      // the pick (and the bus trims, which the component does not touch).
      state.values = Object.fromEntries(Object.entries(state.values).filter(([key]) => key.startsWith("bus.")));
      state.values[COMPONENT_ID] = index;
      scheduleRefresh(true);
    };
  });
}

// A slider drag must not rebuild the inputs under the cursor, so a plain change
// only refreshes the derived readouts. Picking a different mixture component
// swaps which parameters exist, so that one does need a full rebuild.
function scheduleRefresh(structural) {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => refreshSchema(structural), 120);
}

async function refreshSchema(structural) {
  const { ok, payload } = await postJson("/api/effect", {
    recipe: state.recipe.id,
    effect: el("effect").value,
    values: state.values,
  });
  if (!ok) return;
  state.schema = payload;
  if (structural) {
    const component = state.values[COMPONENT_ID];
    state.values = defaultsOf(payload);
    if (component !== undefined) state.values[COMPONENT_ID] = component;
    renderControls();
  } else {
    [...payload.bus_controls, ...payload.controls].forEach((control) => {
      if (control.kind !== "readonly") return;
      const node = document.querySelector(`[data-ro="${control.id}"]`);
      if (node) node.textContent = fmt(control.default);
    });
  }
  renderResolved();
}

// ---------------------------------------------------------------- output
function paramsTable(params) {
  const rows = Object.entries(params || {});
  if (!rows.length) return '<p class="muted">no params; the operator runs at its own defaults</p>';
  return `<pre>${escapeHtml(rows.map(([key, value]) => `${key.padEnd(22)} ${fmt(value)}`).join("\\n"))}</pre>`;
}

function renderResolved(result) {
  const schema = state.schema;
  const parts = [];
  const tag = schema.bus ? `${schema.operator} on a bus` : schema.operator;
  parts.push(layer("what will be applied", tag, paramsTable(schema.params)));

  if (schema.bus) {
    const trims = Object.entries(schema.bus).map(([key, value]) => `${key.padEnd(14)} ${fmt(value)}`).join("\\n");
    parts.push(layer("bus trims", el("emulate-bus").checked ? "emulated" : "ignored", `<pre>${escapeHtml(trims)}</pre>`));
  }
  if ((schema.notes || []).length) {
    parts.push(layer("notes", "", `<ul>${schema.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`));
  }
  if (result) {
    const warnings = (result.warnings || []).length
      ? `<ul>${result.warnings.map((warning) => `<li class="warn">${escapeHtml(warning)}</li>`).join("")}</ul>`
      : '<p class="ok">no warnings</p>';
    parts.push(layer("render", `peak ${fmt(result.peak)}${result.bus_applied ? " · through the bus" : ""}`, warnings));
  }
  el("output").innerHTML = parts.join("");
}

// ---------------------------------------------------------------- actions
async function apply() {
  if (!state.upload) {
    el("file-info").innerHTML = '<span class="bad">choose a .wav first</span>';
    return;
  }
  const button = el("apply");
  button.disabled = true;
  button.textContent = "applying…";
  const { ok, payload } = await postJson("/api/apply", {
    upload_id: state.upload.upload_id,
    recipe: state.recipe.id,
    effect: el("effect").value,
    values: state.values,
    emulate_bus: el("emulate-bus").checked,
  });
  button.disabled = false;
  button.textContent = "apply effect";
  if (!ok) {
    el("output").innerHTML = layer("error", "", `<pre class="bad">${escapeHtml(payload.detail)}</pre>`);
    return;
  }
  state.schema.params = payload.params;
  state.schema.bus = payload.bus;
  renderResolved(payload);
  el("processed").src = `/api/audio/${state.upload.upload_id}/processed?v=${payload.version}`;
}

el("file").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  el("file-info").textContent = "uploading…";
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const payload = await response.json().catch(() => ({ detail: "the server sent a non-JSON reply" }));
  if (!response.ok) {
    el("file-info").innerHTML = `<span class="bad">${escapeHtml(payload.detail)}</span>`;
    return;
  }
  state.upload = payload;
  const truncated = payload.truncated ? ` <span class="warn">(truncated to ${payload.max_seconds}s)</span>` : "";
  el("file-info").innerHTML =
    `${escapeHtml(payload.name)} · ${payload.sample_rate} Hz · ${payload.channels} ch · ${payload.seconds.toFixed(2)}s${truncated}`;
  el("original").src = `/api/audio/${payload.upload_id}/original`;
  el("processed").removeAttribute("src");
};

el("recipe").onchange = selectRecipe;
el("effect").onchange = loadEffect;
el("apply").onclick = apply;
el("reset").onclick = () => {
  state.values = defaultsOf(state.schema);
  renderControls();
  refreshSchema(false);
};
el("emulate-bus").onchange = () => renderResolved();

loadCatalog();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
