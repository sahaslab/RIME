import sys
import os
import argparse
import hashlib
import importlib
import json
import signal
import shutil
import subprocess
import time
import tomllib
import uuid
import numpy as np
import torch
import soundfile
import uvicorn
import yaml
from pathlib import Path
from collections.abc import Mapping
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the local audit UI. Default: %(default)s")
    parser.add_argument("--port", type=int, default=8787, help="Port for the local audit UI. Default: %(default)s")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root. Default: %(default)s")
    args = parser.parse_args()

    APP_STATE["repo_root"] = args.repo_root.expanduser().resolve()
    uvicorn.run(app, host=args.host, port=args.port)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = DEFAULT_REPO_ROOT / "audit_ui"
AD_HOC_SKIP_OPERATORS = {"separate_audio", "mix_stems"}
AD_HOC_SKIP_TAGS = {"separation", "mix", "pitch"}
CONFIG_FILES = {
    "constraints": Path("configs/ground_truth/constraints.yaml"),
    "datasets_mtg_jamendo": Path("configs/ground_truth/datasets/mtg_jamendo.yaml"),
    "distributions": Path("configs/ground_truth/distributions.yaml"),
    "motifs": Path("configs/ground_truth/motifs.yaml"),
    "operators": Path("configs/ground_truth/operators.yaml"),
    "recipes": Path("configs/ground_truth/recipes.yaml"),
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
AD_HOC_MANIFEST_PATH = Path("ad_hoc/ad_hoc_manifest.jsonl")
COUNTERFACTUAL_ROOT = Path("counterfactual")
ISOLATED_ROOT = Path("isolated_sources")
ISOLATED_MANIFEST_PATH = ISOLATED_ROOT / "isolated_manifest.jsonl"
MIN_STEM_RMS_RATIO = 0.02
MIN_STEM_RMS = 1e-4
APP_STATE: dict[str, Any] = {
    "repo_root": DEFAULT_REPO_ROOT,
    "jobs": {},
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(APP_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((APP_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/configs")
def list_configs() -> dict[str, Any]:
    repo_root = repo_root_path()
    configs = []
    for config_name, relative_path in CONFIG_FILES.items():
        path = repo_root / relative_path
        configs.append(
            {
                "name": config_name,
                "path": str(path),
                "exists": path.exists(),
                "mtime": path.stat().st_mtime if path.exists() else None,
            }
        )
    return {"configs": configs}


@app.get("/api/configs/{config_name}")
def get_config(config_name: str) -> dict[str, Any]:
    path = config_path(config_name)
    content = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content) or {}
    return {
        "name": config_name,
        "path": str(path),
        "content": content,
        "summary": config_summary(config_name, parsed, content),
        "anchors": yaml_anchors(config_name, parsed, content),
    }


@app.put("/api/configs/{config_name}")
def save_config(
    config_name: str,
    body: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    path = config_path(config_name)
    content = str(body["content"])
    yaml.safe_load(content)
    path.write_text(content, encoding="utf-8")
    parsed = yaml.safe_load(content) or {}
    return {
        "name": config_name,
        "path": str(path),
        "summary": config_summary(config_name, parsed, content),
        "anchors": yaml_anchors(config_name, parsed, content),
        "saved_at": time.time(),
    }


@app.get("/api/artifacts")
def artifacts() -> dict[str, Any]:
    validation_root = validation_root_path()
    render_root = validation_root / "renders"
    counterfactual_root = validation_root / COUNTERFACTUAL_ROOT
    all_plans_path = validation_root / "all_plans.jsonl"
    selected_plans_path = validation_root / "subsampled_plans.jsonl"
    selected_tracks = read_jsonl(validation_root / "selected_tracks.jsonl", limit=5000)
    analysis_manifest = read_jsonl(validation_root / "analysis_manifest.jsonl", limit=5000)
    plans = read_jsonl(all_plans_path, limit=100000)
    selected_plans = read_jsonl(selected_plans_path, limit=100000)
    plan_sample_rows = selected_plans if selected_plans else plans
    render_manifest = read_jsonl(render_root / "render_manifest.jsonl", limit=20000)
    counterfactual_manifest = read_many_jsonl(counterfactual_root / "manifests", limit=20000)
    ad_hoc_manifest = read_jsonl(validation_root / AD_HOC_MANIFEST_PATH, limit=20000)
    isolated_manifest = read_jsonl(validation_root / ISOLATED_MANIFEST_PATH, limit=20000)
    source_audio = media_files(validation_root / "audio", validation_root)
    isolated_audio = media_files(validation_root / ISOLATED_ROOT, validation_root)
    rendered_audio = media_files(render_root, validation_root)
    counterfactual_audio = media_files(counterfactual_root, validation_root)
    ad_hoc_audio = media_files(validation_root / "ad_hoc", validation_root)
    return {
        "validation_root": str(validation_root),
        "summary": read_json(validation_root / "summary.json"),
        "selected_tracks": selected_tracks,
        "analysis_manifest": analysis_manifest,
        "plan_count": count_lines(all_plans_path),
        "subsampled_plan_count": count_lines(selected_plans_path),
        "render_plans_path": str(selected_plans_path if selected_plans_path.exists() else all_plans_path),
        "plan_sample": plan_sample_rows[:500],
        "render_manifest": render_manifest,
        "counterfactual_manifest": counterfactual_manifest,
        "ad_hoc_manifest": ad_hoc_manifest,
        "isolated_manifest": isolated_manifest,
        "isolated_cache_status": demucs_cache_status(analysis_manifest),
        "source_audio": source_audio,
        "isolated_audio": isolated_audio,
        "rendered_audio": rendered_audio,
        "counterfactual_audio": counterfactual_audio,
        "ad_hoc_audio": ad_hoc_audio,
        "track_groups": build_track_groups(
            selected_tracks=selected_tracks,
            analysis_manifest=analysis_manifest,
            plans=plans,
            render_manifest=render_manifest,
            counterfactual_manifest=counterfactual_manifest,
            ad_hoc_manifest=ad_hoc_manifest,
            isolated_manifest=isolated_manifest,
            source_audio=source_audio,
            isolated_audio=isolated_audio,
            rendered_audio=rendered_audio,
            counterfactual_audio=counterfactual_audio,
            ad_hoc_audio=ad_hoc_audio,
        ),
    }


@app.post("/api/isolated/materialize")
def materialize_isolated(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    return materialize_isolated_sources(force=bool(body.get("force", False)))


@app.post("/api/reset")
def reset_artifacts(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    scopes = [str(scope) for scope in body.get("scopes", [])]
    return reset_validation_artifacts(scopes)


@app.get("/api/operators")
def operators() -> dict[str, Any]:
    parsed = yaml.safe_load(config_path("operators").read_text(encoding="utf-8")) or {}
    specs = []
    for operator in parsed.get("operators", []):
        tags = list(operator.get("tags", []))
        if operator.get("name") in AD_HOC_SKIP_OPERATORS:
            continue
        if set(tags) & AD_HOC_SKIP_TAGS:
            continue
        specs.append(
            {
                "name": operator.get("name"),
                "aliases": operator.get("aliases", []),
                "tags": tags,
                "params": operator.get("params", []),
            }
        )
    return {"operators": specs}


@app.post("/api/fx/apply")
def apply_fx(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    source_path = artifact_path(str(body["source_relative_path"]))
    operator_name = str(body["operator"])
    params = dict(body.get("params", {}))
    label = slugify(str(body.get("label", operator_name)))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    graph_spec = [
        {
            "kind": "chain",
            "prefix": "ad_hoc",
            "source": "audio",
            "output": "final_audio",
            "steps": [
                {
                    "name": "fx",
                    "operator": operator_name,
                    "params": params,
                }
            ],
        }
    ]
    return render_ad_hoc_graph(
        source_path=source_path,
        source_relative_path=str(body["source_relative_path"]),
        graph_spec=graph_spec,
        label=label,
        timestamp=timestamp,
        kind="quick_fx",
        operator=operator_name,
        params=params,
    )


@app.post("/api/fx/apply_graph")
def apply_graph(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    source_relative_path = str(body["source_relative_path"])
    source_path = artifact_path(source_relative_path)
    label = slugify(str(body.get("label", "counterfactual")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    graph_spec = list(body["graph_spec"])
    return render_ad_hoc_graph(
        source_path=source_path,
        source_relative_path=source_relative_path,
        graph_spec=graph_spec,
        label=label,
        timestamp=timestamp,
        kind="graph_rerun",
        operator=None,
        params={},
    )


@app.get("/api/artifacts/file/{relative_path:path}")
def artifact_file(relative_path: str) -> FileResponse:
    return FileResponse(artifact_path(relative_path))


@app.post("/api/jobs")
def start_job(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(body["action"])
    command = job_command(action, body)
    job_id = str(uuid.uuid4())
    log_path = validation_root_path() / "jobs" / ("%s.log" % job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root_path()),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    APP_STATE["jobs"][job_id] = {
        "id": job_id,
        "action": action,
        "command": command,
        "pid": process.pid,
        "process": process,
        "log_path": log_path,
        "started_at": time.time(),
    }
    return job_snapshot(job_id)


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [job_snapshot(job_id) for job_id in sorted(APP_STATE["jobs"])]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return job_snapshot(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = APP_STATE["jobs"][job_id]
    process = job["process"]
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    return job_snapshot(job_id)


def repo_root_path() -> Path:
    return Path(APP_STATE["repo_root"]).resolve()


def validation_root_path() -> Path:
    return repo_root_path() / "derived" / "validation" / "mtg_jamendo_10"


def config_path(config_name: str) -> Path:
    if config_name not in CONFIG_FILES:
        raise ValueError("Unknown config '%s'." % config_name)
    return (repo_root_path() / CONFIG_FILES[config_name]).resolve()


def config_summary(
    config_name: str,
    parsed: Mapping[str, Any],
    content: str,
) -> dict[str, Any]:
    if config_name == "recipes":
        recipes = list(parsed.get("recipes", []))
        return {
            "kind": "recipes",
            "count": len(recipes),
            "items": [
                {
                    "id": recipe.get("id"),
                    "description": recipe.get("description"),
                    "weight": recipe.get("weight"),
                    "tags": recipe.get("tags", []),
                    "graph_blocks": len(recipe.get("graph", [])),
                }
                for recipe in recipes
            ],
        }
    if config_name == "motifs":
        motifs = dict(parsed.get("motifs", {}))
        return {
            "kind": "motifs",
            "count": len(motifs),
            "items": [
                {
                    "id": motif_id,
                    "order_profile": motif.get("order_profile"),
                    "steps": [step.get("operator") for step in motif.get("steps", [])],
                }
                for motif_id, motif in motifs.items()
            ],
        }
    if config_name == "constraints":
        constraints = dict(parsed.get("constraints", {}))
        policies = list(constraints.get("pattern_policies", []))
        profiles = dict(constraints.get("chain_order", {}).get("profiles", {}))
        return {
            "kind": "constraints",
            "policy_count": len(policies),
            "profile_count": len(profiles),
            "items": [
                {
                    "id": policy.get("name"),
                    "action": policy.get("action"),
                    "multiplier": policy.get("multiplier"),
                }
                for policy in policies
            ],
            "profiles": sorted(profiles),
        }
    if config_name == "operators":
        operators = list(parsed.get("operators", []))
        return {
            "kind": "operators",
            "count": len(operators),
            "items": [
                {
                    "id": operator.get("name"),
                    "aliases": operator.get("aliases", []),
                    "tags": operator.get("tags", []),
                    "params": operator.get("params", []),
                }
                for operator in operators
            ],
        }
    if config_name == "distributions":
        distributions = dict(parsed.get("distributions", {}))
        return {
            "kind": "distributions",
            "count": len(distributions),
            "items": distribution_items(distributions),
        }
    return {
        "kind": config_name,
        "count": len(content.splitlines()),
        "items": [],
    }


def distribution_items(distributions: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, value in distributions.items():
        items.append(
            {
                "id": key,
                "children": sorted(value.keys()) if isinstance(value, Mapping) else [],
            }
        )
    return items


def yaml_anchors(
    config_name: str,
    parsed: Mapping[str, Any],
    content: str,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    lines = content.splitlines()
    if config_name == "recipes":
        for recipe in parsed.get("recipes", []):
            anchors.append(anchor_for_token(lines, str(recipe.get("id")), "recipe"))
    elif config_name == "motifs":
        for motif_id in dict(parsed.get("motifs", {})):
            anchors.append(anchor_for_token(lines, str(motif_id), "motif"))
    elif config_name == "constraints":
        for policy in parsed.get("constraints", {}).get("pattern_policies", []):
            anchors.append(anchor_for_token(lines, str(policy.get("name")), "policy"))
    elif config_name == "operators":
        for operator in parsed.get("operators", []):
            anchors.append(anchor_for_token(lines, str(operator.get("name")), "operator"))
    elif config_name == "distributions":
        for distribution_id in dict(parsed.get("distributions", {})):
            anchors.append(anchor_for_token(lines, str(distribution_id), "distribution"))
    return [anchor for anchor in anchors if anchor["line"] is not None]


def anchor_for_token(
    lines: list[str],
    token: str,
    kind: str,
) -> dict[str, Any]:
    line_number: int | None = None
    for index, line in enumerate(lines, start=1):
        if token in line:
            line_number = index
            break
    return {
        "id": token,
        "kind": kind,
        "line": line_number,
    }


def job_command(
    action: str,
    body: Mapping[str, Any],
) -> list[str]:
    repo_root = repo_root_path()
    validation_root = validation_root_path()
    python_bin = sys.executable
    if action == "prepare_subset":
        command = [
            python_bin,
            str(repo_root / "scripts" / "prepare_mtg_jamendo_validation_subset.py"),
            "--output-root",
            str(validation_root),
            "--num-tracks",
            str(int(body.get("num_tracks", 10))),
            "--seed",
            str(int(body.get("seed", 0))),
            "--candidate-scan-limit",
            str(int(body.get("candidate_scan_limit", 1500))),
            "--dataset-config",
            str(repo_root / "configs" / "ground_truth" / "datasets" / "mtg_jamendo.yaml"),
            "--config-dir",
            str(repo_root / "configs" / "ground_truth"),
        ]
        if body.get("force_audio"):
            command.append("--force-audio")
        if body.get("max_variants_per_recipe") not in (None, ""):
            command.extend(["--max-variants-per-recipe", str(int(body["max_variants_per_recipe"]))])
        command.extend(["--random-plans-per-clip", str(int(body.get("random_plans_per_clip", 8)))])
        return command
    if action == "generate_plans":
        command = [
            python_bin,
            str(repo_root / "scripts" / "generate_ground_truth_plans.py"),
            "--analysis-path",
            str(validation_root / "analysis_manifest.jsonl"),
            "--output-path",
            str(validation_root / "all_plans.jsonl"),
            "--config-dir",
            str(repo_root / "configs" / "ground_truth"),
            "--num-workers",
            str(int(body.get("num_workers", 1))),
        ]
        if body.get("max_variants_per_recipe") not in (None, ""):
            command.extend(["--max-variants-per-recipe", str(int(body["max_variants_per_recipe"]))])
        command.extend(["--random-plans-per-clip", str(int(body.get("random_plans_per_clip", 8)))])
        return command
    if action == "subsample_plans":
        command = [
            python_bin,
            str(repo_root / "scripts" / "subsample_ground_truth_plans.py"),
            "--plans-path",
            str(validation_root / "all_plans.jsonl"),
            "--output-path",
            str(validation_root / "subsampled_plans.jsonl"),
            "--policy",
            str(body.get("subsample_policy", "stratified_random")),
            "--limit",
            str(int(body.get("subsample_limit", 200))),
            "--max-per-source-recipe",
            str(int(body.get("max_per_source_recipe", 3))),
            "--seed",
            str(int(body.get("seed", 0))),
            "--cache-dir",
            str(validation_root / "subsampled_plans.cache"),
        ]
        if body.get("subsample_device") not in (None, ""):
            command.extend(["--device", str(body["subsample_device"])])
        return command
    if action == "render":
        plans_path = validation_root / "subsampled_plans.jsonl"
        if not plans_path.exists():
            plans_path = validation_root / "all_plans.jsonl"
        command = [
            python_bin,
            str(repo_root / "scripts" / "render_permissible_plans.py"),
            "--plans-path",
            str(plans_path),
            "--output-root",
            str(validation_root / "renders"),
            "--config-dir",
            str(repo_root / "configs" / "ground_truth"),
            "--separation-cache-dir",
            str(validation_root / "demucs_cache"),
        ]
        if body.get("limit") not in (None, ""):
            command.extend(["--limit", str(int(body["limit"]))])
        if body.get("max_audio_seconds") not in (None, ""):
            command.extend(["--max-audio-seconds", str(float(body["max_audio_seconds"]))])
        if body.get("overwrite"):
            command.append("--overwrite")
        if body.get("skip_harmony"):
            command.append("--skip-harmony")
        return command
    if action == "render_graph":
        plan_path, manifest_path = write_counterfactual_plan(body)
        command = [
            python_bin,
            str(repo_root / "scripts" / "render_permissible_plans.py"),
            "--plans-path",
            str(plan_path),
            "--output-root",
            str(validation_root / COUNTERFACTUAL_ROOT),
            "--manifest-path",
            str(manifest_path),
            "--config-dir",
            str(repo_root / "configs" / "ground_truth"),
            "--separation-cache-dir",
            str(validation_root / "demucs_cache"),
            "--limit",
            "1",
            "--overwrite",
            "--disable-shuffle",
        ]
        if body.get("max_audio_seconds") not in (None, ""):
            command.extend(["--max-audio-seconds", str(float(body["max_audio_seconds"]))])
        return command
    if action == "coverage":
        plans_path = validation_root / "subsampled_plans.jsonl"
        if not plans_path.exists():
            plans_path = validation_root / "all_plans.jsonl"
        return [
            python_bin,
            str(repo_root / "scripts" / "plan_coverage.py"),
            "--plans-path",
            str(plans_path),
            "--output-path",
            str(validation_root / "plan_coverage.json"),
            "--config-dir",
            str(repo_root / "configs" / "ground_truth"),
        ]
    raise ValueError("Unknown job action '%s'." % action)


def write_counterfactual_plan(body: Mapping[str, Any]) -> tuple[Path, Path]:
    validation_root = validation_root_path()
    source_relative_path = str(body["source_relative_path"])
    source_path = artifact_path(source_relative_path)
    graph_spec = list(body["graph_spec"])
    label = slugify(str(body.get("label", "counterfactual")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clip_id = clip_id_for_relative_path(source_relative_path) or source_path.stem
    plan_id = "counterfactual_%s_%s" % (timestamp, label)
    plan_path = validation_root / COUNTERFACTUAL_ROOT / "plans" / ("%s.jsonl" % plan_id)
    manifest_path = validation_root / COUNTERFACTUAL_ROOT / "manifests" / ("%s.jsonl" % plan_id)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "plan_id": plan_id,
        "clip_id": clip_id,
        "recipe_id": "counterfactual",
        "recipe_tags": ["counterfactual"],
        "target_stem": target_from_graph(graph_spec),
        "target_family": target_from_graph(graph_spec),
        "separation_target": target_from_graph(graph_spec),
        "graph_description": describe_graph_spec(graph_spec),
        "graph_spec": graph_spec,
        "bindings": {
            "source_relative_path": source_relative_path,
            "source_label": str(body.get("source_label", source_relative_path)),
            "counterfactual_label": label,
        },
        "audio_path": str(source_path),
    }
    plan_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, manifest_path


def reset_validation_artifacts(scopes: list[str]) -> dict[str, Any]:
    validation_root = validation_root_path()
    reset_map = {
        "plans": [
            validation_root / "all_plans.jsonl",
            validation_root / "subsampled_plans.jsonl",
            validation_root / "subsampled_plans.cache",
            validation_root / "plan_coverage.json",
        ],
        "renders": [
            validation_root / "renders",
        ],
        "counterfactual": [
            validation_root / COUNTERFACTUAL_ROOT,
        ],
        "quick_fx": [
            validation_root / "ad_hoc",
        ],
        "isolated": [
            validation_root / ISOLATED_ROOT,
        ],
        "demucs_cache": [
            validation_root / "demucs_cache",
        ],
    }
    deleted: list[str] = []
    for scope in scopes:
        for path in reset_map.get(scope, []):
            resolved = path.resolve()
            if validation_root not in resolved.parents and resolved != validation_root:
                raise ValueError("Reset path escapes validation root.")
            if not resolved.exists():
                continue
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            deleted.append(str(resolved))
    return {
        "scopes": scopes,
        "deleted": deleted,
    }


def job_snapshot(job_id: str) -> dict[str, Any]:
    job = APP_STATE["jobs"][job_id]
    process = job["process"]
    return_code = process.poll()
    status = "running" if return_code is None else "finished"
    log_path = Path(job["log_path"])
    return {
        "id": job_id,
        "action": job["action"],
        "command": job["command"],
        "pid": job["pid"],
        "status": status,
        "return_code": return_code,
        "started_at": job["started_at"],
        "elapsed_s": round(time.time() - float(job["started_at"]), 1),
        "log": tail_text(log_path, 30000),
    }


def tail_text(
    path: Path,
    max_chars: int,
) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(
    path: Path,
    limit: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            line = line.strip()
            if line:
                row = read_jsonl_line(line, path, index + 1)
                if row is not None:
                    rows.append(row)
    return rows


def read_jsonl_line(
    line: str,
    path: Path,
    line_number: int,
) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        decoder = json.JSONDecoder()
        try:
            row, end = decoder.raw_decode(line)
        except json.JSONDecodeError:
            print("Skipping malformed JSONL row %s:%d: %s" % (path, line_number, error), flush=True)
            return None
        if not isinstance(row, dict):
            print("Skipping non-object JSONL row %s:%d" % (path, line_number), flush=True)
            return None
        print(
            "Recovered first JSON object from malformed JSONL row %s:%d; discarded %d trailing characters"
            % (path, line_number, len(line) - end),
            flush=True,
        )
        return row


def read_many_jsonl(
    root: Path,
    limit: int,
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        for row in read_jsonl(path, limit=max(limit - len(rows), 0)):
            row["_manifest_path"] = str(path)
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def media_files(
    root: Path,
    validation_root: Path,
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        relative_path = path.relative_to(validation_root)
        files.append(
            {
                "name": path.name,
                "relative_path": str(relative_path),
                "url": "/api/artifacts/file/%s" % relative_path,
                "size": path.stat().st_size,
            }
        )
    return files


def build_track_groups(
    selected_tracks: list[dict[str, Any]],
    analysis_manifest: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    render_manifest: list[dict[str, Any]],
    counterfactual_manifest: list[dict[str, Any]],
    ad_hoc_manifest: list[dict[str, Any]],
    isolated_manifest: list[dict[str, Any]],
    source_audio: list[dict[str, Any]],
    isolated_audio: list[dict[str, Any]],
    rendered_audio: list[dict[str, Any]],
    counterfactual_audio: list[dict[str, Any]],
    ad_hoc_audio: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation_root = validation_root_path()
    source_by_relative = {row["relative_path"]: row for row in source_audio}
    plan_by_id = {str(row.get("plan_id")): row for row in plans if row.get("plan_id") is not None}
    selected_by_id = {str(row.get("track_id")): row for row in selected_tracks}
    analysis_by_id = {str(row.get("clip_id")): row for row in analysis_manifest}
    groups: dict[str, dict[str, Any]] = {}

    for track in selected_tracks:
        clip_id = str(track["track_id"])
        relative_path = str(track.get("relative_path", "audio/%s.wav" % clip_id))
        groups[clip_id] = {
            "clip_id": clip_id,
            "title": clip_id,
            "source": source_by_relative.get(relative_path, file_from_relative(relative_path, validation_root)),
            "source_copy": None,
            "metadata": track,
            "analysis": analysis_by_id.get(clip_id, {}).get("analysis", {}),
            "targets": track.get("target_candidates", []),
            "isolated": [],
            "baselines": [],
            "renders": [],
            "counterfactuals": [],
            "ad_hoc": [],
        }

    seen_baselines: set[str] = set()
    for manifest_row in render_manifest:
        plan = plan_by_id.get(str(manifest_row.get("plan_id")), {})
        row = merged_manifest_row(manifest_row, plan)
        clip_id = str(row.get("clip_id") or clip_id_for_any_path(row.get("audio_path")) or "unknown")
        group = ensure_track_group(groups, clip_id, selected_by_id, analysis_by_id, source_by_relative, validation_root)
        source_copy = file_from_absolute(row.get("source_copy_path"), validation_root)
        if source_copy is not None:
            group["source_copy"] = source_copy
        baseline = file_from_absolute(row.get("baseline_path"), validation_root)
        if baseline is not None and baseline["relative_path"] not in seen_baselines:
            baseline.update(
                {
                    "kind": "baseline",
                    "label": "No-FX remix baseline",
                    "target": target_from_row(row),
                    "detail": "Separated %s remixed back with the residual, before planned FX." % target_from_row(row),
                    "graph_spec": row.get("graph_spec", []),
                }
            )
            group["baselines"].append(baseline)
            seen_baselines.add(baseline["relative_path"])
        output = file_from_absolute(row.get("output_path"), validation_root)
        if output is not None:
            output.update(render_item_metadata(row, "render"))
            group["renders"].append(output)

    for manifest_row in counterfactual_manifest:
        row = dict(manifest_row)
        clip_id = str(row.get("clip_id") or clip_id_for_any_path(row.get("audio_path")) or "unknown")
        group = ensure_track_group(groups, clip_id, selected_by_id, analysis_by_id, source_by_relative, validation_root)
        output = file_from_absolute(row.get("output_path"), validation_root)
        if output is not None:
            output.update(render_item_metadata(row, "counterfactual"))
            group["counterfactuals"].append(output)

    isolated_by_relative = {str(row.get("relative_path")): row for row in isolated_manifest}
    for file_row in isolated_audio:
        if Path(file_row["relative_path"]).name == ISOLATED_MANIFEST_PATH.name:
            continue
        manifest_row = isolated_by_relative.get(str(file_row["relative_path"]))
        if manifest_row is None:
            continue
        clip_id = str(manifest_row.get("clip_id") or clip_id_for_relative_path(file_row["relative_path"]) or "unknown")
        group = ensure_track_group(groups, clip_id, selected_by_id, analysis_by_id, source_by_relative, validation_root)
        source_name = str(manifest_row.get("source_name") or Path(file_row["name"]).stem)
        enriched = dict(file_row)
        enriched.update(
            {
                "kind": "isolated_source",
                "label": "Isolated %s stem" % source_name,
                "detail": "Demucs source cached from the original full mix.",
                    "source_name": source_name,
                    "target": source_name,
                    "clip_id": clip_id,
                    "sample_rate": manifest_row.get("sample_rate"),
                    "cache_path": manifest_row.get("cache_path"),
                }
        )
        group["isolated"].append(enriched)

    ad_hoc_by_relative = {str(row.get("relative_path")): row for row in ad_hoc_manifest}
    for file_row in ad_hoc_audio:
        manifest_row = ad_hoc_by_relative.get(str(file_row["relative_path"]), {})
        source_relative_path = str(manifest_row.get("source_relative_path", ""))
        clip_id = str(manifest_row.get("clip_id") or clip_id_for_relative_path(source_relative_path) or "unknown")
        group = ensure_track_group(groups, clip_id, selected_by_id, analysis_by_id, source_by_relative, validation_root)
        enriched = dict(file_row)
        enriched.update(
            {
                "kind": manifest_row.get("kind", "quick_fx"),
                "label": manifest_row.get("label", file_row["name"]),
                "operator": manifest_row.get("operator"),
                "params": manifest_row.get("params", {}),
                "source_relative_path": source_relative_path,
                "source_label": manifest_row.get("source_label", source_relative_path),
                "graph_spec": manifest_row.get("graph_spec", []),
                "created_at": manifest_row.get("created_at"),
            }
        )
        group["ad_hoc"].append(enriched)

    attach_orphan_audio(groups, rendered_audio, "renders", validation_root, plan_by_id)
    attach_orphan_audio(groups, counterfactual_audio, "counterfactuals", validation_root, plan_by_id)
    return sorted(groups.values(), key=lambda row: row["clip_id"])


def ensure_track_group(
    groups: dict[str, dict[str, Any]],
    clip_id: str,
    selected_by_id: Mapping[str, dict[str, Any]],
    analysis_by_id: Mapping[str, dict[str, Any]],
    source_by_relative: Mapping[str, dict[str, Any]],
    validation_root: Path,
) -> dict[str, Any]:
    if clip_id not in groups:
        track = selected_by_id.get(clip_id, {})
        relative_path = str(track.get("relative_path", "audio/%s.wav" % clip_id))
        groups[clip_id] = {
            "clip_id": clip_id,
            "title": clip_id,
            "source": source_by_relative.get(relative_path, file_from_relative(relative_path, validation_root)),
            "source_copy": None,
            "metadata": track,
            "analysis": analysis_by_id.get(clip_id, {}).get("analysis", {}),
            "targets": track.get("target_candidates", []),
            "isolated": [],
            "baselines": [],
            "renders": [],
            "counterfactuals": [],
            "ad_hoc": [],
        }
    return groups[clip_id]


def render_item_metadata(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    target = target_from_row(row)
    recipe_id = row.get("recipe_id") or "unknown_recipe"
    plan_id = row.get("plan_id") or "unknown_plan"
    return {
        "kind": kind,
        "label": "%s | target=%s" % (recipe_id, target),
        "detail": "%s | %s" % (plan_id, params_summary(row.get("graph_spec", []))),
        "plan_id": plan_id,
        "recipe_id": recipe_id,
        "target": target,
        "target_family": row.get("target_family"),
        "recipe_tags": row.get("recipe_tags", []),
        "graph_description": row.get("graph_description") or describe_graph_spec(row.get("graph_spec", [])),
        "graph_spec": row.get("graph_spec", []),
        "status": row.get("status"),
        "params": graph_params(row.get("graph_spec", [])),
    }


def merged_manifest_row(
    manifest_row: Mapping[str, Any],
    plan_row: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(plan_row)
    merged.update(dict(manifest_row))
    if not merged.get("graph_spec") and plan_row.get("graph_spec"):
        merged["graph_spec"] = plan_row["graph_spec"]
    return merged


def attach_orphan_audio(
    groups: dict[str, dict[str, Any]],
    files: list[dict[str, Any]],
    key: str,
    validation_root: Path,
    plan_by_id: Mapping[str, dict[str, Any]],
) -> None:
    known = {
        item["relative_path"]
        for group in groups.values()
        for section in ["baselines", "renders", "counterfactuals", "ad_hoc"]
        for item in group.get(section, [])
    }
    for file_row in files:
        if file_row["relative_path"] in known:
            continue
        clip_id = clip_id_for_relative_path(file_row["relative_path"]) or "unknown"
        group = groups.setdefault(
            clip_id,
            {
                "clip_id": clip_id,
                "title": clip_id,
                "source": file_from_relative("audio/%s.wav" % clip_id, validation_root),
                "source_copy": None,
                "metadata": {},
                "analysis": {},
                "targets": [],
                "isolated": [],
                "baselines": [],
                "renders": [],
                "counterfactuals": [],
                "ad_hoc": [],
            },
        )
        if file_row["name"].startswith("source."):
            group["source_copy"] = file_row
            continue
        if file_row["name"].startswith("original__"):
            baseline = dict(file_row)
            target = Path(file_row["name"]).stem.replace("original__", "")
            baseline.update(
                {
                    "kind": "baseline",
                    "label": "No-FX remix baseline",
                    "target": target,
                    "detail": "Separated %s remixed back with the residual, before planned FX." % target,
                }
            )
            group["baselines"].append(baseline)
            continue
        plan = plan_by_id.get(Path(file_row["name"]).stem)
        orphan = dict(file_row)
        if plan is not None:
            orphan.update(render_item_metadata(plan, "render"))
        else:
            orphan.update({"kind": "unmatched", "label": file_row["name"], "detail": "No manifest metadata found."})
        group[key].append(orphan)


def render_ad_hoc_graph(
    source_path: Path,
    source_relative_path: str,
    graph_spec: list[dict[str, Any]],
    label: str,
    timestamp: str,
    kind: str,
    operator: str | None,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = validation_root_path() / "ad_hoc" / ("%s__%s.wav" % (timestamp, label))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_tensor, sample_rate = load_audio_tensor(source_path)
    compiler = runtime_plan_compiler().from_directory(repo_root_path() / "configs" / "ground_truth")
    graph = compiler.compile_graph_spec(graph_spec)
    outputs = graph.run(audio=audio_tensor, sr=sample_rate, device=torch.device("cpu"))
    save_audio_tensor(outputs["final_audio"], sample_rate, output_path)
    relative_path = output_path.relative_to(validation_root_path())
    row = {
        "kind": kind,
        "label": label,
        "operator": operator,
        "params": dict(params),
        "source_relative_path": source_relative_path,
        "source_label": source_label_for_relative(source_relative_path),
        "clip_id": clip_id_for_relative_path(source_relative_path),
        "output_path": str(output_path),
        "relative_path": str(relative_path),
        "url": "/api/artifacts/file/%s" % relative_path,
        "sample_rate": sample_rate,
        "graph_spec": graph_spec,
        "created_at": time.time(),
    }
    append_jsonl(validation_root_path() / AD_HOC_MANIFEST_PATH, row)
    return row


def materialize_isolated_sources(force: bool) -> dict[str, Any]:
    validation_root = validation_root_path()
    rows = read_jsonl(validation_root / "analysis_manifest.jsonl", limit=5000)
    cache_roots = demucs_source_cache_roots()
    manifest_rows: list[dict[str, Any]] = []
    missing_cache: list[dict[str, Any]] = []
    rejected_low_energy: list[dict[str, Any]] = []
    written = 0
    reused = 0
    for row in rows:
        clip_id = str(row["clip_id"])
        allowed_sources = allowed_demucs_sources_for_analysis(row)
        if not allowed_sources:
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(row.get("audio_path")),
                    "reason": "missing_metadata_targets",
                }
            )
            continue
        audio_path = Path(str(row["audio_path"])).expanduser().resolve()
        if not audio_path.exists():
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "reason": "missing_audio",
                }
            )
            continue
        sample_rate = int(soundfile.info(str(audio_path)).samplerate)
        cache_path = demucs_sources_cache_path(audio_path, sample_rate, cache_roots)
        if not cache_path.exists():
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "expected_cache_path": str(cache_path),
                    "candidate_cache_paths": [
                        str(path)
                        for path in demucs_sources_cache_candidates(audio_path, sample_rate, cache_roots)
                    ],
                }
            )
            continue
        payload = torch.load(cache_path, map_location="cpu")
        sources = dict(payload["sources"])
        mixture_rms = rms_audio(sum_source_audio(sources))
        for source_name, source_audio in sources.items():
            if str(source_name) not in allowed_sources:
                continue
            stats = source_energy_stats(source_audio, mixture_rms)
            if not stats["active"]:
                rejected_low_energy.append(
                    {
                        "clip_id": clip_id,
                        "source_name": str(source_name),
                        "rms": stats["rms"],
                        "rms_ratio": stats["rms_ratio"],
                    }
                )
                continue
            output_path = validation_root / ISOLATED_ROOT / clip_id / ("%s.wav" % slugify(str(source_name)))
            if output_path.exists() and not force:
                reused += 1
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio_tensor(source_audio, sample_rate, output_path)
                written += 1
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "source_name": str(source_name),
                    "metadata_allowed_sources": sorted(allowed_sources),
                    "audio_path": str(audio_path),
                    "cache_path": str(cache_path),
                    "output_path": str(output_path),
                    "relative_path": str(output_path.relative_to(validation_root)),
                    "sample_rate": sample_rate,
                    "rms": stats["rms"],
                    "rms_ratio": stats["rms_ratio"],
                }
            )
    write_jsonl_file(validation_root / ISOLATED_MANIFEST_PATH, manifest_rows)
    cache_hit_clip_ids = sorted({row["clip_id"] for row in manifest_rows})
    visible_rows = read_jsonl(validation_root / ISOLATED_MANIFEST_PATH, limit=20000)
    return {
        "written": written,
        "reused": reused,
        "missing_cache": missing_cache,
        "rejected_low_energy": rejected_low_energy,
        "track_count": len(rows),
        "cache_hit_tracks": len(cache_hit_clip_ids),
        "isolated_count": len(visible_rows),
        "cache_roots": [str(root) for root in cache_roots],
        "manifest_path": str(validation_root / ISOLATED_MANIFEST_PATH),
    }


def demucs_cache_status(analysis_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    cache_roots = demucs_source_cache_roots()
    cache_hits: list[dict[str, Any]] = []
    missing_cache: list[dict[str, Any]] = []
    for row in analysis_manifest:
        clip_id = str(row["clip_id"])
        allowed_sources = allowed_demucs_sources_for_analysis(row)
        if not allowed_sources:
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(row.get("audio_path")),
                    "reason": "missing_metadata_targets",
                }
            )
            continue
        audio_path = Path(str(row["audio_path"])).expanduser().resolve()
        if not audio_path.exists():
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "reason": "missing_audio",
                }
            )
            continue
        sample_rate = int(soundfile.info(str(audio_path)).samplerate)
        cache_path = demucs_sources_cache_path(audio_path, sample_rate, cache_roots)
        if cache_path.exists():
            cache_hits.append(
                {
                    "clip_id": clip_id,
                    "cache_path": str(cache_path),
                    "metadata_allowed_sources": sorted(allowed_sources),
                }
            )
        else:
            missing_cache.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "expected_cache_path": str(cache_path),
                    "candidate_cache_paths": [
                        str(path)
                        for path in demucs_sources_cache_candidates(audio_path, sample_rate, cache_roots)
                    ],
                }
            )
    return {
        "track_count": len(analysis_manifest),
        "cache_hit_tracks": len(cache_hits),
        "missing_cache": missing_cache,
        "cache_roots": [str(root) for root in cache_roots],
    }


def allowed_demucs_sources_for_analysis(row: Mapping[str, Any]) -> set[str]:
    analysis = row.get("analysis") or {}
    dataset = analysis.get("dataset") or {}
    dataset_name = dataset.get("name")
    instrument_tags = analysis.get("instrument_tags") or []
    if dataset_name is None or not instrument_tags:
        return set()
    if dataset_name == "mtg_jamendo":
        dataset_config = yaml.safe_load(config_path("datasets_mtg_jamendo").read_text(encoding="utf-8")) or {}
        demucs_target_map = dataset_config.get("dataset", {}).get("demucs_target_map", {})
        return {
            str(demucs_target_map[instrument_tag])
            for instrument_tag in instrument_tags
            if instrument_tag in demucs_target_map
        }
    targets = analysis.get("target_candidates") or []
    allowed = set()
    for target in targets:
        source_name = target.get("separation_target") or target.get("stem")
        if source_name not in (None, ""):
            allowed.add(str(source_name))
    return allowed


def source_energy_stats(source_audio: Any, mixture_rms: float) -> dict[str, Any]:
    source_rms = rms_audio(source_audio)
    rms_ratio = source_rms / max(mixture_rms, MIN_STEM_RMS)
    return {
        "rms": source_rms,
        "rms_ratio": rms_ratio,
        "active": source_rms >= MIN_STEM_RMS and rms_ratio >= MIN_STEM_RMS_RATIO,
    }


def rms_audio(audio_data: Any) -> float:
    if isinstance(audio_data, torch.Tensor):
        audio_tensor = audio_data.detach().cpu().float()
    else:
        audio_tensor = torch.as_tensor(audio_data).float()
    return float(torch.sqrt(torch.mean(audio_tensor * audio_tensor)).item())


def sum_source_audio(sources: Mapping[str, Any]) -> torch.Tensor:
    tensors = []
    for source_audio in sources.values():
        if isinstance(source_audio, torch.Tensor):
            tensors.append(source_audio.detach().cpu().float())
        else:
            tensors.append(torch.as_tensor(source_audio).float())
    return torch.stack(tensors, dim=0).sum(dim=0)


def demucs_source_cache_roots() -> list[Path]:
    validation_root = validation_root_path()
    roots = [
        validation_root / "demucs_cache",
        validation_root / "renders" / "_demucs_cache",
    ]
    manifests = read_jsonl(validation_root / "renders" / "render_manifest.jsonl", limit=20000)
    manifests.extend(read_many_jsonl(validation_root / COUNTERFACTUAL_ROOT / "manifests", limit=20000))
    for row in manifests:
        separation_cache_dir = row.get("separation_cache_dir")
        if separation_cache_dir not in (None, ""):
            roots.append(Path(str(separation_cache_dir)).expanduser().resolve())
    return unique_paths(roots)


def demucs_sources_cache_path(
    audio_path: Path,
    sample_rate: int,
    cache_roots: list[Path] | None = None,
) -> Path:
    candidates = demucs_sources_cache_candidates(audio_path, sample_rate, cache_roots)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def demucs_sources_cache_candidates(
    audio_path: Path,
    sample_rate: int,
    cache_roots: list[Path] | None = None,
) -> list[Path]:
    validation_root = validation_root_path()
    audio_stat = audio_path.stat()
    server_config = load_server_config()
    payload = {
        "audio_path": str(audio_path),
        "audio_size": audio_stat.st_size,
        "audio_mtime_ns": audio_stat.st_mtime_ns,
        "description": None,
        "sample_rate": sample_rate,
        "separation_backend": server_config.get("separation_backend"),
        "demucs_model": server_config.get("demucs_model"),
        "sam_model": server_config.get("sam_model"),
        "prefix": "demucs_sources",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    roots = cache_roots if cache_roots is not None else demucs_source_cache_roots()
    candidates = []
    for root in roots:
        if root.name == "demucs_sources":
            candidates.append(root / ("%s.pt" % digest))
        else:
            candidates.append(root / "demucs_sources" / ("%s.pt" % digest))
    if not candidates:
        candidates.append(validation_root / "demucs_cache" / "demucs_sources" / ("%s.pt" % digest))
    return unique_paths(candidates)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique = []
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def load_server_config() -> dict[str, Any]:
    config_path = repo_root_path() / "zero_shot_agent.toml"
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return dict(raw_config.get("server", raw_config))


def runtime_plan_compiler() -> Any:
    if str(repo_root_path()) not in sys.path:
        sys.path.insert(0, str(repo_root_path()))
    runtime_module = importlib.import_module("ground_truth.runtime")
    return runtime_module.RuntimePlanCompiler


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def write_jsonl_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def file_from_absolute(value: Any, validation_root: Path) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        return None
    if validation_root not in path.parents and path != validation_root:
        return None
    return file_from_relative(str(path.relative_to(validation_root)), validation_root)


def file_from_relative(relative_path: str, validation_root: Path) -> dict[str, Any]:
    path = validation_root / relative_path
    return {
        "name": path.name,
        "relative_path": relative_path,
        "url": "/api/artifacts/file/%s" % relative_path,
        "size": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
    }


def clip_id_for_any_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return clip_id_for_relative_path(str(path))


def clip_id_for_relative_path(relative_path: str) -> str | None:
    if not relative_path:
        return None
    parts = Path(relative_path).parts
    if len(parts) >= 2 and parts[0] in {"renders", "counterfactual"}:
        return parts[1]
    if len(parts) >= 2 and parts[0] == str(ISOLATED_ROOT):
        return parts[1]
    if len(parts) >= 2 and parts[0] == "audio":
        return Path(parts[1]).stem
    return None


def target_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get("target_stem") or row.get("separation_target") or target_from_graph(row.get("graph_spec", [])) or "unknown")


def target_from_graph(graph_spec: Any) -> str | None:
    for block in graph_spec or []:
        if block.get("kind") == "separate":
            return str(block.get("description"))
    return None


def graph_params(graph_spec: Any) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for block in graph_spec or []:
        if block.get("kind") == "send_return":
            params.append(
                {
                    "block": block.get("name"),
                    "step": "send_return",
                    "operator": "send_return",
                    "params": {
                        "dry_level": block.get("dry_level", 1.0),
                        "send_level": block.get("send_level", 1.0),
                        "return_level": block.get("return_level", 1.0),
                    },
                }
            )
        for step in block.get("steps", []):
            params.append(
                {
                    "block": block.get("name") or block.get("prefix"),
                    "step": step.get("name"),
                    "operator": step.get("operator"),
                    "params": step.get("params", {}),
                }
            )
        if block.get("kind") == "step":
            params.append(
                {
                    "block": block.get("name"),
                    "step": block.get("name"),
                    "operator": block.get("operator"),
                    "params": block.get("params", {}),
                }
            )
    return params


def params_summary(graph_spec: Any) -> str:
    chunks = []
    for item in graph_params(graph_spec):
        params = item.get("params", {})
        rendered = ", ".join("%s=%s" % (key, value) for key, value in sorted(params.items()))
        chunks.append("%s(%s)" % (item.get("operator"), rendered))
    return " | ".join(chunks) if chunks else "no params"


def describe_graph_spec(graph_spec: Any) -> str:
    lines = []
    for block in graph_spec or []:
        kind = block.get("kind")
        if kind == "separate":
            lines.append("separate target=%s" % block.get("description"))
        elif kind in {"chain", "send_return"}:
            lines.append("%s %s: %s" % (kind, block.get("name") or block.get("prefix"), params_summary([block])))
        elif kind == "mix":
            lines.append("mix stem=%s residual=%s" % (block.get("stem"), block.get("residual")))
        elif kind == "step":
            lines.append("%s: %s" % (block.get("name"), params_summary([block])))
    return "\n".join(lines)


def source_label_for_relative(relative_path: str) -> str:
    clip_id = clip_id_for_relative_path(relative_path)
    if relative_path.startswith("audio/"):
        return "%s full mix" % clip_id
    if relative_path.startswith("renders/"):
        return "%s rendered file: %s" % (clip_id, Path(relative_path).name)
    if relative_path.startswith("counterfactual/"):
        return "%s counterfactual file: %s" % (clip_id, Path(relative_path).name)
    if relative_path.startswith("%s/" % ISOLATED_ROOT):
        return "%s isolated source: %s" % (clip_id, Path(relative_path).stem)
    if relative_path.startswith("ad_hoc/"):
        return "ad hoc file: %s" % Path(relative_path).name
    return relative_path


def artifact_path(relative_path: str) -> Path:
    validation_root = validation_root_path()
    path = (validation_root / relative_path).resolve()
    if validation_root not in path.parents and path != validation_root:
        raise ValueError("Artifact path escapes validation root.")
    return path


def load_audio_tensor(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = soundfile.read(
        path,
        always_2d=True,
        dtype="float32",
    )
    return torch.from_numpy(audio.T.copy()), int(sample_rate)


def save_audio_tensor(
    audio_data: Any,
    sample_rate: int,
    output_path: Path,
) -> None:
    if isinstance(audio_data, torch.Tensor):
        audio_array = audio_data.detach().cpu().float().numpy()
    else:
        audio_array = np.asarray(audio_data, dtype=np.float32)
    if audio_array.ndim == 1:
        audio_array = audio_array[np.newaxis, :]
    soundfile.write(output_path, np.ascontiguousarray(audio_array.T), sample_rate)


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned or "fx"


if __name__ == "__main__":
    main()
