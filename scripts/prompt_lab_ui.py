"""Diagnostic UI for effect-chain prompt generation.

Pick a chain of operators, type parameter values, and read the edit graph the
pipeline builds from it followed by the prompt generated at each abstraction
level of the ladder, with the exact model request behind each one.

Nothing about the prompt wording lives here. The prompt script is loaded fresh
from disk on every request, the ladder and chain reader it needs are rebuilt
through its own imported symbols, and its LLM calls are intercepted at
`litellm.completion` -- so edited prompts, renamed helpers, new ladder levels,
retuned bands, and new operators all show up without touching this file or
restarting the server.

Parameter values are checked two ways: the planner's own `validate_graph_spec`
(a hard error), and the sampling priors in distributions.yaml (a warning, since
an off-prior value renders fine but is not one the pipeline would produce).

Usage:
    python scripts/prompt_lab_ui.py --port 8788

Live generation needs the prompt script's own credentials (GEMINI_API_KEY in a
.env file or the environment). The UI's "dry run" checkbox skips every network
call and stubs the model replies, which is enough to read the exact prompt text
at each layer without a key.
"""

import argparse
import copy
import getpass
import importlib.util
import inspect
import json
import os
import re
import socket
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SCRIPT = Path("scripts/generate_ground_truth_prompts.py")
DEFAULT_CONFIG_DIR = Path("configs/ground_truth")
DATASET_CONFIG_PATH = Path("configs/ground_truth/datasets/mtg_jamendo.yaml")
PROMPT_MODULE_NAME = "_prompt_lab_prompt_script"
# Operators that are structural rather than effects; the chain builder emits
# them itself, so they are hidden from the effect picker.
STRUCTURAL_OPERATORS = {"separate_audio", "mix_stems"}
# Entrypoints tried in order when calling into the prompt script.
ENTRYPOINT_CANDIDATES = ("build_prompt_chain", "build_prompts", "generate_prompt_chain")
# Keys checked in order when pulling the prompt ladder out of the result.
VARIANT_KEY_CANDIDATES = ("prompt_variants", "prompts", "descriptions", "variants")
# Keys checked in order for per-level detail (name, checks, retry count).
LEVEL_KEY_CANDIDATES = ("prompt_levels", "levels", "prompt_entries")
DRY_RUN_TEMPLATE = "[dry-run stage %d output: the real model reply would appear here]"
# Dry runs make no network calls, so API-key guards inside the prompt script are
# satisfied with placeholders. Names are scraped from the script itself so a
# switch to another provider keeps working.
API_KEY_RE = re.compile(r"""getenv\(\s*["']([A-Z0-9_]*API_KEY[A-Z0-9_]*)["']""")
DRY_RUN_API_KEY_NAMES = ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
DRY_RUN_API_KEY_VALUE = "prompt-lab-dry-run"

APP_STATE: dict[str, Any] = {
    "repo_root": DEFAULT_REPO_ROOT,
    "prompt_script": DEFAULT_REPO_ROOT / DEFAULT_PROMPT_SCRIPT,
    "config_dir": DEFAULT_REPO_ROOT / DEFAULT_CONFIG_DIR,
}
PROMPT_SCRIPT_LOCK = threading.Lock()

app = FastAPI()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host. Default: %(default)s")
    parser.add_argument("--port", type=int, default=8788, help="Bind port. Default: %(default)s")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root. Default: %(default)s")
    parser.add_argument(
        "--prompt-script",
        type=Path,
        default=None,
        help="Prompt generation script to exercise. Default: <repo-root>/%s" % DEFAULT_PROMPT_SCRIPT,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Ground-truth config directory. Default: <repo-root>/%s" % DEFAULT_CONFIG_DIR,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    APP_STATE["repo_root"] = repo_root
    APP_STATE["prompt_script"] = (
        args.prompt_script.expanduser().resolve()
        if args.prompt_script is not None
        else repo_root / DEFAULT_PROMPT_SCRIPT
    )
    APP_STATE["config_dir"] = (
        args.config_dir.expanduser().resolve()
        if args.config_dir is not None
        else repo_root / DEFAULT_CONFIG_DIR
    )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print_access_hint(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


def print_access_hint(host: str, port: int) -> None:
    """Print the SSH tunnel needed to reach this server from a laptop.

    The UI is only useful with a browser on it, and on a cluster the browser is
    never on the same machine. Compute nodes are not reachable directly, so a
    job step gets the jump-host form.
    """
    node = socket.getfqdn()
    user = getpass.getuser()
    job_id = os.environ.get("SLURM_JOB_ID")
    submit_host = os.environ.get("SLURM_SUBMIT_HOST", "<login-host>")
    print("prompt lab serving on %s:%d (node %s)" % (host, port, node), flush=True)
    if job_id:
        print(
            "  tunnel:  ssh -N -J %s@%s -L %d:127.0.0.1:%d %s@%s   # slurm job %s"
            % (user, submit_host, port, port, user, node, job_id),
            flush=True,
        )
    else:
        print("  tunnel:  ssh -N -L %d:127.0.0.1:%d %s@%s" % (port, port, user, node), flush=True)
    print("  open:    http://127.0.0.1:%d" % port, flush=True)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "  warning: bound to %s, so anyone on this network can drive the UI and spend your API key."
            % host,
            flush=True,
        )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    """Everything the UI needs to draw itself, re-read from disk each call."""
    return {
        "repo_root": str(repo_root_path()),
        "config_dir": str(config_dir_path()),
        "operators": operator_specs(),
        "order_profiles": order_profiles(),
        "targets": target_names(),
        "param_priors": param_priors(),
        "prompt_script": prompt_script_info(),
    }


@app.post("/api/build")
def build(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Layers 0 and 1 only: no model calls, so it is safe to spam."""
    try:
        graph_spec = build_graph_spec(body)
    except Exception as error:
        return {"error": format_error(error)}
    return {
        "graph_spec": graph_spec,
        "graph_description": describe_graph_spec(graph_spec),
        "validation": validate_graph_spec(graph_spec),
        "param_warnings": check_param_priors(graph_spec) + check_send_bus_wetness(graph_spec),
    }


@app.post("/api/prompts")
def prompts(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Every layer, including the prompt script's LLM stages."""
    try:
        graph_spec = build_graph_spec(body)
    except Exception as error:
        return {"error": format_error(error)}
    payload: dict[str, Any] = {
        "graph_spec": graph_spec,
        "graph_description": describe_graph_spec(graph_spec),
        "validation": validate_graph_spec(graph_spec),
        "param_warnings": check_param_priors(graph_spec) + check_send_bus_wetness(graph_spec),
    }
    payload.update(
        run_prompt_script(
            graph_spec=graph_spec,
            graph_description=payload["graph_description"],
            body=body,
        )
    )
    return payload


def repo_root_path() -> Path:
    return Path(APP_STATE["repo_root"]).resolve()


def config_dir_path() -> Path:
    return Path(APP_STATE["config_dir"]).resolve()


def prompt_script_path() -> Path:
    return Path(APP_STATE["prompt_script"]).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def operator_specs() -> list[dict[str, Any]]:
    parsed = load_yaml(config_dir_path() / "operators.yaml")
    specs: list[dict[str, Any]] = []
    for operator in parsed.get("operators", []):
        name = operator.get("name")
        if name in STRUCTURAL_OPERATORS:
            continue
        specs.append(
            {
                "name": name,
                "aliases": list(operator.get("aliases", [])),
                "tags": list(operator.get("tags", [])),
                "params": list(operator.get("params", [])),
            }
        )
    return specs


def order_profiles() -> list[str]:
    constraints = load_yaml(config_dir_path() / "constraints.yaml").get("constraints", {})
    profiles = constraints.get("chain_order", {}).get("profiles", {})
    return sorted(profiles)


def target_names() -> list[str]:
    """Separation targets worth offering, taken from the dataset config."""
    dataset = load_yaml(repo_root_path() / DATASET_CONFIG_PATH).get("dataset", {})
    targets = set(dataset.get("demucs_target_map", {}).values())
    targets.update({"vocals", "drums", "bass", "guitar", "piano", "other"})
    return sorted(str(target) for target in targets)


def distribution_leaves() -> dict[str, dict[str, Any]]:
    """Flatten distributions.yaml to `dotted.path -> leaf spec`.

    Leaves are the nodes carrying `type`; everything above them is grouping, and
    only leaves can be named by a `sample:` reference.
    """
    parsed = load_yaml(config_dir_path() / "distributions.yaml").get("distributions", {})
    leaves: dict[str, dict[str, Any]] = {}

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, Mapping):
            return
        if "type" in node:
            leaves[prefix] = dict(node)
            return
        for key, child in node.items():
            walk(child, "%s.%s" % (prefix, key) if prefix else str(key))

    walk(parsed, "")
    return leaves


def collect_sample_paths(value: Any, derived_via: str | None = None) -> list[dict[str, Any]]:
    """Pull `sample:` references out of one param spec, noting any transform.

    A reference under `tempo_sync`/`scale` samples in different units than the
    param itself, so it is recorded as derived and never range-checked.
    """
    references: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "sample" and isinstance(child, str):
                references.append({"path": child, "derived_via": derived_via})
                continue
            references.extend(collect_sample_paths(child, derived_via or str(key)))
    elif isinstance(value, list):
        for child in value:
            references.extend(collect_sample_paths(child, derived_via))
    return references


def walk_operator_steps(node: Any) -> list[Mapping[str, Any]]:
    """Find every `{operator, params}` step anywhere in a config tree.

    Recipes nest steps in graph blocks, motifs in `steps`, poisons elsewhere
    again; walking for the shape rather than the path keeps this working when
    the config layout moves.
    """
    steps: list[Mapping[str, Any]] = []
    if isinstance(node, Mapping):
        if "operator" in node and isinstance(node.get("params"), Mapping):
            steps.append(node)
        for child in node.values():
            steps.extend(walk_operator_steps(child))
    elif isinstance(node, list):
        for child in node:
            steps.extend(walk_operator_steps(child))
    return steps


def param_priors() -> dict[str, dict[str, dict[str, Any]]]:
    """Index `operator -> param -> prior`, as the planner itself binds them.

    Config usage is the source of truth for which distribution belongs to which
    operator param, since distributions.yaml is organized by workflow rather
    than by operator. Params never sampled anywhere fall back to a
    same-named leaf, flagged so the UI can say the match is only by name.
    """
    leaves = distribution_leaves()
    priors: dict[str, dict[str, dict[str, Any]]] = {}
    for file_name in ("motifs.yaml", "recipes.yaml"):
        for step in walk_operator_steps(load_yaml(config_dir_path() / file_name)):
            operator = str(step["operator"])
            for param, value in dict(step["params"]).items():
                prior = priors.setdefault(operator, {}).setdefault(
                    str(param),
                    {"distributions": [], "literals": [], "matched_by_name": False},
                )
                for reference in collect_sample_paths(value):
                    spec = leaves.get(reference["path"])
                    if spec is None:
                        continue
                    entry = {
                        "path": reference["path"],
                        "derived_via": reference["derived_via"],
                        **summarize_distribution(spec),
                    }
                    if entry not in prior["distributions"]:
                        prior["distributions"].append(entry)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value not in prior["literals"]:
                        prior["literals"].append(value)

    for operator in operator_specs():
        for param in operator["params"]:
            prior = priors.setdefault(operator["name"], {}).setdefault(
                str(param),
                {"distributions": [], "literals": [], "matched_by_name": False},
            )
            if prior["distributions"] or prior["literals"]:
                continue
            for path, spec in leaves.items():
                if path.rsplit(".", 1)[-1] == str(param):
                    prior["matched_by_name"] = True
                    prior["distributions"].append(
                        {"path": path, "derived_via": None, **summarize_distribution(spec)}
                    )
    return priors


def summarize_distribution(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one distribution spec to what a range check needs."""
    kind = str(spec.get("type", ""))
    if kind == "uniform":
        return {
            "type": kind,
            "low": spec.get("low"),
            "high": spec.get("high"),
            "values": [],
        }
    if kind == "choice":
        values = [
            entry.get("value") if isinstance(entry, Mapping) else entry
            for entry in spec.get("values", [])
        ]
        numeric = [value for value in values if isinstance(value, (int, float))]
        return {
            "type": kind,
            "low": min(numeric) if numeric else None,
            "high": max(numeric) if numeric else None,
            "values": values,
        }
    return {"type": kind or "unknown", "low": spec.get("low"), "high": spec.get("high"), "values": []}


def check_param_priors(graph_spec: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Warn where a typed value falls outside what the planner would sample.

    These priors shape sampling rather than gate rendering, so everything here
    is a warning: the value is renderable, it just is not one the ground-truth
    pipeline would ever produce.
    """
    priors = param_priors()
    warnings: list[dict[str, Any]] = []
    for step in walk_operator_steps({"blocks": list(graph_spec)}):
        operator = str(step["operator"])
        for param, value in dict(step["params"]).items():
            prior = priors.get(operator, {}).get(str(param))
            if prior is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            direct = [entry for entry in prior["distributions"] if entry.get("derived_via") is None]
            if not direct:
                continue
            warning = prior_warning(operator, str(param), float(value), direct, prior["matched_by_name"])
            if warning is not None:
                warnings.append(warning)
    return warnings


def check_send_bus_wetness(graph_spec: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Warn when fx on a send bus are not fully wet.

    `add_send_return` already passes the dry signal around the chain, so an fx
    that also blends its own dry doubles the unprocessed signal. The motifs set
    reverb to `wet_level: 1.0, dry_level: 0.0` and delay to `mix: 1.0` for this
    reason. Checked by param name, since that is what carries the blend.
    """
    fully_wet = {"dry_level": 0.0, "wet_level": 1.0, "mix": 1.0}
    warnings: list[dict[str, Any]] = []
    for block in graph_spec or []:
        if block.get("kind") != "send_return":
            continue
        for step in block.get("steps", []):
            for param, value in dict(step.get("params", {})).items():
                expected = fully_wet.get(str(param))
                if expected is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if float(value) == expected:
                    continue
                warnings.append(
                    {
                        "operator": str(step.get("operator", "")),
                        "param": str(param),
                        "value": value,
                        "severity": "send_bus_blend",
                        "message": "%s: %s=%s on send bus '%s' blends its own dry signal, but the bus "
                        "already passes dry at dry_level=%s. Set %s=%s so the dry is not doubled."
                        % (
                            step.get("operator", ""),
                            param,
                            format_number(float(value)),
                            block.get("name", "send_return"),
                            format_number(float(block.get("dry_level", 1.0))),
                            param,
                            format_number(expected),
                        ),
                    }
                )
    return warnings


def prior_warning(
    operator: str,
    param: str,
    value: float,
    entries: Sequence[Mapping[str, Any]],
    matched_by_name: bool,
) -> dict[str, Any] | None:
    lows = [entry["low"] for entry in entries if entry.get("low") is not None]
    highs = [entry["high"] for entry in entries if entry.get("high") is not None]
    if not lows or not highs:
        return None
    low = min(float(bound) for bound in lows)
    high = max(float(bound) for bound in highs)
    paths = ", ".join(str(entry["path"]) for entry in entries)
    source = "matched by param name" if matched_by_name else "used by the planner here"
    if value < low or value > high:
        return {
            "operator": operator,
            "param": param,
            "value": value,
            "severity": "outside_range",
            "message": "%s: %s=%s is outside the prior range %s to %s (%s; %s)."
            % (operator, param, value, format_number(low), format_number(high), paths, source),
        }
    choices = [
        float(choice)
        for entry in entries
        if entry.get("type") == "choice"
        for choice in entry.get("values", [])
        if isinstance(choice, (int, float))
    ]
    if choices and all(entry.get("type") == "choice" for entry in entries) and value not in choices:
        rendered = ", ".join(format_number(choice) for choice in sorted(set(choices)))
        return {
            "operator": operator,
            "param": param,
            "value": value,
            "severity": "off_grid",
            "message": "%s: %s=%s is in range but not one of the sampled values {%s} (%s; %s)."
            % (operator, param, value, rendered, paths, source),
        }
    return None


def format_number(value: float) -> str:
    return ("%g" % value) if isinstance(value, (int, float)) else str(value)


def build_graph_spec(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Assemble a graph spec from the UI form, or take a pasted one verbatim."""
    pasted = str(body.get("graph_spec_json", "")).strip()
    if pasted:
        graph_spec = json.loads(pasted)
        if not isinstance(graph_spec, list):
            raise ValueError("Pasted graph spec must be a JSON list of blocks.")
        return graph_spec

    steps = build_steps(body.get("steps", []))
    if not steps:
        raise ValueError("Add at least one effect to the chain.")

    target = str(body.get("target", "vocals")).strip() or "vocals"
    block_kind = str(body.get("block_kind", "chain"))
    if block_kind not in {"chain", "send_return"}:
        raise ValueError("Unsupported block kind '%s'." % block_kind)
    separate = bool(body.get("separate_target", True))
    order_profile = str(body.get("order_profile", "")).strip()

    source = "target_stem" if separate else "audio"
    output = "processed_stem" if separate else "final_audio"
    graph_spec: list[dict[str, Any]] = []
    if separate:
        graph_spec.append(
            {
                "kind": "separate",
                "name": "isolate_target",
                "source": "audio",
                "description": target,
                "outputs": ["target_stem", "residual"],
            }
        )

    if block_kind == "chain":
        block: dict[str, Any] = {
            "kind": "chain",
            "prefix": str(body.get("block_name", "target_fx")) or "target_fx",
            "source": source,
            "output": output,
            "steps": steps,
        }
    else:
        block = {
            "kind": "send_return",
            "name": str(body.get("block_name", "target_send")) or "target_send",
            "source": source,
            "output": output,
            "dry_level": float(body.get("dry_level", 1.0)),
            "send_level": float(body.get("send_level", 0.5)),
            "return_level": float(body.get("return_level", 0.85)),
            "steps": steps,
        }
    if order_profile:
        block["order_profile"] = order_profile
    graph_spec.append(block)

    if separate:
        graph_spec.append(
            {
                "kind": "mix",
                "name": "remix_target",
                "stem": output,
                "residual": "residual",
                "output": "final_audio",
            }
        )
    return graph_spec


def build_steps(raw_steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        operator = str(raw_step.get("operator", "")).strip()
        if not operator:
            continue
        params = {
            str(name): coerce_param(value)
            for name, value in dict(raw_step.get("params", {})).items()
            if str(value).strip() != ""
        }
        steps.append(
            {
                "name": str(raw_step.get("name", "")).strip() or "fx_%d" % index,
                "operator": operator,
                "params": params,
            }
        )
    return steps


def coerce_param(value: Any) -> Any:
    """Read a form field as JSON when it looks like JSON, else as a string."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def describe_graph_spec(graph_spec: Sequence[Mapping[str, Any]]) -> str:
    from ground_truth.symbolic_graph import SymbolicEditGraph

    return SymbolicEditGraph.describe_blocks(graph_spec)


def validate_graph_spec(graph_spec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Run the planner's own validation so ordering/param errors surface here."""
    try:
        from ground_truth.planner import GroundTruthPlanner

        planner = GroundTruthPlanner.from_directory(config_dir_path())
        planner.validate_graph_spec(graph_spec)
    except Exception as error:
        return {"ok": False, "error": format_error(error)}
    return {"ok": True, "error": None}


class LenientRecord(dict):
    """Plan-row stand-in that never raises on a key the UI does not know about."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.missing_keys: list[str] = []

    def __missing__(self, key: str) -> str:
        if key not in self.missing_keys:
            self.missing_keys.append(key)
        return ""


class StubMessage:
    def __init__(self, content: str):
        self.content = content


class StubChoice:
    def __init__(self, content: str):
        self.message = StubMessage(content)


class StubResponse:
    """Minimal stand-in for a litellm ModelResponse in dry-run mode."""

    def __init__(self, content: str):
        self.choices = [StubChoice(content)]

    def __getitem__(self, key: str) -> Any:
        if key == "choices":
            return [{"message": {"content": self.choices[0].message.content}}]
        raise KeyError(key)


class CompletionRecorder:
    """Stands in for `litellm.completion` and logs every call the script makes."""

    def __init__(self, passthrough: Any, dry_run: bool):
        self.passthrough = passthrough
        self.dry_run = dry_run
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and "model" not in kwargs:
            kwargs["model"] = args[0]
        messages = list(kwargs.get("messages", []))
        stage = len(self.calls) + 1
        record: dict[str, Any] = {
            "stage": stage,
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "messages": [
                {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
                for message in messages
            ],
            "response": None,
            "error": None,
        }
        self.calls.append(record)
        if self.dry_run:
            content = DRY_RUN_TEMPLATE % stage
            record["response"] = content
            record["dry_run"] = True
            return StubResponse(content)
        try:
            response = self.passthrough(**kwargs)
        except Exception as error:
            record["error"] = format_error(error)
            raise
        record["response"] = response_text(response)
        return response


def response_text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content)
    except Exception:
        return repr(response)


def load_prompt_module(recorder: CompletionRecorder | None) -> Any:
    """Load the prompt script fresh from disk, LLM calls optionally intercepted.

    The script binds `completion` at import time (`from litellm import
    completion`), so patching `litellm.completion` before executing the module
    is what makes interception survive renames of the script's own helpers.
    """
    path = prompt_script_path()
    if not path.exists():
        raise FileNotFoundError("Prompt script '%s' does not exist." % path)
    if str(repo_root_path()) not in sys.path:
        sys.path.insert(0, str(repo_root_path()))

    import litellm

    original_completion = litellm.completion
    if recorder is not None:
        recorder.passthrough = original_completion
        litellm.completion = recorder
    try:
        spec = importlib.util.spec_from_file_location(PROMPT_MODULE_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[PROMPT_MODULE_NAME] = module
        spec.loader.exec_module(module)
    finally:
        litellm.completion = original_completion
        sys.modules.pop(PROMPT_MODULE_NAME, None)
    return module


def prompt_script_info() -> dict[str, Any]:
    path = prompt_script_path()
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "model": None,
        "max_rounds": None,
        "entrypoint": None,
        "functions": [],
        "error": None,
    }
    try:
        with PROMPT_SCRIPT_LOCK:
            module = load_prompt_module(recorder=None)
    except Exception as error:
        info["error"] = format_error(error)
        return info
    info["model"] = getattr(module, "MODEL_NAME", None)
    info["abstraction"] = abstraction_info(module)
    entrypoint = find_entrypoint(module)
    info["entrypoint"] = None if entrypoint is None else entrypoint.__name__
    info["functions"] = [
        "%s%s" % (name, inspect.signature(value))
        for name, value in vars(module).items()
        if inspect.isfunction(value) and getattr(value, "__module__", "") == PROMPT_MODULE_NAME
    ]
    return info


def abstraction_info(module: Any) -> dict[str, Any]:
    """Describe the abstraction ladder the script would generate, if it has one.

    Levels are declared in config, so the UI reads them off the ladder rather
    than assuming a fixed number of rewrite rounds.
    """
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "levels": [],
        "max_attempts": int(getattr(module, "MAX_ATTEMPTS", getattr(module, "MAX_ROUNDS", 3)) or 1),
        "error": None,
    }
    resolver = ArgumentResolver(module=module, record=LenientRecord({}), body={})
    try:
        ladder = resolver.ladder()
        levels = ladder.levels()
    except Exception as error:
        info["error"] = format_error(error)
        return info
    info["available"] = True
    info["version"] = getattr(ladder, "version", None)
    info["levels"] = [
        {
            "id": level.id,
            "name": str(level.name),
            "derives_from": level.derives_from,
            "constraints": dict(getattr(level, "constraints", {})),
            "max_tokens": getattr(level, "max_tokens", None),
        }
        for level in levels
    ]
    return info


def find_entrypoint(module: Any) -> Any:
    """Locate the per-plan prompt builder, tolerating a renamed function."""
    for name in ENTRYPOINT_CANDIDATES:
        candidate = getattr(module, name, None)
        if inspect.isfunction(candidate):
            return candidate
    for value in vars(module).values():
        if not inspect.isfunction(value) or getattr(value, "__module__", "") != PROMPT_MODULE_NAME:
            continue
        if "record" in inspect.signature(value).parameters:
            return value
    return None


def patch_dry_run_api_keys() -> dict[str, str | None]:
    """Give the script placeholder credentials so dry runs need no real keys."""
    names = set(DRY_RUN_API_KEY_NAMES)
    path = prompt_script_path()
    if path.exists():
        names.update(API_KEY_RE.findall(path.read_text(encoding="utf-8")))
    previous: dict[str, str | None] = {}
    for name in sorted(names):
        if os.environ.get(name):
            continue
        previous[name] = os.environ.get(name)
        os.environ[name] = DRY_RUN_API_KEY_VALUE
    return previous


def unpatch_dry_run_api_keys(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def run_prompt_script(
    graph_spec: Sequence[Mapping[str, Any]],
    graph_description: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Call the prompt script for one synthetic plan row and record every stage."""
    dry_run = bool(body.get("dry_run", False))
    recorder = CompletionRecorder(passthrough=None, dry_run=dry_run)
    record = LenientRecord(
        {
            "graph_description": graph_description,
            "graph_spec": copy.deepcopy(list(graph_spec)),
            "audio_path": "prompt_lab/synthetic.wav",
            "clip_id": "prompt_lab_clip",
            "plan_id": "prompt_lab_plan",
            "target_stem": str(body.get("target", "vocals")),
            "target_family": str(body.get("target", "vocals")),
            "separation_target": str(body.get("target", "vocals")),
            "recipe_id": "prompt_lab",
            "recipe_tags": ["prompt_lab"],
        }
    )
    payload: dict[str, Any] = {
        "dry_run": dry_run,
        "llm_calls": [],
        "prompt_variants": [],
        "prompt_levels": [],
        "entrypoint": None,
        "entrypoint_args": {},
        "missing_record_keys": [],
        "result": None,
        "error": None,
    }
    try:
        with PROMPT_SCRIPT_LOCK:
            restore_environment = patch_dry_run_api_keys() if dry_run else {}
            try:
                module = load_prompt_module(recorder=recorder)
                entrypoint = find_entrypoint(module)
                if entrypoint is None:
                    raise ValueError(
                        "No prompt entrypoint found in '%s'. Expected one of %s or a function taking 'record'."
                        % (prompt_script_path(), ", ".join(ENTRYPOINT_CANDIDATES))
                    )
                arguments = entrypoint_arguments(entrypoint, record=record, body=body, module=module)
                payload["entrypoint"] = entrypoint.__name__
                payload["entrypoint_args"] = {
                    name: summarize_argument(value)
                    for name, value in arguments.items()
                    if value is not record
                }
                result = entrypoint(**arguments)
            finally:
                unpatch_dry_run_api_keys(restore_environment)
    except Exception as error:
        payload["error"] = format_error(error)
    else:
        payload["result"] = jsonable(result)
        payload["prompt_variants"] = extract_variants(result)
        payload["prompt_levels"] = extract_level_entries(result)
    payload["llm_calls"] = recorder.calls
    payload["missing_record_keys"] = record.missing_keys
    return payload


class ArgumentResolver:
    """Builds whatever the prompt entrypoint declares, lazily and by name.

    The script assembles its ladder, band lexicon and chain reader in `main()`,
    wrapped in plan-file I/O the UI must not run. So those objects are rebuilt
    here through the module's *own* imported symbols rather than reimplemented:
    edits to the abstraction configs or to those constructors take effect on the
    next request, and only the argument names are coupled.
    """

    def __init__(self, module: Any, record: LenientRecord, body: Mapping[str, Any]):
        self.module = module
        self.record = record
        self.body = body
        self.cache: dict[str, Any] = {}

    def providers(self) -> dict[str, Any]:
        graph_description = str(self.record["graph_description"])
        return {
            "record": lambda: self.record,
            "row": lambda: self.record,
            "plan": lambda: self.record,
            "graph": lambda: graph_description,
            "edit_graph": lambda: graph_description,
            "graph_description": lambda: graph_description,
            "config_dir": config_dir_path,
            "registry": self.registry,
            "operator_registry": self.registry,
            "ladder": self.ladder,
            "lexicon": self.lexicon,
            "reader": self.reader,
            "levels": self.levels,
            "profile": self.profile,
            "max_attempts": self.max_attempts,
            "max_rounds": self.max_attempts,
            "rounds": self.max_attempts,
        }

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call one of the module's own imported symbols."""
        attribute = getattr(self.module, name, None)
        if attribute is None:
            raise ValueError(
                "Prompt script '%s' does not expose '%s', which this UI needs to rebuild "
                "the generator's setup." % (prompt_script_path(), name)
            )
        return attribute(*args, **kwargs)

    def registry(self) -> Any:
        if "registry" not in self.cache:
            self.cache["registry"] = self.call("load_operator_registry", config_dir_path())
        return self.cache["registry"]

    def abstraction_config(self) -> tuple[Any, Any]:
        if "abstraction" not in self.cache:
            self.cache["abstraction"] = self.call(
                "load_abstraction_config",
                config_dir=config_dir_path(),
                registry=self.registry(),
            )
        return self.cache["abstraction"]

    def ladder(self) -> Any:
        return self.abstraction_config()[0]

    def lexicon(self) -> Any:
        return self.abstraction_config()[1]

    def reader(self) -> Any:
        if "reader" not in self.cache:
            self.cache["reader"] = self.call(
                "ChainReader", lexicon=self.lexicon(), registry=self.registry()
            )
        return self.cache["reader"]

    def levels(self) -> Any:
        requested = [int(level) for level in self.body.get("levels", []) or []]
        ladder = self.ladder()
        return ladder.select(requested) if requested else ladder.levels()

    def profile(self) -> Any:
        return self.reader().profile(self.record.get("graph_spec") or [])

    def max_attempts(self) -> int:
        default = getattr(self.module, "MAX_ATTEMPTS", getattr(self.module, "MAX_ROUNDS", 3))
        return int(self.body.get("max_attempts") or default or 1)


def entrypoint_arguments(
    entrypoint: Any,
    record: LenientRecord,
    body: Mapping[str, Any],
    module: Any,
) -> dict[str, Any]:
    """Fill only the parameters the entrypoint actually declares."""
    providers = ArgumentResolver(module=module, record=record, body=body).providers()
    arguments: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in inspect.signature(entrypoint).parameters.items():
        if name in providers:
            arguments[name] = providers[name]()
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise ValueError(
            "Entrypoint '%s' requires argument(s) this UI cannot supply: %s. Add a provider "
            "for it in ArgumentResolver.providers()." % (entrypoint.__name__, ", ".join(missing))
        )
    return arguments


def extract_level_entries(result: Any) -> list[dict[str, Any]]:
    """Pull per-level detail out of the result when the script reports it.

    Falls back to nothing, in which case the UI renders the plain variant list.
    """
    if not isinstance(result, Mapping):
        return []
    for key in LEVEL_KEY_CANDIDATES:
        value = result.get(key)
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and value
            and all(isinstance(entry, Mapping) and "text" in entry for entry in value)
        ):
            return [normalize_level_entry(entry, index) for index, entry in enumerate(value)]
    return []


def normalize_level_entry(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    level = entry.get("abstraction_level", entry.get("level", entry.get("id", index)))
    violations = entry.get("violations") or []
    return {
        "level": level,
        "name": str(entry.get("name", "")),
        "text": str(entry.get("text", "")),
        "derived_from": entry.get("derived_from", entry.get("derives_from")),
        "attempts": entry.get("attempts"),
        "checks_passed": entry.get("checks_passed"),
        "violations": [str(violation) for violation in violations],
    }


def summarize_argument(value: Any) -> Any:
    """Describe one entrypoint argument for the response.

    Arguments are now rich objects (a ladder, a chain reader, level dataclasses)
    that no JSON encoder can take, and the UI only ever needed a note of what
    was passed.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        identifiers = [getattr(item, "id", None) for item in value]
        if all(identifier is not None for identifier in identifiers):
            return "%d levels: %s" % (len(value), ", ".join(str(item) for item in identifiers))
        return "%s[%d]" % (type(value).__name__, len(value))
    version = getattr(value, "version", None)
    return "%s%s" % (type(value).__name__, "" if version is None else " (version %s)" % version)


def extract_variants(result: Any) -> list[str]:
    if isinstance(result, str):
        return [result]
    if isinstance(result, Sequence) and all(isinstance(item, str) for item in result):
        return list(result)
    if not isinstance(result, Mapping):
        return []
    for key in VARIANT_KEY_CANDIDATES:
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [str(item) for item in value]
    for value in result.values():
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            return list(value)
    return []


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return json.loads(json.dumps(value, default=str))
    return value


def format_error(error: Exception) -> str:
    return "".join(traceback.format_exception_only(type(error), error)).strip() + "\n\n" + traceback.format_exc()


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Prompt lab</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 10px 16px; border-bottom: 1px solid #8884; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; }
  header .meta { opacity: 0.7; font-size: 12px; }
  main { display: grid; grid-template-columns: minmax(360px, 460px) 1fr; gap: 0; height: calc(100vh - 48px); }
  .col { overflow-y: auto; padding: 14px 16px; }
  .col + .col { border-left: 1px solid #8884; }
  fieldset { border: 1px solid #8884; border-radius: 6px; margin: 0 0 12px; padding: 10px 12px; }
  legend { padding: 0 4px; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }
  label { display: block; font-size: 11px; opacity: 0.75; margin-bottom: 2px; }
  input, select, textarea, button { font: inherit; padding: 3px 6px; border-radius: 4px; border: 1px solid #8886; background: transparent; color: inherit; width: 100%; }
  textarea { min-height: 70px; resize: vertical; }
  button { cursor: pointer; width: auto; }
  button.primary { border-color: currentColor; font-weight: 600; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: end; margin-bottom: 8px; }
  .row > div { flex: 1 1 120px; }
  .inline { display: flex; gap: 6px; align-items: center; }
  .inline input[type=checkbox] { width: auto; }
  .step { border: 1px solid #8886; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
  .step-head { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; }
  .step-head .idx { opacity: 0.6; min-width: 18px; }
  .step-head select { flex: 1; }
  .step-head button { padding: 2px 7px; }
  .tags { font-size: 11px; opacity: 0.6; margin-bottom: 6px; }
  .params { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 6px; }
  .hint { font-size: 10px; opacity: 0.55; margin-top: 2px; }
  .levels { display: flex; flex-wrap: wrap; gap: 4px 12px; }
  .levels label { display: flex; gap: 4px; align-items: center; margin: 0; font-size: 12px; opacity: 1; }
  .levels input { width: auto; }
  .routing { font-size: 11px; opacity: 0.75; background: #8881; border-radius: 4px; padding: 7px 9px; margin: 0 0 8px; }
  input.off-prior { border-color: #c62828; }
  .warn { color: #b26a00; }
  .layer { border: 1px solid #8884; border-radius: 6px; margin-bottom: 12px; overflow: hidden; }
  .layer > h2 { font-size: 12px; margin: 0; padding: 7px 10px; background: #8881; display: flex; justify-content: space-between; gap: 10px; }
  .layer > h2 span.tag { font-weight: 400; opacity: 0.7; }
  pre { margin: 0; padding: 10px; white-space: pre-wrap; word-break: break-word; overflow-x: auto; }
  .sub { border-top: 1px solid #8884; }
  .sub h3 { font-size: 11px; margin: 0; padding: 5px 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; }
  details.sub > summary { font-size: 11px; padding: 5px 10px; opacity: 0.6; cursor: pointer; }
  details.sub[open] > summary { opacity: 0.8; }
  .ok { color: #2e7d32; }
  .bad { color: #c62828; }
  .muted { opacity: 0.6; }
  .badge { font-size: 11px; border: 1px solid currentColor; border-radius: 999px; padding: 0 7px; }
</style>
</head>
<body>
<header>
  <h1>Prompt lab</h1>
  <div class="meta" id="script-meta">loading catalog...</div>
</header>
<main>
  <div class="col">
    <fieldset>
      <legend>chain</legend>
      <div class="row">
        <div>
          <label>separation target</label>
          <select id="target"></select>
          <input id="target-custom" placeholder="instrument label, e.g. electricguitar" hidden>
        </div>
        <div>
          <label>block kind</label>
          <select id="block-kind">
            <option value="chain">chain (serial: everything hits the fx)</option>
            <option value="send_return">send_return (parallel bus: dry stays clean)</option>
          </select>
        </div>
      </div>
      <pre id="routing" class="routing"></pre>
      <div class="row">
        <div>
          <label>block name</label>
          <input id="block-name" value="target_fx">
        </div>
        <div>
          <label>order profile (validation)</label>
          <select id="order-profile"></select>
        </div>
      </div>
      <div class="row" id="levels-row" hidden>
        <div><label>dry level (untouched signal)</label><input id="dry-level" value="1.0"></div>
        <div><label>send level (into the fx)</label><input id="send-level" value="0.5"></div>
        <div><label>return level (fx back into the mix)</label><input id="return-level" value="0.85"></div>
      </div>
      <div class="inline">
        <input type="checkbox" id="separate-target" checked>
        <label for="separate-target" style="margin:0">wrap in separate -&gt; fx -&gt; mix</label>
      </div>
    </fieldset>

    <fieldset>
      <legend id="effects-legend">effects</legend>
      <div id="steps"></div>
      <div class="row" style="margin:0">
        <div style="flex:1 1 200px">
          <select id="add-operator"></select>
        </div>
        <div style="flex:0 0 auto"><button id="add-step">add effect</button></div>
      </div>
    </fieldset>

    <fieldset>
      <legend>prompt generation</legend>
      <div class="row">
        <div style="flex:1 1 100%">
          <label>abstraction levels (levels they derive from are generated too)</label>
          <div id="levels" class="levels"></div>
        </div>
      </div>
      <div class="row">
        <div>
          <label>attempts per level</label>
          <input id="max-attempts" type="number" min="1" value="3">
        </div>
        <div class="inline" style="flex:1 1 160px">
          <input type="checkbox" id="dry-run">
          <label for="dry-run" style="margin:0">dry run (no API calls)</label>
        </div>
      </div>
      <div class="row" style="margin:0">
        <div style="flex:0 0 auto"><button id="build">build graph only</button></div>
        <div style="flex:0 0 auto"><button id="generate" class="primary">generate prompts</button></div>
        <div style="flex:0 0 auto"><button id="reload">reload catalog</button></div>
      </div>
    </fieldset>

    <fieldset>
      <legend>override (advanced)</legend>
      <label>paste a graph_spec JSON list to bypass the builder</label>
      <textarea id="graph-spec-json" placeholder="[]"></textarea>
    </fieldset>
  </div>

  <div class="col" id="output">
    <p class="muted">Pick effects, then generate.</p>
  </div>
</main>

<script>
const state = { operators: [], steps: [], priors: {}, catalog: null };
const CUSTOM_TARGET = "__custom__";
// Order profiles that suit each block kind, best first. The default profile is a
// serial one, so a correct send bus (delay -> reverb -> wet-path filter) fails
// chain-order validation until the profile matches the routing.
const PROFILE_PREFERENCE = { chain: ["shaping", "serial_fx"], send_return: ["send_space", "send_texture"] };

const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));

async function loadCatalog() {
  const response = await fetch("/api/catalog");
  const catalog = await response.json();
  state.catalog = catalog;
  state.operators = catalog.operators || [];
  state.priors = catalog.param_priors || {};

  // A datalist here read as a fixed field: browsers show no dropdown affordance,
  // so the separable stems were invisible. Custom keeps arbitrary instrument
  // labels available, which is what real plans carry as the separate description.
  el("target").innerHTML = (catalog.targets || [])
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .concat([`<option value="${CUSTOM_TARGET}">custom instrument label…</option>`])
    .join("");
  el("order-profile").innerHTML = ['<option value="">(default)</option>']
    .concat((catalog.order_profiles || []).map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`))
    .join("");
  el("add-operator").innerHTML = state.operators
    .map((operator) => `<option value="${escapeHtml(operator.name)}">${escapeHtml(operator.name)} [${escapeHtml((operator.tags || []).join(","))}]</option>`)
    .join("");

  const script = catalog.prompt_script || {};
  const abstraction = script.abstraction || { levels: [], max_attempts: 3 };
  if (abstraction.max_attempts) el("max-attempts").value = abstraction.max_attempts;
  renderLevels(abstraction);

  const meta = [];
  if (script.error) {
    meta.push(`<span class="bad">prompt script failed to load: ${escapeHtml(script.error.split("\\n")[0])}</span>`);
  } else {
    meta.push(`${escapeHtml(script.path)} &middot; model <b>${escapeHtml(script.model)}</b> &middot; entrypoint <b>${escapeHtml(script.entrypoint)}</b>`);
    meta.push(`${state.operators.length} operators`);
    if (abstraction.available) {
      meta.push(`ladder v${escapeHtml(abstraction.version)} &middot; ${abstraction.levels.length} levels`);
    } else if (abstraction.error) {
      meta.push(`<span class="bad">ladder unavailable: ${escapeHtml(abstraction.error.split("\\n")[0])}</span>`);
    }
  }
  el("script-meta").innerHTML = meta.join(" &middot; ");
  applyProfileDefault();
  renderSteps();
}

function renderLevels(abstraction) {
  const container = el("levels");
  if (!abstraction.levels.length) {
    container.innerHTML = '<span class="muted" style="font-size:11px">no ladder found; the script decides which levels to emit</span>';
    return;
  }
  container.innerHTML = abstraction.levels.map((level) => {
    const from = level.derives_from === "graph" ? "from graph" : `from ${level.derives_from}`;
    return `<label title="${escapeHtml(from)}">
      <input type="checkbox" class="level" value="${escapeHtml(level.id)}" checked>
      ${escapeHtml(level.id)} ${escapeHtml(level.name)}
    </label>`;
  }).join("");
}

function selectedLevels() {
  const boxes = Array.from(document.querySelectorAll("input.level"));
  const checked = boxes.filter((box) => box.checked).map((box) => Number(box.value));
  // Empty means "everything the ladder declares", so do not send a selection
  // when all boxes are ticked.
  return checked.length === boxes.length ? [] : checked;
}

function operatorSpec(name) {
  return state.operators.find((operator) => operator.name === name) || { name, tags: [], params: [] };
}

function paramPrior(operator, param) {
  return ((state.priors[operator] || {})[param]) || null;
}

// Bounds the planner samples within, unioned over every distribution bound to
// this param. Transformed references (tempo_sync, scale) sample in other units,
// so they are excluded here exactly as they are server-side.
function priorBounds(prior) {
  if (!prior) return null;
  const direct = prior.distributions.filter((entry) => entry.derived_via === null);
  const lows = direct.map((entry) => entry.low).filter((value) => value !== null);
  const highs = direct.map((entry) => entry.high).filter((value) => value !== null);
  if (!lows.length || !highs.length) return null;
  return { low: Math.min(...lows), high: Math.max(...highs) };
}

function priorHint(prior) {
  if (!prior) return "";
  const parts = prior.distributions.map((entry) => {
    const label = entry.derived_via ? ` via ${entry.derived_via}` : "";
    if (entry.type === "choice" && entry.values.length) return `{${entry.values.join(", ")}}${label}`;
    if (entry.low !== null && entry.high !== null) return `${entry.low}–${entry.high}${label}`;
    return `${entry.type}${label}`;
  });
  if (!parts.length && prior.literals.length) return `planner uses ${prior.literals.join(", ")}`;
  if (!parts.length) return "";
  return (prior.matched_by_name ? "~ " : "") + parts.join(" | ");
}

function markPriorState(input, operator, param) {
  const bounds = priorBounds(paramPrior(operator, param));
  const value = Number(input.value);
  const numeric = input.value.trim() !== "" && Number.isFinite(value);
  input.classList.toggle("off-prior", Boolean(bounds) && numeric && (value < bounds.low || value > bounds.high));
}

function renderSteps() {
  const container = el("steps");
  renderRouting();
  if (!state.steps.length) {
    container.innerHTML = '<p class="muted" style="margin:0 0 8px">No effects yet.</p>';
    return;
  }
  container.innerHTML = state.steps.map((step, index) => {
    const spec = operatorSpec(step.operator);
    const options = state.operators
      .map((operator) => `<option value="${escapeHtml(operator.name)}"${operator.name === step.operator ? " selected" : ""}>${escapeHtml(operator.name)}</option>`)
      .join("");
    const params = (spec.params || []).map((param) => {
      const hint = priorHint(paramPrior(step.operator, param));
      return `
      <div>
        <label>${escapeHtml(param)}</label>
        <input data-index="${index}" data-param="${escapeHtml(param)}" class="param" value="${escapeHtml(step.params[param] ?? "")}">
        ${hint ? `<div class="hint">${escapeHtml(hint)}</div>` : ""}
      </div>`;
    }).join("");
    return `
      <div class="step">
        <div class="step-head">
          <span class="idx">${index + 1}</span>
          <select data-index="${index}" class="operator">${options}</select>
          <button data-index="${index}" class="up" title="move up">&uarr;</button>
          <button data-index="${index}" class="down" title="move down">&darr;</button>
          <button data-index="${index}" class="remove" title="remove">&times;</button>
        </div>
        <div class="tags">tags: ${escapeHtml((spec.tags || []).join(", ")) || "none"}${(spec.params || []).length ? "" : " &middot; no parameters"}</div>
        <div class="params">${params}</div>
      </div>`;
  }).join("");

  container.querySelectorAll("select.operator").forEach((node) => {
    node.onchange = (event) => {
      const index = Number(event.target.dataset.index);
      state.steps[index] = { operator: event.target.value, params: {} };
      renderSteps();
    };
  });
  container.querySelectorAll("input.param").forEach((node) => {
    const { index, param } = node.dataset;
    markPriorState(node, state.steps[Number(index)].operator, param);
    node.oninput = (event) => {
      state.steps[Number(index)].params[param] = event.target.value;
      markPriorState(event.target, state.steps[Number(index)].operator, param);
    };
  });
  container.querySelectorAll("button.remove").forEach((node) => {
    node.onclick = (event) => {
      state.steps.splice(Number(event.target.dataset.index), 1);
      renderSteps();
    };
  });
  container.querySelectorAll("button.up").forEach((node) => {
    node.onclick = (event) => moveStep(Number(event.target.dataset.index), -1);
  });
  container.querySelectorAll("button.down").forEach((node) => {
    node.onclick = (event) => moveStep(Number(event.target.dataset.index), 1);
  });
}

function moveStep(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= state.steps.length) return;
  const [step] = state.steps.splice(index, 1);
  state.steps.splice(target, 0, step);
  renderSteps();
}

function currentTarget() {
  const selected = el("target").value;
  return selected === CUSTOM_TARGET ? el("target-custom").value.trim() : selected;
}

// Mirrors add_send_return() in graph/edit_graph.py: the dry signal is trimmed
// and passed through untouched, a parallel copy is trimmed into the fx chain,
// and the wet result is trimmed again before the two are summed.
function renderRouting() {
  const isSend = el("block-kind").value === "send_return";
  const source = el("separate-target").checked ? "target_stem" : "audio";
  const output = el("separate-target").checked ? "processed_stem" : "final_audio";
  const chain = state.steps.length
    ? state.steps.map((step) => step.operator.replace(/^apply_/, "").replace(/_effect$/, "")).join(" -> ")
    : "(no effects yet)";
  const lines = [];
  if (isSend) {
    lines.push(`dry path    ${source} x ${el("dry-level").value || 1}  (never touches the fx)`);
    lines.push(`send path   ${source} x ${el("send-level").value || 1} -> ${chain} -> x ${el("return-level").value || 1}`);
    lines.push(`output      dry + wet -> ${output}`);
    if (el("separate-target").checked) lines.push(`then        ${output} + residual -> final_audio`);
  } else {
    lines.push(`${source} -> ${chain} -> ${output}`);
    if (el("separate-target").checked) lines.push(`${output} + residual -> final_audio`);
  }
  el("routing").textContent = lines.join("\\n");
  el("effects-legend").textContent = isSend ? "effects (wet / send path only)" : "effects (serial path)";
}

// A send bus carries its own dry signal, so the fx on it should be fully wet.
function applyProfileDefault() {
  const preferred = PROFILE_PREFERENCE[el("block-kind").value] || [];
  const available = Array.from(el("order-profile").options).map((option) => option.value);
  const choice = preferred.find((name) => available.includes(name));
  if (choice) el("order-profile").value = choice;
}

function requestBody() {
  return {
    target: currentTarget(),
    block_kind: el("block-kind").value,
    block_name: el("block-name").value,
    order_profile: el("order-profile").value,
    separate_target: el("separate-target").checked,
    dry_level: Number(el("dry-level").value || 1),
    send_level: Number(el("send-level").value || 0.5),
    return_level: Number(el("return-level").value || 0.85),
    max_attempts: Number(el("max-attempts").value || 1),
    levels: selectedLevels(),
    dry_run: el("dry-run").checked,
    graph_spec_json: el("graph-spec-json").value,
    steps: state.steps.map((step) => ({ operator: step.operator, params: step.params })),
  };
}

async function post(path) {
  el("output").innerHTML = '<p class="muted">running...</p>';
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody()),
  });
  render(await response.json());
}

function layer(title, tag, body, extraClass) {
  return `
    <div class="layer">
      <h2>${escapeHtml(title)}<span class="tag ${extraClass || ""}">${tag || ""}</span></h2>
      ${body}
    </div>`;
}

function details(summary, body) {
  return `<details class="sub"><summary>${escapeHtml(summary)}</summary>${body}</details>`;
}

function render(payload) {
  const parts = [];
  const validation = payload.validation || {};
  const variants = payload.prompt_variants || [];
  const calls = payload.llm_calls || [];

  if (payload.graph_description !== undefined) {
    const priorWarnings = payload.param_warnings || [];
    let body = `<pre>${escapeHtml(payload.graph_description)}</pre>`;
    if (validation.ok === false) {
      body += `<div class="sub"><h3>planner validation failed</h3><pre class="bad">${escapeHtml(validation.error)}</pre></div>`;
    }
    if (priorWarnings.length) {
      const lines = priorWarnings
        .map((warning) => warning.message)
        .join("\\n");
      body += `<div class="sub"><h3>warnings</h3><pre class="warn">${escapeHtml(lines)}</pre></div>`;
    }
    if (payload.graph_spec) {
      body += details("graph_spec JSON", `<pre>${escapeHtml(JSON.stringify(payload.graph_spec, null, 2))}</pre>`);
    }
    const badges = [];
    if (validation.ok === false) badges.push('<span class="badge bad">invalid</span>');
    if (priorWarnings.length) badges.push(`<span class="badge warn">${priorWarnings.length} off-prior</span>`);
    parts.push(layer("edit graph", badges.join(" "), body));
  }

  // Prefer the script's own per-level report; fall back to the plain list.
  const levels = (payload.prompt_levels || []).length
    ? payload.prompt_levels
    : variants.map((text, index) => ({ level: index, name: "", text, violations: [] }));
  let callCursor = 0;
  levels.forEach((level, index) => {
    // Each level can retry, so consume calls in order rather than by index.
    const attempts = Number(level.attempts) || 1;
    const levelCalls = calls.slice(callCursor, callCursor + attempts);
    callCursor += attempts;
    const notes = [];
    if (index === 0) notes.push("most technical");
    if (index === levels.length - 1 && levels.length > 1) notes.push("most abstract");
    if (attempts > 1) notes.push(`${attempts} attempts`);
    if (level.checks_passed === false) notes.push('<span class="badge warn">checks failed</span>');
    if (payload.dry_run) notes.push("dry run");

    let body = `<pre>${escapeHtml(level.text)}</pre>`;
    if ((level.violations || []).length) {
      const note = payload.dry_run
        ? "\\n(expected in a dry run: the checks run against stub text, not model output)"
        : "";
      body += `<div class="sub"><h3>hard checks failed</h3><pre class="warn">${escapeHtml(level.violations.join("\\n") + note)}</pre></div>`;
    }
    levelCalls.forEach((call, attempt) => {
      const messages = (call.messages || []).map((message) => `
        <h3>${escapeHtml(message.role)}</h3><pre>${escapeHtml(message.content)}</pre>`).join("");
      const label = levelCalls.length > 1 ? ` (attempt ${attempt + 1})` : "";
      body += details(`prompt sent to ${call.model || "the model"}${label}`, messages);
    });
    // The title is escaped by layer(), so use a literal separator here.
    const title = level.name ? `level ${level.level} \\u00b7 ${level.name}` : `level ${level.level}`;
    parts.push(layer(title, notes.join(" &middot; "), body));
  });

  if (payload.error) {
    parts.push(layer("generation failed", "", `<pre class="bad">${escapeHtml(payload.error)}</pre>`));
  }
  if ((payload.missing_record_keys || []).length) {
    parts.push(layer("prompt script read plan fields this UI does not supply", "substituted empty strings",
      `<pre>${escapeHtml(payload.missing_record_keys.join("\\n"))}</pre>`));
  }
  el("output").innerHTML = parts.join("") || '<p class="muted">nothing to show</p>';
}

el("add-step").onclick = () => {
  const operator = el("add-operator").value;
  if (!operator) return;
  state.steps.push({ operator, params: {} });
  renderSteps();
};
el("block-kind").onchange = () => {
  const isSend = el("block-kind").value === "send_return";
  el("levels-row").hidden = !isSend;
  el("block-name").value = isSend ? "target_send" : "target_fx";
  applyProfileDefault();
  renderRouting();
};
el("target").onchange = () => {
  el("target-custom").hidden = el("target").value !== CUSTOM_TARGET;
  if (!el("target-custom").hidden) el("target-custom").focus();
};
["dry-level", "send-level", "return-level", "separate-target"].forEach((id) => {
  el(id).oninput = renderRouting;
  el(id).onchange = renderRouting;
});
el("build").onclick = () => post("/api/build");
el("generate").onclick = () => post("/api/prompts");
el("reload").onclick = loadCatalog;

loadCatalog();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
