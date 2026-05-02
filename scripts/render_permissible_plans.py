#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import asyncio
import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping, Iterator, Sequence
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def format_seconds(value: float) -> str:
    return f"{value:.1f}s"


def log_event(message: str) -> None:
    print(f"[render] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def iter_plan_rows(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def iter_plan_batches(
    path: Path,
    limit: int | None,
    *,
    group_by_audio: bool,
) -> list[list[dict[str, Any]]]:
    if not group_by_audio:
        return [[row] for row in iter_plan_rows(path, limit)]

    grouped_rows: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(iter_plan_rows(path, limit), start=1):
        audio_path = row.get("audio_path")
        group_key = str(audio_path) if audio_path else f"__missing_audio__:{index}"
        grouped_rows.setdefault(group_key, []).append(row)
    return list(grouped_rows.values())


def shuffle_plan_batches(
    plan_batches: list[list[dict[str, Any]]],
    seed: int,
) -> list[list[dict[str, Any]]]:
    shuffled = list(plan_batches)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def baseline_description(row: Mapping[str, Any]) -> str | None:
    for block in row.get("graph_spec", []):
        if block.get("kind") == "separate":
            return str(block.get("description"))
    return None


def slugify_filename(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned or "full_mix"


def baseline_output_path(row: Mapping[str, Any], clip_output_dir: Path) -> Path:
    description = baseline_description(row)
    if description is None:
        return clip_output_dir / "original.wav"
    return clip_output_dir / f"original__{slugify_filename(description)}.wav"


def source_copy_path(audio_path: Path, clip_output_dir: Path) -> Path:
    return clip_output_dir / f"source{audio_path.suffix}"


def render_manifest_metadata(
    row: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    baseline_path: Path,
    source_path_copy: Path,
) -> dict[str, Any]:
    return {
        "plan_id": row.get("plan_id"),
        "clip_id": row.get("clip_id"),
        "recipe_id": row.get("recipe_id"),
        "recipe_tags": row.get("recipe_tags", []),
        "target_stem": row.get("target_stem"),
        "target_family": row.get("target_family"),
        "separation_target": row.get("separation_target"),
        "graph_description": row.get("graph_description"),
        "graph_spec": row.get("graph_spec", []),
        "bindings": row.get("bindings", {}),
        "applied_policies": row.get("applied_policies", []),
        "weight": row.get("weight"),
        "audio_path": str(source_path),
        "output_path": str(output_path),
        "baseline_path": str(baseline_path),
        "source_copy_path": str(source_path_copy),
    }


def graph_has_operator(
    graph_spec: Sequence[Mapping[str, Any]],
    operator_name: str,
) -> bool:
    for block in graph_spec:
        if block.get("operator") == operator_name:
            return True
        for step in block.get("steps", []):
            if step.get("operator") == operator_name:
                return True
    return False


def parse_call_tool_result(result: Any) -> dict[str, Any]:
    is_error = bool(getattr(result, "isError", False))
    content = getattr(result, "content", result)
    if isinstance(content, dict):
        payload = dict(content)
        if is_error and "status" not in payload:
            payload["status"] = "error"
        return payload
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                payload = dict(item)
                if is_error and "status" not in payload:
                    payload["status"] = "error"
                return payload
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        for text in texts:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = dict(parsed)
                if is_error and "status" not in payload:
                    payload["status"] = "error"
                return payload
        if texts:
            message = texts[0]
            if is_error or message.startswith("Error executing tool"):
                return {"status": "error", "error": message}
            return {"raw_result": message}
    if isinstance(result, dict):
        payload = dict(result)
        if is_error and "status" not in payload:
            payload["status"] = "error"
        return payload
    raise ValueError(f"Unsupported MCP tool result: {result!r}")


@dataclass
class BatchRenderResult:
    batch_index: int
    batch_count: int
    clip_id: str
    audio_path: str | None
    rows: list[dict[str, Any]]
    rendered: int
    skipped: int
    failed: int
    stop_requested: bool = False


@dataclass
class WorkerComplete:
    worker_id: int


async def render_plan_row(
    *,
    session: Any,
    row: Mapping[str, Any],
    output_root: Path,
    config_dir: Path,
    separation_cache_dir: Path | None,
    max_audio_seconds: float | None,
    overwrite: bool,
) -> tuple[dict[str, Any], str]:
    plan_id = str(row.get("plan_id"))
    clip_id = str(row.get("clip_id", "unknown_clip"))
    audio_path = row.get("audio_path")
    if not audio_path:
        raise ValueError(
            "Plan '%s' is missing audio_path. Rebuild the analysis manifest with --audio-root and then regenerate permissible plans."
            % plan_id
        )
    source_path = Path(str(audio_path)).expanduser().resolve()
    clip_output_dir = output_root / clip_id
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = clip_output_dir / ("%s.wav" % plan_id)
    baseline_path = baseline_output_path(row, clip_output_dir)
    source_path_copy = source_copy_path(source_path, clip_output_dir)
    metadata = render_manifest_metadata(
        row,
        source_path,
        output_path,
        baseline_path,
        source_path_copy,
    )
    result = await session.call_tool(
        "render_ground_truth_plan",
        arguments={
            "audio_file": str(source_path),
            "graph_spec": list(row["graph_spec"]),
            "output_path": str(output_path),
            "config_dir": str(config_dir),
            "separation_cache_dir": str(separation_cache_dir) if separation_cache_dir is not None else None,
            "max_audio_seconds": max_audio_seconds,
            "overwrite": bool(overwrite),
            "baseline_output_path": str(baseline_path),
            "source_copy_path": str(source_path_copy),
        },
    )
    payload = parse_call_tool_result(result)
    status = str(payload.get("status", ""))
    if status not in {"rendered", "skipped_existing", "error"} and not payload.get("error"):
        status = "error"
        payload.setdefault("status", "error")
        payload.setdefault("error", "Missing explicit tool status in MCP response")
    result_row = dict(metadata)
    result_row.update(payload)
    result_row.setdefault("plan_id", plan_id)
    result_row.setdefault("clip_id", clip_id)
    result_row.setdefault("audio_path", str(source_path))
    result_row.setdefault("max_audio_seconds", max_audio_seconds)
    return result_row, status


def error_manifest_row(
    *,
    row: Mapping[str, Any],
    output_root: Path,
    error: Exception,
    max_audio_seconds: float | None,
) -> dict[str, Any]:
    plan_id = str(row.get("plan_id"))
    clip_id = str(row.get("clip_id", "unknown_clip"))
    audio_path = row.get("audio_path")
    result_row = {
        "plan_id": plan_id,
        "clip_id": clip_id,
        "audio_path": str(audio_path) if audio_path else None,
        "status": "error",
        "error": str(error),
        "max_audio_seconds": max_audio_seconds,
    }
    if audio_path:
        source_path = Path(str(audio_path)).expanduser().resolve()
        clip_output_dir = output_root / clip_id
        output_path = clip_output_dir / ("%s.wav" % plan_id)
        baseline_path = baseline_output_path(row, clip_output_dir)
        source_path_copy = source_copy_path(source_path, clip_output_dir)
        result_row.update(
            render_manifest_metadata(
                row,
                source_path,
                output_path,
                baseline_path,
                source_path_copy,
            )
        )
        result_row["status"] = "error"
        result_row["error"] = str(error)
        result_row["max_audio_seconds"] = max_audio_seconds
    return result_row


async def render_clip_batch(
    *,
    worker_id: int,
    batch_index: int,
    batch_count: int,
    batch: list[Mapping[str, Any]],
    session: Any,
    output_root: Path,
    config_dir: Path,
    separation_cache_dir: Path | None,
    max_audio_seconds: float | None,
    overwrite: bool,
    skip_harmony: bool,
    fail_fast: bool,
) -> BatchRenderResult:
    first_row = batch[0]
    clip_id = str(first_row.get("clip_id", "unknown_clip"))
    audio_path = first_row.get("audio_path")
    log_event(
        "worker=%d starting clip batch %d/%d | clip_id=%s | plans=%d | audio_path=%s"
        % (
            worker_id,
            batch_index,
            batch_count,
            clip_id,
            len(batch),
            audio_path,
        )
    )
    rendered = 0
    skipped = 0
    failed = 0
    result_rows: list[dict[str, Any]] = []
    stop_requested = False

    for row in batch:
        plan_id = str(row.get("plan_id"))
        clip_id = str(row.get("clip_id", "unknown_clip"))
        audio_path = row.get("audio_path")
        try:
            if skip_harmony and graph_has_operator(list(row["graph_spec"]), "apply_harmony_effect"):
                result_row = error_manifest_row(
                    row=row,
                    output_root=output_root,
                    error=RuntimeError("Skipped by --skip-harmony"),
                    max_audio_seconds=max_audio_seconds,
                )
                result_row["status"] = "skipped_harmony"
                skipped += 1
                result_rows.append(result_row)
                continue
            result_row, status = await render_plan_row(
                session=session,
                row=row,
                output_root=output_root,
                config_dir=config_dir,
                separation_cache_dir=separation_cache_dir,
                max_audio_seconds=max_audio_seconds,
                overwrite=overwrite,
            )
            if status == "rendered":
                rendered += 1
            elif status == "skipped_existing":
                skipped += 1
            else:
                failed += 1
        except Exception as error:
            failed += 1
            print(
                "ERROR worker=%d plan_id=%s clip_id=%s audio_path=%s :: %s"
                % (worker_id, plan_id, clip_id, audio_path, error),
                file=sys.stderr,
                flush=True,
            )
            result_row = error_manifest_row(
                row=row,
                output_root=output_root,
                error=error,
                max_audio_seconds=max_audio_seconds,
            )
            if fail_fast:
                stop_requested = True
                result_rows.append(result_row)
                break
        result_rows.append(result_row)

    log_event(
        "worker=%d finished clip batch %d/%d | clip_id=%s | rendered=%d | skipped=%d | failed=%d"
        % (
            worker_id,
            batch_index,
            batch_count,
            first_row.get("clip_id", "unknown_clip"),
            rendered,
            skipped,
            failed,
        )
    )
    return BatchRenderResult(
        batch_index=batch_index,
        batch_count=batch_count,
        clip_id=str(first_row.get("clip_id", "unknown_clip")),
        audio_path=str(audio_path) if audio_path is not None else None,
        rows=result_rows,
        rendered=rendered,
        skipped=skipped,
        failed=failed,
        stop_requested=stop_requested,
    )


async def render_worker(
    *,
    worker_id: int,
    batch_count: int,
    plan_batches: list[list[dict[str, Any]]],
    batch_state: dict[str, int],
    batch_lock: asyncio.Lock,
    stop_event: asyncio.Event,
    result_queue: "asyncio.Queue[BatchRenderResult | WorkerComplete]",
    server_params: Any,
    output_root: Path,
    config_dir: Path,
    separation_cache_dir: Path | None,
    max_audio_seconds: float | None,
    overwrite: bool,
    skip_harmony: bool,
    fail_fast: bool,
    stdio_client: Any,
    ClientSession: Any,
) -> None:
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                while True:
                    async with batch_lock:
                        next_batch_index = batch_state["next_batch_index"]
                        if stop_event.is_set() or next_batch_index >= batch_count:
                            break
                        batch_state["next_batch_index"] += 1
                    batch_index = next_batch_index + 1
                    batch = plan_batches[next_batch_index]
                    batch_result = await render_clip_batch(
                        worker_id=worker_id,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        batch=batch,
                        session=session,
                        output_root=output_root,
                        config_dir=config_dir,
                        separation_cache_dir=separation_cache_dir,
                        max_audio_seconds=max_audio_seconds,
                        overwrite=overwrite,
                        skip_harmony=skip_harmony,
                        fail_fast=fail_fast,
                    )
                    await result_queue.put(batch_result)
                    if batch_result.stop_requested:
                        stop_event.set()
                        break
    finally:
        await result_queue.put(WorkerComplete(worker_id=worker_id))


async def render_batches(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    try:
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The render client requires the `mcp` package. Run this script through the project environment or set PYTHON_BIN to the environment that has MCP installed."
        ) from error

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    separation_cache_dir = None
    if not args.disable_separation_cache:
        separation_cache_dir = (
            args.separation_cache_dir.expanduser().resolve()
            if args.separation_cache_dir is not None
            else output_root / "_demucs_cache"
        )
        separation_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else output_root / "render_manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest_path = manifest_path.with_name(
        "%s.tmp.%d" % (
            manifest_path.name,
            os.getpid(),
        )
    )

    plan_batches = iter_plan_batches(
        args.plans_path,
        args.limit,
        group_by_audio=not args.disable_audio_batching,
    )
    if not args.disable_shuffle:
        plan_batches = shuffle_plan_batches(plan_batches, args.seed)
    total_plans = sum(len(batch) for batch in plan_batches)
    if args.workers > 1 and args.disable_audio_batching:
        raise ValueError(
            "--workers > 1 requires clip-level audio batching so concurrent workers do not race on shared clip artifacts. Remove --disable-audio-batching or use --workers 1."
        )
    worker_count = 0 if not plan_batches else min(args.workers, len(plan_batches))

    server_params = StdioServerParameters(
        command=args.python_bin,
        args=[str(args.server_script)],
        env=os.environ.copy(),
    )

    log_event(
        "starting render | plans=%d | clip_batches=%d | workers=%d | output_root=%s | manifest=%s | server=%s | separation_cache=%s | max_audio_seconds=%s | shuffle=%s | seed=%d"
        % (
            total_plans,
            len(plan_batches),
            max(worker_count, 1),
            output_root,
            manifest_path,
            args.server_script,
            separation_cache_dir,
            args.max_audio_seconds,
            not args.disable_shuffle,
            args.seed,
        )
    )

    started_at = time.perf_counter()
    successes = 0
    skips = 0
    failures = 0
    processed = 0
    next_progress_report = 25
    fail_fast_triggered = False

    with temp_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        if worker_count == 0:
            pass
        else:
            result_queue: asyncio.Queue[BatchRenderResult | WorkerComplete] = asyncio.Queue()
            batch_lock = asyncio.Lock()
            stop_event = asyncio.Event()
            batch_state = {"next_batch_index": 0}
            worker_tasks = [
                asyncio.create_task(
                    render_worker(
                        worker_id=worker_id,
                        batch_count=len(plan_batches),
                        plan_batches=plan_batches,
                        batch_state=batch_state,
                        batch_lock=batch_lock,
                        stop_event=stop_event,
                        result_queue=result_queue,
                        server_params=server_params,
                        output_root=output_root,
                        config_dir=args.config_dir,
                        separation_cache_dir=separation_cache_dir,
                        max_audio_seconds=args.max_audio_seconds,
                        overwrite=bool(args.overwrite),
                        skip_harmony=bool(args.skip_harmony),
                        fail_fast=bool(args.fail_fast),
                        stdio_client=stdio_client,
                        ClientSession=ClientSession,
                    )
                )
                for worker_id in range(1, worker_count + 1)
            ]

            done_workers = 0
            expected_batch_index = 1
            pending_batches: dict[int, BatchRenderResult] = {}
            while done_workers < worker_count:
                queue_item = await result_queue.get()
                if isinstance(queue_item, WorkerComplete):
                    done_workers += 1
                    continue
                pending_batches[queue_item.batch_index] = queue_item
                while expected_batch_index in pending_batches:
                    batch_result = pending_batches.pop(expected_batch_index)
                    for result_row in batch_result.rows:
                        manifest_handle.write(json.dumps(result_row, sort_keys=True))
                        manifest_handle.write("\n")
                    successes += batch_result.rendered
                    skips += batch_result.skipped
                    failures += batch_result.failed
                    processed += len(batch_result.rows)
                    while processed >= next_progress_report:
                        elapsed = time.perf_counter() - started_at
                        print(
                            "Processed %d plans | rendered=%d | skipped=%d | failed=%d | elapsed=%.1fs"
                            % (processed, successes, skips, failures, elapsed),
                            flush=True,
                        )
                        next_progress_report += 25
                    log_event(
                        "finished clip batch %d/%d | clip_id=%s | cumulative_rendered=%d | cumulative_skipped=%d | cumulative_failed=%d"
                        % (
                            batch_result.batch_index,
                            batch_result.batch_count,
                            batch_result.clip_id,
                            successes,
                            skips,
                            failures,
                        )
                    )
                    if batch_result.stop_requested:
                        fail_fast_triggered = True
                    expected_batch_index += 1
            await asyncio.gather(*worker_tasks)

    if fail_fast_triggered:
        raise RuntimeError(
            "Stopping after first rendering error because --fail-fast is enabled."
        )

    os.replace(temp_manifest_path, manifest_path)
    elapsed = time.perf_counter() - started_at
    log_event(
        "finished rendering | rendered=%d | skipped=%d | failed=%d | manifest=%s | elapsed=%s"
        % (successes, skips, failures, manifest_path, format_seconds(elapsed))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=Path("derived/ground_truth/subsampled_plans.jsonl"),
        help="Input JSONL of permissible plans. Default: %(default)s",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/lab/postmaster/generated-audio").expanduser(),
        help="Root directory for rendered audio. Default: %(default)s",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSONL manifest for render results. Default: <output-root>/render_manifest.jsonl",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/ground_truth"),
        help="Ground-truth config directory. Default: %(default)s",
    )
    parser.add_argument(
        "--server-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "backend" / "server.py",
        help="Backend server entrypoint used for queue-backed rendering. Default: %(default)s",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used to launch the backend server. Default: current interpreter",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional plan limit for smoke tests. Default: no limit")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent clip-batch workers. Each worker launches its own backend server process. Default: %(default)s",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for randomized render order. Default: %(default)s")
    parser.add_argument("--disable-shuffle", action="store_true", help="Preserve input file order instead of seeded randomized render order")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered outputs. Default: skip already-rendered plans")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first rendering error")
    parser.add_argument("--skip-harmony", action="store_true", help="Skip plans containing apply_harmony_effect during rendering")
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        help="Optional source duration cap applied before Demucs and rendering. Default: full source",
    )
    parser.add_argument(
        "--separation-cache-dir",
        type=Path,
        default=None,
        help="Persistent Demucs source-cache directory. Default: <output-root>/_demucs_cache",
    )
    parser.add_argument(
        "--disable-separation-cache",
        action="store_true",
        help="Disable persistent separation caching. Default: cache under <output-root>/_demucs_cache",
    )
    parser.add_argument(
        "--disable-audio-batching",
        action="store_true",
        help="Process plans strictly in file order instead of grouping by source audio clip",
    )
    args = parser.parse_args()
    asyncio.run(render_batches(args))


if __name__ == "__main__":
    main()
