import importlib
import sys
import time
import functools
from pathlib import Path
from collections.abc import Mapping, Sequence
from graph.edit_graph import Step
from ground_truth.operators import OperatorRegistry, load_operator_registry
from ground_truth.symbolic_graph import SymbolicEditGraph
from typing import Any

BLOCK_DEFAULT_OPERATORS = {
    "separate": "separate_audio",
    "mix": "mix_stems"
}

# Map block kinds to methods that compile each block type.
BLOCK_COMPILE_HANDLERS = {
    "separate": "_compile_separate_block",
    "mix": "_compile_mix_block",
    "chain": "_compile_chain_block",
    "send_return": "_compile_send_return_block",
    "step": "_compile_step_block"
}


def _runtime_log(message: str) -> None:
    print("[runtime] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), file=sys.stderr, flush=True)


class RuntimePlanCompiler:
    def __init__(self, operator_registry: OperatorRegistry):
        self.operator_registry = operator_registry
        self._callable_cache: dict[str, Any] = {}

    @classmethod
    def from_directory(cls, config_dir: Path | str) -> "RuntimePlanCompiler":
        return cls(load_operator_registry(config_dir))

    def compile_plan(self, plan: Any) -> SymbolicEditGraph:
        return self.compile_graph_spec(plan.graph_spec)

    def compile_graph_spec(self, graph_spec: Sequence[Mapping[str, Any]]) -> SymbolicEditGraph:
        graph = SymbolicEditGraph(graph_spec)
        for block in graph_spec:
            handler_name = BLOCK_COMPILE_HANDLERS.get(block["kind"])
            if handler_name is None:
                raise ValueError("Unsupported graph block kind '%s'." % block["kind"])
            getattr(self, handler_name)(graph, block)
        return graph

    def _compile_separate_block(
        self,
        graph: SymbolicEditGraph,
        block: Mapping[str, Any]
    ) -> None:
        operator_name = block.get("operator", BLOCK_DEFAULT_OPERATORS["separate"])
        fn = self._resolve_callable(operator_name)
        self.operator_registry.resolve(operator_name).validate_params(
            {"description": block["description"]}
        )
        graph.add_step(
            block["name"],
            fn,
            inputs={"audio": block.get("source", "audio")},
            kwargs={"description": block["description"]},
            outputs=list(block.get("outputs", ["stem", "residual"]))
        )

    def _compile_mix_block(
        self,
        graph: SymbolicEditGraph,
        block: Mapping[str, Any]
    ) -> None:
        operator_name = block.get("operator", BLOCK_DEFAULT_OPERATORS["mix"])
        fn = self._resolve_callable(operator_name)
        graph.add_step(
            block["name"],
            fn,
            inputs={"stem": block["stem"], "residual": block["residual"]},
            outputs=block["output"]
        )

    def _compile_chain_block(
        self,
        graph: SymbolicEditGraph,
        block: Mapping[str, Any]
    ) -> None:
        graph.add_chain(
            block["prefix"],
            source=block["source"],
            chain=self._compile_chain_steps(block.get("steps", [])),
            output=block["output"]
        )

    def _compile_send_return_block(
        self,
        graph: SymbolicEditGraph,
        block: Mapping[str, Any]
    ) -> None:
        graph.add_send_return(
            block["name"],
            source=block["source"],
            chain=self._compile_chain_steps(block.get("steps", [])),
            output=block["output"],
            dry_level=float(block.get("dry_level", 1.0)),
            send_level=float(block.get("send_level", 1.0)),
            return_level=float(block.get("return_level", 1.0))
        )

    def _compile_step_block(
        self,
        graph: SymbolicEditGraph,
        block: Mapping[str, Any]
    ) -> None:
        operator_name = block["operator"]
        fn = self._resolve_callable(operator_name)
        params = dict(block.get("params", {}))
        self.operator_registry.resolve(operator_name).validate_params(params)
        graph.add_step(
            block["name"],
            fn,
            inputs=dict(block.get("inputs", {})),
            kwargs=params,
            outputs=block.get("outputs")
        )

    def _compile_chain_steps(self, steps: Sequence[Mapping[str, Any]]) -> list[Step]:
        compiled_steps: list[Step] = []
        for step in steps:
            operator_name = step["operator"]
            operator = self.operator_registry.resolve(operator_name)
            params = dict(step.get("params", {}))
            operator.validate_params(params)
            compiled_steps.append(
                Step(
                    name=step["name"],
                    fn=self._resolve_callable(operator_name),
                    kwargs=params,
                    inputs=dict(step.get("inputs", {})),
                    outputs=step.get("outputs"),
                    signal_param=step.get("signal_param", operator.signal_param)
                )
            )
        return compiled_steps

    def _resolve_callable(self, operator_name: str) -> Any:
        if operator_name in self._callable_cache:
            return self._callable_cache[operator_name]

        runtime_spec = self.operator_registry.runtime_spec(operator_name)
        module = importlib.import_module(runtime_spec["module"])
        fn = getattr(module, runtime_spec["callable"])
        wrapped_fn = self._logging_wrapper(operator_name, fn)
        self._callable_cache[operator_name] = wrapped_fn
        return wrapped_fn

    def _logging_wrapper(self, operator_name: str, fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            _runtime_log(
                "operator start | operator=%s | callable=%s.%s | kwargs=%s"
                % (
                    operator_name,
                    getattr(fn, "__module__", "unknown"),
                    getattr(fn, "__name__", "unknown"),
                    sorted(kwargs.keys()),
                )
            )
            result = fn(*args, **kwargs)
            _runtime_log(
                "operator done | operator=%s | elapsed=%.1fs"
                % (operator_name, time.perf_counter() - started_at)
            )
            return result

        return wrapped
