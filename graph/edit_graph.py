from __future__ import annotations
import inspect
from collections import OrderedDict, deque
from dataclasses import field, dataclass
from collections.abc import Mapping, Callable, Sequence, MutableMapping
import numpy as np
import torch
from typing import Any, Optional, get_args, get_origin

OutputSpec = Optional[str | Sequence[str]]


class EditGraphError(RuntimeError):
    """Raised when the edit graph is malformed or cannot be executed."""


@dataclass(frozen=True)
class Step:
    """A convenience spec for a single processing step in a chain or send/return bus.

    Attributes
    ----------
    name:
        Unique name for the step *within the chain*.
    fn:
        Callable to execute.
    kwargs:
        Constant keyword arguments to pass to the callable.
    inputs:
        Extra graph inputs to bind by parameter name. The running signal is bound
        separately through ``signal_param``.
    outputs:
        Optional override for the step's output name(s). When omitted, the graph
        stores the result under the fully-qualified node name.
    signal_param:
        Function argument that should receive the running signal. Defaults to
        ``"audio"`` which matches the uploaded effects / pitch modules.
    """

    name: str
    fn: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: OutputSpec = None
    signal_param: str = "audio"


@dataclass
class _Node:
    name: str
    fn: Callable[..., Any]
    inputs: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    auto_bind: bool = True


class EditGraph:
    """A lightweight DAG executor for audio post-production edit chains.

    The graph stores named values in a shared context. Each node consumes values
    from that context, calls a Python function, and writes its output(s) back to
    the context under new names.

    Design goals:
    - compose arbitrary functions from your separation / effects / pitch / mixing modules
    - support branching and re-joining (for example stem -> FX -> mix with residual)
    - provide a helper for aux-style send/return processing
    - automatically adapt between ``torch.Tensor`` and ``np.ndarray`` when type
      annotations make the target representation clear

    Example
    -------
    ```python
    from edit_graph import EditGraph, Step
    from separation import separate
    from effects import apply_delay, apply_distortion
    from mixing import mix_stem_with_residual

    graph = EditGraph()
    graph.add_step(
        "separate_guitar",
        separate,
        inputs={"model": "model", "processor": "processor", "device": "device", "audio": "audio"},
        kwargs={"description": "guitar"},
        outputs=["guitar", "residual"],
    )
    graph.add_step(
        "delay_guitar",
        apply_delay,
        inputs={"audio": "guitar", "sr": "sr"},
        kwargs={"delay_seconds": 0.35, "mix": 0.4},
        outputs="guitar_delayed",
    )
    graph.add_step(
        "distort_guitar",
        apply_distortion,
        inputs={"audio": "guitar_delayed", "sr": "sr"},
        kwargs={"drive_db": 18.0},
        outputs="guitar_fx",
    )
    graph.add_step(
        "mix_back",
        mix_stem_with_residual,
        inputs={"stem": "guitar_fx", "residual": "residual"},
        outputs="mix",
    )

    outputs = graph.run(audio=audio, sr=sr, model=model, processor=processor, device=device)
    final_audio = outputs["mix"]
    ```

    Send / return example
    ---------------------
    ```python
    graph = EditGraph()
    graph.add_send_return(
        "vocal_bus",
        source="vocals",
        chain=[
            Step("delay", apply_delay, kwargs={"delay_seconds": 0.25, "mix": 1.0}),
            Step("reverb", apply_reverb, kwargs={"room_size": 0.7, "wet_level": 0.8, "dry_level": 0.0}),
        ],
        output="vocals_with_fx",
        dry_level=1.0,
        send_level=0.4,
        return_level=0.8,
    )
    ```
    """

    def __init__(self) -> None:
        self._nodes: OrderedDict[str, _Node] = OrderedDict()
        self._output_to_node: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_step(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        inputs: Mapping[str, str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        outputs: OutputSpec = None,
        auto_bind: bool = True,
    ) -> EditGraph:
        """Add a node to the graph.

        Parameters
        ----------
        name:
            Unique node name.
        fn:
            Callable to execute.
        inputs:
            Mapping from function parameter name -> context key.
        kwargs:
            Constant keyword arguments passed directly to the callable.
        outputs:
            Name or names to store the return value(s) under. If omitted, the
            output is stored under ``name``.
        auto_bind:
            When True, any remaining function parameters are automatically bound
            from the execution context using the same parameter name.
        """
        if name in self._nodes:
            raise EditGraphError(f"A node named '{name}' already exists.")

        normalized_outputs = self._normalize_outputs(name, outputs)
        for out_name in normalized_outputs:
            if out_name in self._output_to_node:
                owner = self._output_to_node[out_name]
                raise EditGraphError(
                    f"Output name '{out_name}' is already produced by node '{owner}'."
                )

        node = _Node(
            name=name,
            fn=fn,
            inputs=dict(inputs or {}),
            kwargs=dict(kwargs or {}),
            outputs=normalized_outputs,
            auto_bind=auto_bind,
        )
        self._nodes[name] = node
        for out_name in normalized_outputs:
            self._output_to_node[out_name] = name
        return self

    def add_chain(
        self,
        prefix: str,
        source: str,
        chain: Sequence[Step],
        *,
        output: str | None = None,
        sr_key: str = "sr",
        default_signal_param: str = "audio",
    ) -> str:
        """Add a linear chain of processing nodes.

        Parameters
        ----------
        prefix:
            Prefix used to namespace internal node names.
        source:
            Context key holding the initial signal.
        chain:
            Sequence of :class:`Step` specs.
        output:
            Final context key for the chain output. Defaults to ``f"{prefix}_out"``.
        sr_key:
            Context key for sample rate. Automatically injected into any step that
            has an ``sr`` parameter and does not already bind it explicitly.
        default_signal_param:
            Fallback parameter name for the running signal when the step spec does
            not override it.
        """
        if not chain:
            raise EditGraphError("add_chain() requires at least one step.")

        current = source
        final_output = output or f"{prefix}_out"

        for index, step in enumerate(chain):
            node_name = f"{prefix}.{step.name}"
            default_step_output = (
                final_output if index == len(chain) - 1 else f"{node_name}.out"
            )
            actual_outputs = step.outputs or default_step_output
            signal_param = step.signal_param or default_signal_param

            step_inputs = dict(step.inputs)
            step_inputs.setdefault(signal_param, current)

            if (
                self._function_accepts(step.fn, "sr")
                and "sr" not in step_inputs
                and "sr" not in step.kwargs
            ):
                step_inputs["sr"] = sr_key

            self.add_step(
                node_name,
                step.fn,
                inputs=step_inputs,
                kwargs=step.kwargs,
                outputs=actual_outputs,
            )

            if isinstance(actual_outputs, str):
                current = actual_outputs
            else:
                if len(actual_outputs) != 1:
                    raise EditGraphError(
                        f"Chain step '{node_name}' must expose exactly one running output; "
                        f"got {len(actual_outputs)} outputs."
                    )
                current = tuple(actual_outputs)[0]

        return final_output

    def add_send_return(
        self,
        name: str,
        source: str,
        chain: Sequence[Step],
        *,
        output: str | None = None,
        dry_level: float = 1.0,
        send_level: float = 1.0,
        return_level: float = 1.0,
        sr_key: str = "sr",
    ) -> str:
        """Add an aux-style send/return branch and mix the wet signal back in.

        This models the common studio pattern:

            dry signal --> (optional dry trim) ----\
                                                  mix --> output
            dry signal --> send trim --> FX chain --> return trim --/

        The graph remains acyclic; this is a *send/return bus*, not a true feedback
        cycle.
        """
        if not chain:
            raise EditGraphError("add_send_return() requires at least one step.")

        dry_signal = source
        if dry_level != 1.0:
            dry_signal = f"{name}.dry"
            self.add_step(
                dry_signal,
                self.scale_signal,
                inputs={"audio": source},
                kwargs={"gain": dry_level},
                outputs=dry_signal,
            )

        wet_source = source
        if send_level != 1.0:
            wet_source = f"{name}.send"
            self.add_step(
                wet_source,
                self.scale_signal,
                inputs={"audio": source},
                kwargs={"gain": send_level},
                outputs=wet_source,
            )

        wet_signal = self.add_chain(
            f"{name}.fx",
            wet_source,
            chain,
            output=f"{name}.wet",
            sr_key=sr_key,
        )

        if return_level != 1.0:
            wet_return = f"{name}.return"
            self.add_step(
                wet_return,
                self.scale_signal,
                inputs={"audio": wet_signal},
                kwargs={"gain": return_level},
                outputs=wet_return,
            )
            wet_signal = wet_return

        mixed_output = output or f"{name}.out"
        self.add_step(
            f"{name}.mix",
            self.mix_signals,
            inputs={"dry": dry_signal, "wet": wet_signal},
            outputs=mixed_output,
        )
        return mixed_output

    def run(self, **initial_context: Any) -> dict[str, Any]:
        """Execute the graph and return the final context dictionary."""
        context: dict[str, Any] = dict(initial_context)
        order = self.execution_order()
        for node_name in order:
            node = self._nodes[node_name]
            result = self._execute_node(node, context)
            self._store_outputs(node, result, context)
        return context

    def execution_order(self) -> list[str]:
        """Return a topologically sorted list of node names."""
        dependencies: dict[str, set[str]] = {
            name: self._dependencies(node) for name, node in self._nodes.items()
        }
        dependents: dict[str, set[str]] = {name: set() for name in self._nodes}
        for node_name, deps in dependencies.items():
            for dep in deps:
                dependents[dep].add(node_name)

        ready = deque(name for name, deps in dependencies.items() if not deps)
        ordered: list[str] = []

        while ready:
            node_name = ready.popleft()
            ordered.append(node_name)
            for dependent in dependents[node_name]:
                dependencies[dependent].remove(node_name)
                if not dependencies[dependent]:
                    ready.append(dependent)

        if len(ordered) != len(self._nodes):
            remaining = [name for name, deps in dependencies.items() if deps]
            raise EditGraphError(
                "The graph contains a cycle or unresolved dependency among nodes: "
                + ", ".join(remaining)
            )
        return ordered

    def describe(self) -> str:
        """Return a human-readable summary of the graph."""
        lines: list[str] = []
        for node_name in self.execution_order():
            node = self._nodes[node_name]
            outputs = ", ".join(node.outputs)
            inputs = ", ".join(
                f"{param}<-{source}" for param, source in node.inputs.items()
            )
            lines.append(
                f"{node_name}: {inputs} => {getattr(node.fn, '__name__', repr(node.fn))} -> {outputs}"
            )
        return "\n".join(lines)

    def __call__(self, **initial_context: Any) -> dict[str, Any]:
        return self.run(**initial_context)

    # ------------------------------------------------------------------
    # Static helpers that are also useful as graph nodes
    # ------------------------------------------------------------------
    @staticmethod
    def scale_signal(audio: Any, gain: float) -> Any:
        """Scale an audio tensor / array by a linear gain."""
        if torch is not None and isinstance(audio, torch.Tensor):
            return audio * gain
        return np.asarray(audio) * gain

    @staticmethod
    def mix_signals(dry: Any, wet: Any) -> Any:
        """Mix two signals while preserving a sensible container type.

        If either side is a torch tensor, the other side is promoted onto the
        same device and dtype before summation. This avoids common send/return
        failures where the dry path stays on CUDA while the wet path temporarily
        passes through numpy-based effects on CPU.
        """
        if torch is not None and (
            isinstance(dry, torch.Tensor) or isinstance(wet, torch.Tensor)
        ):
            reference = dry if isinstance(dry, torch.Tensor) else wet
            assert isinstance(reference, torch.Tensor)
            dry_t = EditGraph._to_torch(dry, like=reference)
            wet_t = EditGraph._to_torch(wet, like=reference)
            return dry_t + wet_t
        return EditGraph._to_numpy(dry) + EditGraph._to_numpy(wet)

    # ------------------------------------------------------------------
    # Internal execution helpers
    # ------------------------------------------------------------------
    def _execute_node(self, node: _Node, context: MutableMapping[str, Any]) -> Any:
        signature = inspect.signature(node.fn)
        call_kwargs: dict[str, Any] = {}

        for param_name, param in signature.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param_name in node.inputs:
                source_key = node.inputs[param_name]
                if source_key not in context:
                    raise EditGraphError(
                        f"Node '{node.name}' expects context key '{source_key}' for parameter '{param_name}', "
                        f"but it is not available."
                    )
                value = context[source_key]
            elif param_name in node.kwargs:
                value = node.kwargs[param_name]
            elif node.auto_bind and param_name in context:
                value = context[param_name]
            elif param.default is not inspect._empty:
                continue
            else:
                raise EditGraphError(
                    f"Node '{node.name}' is missing a value for required parameter '{param_name}'."
                )

            call_kwargs[param_name] = self._adapt_value_for_parameter(
                value, param.annotation
            )

        return node.fn(**call_kwargs)

    def _store_outputs(
        self, node: _Node, result: Any, context: MutableMapping[str, Any]
    ) -> None:
        if not node.outputs:
            return

        if len(node.outputs) == 1:
            context[node.outputs[0]] = result
            return

        if not isinstance(result, (tuple, list)):
            raise EditGraphError(
                f"Node '{node.name}' declares {len(node.outputs)} outputs but returned a non-sequence value."
            )
        if len(result) != len(node.outputs):
            raise EditGraphError(
                f"Node '{node.name}' declares {len(node.outputs)} outputs but returned {len(result)} values."
            )
        for key, value in zip(node.outputs, result):
            context[key] = value

    def _dependencies(self, node: _Node) -> set[str]:
        deps: set[str] = set()

        for source_key in node.inputs.values():
            producer = self._output_to_node.get(source_key)
            if producer is not None and producer != node.name:
                deps.add(producer)

        if node.auto_bind:
            signature = inspect.signature(node.fn)
            for param_name, param in signature.parameters.items():
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if param_name in node.inputs or param_name in node.kwargs:
                    continue
                producer = self._output_to_node.get(param_name)
                if producer is not None and producer != node.name:
                    deps.add(producer)
        return deps

    @staticmethod
    def _normalize_outputs(name: str, outputs: OutputSpec) -> tuple[str, ...]:
        if outputs is None:
            return (name,)
        if isinstance(outputs, str):
            return (outputs,)
        normalized = tuple(outputs)
        if not normalized:
            raise EditGraphError(f"Node '{name}' must expose at least one output name.")
        return normalized

    @staticmethod
    def _function_accepts(fn: Callable[..., Any], param_name: str) -> bool:
        try:
            return param_name in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _adapt_value_for_parameter(value: Any, annotation: Any) -> Any:
        if annotation is inspect._empty:
            return value

        if EditGraph._annotation_contains(annotation, np.ndarray):
            return EditGraph._to_numpy(value)

        if torch is not None and EditGraph._annotation_contains(
            annotation, torch.Tensor
        ):
            return EditGraph._to_torch(value)

        return value

    @staticmethod
    def _annotation_contains(annotation: Any, target: Any) -> bool:
        if annotation is target:
            return True

        if isinstance(annotation, str):
            target_name = getattr(target, "__name__", str(target))
            return target_name in annotation

        origin = get_origin(annotation)
        if origin is None:
            return annotation is target

        return any(
            EditGraph._annotation_contains(arg, target) for arg in get_args(annotation)
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            array = value
        elif torch is not None and isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)

        # Common interoperability fix: remove singleton batch dimensions so that
        # separation outputs like [1, C, T] become [C, T] before effects / mixing.
        while array.ndim > 2 and array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        return array

    @staticmethod
    def _to_torch(value: Any, like: Any = None):
        if torch is None:
            raise EditGraphError(
                "torch is not installed, so values cannot be converted to torch.Tensor."
            )

        reference = like if isinstance(like, torch.Tensor) else None

        if isinstance(value, torch.Tensor):
            tensor = value
            if reference is not None and (
                tensor.device != reference.device or tensor.dtype != reference.dtype
            ):
                tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        else:
            array = np.asarray(value)
            kwargs = {}
            if reference is not None:
                kwargs["device"] = reference.device
                kwargs["dtype"] = reference.dtype
            tensor = torch.as_tensor(array, **kwargs)

        # Common interoperability fix: mono numpy arrays often appear as [T]. The
        # pitch module expects channel-first audio, so promote this to [1, T].
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor
