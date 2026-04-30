import copy
import json
from collections.abc import Mapping, Sequence
from graph.edit_graph import EditGraph
from typing import Any

# Map graph block kinds to the method used for textual output.
BLOCK_DESCRIBE_HANDLERS = {
    "separate": "_describe_separate_block",
    "mix": "_describe_mix_block",
    "chain": "_describe_chain_block",
    "send_return": "_describe_send_return_block",
    "step": "_describe_step_block",
}


class SymbolicEditGraph(EditGraph):
    """EditGraph-compatible wrapper that retains the symbolic graph spec.

    The ground-truth pipeline still serializes graph blocks as dictionaries, but
    compiled plans should integrate with the repo's canonical EditGraph type.
    This class is the bridge: it subclasses EditGraph without modifying the base
    implementation and carries the original block spec for description and
    round-tripping.
    """

    def __init__(self, blocks: Sequence[Mapping[str, Any]]):
        super().__init__()
        self.blocks: list[dict[str, Any]] = [copy.deepcopy(dict(block)) for block in blocks]

    @property
    def graph_spec(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(block) for block in self.blocks]

    def describe(self) -> str:
        return self.describe_blocks(self.blocks)

    @classmethod
    def describe_blocks(cls, blocks: Sequence[Mapping[str, Any]]) -> str:
        helper = cls.__new__(cls)
        return "\n".join(helper._describe_block(block) for block in blocks)

    def _describe_block(self, block: Mapping[str, Any]) -> str:
        kind = block["kind"]
        handler_name = BLOCK_DESCRIBE_HANDLERS.get(kind)
        if handler_name is None:
            raise ValueError("Unsupported graph block kind '%s'." % kind)
        return getattr(self, handler_name)(block)

    def _describe_separate_block(self, block: Mapping[str, Any]) -> str:
        return "%s: %s(%s, description=%s) -> %s" % (
            block["name"],
            block.get("operator", "separate_audio"),
            block.get("source", "audio"),
            json.dumps(block["description"]),
            ", ".join(block.get("outputs", ["stem", "residual"])),
        )

    def _describe_mix_block(self, block: Mapping[str, Any]) -> str:
        return "%s: %s(stem=%s, residual=%s) -> %s" % (
            block["name"],
            block.get("operator", "mix_stems"),
            block["stem"],
            block["residual"],
            block["output"],
        )

    def _describe_chain_block(self, block: Mapping[str, Any]) -> str:
        return "%s: %s -> %s -> %s" % (
            block["prefix"],
            block["source"],
            self._describe_steps(block.get("steps", [])),
            block["output"],
        )

    def _describe_send_return_block(self, block: Mapping[str, Any]) -> str:
        return "%s: send_return(%s, dry=%s, send=%s, return=%s) [%s] -> %s" % (
            block["name"],
            block["source"],
            block.get("dry_level", 1.0),
            block.get("send_level", 1.0),
            block.get("return_level", 1.0),
            self._describe_steps(block.get("steps", [])),
            block["output"],
        )

    def _describe_step_block(self, block: Mapping[str, Any]) -> str:
        return "%s: %s -> %s" % (
            block["name"],
            self._describe_step(block),
            ", ".join(self._normalize_outputs(block["name"], block.get("outputs"))),
        )

    def _describe_steps(self, steps: Sequence[Mapping[str, Any]]) -> str:
        return " -> ".join(self._describe_step(step) for step in steps)

    def _describe_step(self, step: Mapping[str, Any]) -> str:
        params = step.get("params", {})
        if not params:
            return step["operator"]
        rendered_params = ", ".join(
            "%s=%s" % (name, json.dumps(value, sort_keys=True))
            for name, value in sorted(params.items())
        )
        return "%s(%s)" % (step["operator"], rendered_params)

    @staticmethod
    def _normalize_outputs(default_name: str, outputs: Any) -> list[str]:
        if outputs is None:
            return [default_name]
        if isinstance(outputs, str):
            return [outputs]
        return list(outputs)
