from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping
import yaml
from typing import Any


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: tuple[str, ...] = field(default_factory=tuple)
    signal_param: str = "audio"
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def validate_params(self, params: Mapping[str, Any]) -> None:
        invalid = sorted(set(params.keys()) - set(self.params))
        if invalid:
            raise ValueError(
                "Operator '%s' does not accept parameter(s) %s." % (
                    self.name,
                    ", ".join(invalid)
                )
            )


class OperatorRegistry:
    def __init__(self, specs: Mapping[str, OperatorSpec]):
        self._by_name: dict[str, OperatorSpec] = dict(specs)
        self._by_alias: dict[str, OperatorSpec] = {}
        for spec in specs.values():
            for alias in (spec.name, *spec.aliases):
                if alias in self._by_alias:
                    raise ValueError("Duplicate operator alias '%s'." % alias)
                self._by_alias[alias] = spec

    @classmethod
    def from_config(cls, config_path: Path | str) -> "OperatorRegistry":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        definitions = loaded.get("operators", [])
        if not definitions:
            raise ValueError("Operator config '%s' does not define any operators." % path)
        specs = {
            definition["name"]: OperatorSpec(
                name=definition["name"],
                aliases=tuple(definition.get("aliases", [])),
                tags=tuple(definition.get("tags", [])),
                params=tuple(definition.get("params", [])),
                signal_param=definition.get("signal_param", "audio"),
                runtime=dict(definition.get("runtime", {}))
            )
            for definition in definitions
        }
        return cls(specs)

    def resolve(self, name: str) -> OperatorSpec:
        if name not in self._by_alias:
            raise ValueError("Unknown operator '%s'." % name)
        return self._by_alias[name]

    def has(self, name: str) -> bool:
        return name in self._by_alias

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name.keys()))

    def runtime_spec(self, name: str) -> Mapping[str, Any]:
        spec = self.resolve(name)
        if not spec.runtime:
            raise ValueError("Operator '%s' does not define runtime binding metadata." % name)
        return spec.runtime


def load_operator_registry(
    config_dir: Path | str,
    operator_config: Path | str = None
) -> OperatorRegistry:
    if operator_config is not None:
        return OperatorRegistry.from_config(operator_config)
    return OperatorRegistry.from_config(Path(config_dir) / "operators.yaml")
