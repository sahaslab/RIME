from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import yaml
from typing import Any

# Root key expected in summarization.yaml, matching the filename stem the way
# operators.yaml -> `operators` and distributions.yaml -> `distributions` do.
ROOT_KEY = "summarization"

CONFIG_FILE_NAME = "summarization.yaml"


@dataclass(frozen=True)
class ModelSettings:
    """How to reach the summarization model through litellm."""

    name: str
    temperature: float = 0.7
    aws_region_name: str | None = None
    aws_profile_name: str | None = None
    num_retries: int = 2
    timeout_s: float = 120.0

    def completion_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to hand `litellm.completion` for this model.

        Unset AWS settings are omitted rather than passed as None, so litellm
        falls back to the default AWS credential chain instead of being told to
        use no region or no profile.
        """
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "num_retries": self.num_retries,
            "timeout": self.timeout_s,
        }
        if self.aws_region_name:
            kwargs["aws_region_name"] = self.aws_region_name
        if self.aws_profile_name:
            kwargs["aws_profile_name"] = self.aws_profile_name
        return kwargs

    @property
    def is_bedrock(self) -> bool:
        return self.name.startswith("bedrock/")

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any], context: str) -> "ModelSettings":
        name = str(section.get("name") or "").strip()
        if not name:
            raise ValueError("Summarization config '%s' does not name a model." % context)
        temperature = float(section.get("temperature", 0.7))
        if temperature < 0.0:
            raise ValueError(
                "Summarization config '%s' has a negative temperature (%s)." % (
                    context,
                    temperature
                )
            )
        return cls(
            name=name,
            temperature=temperature,
            aws_region_name=_optional_str(section.get("aws_region_name")),
            aws_profile_name=_optional_str(section.get("aws_profile_name")),
            num_retries=int(section.get("num_retries", 2)),
            timeout_s=float(section.get("timeout_s", 120.0))
        )


@dataclass(frozen=True)
class SummarizationConfig:
    """Defaults for one prompt-generation run, before CLI overrides."""

    model: ModelSettings
    plans_path: Path
    output_path: Path
    max_attempts: int
    max_workers: int
    levels: tuple[int, ...]
    subsample_num: int
    seed: int

    @classmethod
    def from_config(cls, config_path: Path | str) -> "SummarizationConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if ROOT_KEY not in loaded:
            raise ValueError(
                "Summarization config '%s' is missing a '%s' root key." % (path, ROOT_KEY)
            )
        section = loaded[ROOT_KEY] or {}
        paths = section.get("paths") or {}
        run = section.get("run") or {}

        max_attempts = int(run.get("max_attempts", 3))
        if max_attempts < 1:
            raise ValueError(
                "Summarization config '%s' sets max_attempts to %d; it must be at least 1." % (
                    path,
                    max_attempts
                )
            )
        max_workers = int(run.get("max_workers", 4))
        if max_workers < 1:
            raise ValueError(
                "Summarization config '%s' sets max_workers to %d; it must be at least 1." % (
                    path,
                    max_workers
                )
            )

        return cls(
            model=ModelSettings.from_mapping(section.get("model") or {}, str(path)),
            plans_path=_required_path(paths, "plans", path),
            output_path=_required_path(paths, "output", path),
            max_attempts=max_attempts,
            max_workers=max_workers,
            levels=_parse_levels(run.get("levels"), path),
            subsample_num=int(run.get("subsample_num", -1)),
            seed=int(run.get("seed", 0))
        )


def load_summarization_config(
    config_dir: Path | str,
    summarization_config: Path | str | None = None
) -> SummarizationConfig:
    if summarization_config is not None:
        return SummarizationConfig.from_config(summarization_config)
    return SummarizationConfig.from_config(Path(config_dir) / CONFIG_FILE_NAME)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_path(paths: Mapping[str, Any], key: str, config_path: Path) -> Path:
    value = _optional_str(paths.get(key))
    if value is None:
        raise ValueError(
            "Summarization config '%s' does not define paths.%s." % (config_path, key)
        )
    return Path(value).expanduser()


def _parse_levels(value: Any, config_path: Path) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            "Summarization config '%s' expects run.levels to be a list of level ids." % config_path
        )
    return tuple(int(item) for item in value)
