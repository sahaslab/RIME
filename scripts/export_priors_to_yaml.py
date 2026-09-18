import copy
import argparse
import difflib
import itertools
import yaml
from pathlib import Path
from typing import Any
from build_corpus import ROOT, PIPELINE_SCRIPTS, digest, read_json, backup_existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "02_priors")
    parser.add_argument(
        "--base-config-dir",
        type=Path,
        default=ROOT / "01_corpus" / "original_yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "03_rime")
    parser.add_argument("--diff-dir", type=Path, default=ROOT / "04_diffs")
    args = parser.parse_args()
    priors = read_json(args.input_dir / "priors.json")
    assert digest(args.input_dir / "presets.jsonl") == priors["presets_sha256"]
    assert priors["scripts_sha256"] == {
        path.name: digest(path) for path in PIPELINE_SCRIPTS
    }, "Scripts changed; rerun extraction"
    originals = {}
    for name, expected in priors["original_sha256"].items():
        path = args.base_config_dir / name
        assert digest(path) == expected, "Original YAML changed: %s; rerun extraction" % name
        originals[name] = path.read_text()
    config = {name: yaml.safe_load(value) for name, value in originals.items()}
    assert not args.output_dir.resolve().is_relative_to(args.base_config_dir.resolve())
    assert not args.base_config_dir.resolve().is_relative_to(args.output_dir.resolve())
    assert args.diff_dir.resolve() != args.output_dir.resolve()
    integrate(config, priors)
    for path in [args.output_dir, args.diff_dir]:
        backup = backup_existing(path)
        if backup:
            print("Previous output: %s" % backup)
        path.mkdir(parents=True)
    provenance = []
    for name, value in config.items():
        if name in {"operators.yaml", "constraints.yaml"}:
            text = originals[name]
        else:
            text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
            text, records = annotate_yaml(text, name, priors, config["motifs.yaml"]["motifs"])
            if name == "distributions.yaml":
                text = "# " + priors["fit_policy"]["generation_constraints_note"] + "\n" + text
            provenance.extend(records)
            assert yaml.safe_load(text) == value
            difference = difflib.unified_diff(
                yaml.safe_dump(
                    yaml.safe_load(originals[name]), sort_keys=False, allow_unicode=True
                ).splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile="original/" + name + " (format normalized)",
                tofile="03_rime/" + name,
            )
            (args.diff_dir / (name + ".diff")).write_text("".join(difference))
        (args.output_dir / name).write_text(text)
    (args.diff_dir / "provenance.txt").write_text("\n".join(provenance) + "\n")
    print(
        "03_rime: %d recipes, %d motifs; original YAML diffs in %s"
        % (
            len(config["recipes.yaml"]["recipes"]),
            len(config["motifs.yaml"]["motifs"]),
            args.diff_dir,
        ),
    )


def annotate_yaml(
    text: str,
    name: str,
    priors: dict[str, Any],
    motifs: dict[str, Any],
) -> tuple[str, list[str]]:
    section = name.removesuffix(".yaml")
    root = yaml.compose(text).value[0][1]
    if section == "distributions":
        entries = distribution_nodes(root)
    elif section == "motifs":
        entries = [
            (key.value, key.start_mark.line, key.start_mark.column) for key, value in root.value
        ]
    else:
        entries = [
            (
                next(value.value for key, value in node.value if key.value == "id"),
                node.start_mark.line,
                0,
            )
            for node in root.value
        ]
    loaded = yaml.safe_load(text)[section]
    current = {row["id"]: row for row in loaded} if section == "recipes" else loaded
    comments, records = {}, []
    for identifier, line, column in entries:
        citations = []
        if section == "distributions":
            if identifier in priors["distributions"]:
                citations.append(
                    upstream_citation(
                        priors["fit_reports"][identifier]["provenance"],
                        priors["fit_reports"][identifier].get("controls", []),
                    )
                )
        else:
            if identifier in priors["recipe_evidence"]:
                evidence = priors["recipe_evidence"][identifier]
                citations.append(
                    upstream_citation(evidence["provenance"]) + "; " + evidence["note"]
                )
            paths = find_references(current[identifier], "sample")
            for motif in find_references(current[identifier], "chain_ref"):
                paths |= find_references(motifs[motif], "sample")
            for path in sorted(paths):
                if path in priors["distributions"]:
                    citations.append(
                        "%s: %s"
                        % (
                            path,
                            upstream_citation(
                                priors["fit_reports"][path]["provenance"],
                                priors["fit_reports"][path].get("controls", []),
                            ),
                        )
                    )
            effect = identifier.removeprefix("target_").removesuffix("_compression")
            if identifier.endswith("_compression") and effect in priors["combinations"]:
                citations.append(
                    "Effect co-occurrence: "
                    + upstream_citation(priors["combinations"][effect]["provenance"])
                )
        if not citations:
            continue
        description = "; ".join(citations) + "."
        comments[line] = " " * column + "# " + description
        records.append("%s %s | %s" % (name, identifier, description))
    lines = text.splitlines()
    for line, comment in sorted(comments.items(), reverse=True):
        lines.insert(line, comment)
    return "\n".join(lines) + "\n", records


def upstream_citation(provenance: list[dict[str, Any]], controls: list[str] | None = None) -> str:
    labels = {"mixparams": "MixParams", "socialfx": "SocialFX"}
    scope = " for " + ", ".join(controls) if controls else ""
    citations = []
    for source in provenance:
        if "lines" in source:
            citations.append(
                "Inferred from %s, see lines %d–%d in the original file %s for provenance"
                % (source["source"], source["lines"][0], source["lines"][1], source["url"])
            )
            continue
        ranges = []
        for key, group in itertools.groupby(
            enumerate(source["rows"]), lambda pair: pair[1] - pair[0]
        ):
            values = [value for index, value in group]
            ranges.append(
                str(values[0]) if len(values) == 1 else "%d-%d" % (values[0], values[-1])
            )
        citations.append(
            "Inferred from %s%s, see zero-based rows %s in the original file %s for provenance"
            % (
                labels.get(source["source"], source["source"]),
                scope,
                ",".join(ranges),
                source["url"],
            )
        )
    return "; ".join(citations)


def distribution_nodes(node: yaml.MappingNode, prefix: str = "") -> list[tuple[str, int, int]]:
    entries = []
    for key, value in node.value:
        path = prefix + "." + key.value if prefix else key.value
        if any(child.value == "type" for child, spec in value.value):
            entries.append((path, key.start_mark.line, key.start_mark.column))
        else:
            entries.extend(distribution_nodes(value, path))
    return entries


def find_references(value: Any, key: str) -> set[str]:
    if isinstance(value, dict):
        found = {value[key]} if isinstance(value.get(key), str) else set()
        return found | {item for child in value.values() for item in find_references(child, key)}
    if isinstance(value, list):
        return {item for child in value for item in find_references(child, key)}
    return set()


def integrate(config: dict[str, Any], priors: dict[str, Any]) -> None:
    distributions = config["distributions.yaml"]["distributions"]
    motifs = config["motifs.yaml"]["motifs"]
    recipes = config["recipes.yaml"]["recipes"]
    continuous_defaults(distributions)
    # Apply the same requested ratio ceiling to the authored parallel bus.
    ratio = distributions["drums"]["parallel_compression"]["ratio"]
    ratio["high"] = min(ratio["high"], priors["fit_policy"]["domains"]["ratio"][1])
    for path, model in priors["distributions"].items():
        branch = distributions
        parts = path.split(".")
        for part in parts[:-1]:
            branch = branch.setdefault(part, {})
        branch[parts[-1]] = copy.deepcopy(model)
    # Reference the same authored timing priors in both compression recipes.
    # Ambiguous source timing units never replace these defaults.
    for group, motif_name in [
        ("vocals.compression", "vocal_control_compression"),
        ("dynamics.compression", "control_compression"),
    ]:
        path = group + ".settings"
        if path not in priors["distributions"]:
            continue
        branch = distributions
        for part in group.split("."):
            branch = branch[part]
        branch["settings"]["parameters"]["attack_ms"] = {"sample": "vocals.compression.attack_ms"}
        branch["settings"]["parameters"]["release_ms"] = {"sample": "vocals.compression.release_ms"}
        motifs[motif_name] = {
            "order_profile": "shaping",
            "steps": [
                {
                    "name": "compressor",
                    "operator": "apply_compressor_effect",
                    "params": {"sample": path},
                }
            ],
        }
        if group.startswith("dynamics."):
            recipes.append(
                recipe(
                    motif_name,
                    "Control target dynamics with compression.",
                    ["dynamics", "compression"],
                    exclude_vocals=True,
                ),
            )
    for effect in ["highpass", "lowpass", "highshelf", "lowshelf", "peak"]:
        path = "tone.equalization.%s.settings" % effect
        scalar_path = "tone.equalization.%s.cutoff_frequency_hz" % effect
        if path not in priors["distributions"] and scalar_path not in priors["distributions"]:
            continue
        params = (
            {"sample": path}
            if path in priors["distributions"]
            else {"cutoff_frequency_hz": {"sample": scalar_path}}
        )
        name = "target_" + effect
        motifs[name] = {
            "order_profile": "eq_cleanup",
            "steps": [
                {
                    "name": effect,
                    "operator": "apply_%s_filter" % effect,
                    "params": params,
                }
            ],
        }
        recipes.append(
            recipe(name, "Shape the target with %s filtering." % effect, ["eq"]),
        )
        # Co-occurrence supplies evidence for combining effects. Ordering is
        # explicitly inherited from RIME's shaping profile, not inferred from
        # unordered channel snapshots.
        if effect in priors["combinations"] and "control_compression" in motifs:
            name = "target_%s_compression" % effect
            steps = copy.deepcopy(
                motifs["target_" + effect]["steps"] + motifs["control_compression"]["steps"],
            )
            motifs[name] = {"order_profile": "shaping", "steps": steps}
            recipes.append(
                recipe(
                    name,
                    "Shape and compress the target using the existing EQ-before-dynamics order.",
                    ["eq", "dynamics", "compression"],
                ),
            )
    integrate_time_recipes(config, priors["recipe_evidence"])
    assert len({item["id"] for item in recipes}) == len(recipes)


def continuous_defaults(distributions: dict[str, Any], prefix: str = "") -> None:
    # Musical subdivisions and pitch intervals are genuinely discrete.
    discrete = {"space.shared_send.delay.synced.beats", "vocals.harmony.intervals"}
    for name, spec in distributions.items():
        path = prefix + "." + name if prefix else name
        if spec.get("type") in {"uniform", "int_uniform", "log_uniform"}:
            spec.pop("samples", None)
        if "type" not in spec:
            continuous_defaults(spec, path)
        elif spec["type"] == "choice" and path not in discrete:
            values = [item["value"] for item in spec["values"]]
            assert all(isinstance(value, (int, float)) for value in values), path
            low, high = min(values), max(values)
            assert low < high, "A single setting does not establish a range: %s" % path
            logarithmic = name.endswith(("_hz", "_ms")) or name in {"q", "ratio"}
            distributions[name] = {
                "type": "log_uniform" if logarithmic else "uniform",
                "low": low,
                "high": high,
            }


def integrate_time_recipes(config: dict[str, Any], evidence: dict[str, Any]) -> None:
    motifs = config["motifs.yaml"]["motifs"]
    recipes = config["recipes.yaml"]["recipes"]
    space = motifs["band_limited_space_send_free"]["steps"]
    steps = {
        "delay": copy.deepcopy(space[0]),
        "reverb": copy.deepcopy(space[1]),
        "chorus": copy.deepcopy(motifs["modulation_texture"]["steps"][0]),
        "limiter": {
            "name": "limiter",
            "operator": "apply_limiter_effect",
            "params": {"threshold_db": -10.0, "release_ms": 100.0},
        },
    }
    for name, item in evidence.items():
        if name == "modulation_texture":
            continue
        effect = item["effects"][-1]
        wet = item["routing"] == "send"
        step = copy.deepcopy(steps[effect])
        if effect in {"delay", "chorus"}:
            step["params"]["mix"] = 1.0 if wet else {"sample": "space.shared_send.send_level"}
        elif effect == "reverb" and not wet:
            step["params"]["dry_level"] = 1.0
            step["params"]["wet_level"] = {"sample": "space.shared_send.send_level"}
        motif_name = "target_%s_%s" % (effect, item["routing"])
        if effect == "limiter":
            motif_name = name
            motif_steps = copy.deepcopy(motifs["retro_tilt_eq"]["steps"]) + [step]
            motifs[motif_name] = {"order_profile": "shaping", "steps": motif_steps}
        else:
            motifs[motif_name] = {"order_profile": "serial_fx", "steps": [step]}
        tags = list(item["effects"]) + (["send_return"] if wet else [])
        if effect in {"delay", "reverb"}:
            tags.append("time_based")
        if effect == "chorus":
            tags.append("modulation")
        generated = recipe(
            name,
            "Apply %s using %s routing." % (" then ".join(item["effects"]), item["routing"]),
            tags,
        )
        block = generated["graph"][1]
        block["chain_ref"] = motif_name
        if wet:
            block.update(
                {
                    "kind": "send_return",
                    "name": name,
                    "dry_level": 1.0,
                    "send_level": {"sample": "space.shared_send.send_level"},
                    "return_level": {"sample": "space.shared_send.return_level"},
                }
            )
        if len(item["effects"]) > 1 and effect != "limiter":
            before = item["effects"][0]
            motif = "retro_tilt_eq" if before == "eq" else "control_compression"
            prefix = {
                "kind": "chain",
                "prefix": "prepare_target",
                "source": "target_stem",
                "output": "prepared_stem",
                "chain_ref": motif,
            }
            generated["graph"].insert(1, prefix)
            block["source"] = "prepared_stem"
        recipes.append(generated)


def recipe(
    name: str,
    description: str,
    tags: list[str],
    exclude_vocals: bool = False,
) -> dict[str, Any]:
    excluded = ["other", "vocals"] if exclude_vocals else ["other"]
    return {
        "id": name,
        "description": description,
        "tags": tags,
        "weight": 1.0,
        "bindings": {
            "target_candidate": {"each_from": "metadata.analysis.target_candidates"},
            "target_description": {"ref": "bindings.target_candidate.stem"},
            "target_family": {"ref": "bindings.target_candidate.family"},
        },
        "when": {
            "all": [{"exists": {"path": "bindings.target_description"}}]
            + [{"neq": {"path": "bindings.target_family", "value": family}} for family in excluded]
        },
        "graph": [
            {
                "kind": "separate",
                "name": "isolate_target",
                "source": "audio",
                "description": "${bindings.target_description}",
                "outputs": ["target_stem", "residual"],
            },
            {
                "kind": "chain",
                "prefix": name,
                "source": "target_stem",
                "output": "processed_stem",
                "chain_ref": name,
            },
            {
                "kind": "mix",
                "name": "remix_target",
                "stem": "processed_stem",
                "residual": "residual",
                "output": "final_audio",
            },
        ],
    }


if __name__ == "__main__":
    main()
