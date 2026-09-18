import json
import math
import re
import io
import csv
import zipfile
import argparse
import itertools
import xml.etree.ElementTree
import numpy
import pandas
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from scipy.stats import truncnorm, beta as beta_distribution
from scipy.optimize import minimize
from scipy.special import betainc
from tqdm import tqdm
from build_corpus import (
    ROOT,
    PIPELINE_SCRIPTS,
    ASSETS,
    SOURCES,
    read_json,
    write_json,
    digest,
    stable_id,
    numeric,
    flatten,
    backup_existing,
)


def main():
    parser = argparse.ArgumentParser(
        epilog="Output: 02_priors/presets.jsonl contains every native record; priors.json contains pooled fits, cluster diagnostics, chain counts and source text. Descriptors are observed text, not names invented for fitted clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "01_corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "02_priors",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Fit selected splits while retaining all native records",
    )
    args = parser.parse_args()
    inventory = read_json(args.corpus_dir / "inventory.json")
    assert inventory == {
        "sources": SOURCES,
        "assets": ASSETS,
    }, "The corpus inventory differs from the pinned sources"
    records = []
    for asset in tqdm(inventory["assets"], desc="Reading corpus"):
        assert digest(args.corpus_dir / asset["path"]) == asset["sha256"]
        adapter = inventory["sources"][asset["source"]]["adapter"]
        records.extend(
            extract_asset(
                args.corpus_dir,
                asset,
                adapter,
            ),
        )
    assert len({row["id"] for row in records}) == len(records)
    link_preset_assets(records, inventory)
    selected = [row for row in records if args.splits is None or row["split"] in args.splits]
    assert selected, "No records selected"
    observations = parameter_observations(selected)
    parameters, translation = translate(
        selected,
        inventory,
        MAPPINGS,
    )
    original_dir = args.corpus_dir / "original_yaml"
    originals = yaml.safe_load((original_dir / "distributions.yaml").read_text())["distributions"]
    fitted = integrate_parameters(parameters, originals)
    result = {
        "distributions": fitted["distributions"],
        "fit_reports": fitted["reports"],
        "combinations": fitted["combinations"],
        "recipe_evidence": extract_recipe_evidence(selected, args.corpus_dir),
        "original_sha256": {
            path.name: digest(path) for path in sorted(original_dir.glob("*.yaml"))
        },
        "chains": summarize_structure(selected),
        "native_parameters": summarize_parameters(observations),
        "language": summarize_language(selected, LEXICON),
        "vocabulary": summarize_vocabulary(selected),
        "published_chains": PUBLISHED_CHAINS,
        "translation": translation,
        "splits": args.splits,
        "records": len(records),
        "fit_policy": FIT_POLICY,
        "input_sha256": digest(args.corpus_dir / "inventory.json"),
        "scripts_sha256": {path.name: digest(path) for path in PIPELINE_SCRIPTS},
    }
    backup = backup_existing(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "presets.jsonl").open("w", encoding="utf-8") as handle:
        for row in tqdm(records, desc="Writing native presets"):
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
            )
    result["presets_sha256"] = digest(args.output_dir / "presets.jsonl")
    write_json(args.output_dir / "priors.json", result)
    print(
        "02_priors: %d native records, %d integrated distributions"
        % (len(records), len(fitted["distributions"])),
    )
    if backup:
        print("Previous output: %s" % backup)


def extract_asset(
    corpus: Path,
    asset: dict[str, Any],
    adapter: str,
) -> list[dict[str, Any]]:
    path = corpus / asset["path"]
    suffix = path.suffix.lower()
    if suffix == ".parquet" and adapter in {"mixparams", "socialfx", "mixassist"}:
        frame = pandas.read_parquet(path)
        rows = json.loads(frame.to_json(orient="records", double_precision=15))
        functions = {
            "mixparams": mixparams,
            "socialfx": socialfx,
            "mixassist": mixassist,
        }
        return [
            functions[adapter](
                row,
                asset,
                index,
            )
            for index, row in enumerate(
                tqdm(
                    rows,
                    desc=asset["path"],
                    leave=False,
                ),
            )
        ]
    if adapter == "calf" and path.name == "presets.xml":
        presets = xml.etree.ElementTree.parse(path).getroot().findall("preset")
        return [
            calf(
                preset,
                asset,
                index,
            )
            for index, preset in enumerate(presets)
        ]
    if adapter == "easyeffects" and suffix == ".json":
        return easyeffects(read_json(path), asset)
    if adapter == "timbral" and suffix == ".zip":
        return timbral(path, asset)
    return []


def record(
    raw: dict[str, Any],
    asset: dict[str, Any],
    index: int,
    kind: str,
) -> dict[str, Any]:
    return {
        "id": stable_id([asset["source"], asset["path"], asset["sha256"], index]),
        "source": asset["source"],
        "asset": asset["path"],
        "asset_sha256": asset["sha256"],
        "row_index": index,
        "kind": kind,
        "split": Path(asset["path"]).stem.split("-000")[0],
        "context": {},
        "descriptors": [],
        "instances": [],
        "chain": None,
        "issues": [],
        "raw": raw,
    }


def instance(
    plugin: str,
    parameters: dict[str, Any],
    index: int,
    subtype: str = "",
) -> dict[str, Any]:
    family = plugin.split("#")[0]
    return {
        "instance_id": str(index),
        "plugin": plugin,
        "effect": FAMILIES.get(family, family),
        "subtype": subtype,
        "enabled": True,
        "routing": "unknown",
        "parameters": parameters,
    }


def mixparams(
    raw: dict[str, Any],
    asset: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    result = record(
        raw,
        asset,
        index,
        "channel_state",
    )
    parameters = json.loads(raw["parameters"])
    result["context"] = {
        "song": raw["song_name"],
        "mix": raw["mix_name"],
        "artist": raw["artist_name"],
        "instrument": raw["track_instrument_type"],
        "instrument_subtype": raw["track_instrument_subtype"],
        "genre": raw["genre"],
        "track": raw["track_name"],
        "channel_mode": raw["channel_mode"],
    }
    for effect, settings in parameters.items():
        if effect in MIX_EFFECTS:
            for position, values in enumerate(settings):
                item = instance(
                    effect,
                    values,
                    position,
                    str(values.get("type", "")),
                )
                item["instance_id"] = "%s/%d" % (effect, position)
                if "send" in values:
                    item["routing"] = "send" if str(values["send"]).lower() == "true" else "insert"
                result["instances"].append(item)
        elif effect in {"gain", "pan"}:
            item = instance(
                effect,
                {effect: settings},
                0,
            )
            item["instance_id"] = effect
            result["instances"].append(item)
        else:
            result["issues"].append(
                "Unclassified channel field retained in raw: %s" % effect,
            )
    result["issues"].append(
        "Channel snapshot does not establish cross-effect insert order",
    )
    return result


def socialfx(
    raw: dict[str, Any],
    asset: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    result = record(
        raw,
        asset,
        index,
        "descriptor_setting",
    )
    result["context"] = {"source_id": raw["id"], "extra": raw.get("extra")}
    result["descriptors"] = [" ".join(str(raw.get("text") or "").lower().split())]
    keys, values = raw["param_keys"], raw["param_values"]
    if len(keys) != len(values) or len(set(keys)) != len(keys):
        result["issues"].append(
            "Ambiguous parameter schema; keys and values retained without pairing",
        )
        return result
    plugin = result["split"]
    item = instance(
        plugin,
        dict(zip(keys, values)),
        0,
    )
    item["routing"] = "single_effect"
    result["instances"] = [item]
    return result


def mixassist(
    raw: dict[str, Any],
    asset: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    result = record(
        raw,
        asset,
        index,
        "dialogue",
    )
    result["context"] = {
        "conversation": raw["conversation_id"],
        "turn": raw["turn_id"],
        "topic": raw["topic"],
        "has_content": raw["has_content"],
    }
    return result


def calf(
    preset: xml.etree.ElementTree.Element,
    asset: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    raw = {
        "attributes": dict(preset.attrib),
        "xml": xml.etree.ElementTree.tostring(preset, encoding="unicode"),
    }
    result = record(
        raw,
        asset,
        index,
        "factory_preset",
    )
    result["split"] = "factory"
    result["context"] = {"name": preset.attrib["name"]}
    result["descriptors"] = [preset.attrib["name"].lower()]
    parameters = {node.attrib["name"]: node.attrib["value"] for node in preset.findall("param")}
    item = instance(
        preset.attrib["plugin"],
        parameters,
        0,
    )
    item["routing"] = "single_effect"
    item["enabled"] = parameters.get("bypass", "0") == "0" and parameters.get("on", "1") == "1"
    result["instances"] = [item]
    if item["plugin"] in {"monosynth", "organ", "wavetable"}:
        result["kind"] = "instrument_preset"
    return result


def easyeffects(raw: dict[str, Any], asset: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for direction in ["input", "output"]:
        if direction not in raw:
            continue
        pipeline = raw[direction]
        result = record(
            raw,
            asset,
            len(results),
            "chain_preset",
        )
        result["split"] = "community"
        result["context"] = {"name": Path(asset["path"]).stem, "direction": direction}
        result["descriptors"] = [result["context"]["name"].lower()]
        if "plugins_order" not in pipeline:
            result["issues"].append(
                "Missing explicit plugins_order; native document retained",
            )
            results.append(result)
            continue
        order = pipeline["plugins_order"]
        assert isinstance(order, list)
        result["chain"] = []
        for index, plugin in enumerate(order):
            if plugin not in pipeline:
                result["issues"].append("Ordered plugin has no settings: %s" % plugin)
                result["chain"] = None
                continue
            values = pipeline[plugin]
            item = instance(
                plugin,
                values,
                index,
            )
            item["routing"] = "insert"
            item["enabled"] = not values.get("bypass", False) and values.get(
                "state",
                True,
            )
            result["instances"].append(item)
            if item["enabled"] and result["chain"] is not None:
                result["chain"].append(item["instance_id"])
        results.append(result)
    if not results:
        result = record(
            raw,
            asset,
            0,
            "unsupported_preset_document",
        )
        result["issues"].append("JSON has no supported input/output preset pipeline")
        results.append(result)
    return results


def timbral(path: Path, asset: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    with zipfile.ZipFile(path) as archive:
        for name in tqdm(
            sorted(archive.namelist()),
            desc="Reading timbral tables",
            leave=False,
        ):
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            text = archive.read(name).decode("utf-8-sig")
            delimiter = ";" if Path(name).name == "FreeSound search data.csv" else ","
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
            result = record(
                {"member": name, "rows": rows},
                asset,
                len(results),
                "vocabulary_table",
            )
            result["split"] = "vocabulary"
            results.append(result)
    return results


def link_preset_assets(records: list[dict[str, Any]], lock: dict[str, Any]) -> None:
    lookup = defaultdict(list)
    for asset in lock["assets"]:
        lookup[(asset["source"], Path(asset["path"]).name)].append(asset["path"])
    for row in tqdm(records, desc="Linking preset dependencies"):
        if row["kind"] != "chain_preset":
            continue
        references = []
        for field, value in flatten(row["raw"]):
            if not isinstance(value, str) or not value:
                continue
            if not field.endswith(
                ("kernel-path", "kernel-name", "ir-path", "model-path"),
            ):
                continue
            name = Path(value).name
            candidates = lookup[(row["source"], name)]
            if not Path(name).suffix:
                candidates = lookup[(row["source"], name + ".irs")]
            references.append(
                {"field": field, "value": value, "candidate_assets": candidates},
            )
            if len(candidates) != 1:
                row["issues"].append(
                    "Unresolved or ambiguous preset dependency: %s" % value,
                )
        row["asset_references"] = references


def parameter_observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for row in tqdm(records, desc="Extracting every scalar control"):
        for item in row["instances"]:
            for parameter, value in flatten(item["parameters"]):
                observations.append(
                    {
                        "record_id": row["id"],
                        "instance_id": item["instance_id"],
                        "source": row["source"],
                        "split": row["split"],
                        "kind": row["kind"],
                        "context": row["context"],
                        "descriptors": row["descriptors"],
                        "plugin": item["plugin"].split("#")[0],
                        "effect": item["effect"],
                        "subtype": item["subtype"],
                        "routing": item["routing"],
                        "enabled": item["enabled"],
                        "parameter": parameter,
                        "unit": "source_native",
                        "value": value,
                        "numeric_value": numeric(value),
                    },
                )
    return observations


def summarize_parameters(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in tqdm(observations, desc="Grouping control priors"):
        if not row["enabled"] or row["kind"] == "instrument_preset":
            continue
        key = tuple(
            row[field]
            for field in [
                "source",
                "plugin",
                "effect",
                "subtype",
                "parameter",
                "unit",
                "routing",
            ]
        )
        groups[key + ("all", "all")].append(row)
    results = []
    for key, rows in tqdm(sorted(groups.items()), desc="Summarizing control priors"):
        fields = [
            "source",
            "plugin",
            "effect",
            "subtype",
            "parameter",
            "unit",
            "routing",
            "condition",
            "condition_value",
        ]
        result = dict(zip(fields, key))
        counts = Counter(json.dumps(row["value"], sort_keys=True) for row in rows)
        result["n"] = len(rows)
        result["records_n"] = len({row["record_id"] for row in rows})
        result["songs_n"] = len(
            {row["context"]["song"] for row in rows if "song" in row["context"]},
        )
        result["artists_n"] = len(
            {row["context"]["artist"] for row in rows if "artist" in row["context"]},
        )
        result["empirical_support"] = [
            {"value": json.loads(value), "count": count} for value, count in sorted(counts.items())
        ]
        values = pandas.Series(
            [row["numeric_value"] for row in rows if row["numeric_value"] is not None],
            dtype=float,
        )
        result["numeric_n"] = len(values)
        if len(values):
            result["quantiles"] = {
                str(probability): float(values.quantile(probability))
                for probability in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
            }
        results.append(result)
    return results


def summarize_structure(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in tqdm(records, desc="Grouping structural evidence"):
        if not row["instances"] or row["kind"] == "instrument_preset":
            continue
        conditions = [("all", "all"), ("split", row["split"])]
        conditions.extend(
            (field, str(row["context"][field]))
            for field in ["instrument", "genre"]
            if row["context"].get(field)
        )
        for condition in conditions:
            groups[(row["source"],) + condition].append(row)
    summaries = []
    for key, rows in tqdm(
        sorted(groups.items()),
        desc="Summarizing chains and occurrence",
    ):
        presence = Counter()
        cooccurrence = Counter()
        chains = Counter()
        transitions = Counter()
        positions = Counter()
        send_presence = Counter()
        ordered_n = 0
        for row in tqdm(
            rows,
            desc="Counting structural evidence",
            leave=False,
        ):
            enabled = [item for item in row["instances"] if item["enabled"]]
            effects = sorted({item["effect"] for item in enabled})
            presence.update(effects)
            cooccurrence.update(itertools.combinations(effects, 2))
            send_presence.update(
                {item["effect"] for item in enabled if item["routing"] == "send"},
            )
            if row["chain"] is not None:
                by_id = {item["instance_id"]: item for item in enabled}
                chain = tuple(by_id[name]["effect"] for name in row["chain"])
                chains[chain] += 1
                transitions.update(zip(chain, chain[1:]))
                positions.update(enumerate(chain))
                ordered_n += 1
        lengths = Counter()
        for chain, count in chains.items():
            lengths[len(chain)] += count
        outgoing = Counter()
        for (before, after), count in transitions.items():
            outgoing[before] += count
        position_totals = Counter()
        for (position, effect), count in positions.items():
            position_totals[position] += count
        summaries.append(
            {
                "source": key[0],
                "condition": key[1],
                "condition_value": key[2],
                "records_n": len(rows),
                "ordered_records_n": ordered_n,
                "presence_definition": "At least one enabled reported instance; zero-valued gain/pan states included",
                "presence": dict(sorted(presence.items())),
                "send_presence": dict(sorted(send_presence.items())),
                "cooccurrence": [
                    {"effects": list(pair), "n": count}
                    for pair, count in sorted(cooccurrence.items())
                ],
                "chains": [
                    {
                        "effects": list(chain),
                        "n": count,
                        "probability": count / ordered_n,
                    }
                    for chain, count in sorted(chains.items())
                ],
                "chain_lengths": [
                    {"length": length, "n": count, "probability": count / ordered_n}
                    for length, count in sorted(lengths.items())
                ],
                "adjacent_pairs": [
                    {
                        "effects": list(pair),
                        "n": count,
                        "next_probability": count / outgoing[pair[0]],
                    }
                    for pair, count in sorted(transitions.items())
                ],
                "positions": [
                    {
                        "position": position + 1,
                        "effect": effect,
                        "n": count,
                        "probability": count / position_totals[position],
                    }
                    for (position, effect), count in sorted(positions.items())
                ],
            },
        )
    return {
        "groups": summaries,
        "note": "Each source and condition has its own denominator. Repeated instances remain in ordered chains.",
    }


def summarize_language(records: list[dict[str, Any]], lexicon: dict[str, Any]) -> dict[str, Any]:
    mentions = []
    descriptors = Counter()
    patterns = {
        family: re.compile(
            r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(term) for term in terms),
            re.IGNORECASE,
        )
        for family, terms in lexicon["terms"].items()
    }
    for row in tqdm(records, desc="Extracting descriptor and dialogue evidence"):
        for descriptor in row["descriptors"]:
            if descriptor:
                descriptors[(row["source"], row["split"], descriptor)] += 1
        if row["kind"] != "dialogue" or not row["context"]["has_content"]:
            continue
        for role in ["user", "assistant"]:
            text = row["raw"].get(role) or ""
            for family, pattern in patterns.items():
                matches = list(pattern.finditer(text))
                if matches:
                    mentions.append(
                        {
                            "record_id": row["id"],
                            "source": row["source"],
                            "split": row["split"],
                            "conversation": row["context"]["conversation"],
                            "role": role,
                            "family": family,
                            "matches": [
                                {
                                    "text": match.group(),
                                    "start": match.start(),
                                    "end": match.end(),
                                }
                                for match in matches
                            ],
                        },
                    )
    return {
        "descriptors": [
            {"source": source, "split": split, "descriptor": descriptor, "n": count}
            for (source, split, descriptor), count in sorted(descriptors.items())
        ],
        "mentions": mentions,
        "note": "Dialogue mentions are language evidence. Vocabulary tables remain in records.jsonl with source headers and rows.",
    }


def summarize_vocabulary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = {
        Path(row["raw"]["member"]).name: row
        for row in records
        if row["kind"] == "vocabulary_table"
    }
    if "Dictionary arranged.csv" not in tables:
        return []
    searches = Counter()
    for cells in tqdm(
        tables["FreeSound search data.csv"]["raw"]["rows"],
        desc="Reading search frequencies",
    ):
        if not cells:
            continue
        count = numeric(cells[0])
        assert count is not None and count >= 0
        phrase = " ".join(cells[1].lower().split())
        searches[phrase] += count
    phrases = {}
    for cells in tables["CommonSearches.csv"]["raw"]["rows"]:
        if not cells:
            continue
        term = " ".join(cells[0].lower().split())
        accepted = {" ".join(value.lower().split()) for value in cells[1:]}
        accepted -= {"", "x", "no searches"}
        assert all(phrase in searches for phrase in accepted), (
            "Unknown search phrase for %s" % term
        )
        phrases[term] = sorted(accepted)
    result = []
    dictionary = tables["Dictionary arranged.csv"]
    for cells in dictionary["raw"]["rows"]:
        if not cells or not cells[0].strip():
            continue
        term = " ".join(cells[0].lower().split())
        accepted = phrases.get(term, [])
        result.append(
            {
                "source": dictionary["source"],
                "record_id": dictionary["id"],
                "term": term,
                "hierarchy": cells[1:],
                "accepted_searches": accepted,
                "search_count": sum(searches[phrase] for phrase in accepted),
            },
        )
    return result


def translate(
    records: list[dict[str, Any]],
    extraction: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters, report = [], []
    for row in tqdm(records, desc="Projecting source controls into RIME"):
        adapter = extraction["sources"][row["source"]]["adapter"]
        mapped_instances = {}
        for item in row["instances"]:
            if not item["enabled"] or row["kind"] == "instrument_preset":
                continue
            units, expansion_issues = expand_equalizer(item, adapter)
            if adapter == "easyeffects" and item["plugin"].split("#")[0] == "equalizer":
                report.append(
                    {
                        "record_id": row["id"],
                        "instance_id": item["instance_id"],
                        "projection_notes": [
                            "Linked left-channel band cascade and input/output gain; native filter algorithm, slope, balance and other plugin-level controls are not reproduced"
                        ],
                    },
                )
            steps = []
            complete = not expansion_issues
            for unit_index, unit in enumerate(units):
                projected_id = (
                    "%s/%d" % (item["instance_id"], unit_index)
                    if len(units) > 1
                    else item["instance_id"]
                )
                rules = [
                    rule
                    for rule in mapping["rules"]
                    if matches(
                        rule,
                        unit,
                        adapter,
                    )
                ]
                assert len(rules) <= 1, "Ambiguous mapping for %s" % unit["plugin"]
                if not rules:
                    expansion_issues.append(
                        "No mapping: %s/%s" % (unit["plugin"], unit["subtype"]),
                    )
                    complete = False
                    continue
                rule = rules[0]
                values, used, issues = convert_parameters(unit["parameters"], rule)
                for parameter, value in values.items():
                    parameters.append(
                        {
                            "source": row["source"],
                            "record_id": row["id"],
                            "asset": row["asset"],
                            "row_index": row["row_index"],
                            "instance_id": projected_id,
                            "context": row["context"],
                            "split": row["split"],
                            "descriptors": row["descriptors"],
                            "routing": item["routing"],
                            "rule": rule["id"],
                            "operator": rule["operator"],
                            "parameter": parameter,
                            "unit": rule["parameters"][parameter]["unit"],
                            "value": value,
                        },
                    )
                missing = sorted(set(rule["required"]) - set(values))
                if missing:
                    complete = False
                    issues.append(
                        "Missing required RIME controls: %s" % ", ".join(missing),
                    )
                omitted = sorted(set(dict(flatten(unit["parameters"]))) - used)
                report.append(
                    {
                        "record_id": row["id"],
                        "instance_id": item["instance_id"],
                        "rule": rule["id"],
                        "mapped_parameters": values,
                        "omitted_source_fields": omitted,
                        "issues": issues,
                    },
                )
                if not missing:
                    steps.append({"operator": rule["operator"], "params": values})
            if expansion_issues:
                report.append(
                    {
                        "record_id": row["id"],
                        "instance_id": item["instance_id"],
                        "issues": expansion_issues,
                    },
                )
            if complete and steps:
                mapped_instances[item["instance_id"]] = steps
        if row["chain"]:
            if not all(name in mapped_instances for name in row["chain"]):
                report.append(
                    {
                        "record_id": row["id"],
                        "issues": [
                            "Whole-chain projection unavailable; at least one active processor is unsupported"
                        ],
                    },
                )
    return parameters, report


def matches(
    rule: dict[str, Any],
    item: dict[str, Any],
    adapter: str,
) -> bool:
    values = dict(flatten(item["parameters"]))
    return (
        rule["adapter"] == adapter
        and rule["plugin"] == item["plugin"].split("#")[0]
        and ("subtypes" not in rule or item["subtype"] in rule["subtypes"])
        and all(values.get(key) == value for key, value in rule.get("equals", {}).items())
    )


def convert_parameters(
    parameters: dict[str, Any], rule: dict[str, Any]
) -> tuple[dict[str, float], set[str], list[str]]:
    native = dict(flatten(parameters))
    result, used, issues = {}, set(), []
    for target, spec in rule["parameters"].items():
        paths = spec.get("paths", [spec.get("path")])
        present = [path for path in paths if path in native]
        if not present:
            continue
        numbers = [numeric(native[path]) for path in present]
        if None in numbers or len(set(numbers)) != 1:
            issues.append("Nonnumeric or conflicting aliases for %s" % target)
            continue
        value = numbers[0]
        transform = spec.get("transform", "identity")
        assert transform in {"identity", "amplitude_to_db", "scale"}
        if transform == "amplitude_to_db":
            if value <= 0:
                issues.append("Nonpositive amplitude for %s" % target)
                continue
            value = 20 * math.log10(value)
        elif transform == "scale":
            value *= spec["factor"]
        outside_domain = (
            ("min" in spec and value < spec["min"])
            or ("max" in spec and value > spec["max"])
            or ("exclusive_min" in spec and value <= spec["exclusive_min"])
        )
        if outside_domain:
            issues.append("Out-of-domain control: %s" % target)
            continue
        result[target] = value
        used.update(present)
    return result, used, issues


def expand_equalizer(item: dict[str, Any], adapter: str) -> tuple[list[dict[str, Any]], list[str]]:
    if adapter != "easyeffects" or item["plugin"].split("#")[0] != "equalizer":
        return [item], []
    parameters = item["parameters"]
    if parameters.get("split-channels", False):
        return [], ["Independent stereo EQ cannot be represented by RIME serial filters"]
    if "left" not in parameters or "num-bands" not in parameters:
        return [], ["Unsupported EasyEffects equalizer schema"]
    bands = parameters["left"]
    units = []
    issues = []
    for name in ["input-gain", "output-gain"]:
        if name not in parameters:
            issues.append("Missing EQ %s; not defaulted" % name)
    if "input-gain" in parameters:
        units.append(
            {
                "plugin": "eq_gain",
                "subtype": "",
                "parameters": {"gain": parameters["input-gain"]},
            },
        )
    active = [bands["band%d" % index] for index in range(int(parameters["num-bands"]))]
    solo = any(band.get("solo", False) for band in active)
    for band in active:
        if band.get("type") == "Off":
            continue
        if band.get("mute", False) or solo:
            issues.append("EQ mute/solo behavior needs native rendering")
            continue
        units.append({"plugin": "eq_band", "subtype": band["type"], "parameters": band})
    if "output-gain" in parameters:
        units.append(
            {
                "plugin": "eq_gain",
                "subtype": "",
                "parameters": {"gain": parameters["output-gain"]},
            },
        )
    return units, issues


def upstream_provenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets = {asset["path"]: asset for asset in ASSETS}
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["asset"]].add(row["row_index"])
    return [
        {"source": assets[path]["source"], "url": assets[path]["url"], "rows": sorted(indices)}
        for path, indices in sorted(grouped.items())
    ]


def observation_weights(rows: list[dict[str, Any]]) -> numpy.ndarray:
    clusters = [(row["source"], row["context"].get("song") or row["record_id"]) for row in rows]
    counts = Counter(clusters)
    sources = Counter(source for source, cluster in counts)
    weights = numpy.array([1.0 / (counts[key] * sources[key[0]]) for key in clusters])
    return weights / weights.sum()


def fit_distribution(
    rows: list[dict[str, Any]],
    parameter: str,
    policy: dict[str, Any],
    supplied_weights: numpy.ndarray | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    low, high = policy["domains"][parameter]
    accepted = [row for row in rows if low <= row["value"] <= high]
    records = {row["record_id"] for row in accepted}
    contexts = {(row["source"], row["context"].get("song") or row["record_id"]) for row in accepted}
    report = {
        "observations": len(rows),
        "records": len(records),
        "contexts": len(contexts),
        "provenance": upstream_provenance(accepted),
        "source_records": dict(Counter(source for source, record in {(row["source"], row["record_id"]) for row in accepted})),
        "outside_domain": len(rows) - len(accepted),
        "domain": [low, high],
    }
    if len(records) < policy["min_records"] or len(contexts) < policy["min_clusters"]:
        return None, report | {"status": "insufficient_support"}
    values = numpy.array([row["value"] for row in accepted], dtype=float)
    weights = observation_weights(accepted) if supplied_weights is None else supplied_weights / supplied_weights.sum()
    assert len(weights) == len(values)
    context_mass = defaultdict(float)
    for row, weight in zip(accepted, weights):
        context_mass[(row["source"], row["context"].get("song") or row["record_id"])] += float(weight)
    effective_contexts = 1 / sum(mass ** 2 for mass in context_mass.values())
    logarithmic = parameter in policy["log_parameters"]
    a, b = (math.log(low), math.log(high)) if logarithmic else (low, high)
    transformed = numpy.log(values) if logarithmic else values
    unit_values = (transformed - a) / (b - a)
    base = {"low": float(low), "high": float(high), "scale": "log" if logarithmic else "linear"}
    candidates = [{"model": base | {"type": "log_uniform" if logarithmic else "uniform"}, "parameters": 0, "negative_log_likelihood": 0.0, "converged": True}]
    minimum_std = policy["mle"]["minimum_std_fraction"]
    mean = float(weights @ unit_values)
    std = max(minimum_std, float(numpy.sqrt(weights @ ((unit_values - mean) ** 2))))
    normal_starts = [[mean, std], [0.5, 0.5]]
    normal_bounds = [(0, 1), (minimum_std, 3)]
    for family in ["normal", "beta", "power_law", "gaussian_mixture"]:
        if family == "beta" and numpy.any((unit_values <= 0) | (unit_values >= 1)):
            candidates.append({"family": family, "excluded": "An observation lies on the fixed support boundary; point-density Beta MLE is not used for rounded boundary values."})
            continue
        if family == "power_law" and not logarithmic:
            continue
        if family == "normal":
            starts, bounds = normal_starts, normal_bounds
        elif family == "beta":
            starts, bounds = [[1, 1], [2, 2]], [(0.1, 100), (0.1, 100)]
        elif family == "power_law":
            starts, bounds = [[0]], [(-50, 50)]
        else:
            share = policy["mle"]["minimum_component_weight"]
            starts = [[0.5, 0.25, 0.2, 0.75, 0.2], [0.3, mean, std, 0.75, 0.1], [0.7, 0.25, 0.1, mean, std]]
            bounds = [(share, 1 - share)] + normal_bounds + normal_bounds
        fits = [minimize(lambda theta: -float(weights @ mle_log_density(family, theta, unit_values)), start, bounds=bounds, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12}) for start in starts]
        converged = [fit for fit in fits if fit.success and numpy.isfinite(fit.fun)]
        if not converged:
            candidates.append({"family": family, "excluded": "Optimizer did not converge", "messages": [str(fit.message) for fit in fits]})
            continue
        fit = min(converged, key=lambda item: item.fun)
        theta = fit.x
        model = base | {"type": family}
        if family == "normal":
            model.update({"mean": float(a + theta[0] * (b - a)), "std": float(theta[1] * (b - a))})
        elif family == "beta":
            model.update({"alpha": float(theta[0]), "beta": float(theta[1])})
        elif family == "power_law":
            model["exponent"] = float(theta[0] / (b - a) - 1)
        else:
            model["components"] = [{"weight": float(weight), "mean": float(a + location * (b - a)), "std": float(spread * (b - a))} for weight, location, spread in [(theta[0], theta[1], theta[2]), (1 - theta[0], theta[3], theta[4])]]
            model["components"].sort(key=lambda component: component["mean"])
        candidates.append({"model": model, "parameters": len(theta), "negative_log_likelihood": float(fit.fun), "converged": True, "at_numerical_bound": any(abs(value - lower) < 1e-5 or abs(value - upper) < 1e-5 for value, (lower, upper) in zip(theta, bounds))})
    fitted = [candidate for candidate in candidates if "model" in candidate]
    for candidate in fitted:
        candidate["bic_score"] = 2 * effective_contexts * candidate["negative_log_likelihood"] + candidate["parameters"] * math.log(effective_contexts)
    selected = min(fitted, key=lambda candidate: (candidate["bic_score"], candidate["parameters"]))
    return selected["model"], report | {
        "status": "fitted",
        "family": selected["model"]["type"],
        "distinct_values": len(numpy.unique(values)),
        "observed_range": [float(values.min()), float(values.max())],
        "sampling_range": [low, high],
        "effective_contexts": effective_contexts,
        "candidates": candidates,
        "selection": "Minimum BIC-style score using effective context count and normalized weighted log likelihood; this is a clustered-data approximation, not ordinary IID BIC.",
    }


def mle_log_density(family: str, theta: Any, values: numpy.ndarray) -> numpy.ndarray:
    if family == "normal":
        mean, std = theta
        return truncnorm.logpdf(values, -mean / std, (1 - mean) / std, loc=mean, scale=std)
    if family == "beta":
        return beta_distribution.logpdf(values, theta[0], theta[1])
    if family == "power_law":
        rate = float(theta[0])
        log_normalizer = rate / 2 + rate ** 2 / 24 if abs(rate) < 1e-5 else math.log(math.expm1(rate) / rate)
        return rate * values - log_normalizer
    weight, mean_a, std_a, mean_b, std_b = theta
    return numpy.logaddexp(math.log(weight) + mle_log_density("normal", [mean_a, std_a], values), math.log1p(-weight) + mle_log_density("normal", [mean_b, std_b], values))


def integrate_parameters(rows: list[dict[str, Any]], originals: dict[str, Any]) -> dict[str, Any]:
    # Each dataset has equal mass; within it, each song has equal mass.
    # Require support within each source before pooling so a tiny preset bank
    # cannot carry the same weight as a well-sampled dataset.
    # Reject whole shelf settings outside the intended edit magnitude.
    shelf_min, shelf_max = FIT_POLICY["highshelf_gain_magnitude_db"]
    rejected_shelves = {
        (row["source"], row["record_id"], row["instance_id"])
        for row in rows
        if row["operator"] == "apply_highshelf_filter"
        and row["parameter"] == "gain_db"
        and not shelf_min <= abs(row["value"]) <= shelf_max
    }
    rows = [
        row
        for row in rows
        if (row["source"], row["record_id"], row["instance_id"]) not in rejected_shelves
        if row["parameter"] in FIT_POLICY["domains"]
        and row["parameter"] not in FIT_POLICY["blocked_controls"].get(row["source"], [])
        and row["routing"] != "send"
        and FIT_POLICY["domains"][row["parameter"]][0]
        <= row["value"]
        <= FIT_POLICY["domains"][row["parameter"]][1]
    ]
    distributions, reports = {}, {}
    groups = {
        "dynamics.compression": ("apply_compressor_effect", ["threshold_db", "ratio"]),
        "vocals.compression": ("apply_compressor_effect", ["threshold_db", "ratio"]),
        "tone.equalization.highpass": (
            "apply_highpass_filter",
            ["cutoff_frequency_hz"],
        ),
        "tone.equalization.lowpass": ("apply_lowpass_filter", ["cutoff_frequency_hz"]),
        "tone.equalization.highshelf": (
            "apply_highshelf_filter",
            ["cutoff_frequency_hz", "gain_db", "q"],
        ),
        "tone.equalization.lowshelf": (
            "apply_lowshelf_filter",
            ["cutoff_frequency_hz", "gain_db", "q"],
        ),
        "tone.equalization.peak": (
            "apply_peak_filter",
            ["cutoff_frequency_hz", "gain_db", "q"],
        ),
    }
    for name, (operator, controls) in tqdm(
        groups.items(),
        desc="Fitting pooled parameter priors",
    ):
        selected = [row for row in rows if row["operator"] == operator]
        if name.startswith("vocals."):
            selected = [
                row
                for row in selected
                if MAPPINGS["instrument_families"].get(
                    str(row["context"].get("instrument", "")).lower(),
                )
                == "vocals"
            ]
        if len(controls) == 1:
            observations = supported_sources(
                [row for row in selected if row["parameter"] == controls[0]],
            )
            model, report = fit_distribution(observations, controls[0], FIT_POLICY)
            path = name + "." + controls[0]
        else:
            model, report = fit_parameter_set(selected, controls)
            path = name + ".settings"
        reports[path] = report
        if model is not None:
            distributions[path] = model
        # Existing recipes and the random planner share the same scalar priors.
        # Parameter sets reference these definitions rather than copying fits.
        if name == "vocals.compression":
            if model is not None:
                for control in controls:
                    scalar_path = name + "." + control
                    distributions[scalar_path] = model["parameters"][control]
                    reports[scalar_path] = report["partitions"][0]["fits"][control]
                    model["parameters"][control] = {"sample": scalar_path}
    # These are intent-conditioned priors: retain the original retro cutoff
    # range and negative shelf direction. Fader levels are not gain edits;
    # insert compression is not evidence for parallel-bus compression.
    shelf_cuts = {
        (row["record_id"], row["instance_id"])
        for row in rows
        if row["operator"] == "apply_highshelf_filter"
        and row["parameter"] == "gain_db"
        and row["value"] < 0
    }
    for path, operator, control in [
        ("tone.retro.shelf_cut_hz", "apply_highshelf_filter", "cutoff_frequency_hz"),
        ("tone.retro.shelf_gain_db", "apply_highshelf_filter", "gain_db"),
        ("tone.retro.lowpass_hz", "apply_lowpass_filter", "cutoff_frequency_hz"),
    ]:
        base = originals
        for part in path.split("."):
            base = base[part]
        support = [
            item["value"] if isinstance(item, dict) else item for item in base.get("values", [])
        ]
        low, high = (min(support), max(support)) if support else (base["low"], base["high"])
        selected = supported_sources(
            [
                row
                for row in rows
                if row["operator"] == operator
                and row["parameter"] == control
                and low <= row["value"] <= high
                and (
                    operator != "apply_highshelf_filter"
                    or (row["record_id"], row["instance_id"]) in shelf_cuts
                )
            ],
        )
        model, report = fit_distribution(selected, control, FIT_POLICY)
        reports[path] = report | {"intent_range": [low, high]}
        if model is not None:
            distributions[path] = model
    instances = defaultdict(set)
    contexts = {}
    for row in rows:
        instances[row["record_id"]].add(row["operator"])
        contexts[row["record_id"]] = row["context"].get("song") or row["record_id"]
    combinations = {}
    source_lookup = {row["record_id"]: row["source"] for row in rows}
    record_lookup = {row["record_id"]: row for row in rows}
    for effect in ["highpass", "lowpass", "highshelf", "lowshelf", "peak"]:
        matching = [
            key
            for key, operators in instances.items()
            if {"apply_%s_filter" % effect, "apply_compressor_effect"} <= operators
        ]
        if (
            len(matching) >= FIT_POLICY["min_records"]
            and len({contexts[key] for key in matching}) >= FIT_POLICY["min_clusters"]
        ):
            combinations[effect] = {
                "records": len(matching),
                "source_records": dict(Counter(source_lookup[key] for key in matching)),
                "provenance": upstream_provenance([record_lookup[key] for key in matching]),
                "order": "inherited_shaping_profile",
            }
    return {
        "distributions": distributions,
        "reports": reports,
        "combinations": combinations,
    }


def extract_recipe_evidence(records: list[dict[str, Any]], corpus: Path) -> dict[str, Any]:
    evidence = {}
    channels = [row for row in records if row["source"] == "mixparams"]
    for effect in ["delay", "reverb", "chorus"]:
        for routing in ["insert", "send"]:
            matching = []
            explicit = 0
            for row in channels:
                instances = [
                    item
                    for item in row["instances"]
                    if item["enabled"]
                    and item["effect"] == effect
                    and (item["routing"] == "send") == (routing == "send")
                ]
                if instances:
                    matching.append(row)
                    explicit += any(item["routing"] == routing for item in instances)
            if not matching:
                continue
            # The existing modulation recipe already supplies an insert chorus.
            name = (
                "modulation_texture"
                if effect == "chorus" and routing == "insert"
                else "target_%s_%s" % (effect, routing)
            )
            evidence[name] = {
                "effects": [effect],
                "routing": routing,
                "records": len(matching),
                "explicit_routing_records": explicit,
                "provenance": upstream_provenance(matching),
                "note": "Effect presence only; existing RIME parameter priors. "
                + (
                    "Source send flags establish auxiliary routing; bus gains use authored RIME priors."
                    if routing == "send"
                    else "Missing routing flags are retained as unknown evidence; insert routing is an authored RIME choice."
                ),
            }
            if routing != "send" or effect == "chorus":
                continue
            for before in ["eq", "compression"]:
                combined = [
                    row
                    for row in matching
                    if any(
                        item["enabled"] and item["effect"] == before and item["routing"] != "send"
                        for item in row["instances"]
                    )
                ]
                if len(combined) < FIT_POLICY["min_records"]:
                    continue
                evidence["target_%s_%s_send" % (before, effect)] = {
                    "effects": [before, effect],
                    "routing": "send",
                    "records": len(combined),
                    "provenance": upstream_provenance(combined),
                    "note": "Observed co-occurrence and explicit effect send flags; processing the target before the send is authored RIME routing, not recovered source order. EQ is represented by RIME's retro EQ motif. Delay/reverb and bus gains use authored RIME priors; compression uses the bounded fitted prior.",
                }
    assets = {asset["path"]: asset for asset in ASSETS}
    ordered = []
    for row in records:
        if row["chain"] is None:
            continue
        by_id = {item["instance_id"]: item for item in row["instances"]}
        effects = [by_id[name]["effect"] for name in row["chain"]]
        if ("eq", "limiter") not in list(zip(effects, effects[1:])):
            continue
        lines = (corpus / row["asset"]).read_text().splitlines()
        start = next(index for index, line in enumerate(lines) if "\"plugins_order\"" in line)
        end = next(index for index in range(start, len(lines)) if "]" in lines[index])
        ordered.append(
            {
                "source": row["source"],
                "url": assets[row["asset"]]["url"],
                "lines": [start + 1, end + 1],
            }
        )
    if ordered:
        evidence["target_eq_limiter"] = {
            "effects": ["eq", "limiter"],
            "routing": "insert",
            "records": len(ordered),
            "provenance": ordered,
            "note": "Observed adjacent EQ-to-limiter fragment; unsupported surrounding processors are not reproduced or bridged. RIME retro EQ priors and existing limiter defaults (-10 dB, 100 ms) instantiate the fragment; source plugin settings are not transferred.",
        }
    return evidence


def supported_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = defaultdict(set)
    for row in rows:
        records[row["source"]].add(row["record_id"])
    return [row for row in rows if len(records[row["source"]]) >= FIT_POLICY["min_records"]]


def fit_parameter_set(
    rows: list[dict[str, Any]], controls: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    settings, lookup = defaultdict(dict), {}
    for row in rows:
        if row["parameter"] in controls:
            key = (row["source"], row["record_id"], row["instance_id"])
            settings[key][row["parameter"]] = row["value"]
            lookup[key] = row
    keys = sorted(key for key, values in settings.items() if set(values) == set(controls))
    sources = {row["source"] for row in supported_sources([lookup[key] for key in keys])}
    keys = [key for key in keys if key[0] in sources]
    report = {
        "records": len({key[:2] for key in keys}),
        "controls": controls,
        "source_records": dict(Counter(key[0] for key in {key[:2] for key in keys})),
        "provenance": upstream_provenance([lookup[key] for key in keys]),
    }
    if report["records"] < FIT_POLICY["min_records"]:
        return None, report | {"status": "insufficient_support"}
    observations = []
    for key in keys:
        row = lookup[key]
        context = row["context"].copy()
        if not context.get("song"):
            context["song"] = "setting:" + stable_id(settings[key])
        observations.append(row | {"context": context})
    matrix = numpy.array([[settings[key][control] for control in controls] for key in keys])
    weights = observation_weights(observations)
    partitions = [("all", numpy.arange(len(keys)))]
    high_shelf = all(row["operator"] == "apply_highshelf_filter" for row in observations)
    if high_shelf:
        gains = matrix[:, controls.index("gain_db")]
        partitions = [("cut", numpy.flatnonzero(gains < 0)), ("boost", numpy.flatnonzero(gains > 0))]
        partitions = [(name, indices) for name, indices in partitions if len(indices)]
    models, reports = [], []
    for label, indices in partitions:
        members = [observations[index] for index in indices]
        parameters, fits = {}, {}
        for column, control in enumerate(controls):
            column_rows = [row | {"value": float(matrix[index, column])} for row, index in zip(members, indices)]
            policy = FIT_POLICY.copy()
            if high_shelf and control == "gain_db":
                low, high = policy["highshelf_gain_magnitude_db"]
                policy["domains"] = policy["domains"] | {"gain_db": [-high, -low] if label == "cut" else [low, high]}
            model, fit = fit_distribution(column_rows, control, policy)
            if model is None:
                return None, report | {"status": "insufficient_support", "partition": label, "marginal": control}
            parameters[control] = model
            fits[control] = fit
        correlations = parameter_correlations(members, matrix[indices], controls)
        model = {"type": "parameters", "parameters": parameters}
        models.append({"label": label, "weight": float(weights[indices].sum()), "distribution": model})
        reports.append({"label": label, "fits": fits, "correlations": correlations})
    result = models[0]["distribution"] if len(models) == 1 else {"type": "mixture", "components": models}
    # Sign branches are declared categories, not fitted numerical clusters.
    return result, report | {"status": "fitted", "family": "independent_parameters" if len(models) == 1 else "shelf_sign_branches", "partitions": reports}


def parameter_correlations(
    rows: list[dict[str, Any]], matrix: numpy.ndarray, controls: list[str]
) -> list[dict[str, Any]]:
    transformed = matrix.copy()
    for column, control in enumerate(controls):
        if control in FIT_POLICY["log_parameters"]:
            transformed[:, column] = numpy.log(transformed[:, column])
    result = []
    sources = sorted({row["source"] for row in rows})
    for source in (["pooled"] if len(sources) > 1 else []) + sources:
        indices = [index for index, row in enumerate(rows) if source == "pooled" or row["source"] == source]
        members = [rows[index] for index in indices]
        weights = observation_weights(members)
        values = transformed[indices]
        ranks = numpy.empty_like(values)
        for column in range(len(controls)):
            unique, inverse = numpy.unique(values[:, column], return_inverse=True)
            masses = numpy.bincount(inverse, weights=weights)
            ranks[:, column] = (numpy.cumsum(masses) - masses / 2)[inverse]
        covariance = (values - weights @ values).T @ (weights[:, None] * (values - weights @ values))
        rank_covariance = (ranks - weights @ ranks).T @ (weights[:, None] * (ranks - weights @ ranks))
        for first, second in itertools.combinations(range(len(controls)), 2):
            denominator = math.sqrt(covariance[first, first] * covariance[second, second])
            rank_denominator = math.sqrt(rank_covariance[first, first] * rank_covariance[second, second])
            pearson = float(covariance[first, second] / denominator) if denominator > 0 else 0.0
            spearman = float(rank_covariance[first, second] / rank_denominator) if rank_denominator > 0 else 0.0
            result.append({"source": source, "parameters": [controls[first], controls[second]], "covariance": float(covariance[first, second]), "pearson": pearson, "spearman": spearman, "extreme": min(abs(pearson), abs(spearman)) >= FIT_POLICY["mle"]["joint_min_abs_correlation"] and pearson * spearman > 0})
    return result


def model_cdf(model: dict[str, Any], values: Any) -> numpy.ndarray:
    values = numpy.asarray(values, dtype=float)
    kind = model["type"]
    if kind == "choice":
        support = numpy.array([item["value"] for item in model["values"]])
        weights = numpy.array([item["weight"] for item in model["values"]])
        return (values[..., None] >= support).dot(weights / weights.sum())
    if kind == "gaussian_mixture":
        return sum(component["weight"] * model_cdf({"type": "normal", "low": model["low"], "high": model["high"], "scale": model["scale"], "mean": component["mean"], "std": component["std"]}, values) for component in model["components"])
    logarithmic = model.get("scale") == "log"
    low, high = model["low"], model["high"]
    clipped = numpy.clip(
        values,
        low,
        high,
    )
    x = numpy.log(clipped) if logarithmic else clipped
    a, b = (math.log(low), math.log(high)) if logarithmic else (low, high)
    if kind in {"uniform", "log_uniform"}:
        result = (x - a) / (b - a)
    elif kind == "beta":
        result = betainc(model["alpha"], model["beta"], (x - a) / (b - a))
    elif kind == "power_law":
        exponent = model["exponent"] + 1
        result = numpy.log(clipped / low) / math.log(high / low) if abs(exponent) < 1e-10 else numpy.expm1(exponent * numpy.log(clipped / low)) / math.expm1(exponent * math.log(high / low))
    elif kind == "normal":
        mean, std = model["mean"], model["std"]
        result = truncnorm.cdf(
            x,
            (a - mean) / std,
            (b - mean) / std,
            loc=mean,
            scale=std,
        )
    else:
        assert kind == "histogram"
        edges = numpy.array(model["edges"])
        edges = numpy.log(edges) if logarithmic else edges
        result = numpy.interp(
            x,
            edges,
            numpy.r_[0.0, numpy.cumsum(model["weights"])],
        )
    return numpy.where(
        values < low,
        0.0,
        numpy.where(
            values > high,
            1.0,
            result,
        ),
    )


FAMILIES = {
    "compressor": "compression",
    "monocompressor": "compression",
    "comp": "compression",
    "equalizer": "eq",
    "filter": "filter",
    "multibandcompressor": "multiband_compression",
    "multiband_compressor": "multiband_compression",
    "multibandgate": "multiband_gate",
    "bass_enhancer": "bass_enhancer",
    "stereo_tools": "stereo",
    "deesser": "deesser",
}

MIX_EFFECTS = {
    "eq",
    "compression",
    "gate",
    "limiter",
    "reverb",
    "delay",
    "chorus",
    "phaser",
    "flanger",
}

# Fit policy
FIT_POLICY = {
    "generation_constraints_note": "User-selected edit constraints, not empirical source limits: high-shelf gain magnitude 5–18 dB, compressor threshold -40 to -10 dB, ratio 1–10. Raw source records are preserved; fits condition on eligible settings.",
    "highshelf_gain_magnitude_db": [5.0, 18.0],
    "schema_version": 2,
    "min_records": 30,
    "min_clusters": 3,
    "mle": {
        "minimum_std_fraction": 0.02,
        "minimum_component_weight": 0.05,
        "joint_min_abs_correlation": 0.8,
    },
    "log_parameters": [
        "cutoff_frequency_hz",
        "attack_ms",
        "release_ms",
        "ratio",
        "q",
    ],
    "domains": {
        "cutoff_frequency_hz": [
            20.0,
            20000.0,
        ],
        "gain_db": [
            -60.0,
            24.0,
        ],
        "threshold_db": [
            -40.0,
            -10.0,
        ],
        "ratio": [
            1.0,
            10.0,
        ],
        "q": [
            0.1,
            50.0,
        ],
        "attack_ms": [
            0.01,
            2000.0,
        ],
        "release_ms": [
            0.01,
            5000.0,
        ],
    },
    "blocked_controls": {
        "socialfx": [
            "attack_ms",
            "release_ms",
        ],
    },
    "blocked_control_reason": "SocialFX timing fields need source-unit verification; release_ms spans 0.0001 to 0.05 in this mirror. No scale factor is inferred from the range.",
}

# Mappings
MAPPINGS = {
    "schema_version": 1,
    "rules": [
        {
            "id": "mixparams_gain",
            "adapter": "mixparams",
            "plugin": "gain",
            "operator": "apply_gain",
            "required": [
                "gain_db",
            ],
            "parameters": {
                "gain_db": {
                    "path": "gain",
                    "unit": "dB",
                },
            },
            "evidence": "rebuttal/analysis/external/mixparams_analysis.py UNIT_BY_PARAMETER",
        },
        {
            "id": "mixparams_compression",
            "adapter": "mixparams",
            "plugin": "compression",
            "operator": "apply_compressor_effect",
            "required": [
                "threshold_db",
                "ratio",
                "attack_ms",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "paths": [
                        "threshold",
                        "thresh",
                        "thres",
                    ],
                    "unit": "dB",
                },
                "ratio": {
                    "path": "ratio",
                    "unit": "ratio",
                    "min": 1.0,
                },
            },
            "evidence": "Known threshold and ratio semantics; mixed-plugin attack/release units unresolved",
        },
        {
            "id": "mixparams_gate",
            "adapter": "mixparams",
            "plugin": "gate",
            "operator": "apply_noisegate",
            "required": [
                "threshold_db",
                "ratio",
                "attack_ms",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "paths": [
                        "threshold",
                        "thresh",
                        "thres",
                    ],
                    "unit": "dB",
                },
                "ratio": {
                    "path": "ratio",
                    "unit": "ratio",
                    "min": 1.0,
                },
            },
            "evidence": "Timing controls remain source-native until units are established",
        },
        {
            "id": "mixparams_limiter",
            "adapter": "mixparams",
            "plugin": "limiter",
            "operator": "apply_limiter_effect",
            "required": [
                "threshold_db",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "paths": [
                        "threshold",
                        "thresh",
                        "thres",
                    ],
                    "unit": "dB",
                },
            },
            "evidence": "Release units remain source-native",
        },
        {
            "id": "mixparams_highshelf",
            "adapter": "mixparams",
            "plugin": "eq",
            "subtypes": [
                "HS",
            ],
            "operator": "apply_highshelf_filter",
            "required": [
                "cutoff_frequency_hz",
                "gain_db",
                "q",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "value/freq",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
                "gain_db": {
                    "path": "value/gain",
                    "unit": "dB",
                },
                "q": {
                    "path": "value/q",
                    "unit": "Q",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "MixParams EQ subtype and value fields",
        },
        {
            "id": "mixparams_lowshelf",
            "adapter": "mixparams",
            "plugin": "eq",
            "subtypes": [
                "LS",
            ],
            "operator": "apply_lowshelf_filter",
            "required": [
                "cutoff_frequency_hz",
                "gain_db",
                "q",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "value/freq",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
                "gain_db": {
                    "path": "value/gain",
                    "unit": "dB",
                },
                "q": {
                    "path": "value/q",
                    "unit": "Q",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "MixParams EQ subtype and value fields",
        },
        {
            "id": "mixparams_highpass",
            "adapter": "mixparams",
            "plugin": "eq",
            "subtypes": [
                "HP",
            ],
            "operator": "apply_highpass_filter",
            "required": [
                "cutoff_frequency_hz",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "value/freq",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "Cutoff projection; reported slope cannot be represented by this RIME operator",
        },
        {
            "id": "mixparams_lowpass",
            "adapter": "mixparams",
            "plugin": "eq",
            "subtypes": [
                "LP",
            ],
            "operator": "apply_lowpass_filter",
            "required": [
                "cutoff_frequency_hz",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "value/freq",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "Cutoff projection; reported slope cannot be represented by this RIME operator",
        },
        {
            "id": "calf_compression",
            "adapter": "calf",
            "plugin": "monocompressor",
            "operator": "apply_compressor_effect",
            "required": [
                "threshold_db",
                "ratio",
                "attack_ms",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "path": "threshold",
                    "unit": "dB",
                    "transform": "amplitude_to_db",
                },
                "ratio": {
                    "path": "ratio",
                    "unit": "ratio",
                    "min": 1.0,
                    "max": 20.0,
                },
                "attack_ms": {
                    "path": "attack",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
                "release_ms": {
                    "path": "release",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "Calf src/metadata.cpp CALF_PORT_PROPS(monocompressor); gain-scale threshold, millisecond time constants; ratio 21 is an infinity sentinel",
        },
        {
            "id": "socialfx_compression",
            "adapter": "socialfx",
            "plugin": "comp",
            "operator": "apply_compressor_effect",
            "required": [
                "threshold_db",
                "ratio",
                "attack_ms",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "path": "threshold_db",
                    "unit": "dB",
                },
                "ratio": {
                    "path": "ratio",
                    "unit": "ratio",
                    "min": 1.0,
                },
                "attack_ms": {
                    "path": "attack_ms",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
                "release_ms": {
                    "path": "release_ms",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "Named comp controls; attack/release suffixes conflict with observed ranges and are blocked by fit_policy.yaml pending source-unit verification",
        },
        {
            "id": "easyeffects_compression",
            "adapter": "easyeffects",
            "plugin": "compressor",
            "equals": {
                "mode": "Downward",
            },
            "operator": "apply_compressor_effect",
            "required": [
                "threshold_db",
                "ratio",
                "attack_ms",
                "release_ms",
            ],
            "parameters": {
                "threshold_db": {
                    "path": "threshold",
                    "unit": "dB",
                },
                "ratio": {
                    "path": "ratio",
                    "unit": "ratio",
                    "min": 1.0,
                },
                "attack_ms": {
                    "path": "attack",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
                "release_ms": {
                    "path": "release",
                    "unit": "ms",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "EasyEffects schema and src/contents/ui/Compressor.qml use dB, ratio and milliseconds; downward mode required",
        },
        {
            "id": "easyeffects_bell",
            "adapter": "easyeffects",
            "plugin": "eq_band",
            "subtypes": [
                "Bell",
            ],
            "operator": "apply_peak_filter",
            "required": [
                "cutoff_frequency_hz",
                "gain_db",
                "q",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "frequency",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
                "gain_db": {
                    "path": "gain",
                    "unit": "dB",
                },
                "q": {
                    "path": "q",
                    "unit": "Q",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "Named EasyEffects band controls; RIME uses its own filter implementation",
        },
        {
            "id": "easyeffects_highshelf",
            "adapter": "easyeffects",
            "plugin": "eq_band",
            "subtypes": [
                "Hi-shelf",
            ],
            "operator": "apply_highshelf_filter",
            "required": [
                "cutoff_frequency_hz",
                "gain_db",
                "q",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "frequency",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
                "gain_db": {
                    "path": "gain",
                    "unit": "dB",
                },
                "q": {
                    "path": "q",
                    "unit": "Q",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "EasyEffects shelf control projection",
        },
        {
            "id": "easyeffects_lowshelf",
            "adapter": "easyeffects",
            "plugin": "eq_band",
            "subtypes": [
                "Lo-shelf",
            ],
            "operator": "apply_lowshelf_filter",
            "required": [
                "cutoff_frequency_hz",
                "gain_db",
                "q",
            ],
            "parameters": {
                "cutoff_frequency_hz": {
                    "path": "frequency",
                    "unit": "Hz",
                    "exclusive_min": 0.0,
                },
                "gain_db": {
                    "path": "gain",
                    "unit": "dB",
                },
                "q": {
                    "path": "q",
                    "unit": "Q",
                    "exclusive_min": 0.0,
                },
            },
            "evidence": "EasyEffects shelf control projection",
        },
        {
            "id": "easyeffects_eq_gain",
            "adapter": "easyeffects",
            "plugin": "eq_gain",
            "operator": "apply_gain",
            "required": [
                "gain_db",
            ],
            "parameters": {
                "gain_db": {
                    "path": "gain",
                    "unit": "dB",
                },
            },
            "evidence": "EasyEffects input/output gain surrounding the EQ band cascade",
        },
    ],
    "instrument_families": {
        "vocal": "vocals",
        "vocals": "vocals",
        "voice": "vocals",
        "singing": "vocals",
        "drums": "drums",
        "drum": "drums",
        "percussion": "drums",
        "bass": "bass",
        "guitar": "guitar",
        "guitars": "guitar",
        "piano": "piano",
        "keys": "keys",
        "keyboard": "keys",
    },
}

# Lexicon
LEXICON = {
    "terms": {
        "gain": [
            "gain",
            "volume",
            "louder",
            "quieter",
            "bring up",
            "turn down",
        ],
        "eq": [
            "eq",
            "equalizer",
            "equalization",
            "equalisation",
            "high pass",
            "low pass",
            "high shelf",
            "low shelf",
        ],
        "compression": [
            "compress",
            "compressor",
            "compression",
            "compressed",
        ],
        "limiter": [
            "limiter",
            "limiting",
        ],
        "gate": [
            "gate",
            "gating",
            "expander",
        ],
        "reverb": [
            "reverb",
            "reverberation",
            "ambience",
        ],
        "delay": [
            "delay",
            "echo",
            "echoes",
            "slapback",
        ],
        "distortion": [
            "distortion",
            "saturation",
            "overdrive",
        ],
        "modulation": [
            "chorus",
            "phaser",
            "flanger",
        ],
        "stereo": [
            "pan",
            "panning",
            "stereo",
            "width",
        ],
    },
}

# Published chains
PUBLISHED_CHAINS = {
    "source": "dafx_2017",
    "url": "https://www.dafx.de/paper-archive/2017/papers/DAFx17_paper_75.pdf",
    "transcription_origin": "rebuttal/analysis/external/dafx_analysis.py",
    "method": "Versioned manual transcription of published aggregates; no PDF OCR or synthetic individual records",
    "submissions_n": 178,
    "participants_n": 47,
    "support_complete": False,
    "percentages_are_rounded": True,
    "chains": [
        {
            "effects": [
                "eq",
            ],
            "submission_percent": 27.5,
        },
        {
            "effects": [
                "reverb",
            ],
            "submission_percent": 12.5,
        },
        {
            "effects": [
                "compression",
                "eq",
            ],
            "submission_percent": 11.9,
        },
        {
            "effects": [
                "distortion",
            ],
            "submission_percent": 8.9,
        },
        {
            "effects": [
                "eq",
                "compression",
            ],
            "submission_percent": 8.9,
        },
        {
            "effects": [
                "eq",
                "reverb",
            ],
            "submission_percent": 5.3,
        },
    ],
    "length_counts": {
        "1": 90,
        "2": 64,
        "3_or_4": 24,
    },
    "effect_instances": {
        "eq": 124,
        "compression": 72,
        "reverb": 57,
        "distortion": 40,
    },
    "position_probabilities": {
        "eq": [
            0.44,
            0.43,
            0.21,
            0.33,
        ],
        "compression": [
            0.22,
            0.28,
            0.25,
            0.33,
        ],
        "distortion": [
            0.15,
            0.1,
            0.16,
            0.0,
        ],
        "reverb": [
            0.17,
            0.18,
            0.38,
            0.33,
        ],
    },
}


if __name__ == "__main__":
    main()
