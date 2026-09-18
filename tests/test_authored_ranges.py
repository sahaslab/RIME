import sys
import random
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ground_truth.planner import GroundTruthPlanner
from export_priors_to_yaml import apply_authored_ranges


def test_export_restores_authored_ranges() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = {
        name: yaml.safe_load((root / "prior_baseline" / name).read_text())
        for name in ["distributions.yaml", "motifs.yaml"]
    }
    apply_authored_ranges(config)
    active = yaml.safe_load((root / "ground_truth/distributions.yaml").read_text())["distributions"]
    exported = config["distributions.yaml"]["distributions"]
    assert exported["tone"]["modulation"]["chorus"]["centre_delay_ms"] == {
        "type": "uniform", "low": 20.0, "high": 50.0,
    }
    assert exported["tone"]["modulation"]["chorus"]["centre_delay_ms"] == active["tone"]["modulation"]["chorus"]["centre_delay_ms"]
    params = config["motifs.yaml"]["motifs"]["modulation_texture"]["steps"][0]["params"]
    assert params["centre_delay_ms"] == {"sample": "tone.modulation.chorus.centre_delay_ms"}
    assert params["feedback"] == {"sample": "tone.modulation.chorus.feedback"}


def test_random_effects_vary_within_new_ranges() -> None:
    config = Path(__file__).resolve().parents[1] / "configs/ground_truth"
    planner = GroundTruthPlanner.from_directory(config)
    expected = {
        "apply_chorus_effect": {"centre_delay_ms": (20.0, 50.0), "feedback": (0.0, 0.3)},
        "apply_compressor_effect": {"attack_ms": (2.0, 10.0), "release_ms": (40.0, 100.0)},
        "apply_highshelf_filter": {"q": (0.5, 1.0)},
        "apply_lowshelf_filter": {"q": (0.5, 1.0)},
        "apply_limiter_effect": {"threshold_db": (-12.0, -6.0), "release_ms": (60.0, 150.0)},
        "apply_delay_effect": {"mix": (0.3, 0.6)},
        "apply_reverb_effect": {"wet_level": (0.4, 0.7), "dry_level": (0.5, 1.0), "width": (0.5, 1.0)},
    }
    for operator, bounds in tqdm(expected.items(), desc="Checking effect ranges"):
        draws = [
            planner._random_step_for_operator(
                operator,
                {},
                {"family": "drums"},
                random.Random(seed),
                1,
            )["params"]
            for seed in range(100)
        ]
        for parameter, (low, high) in bounds.items():
            values = {draw[parameter] for draw in draws}
            assert len(values) > 1, (operator, parameter)
            assert all(low <= value <= high for value in values)
        if operator == "apply_reverb_effect":
            assert all(draw["freeze_mode"] == 0.0 for draw in draws)
