"""Config loading - a typo in config.yaml should fail loudly, not silently."""

from __future__ import annotations

import pytest
import yaml

from pdfx.config import Config


def test_defaults_are_a_working_configuration():
    cfg = Config()
    assert cfg.ollama.base_url == "http://127.0.0.1:11434"
    assert cfg.ollama.api_mode == "generate", "vision via the OpenAI path is unreliable"
    assert cfg.validation.enabled
    assert cfg.validation.number_locale == "auto"


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "ollama": {"model": "glm-ocr:q8_0", "port": 12345},
                "render": {"dpi": 150},
                "output_dir": "results",
                "hardware_note": "Apple Silicon, 8 GB",
            }
        )
    )
    cfg = Config.load(path)
    assert cfg.ollama.model == "glm-ocr:q8_0"
    assert cfg.ollama.base_url == "http://127.0.0.1:12345"
    assert cfg.render.dpi == 150
    assert cfg.output_dir == "results"
    # Unspecified sections keep their defaults.
    assert cfg.router.min_chars == 60


def test_a_missing_config_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.yaml")


def test_an_unknown_key_is_rejected_with_the_valid_ones(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"ollama": {"modle": "glm-ocr"}}))
    with pytest.raises(ValueError, match="unknown OllamaConfig keys"):
        Config.load(path)


def test_an_unknown_section_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"pipline": {}}))
    with pytest.raises(ValueError, match="unknown top-level config keys"):
        Config.load(path)


def test_the_generated_config_example_is_valid(tmp_path):
    """config.example.yaml must always load - it is the documented starting point."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    assert example.exists(), "config.example.yaml is missing"
    cfg = Config.load(example)
    assert cfg.ollama.model.startswith("glm-ocr")
