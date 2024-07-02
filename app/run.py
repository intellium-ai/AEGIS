import logging
import sys
from pathlib import Path

from primaite.main import run

config_path = Path("../src/primaite/config/_package_data/")
training_config_path = config_path / "training" / "trainable_llm.yaml"
lay_down_config_path = config_path / "lay_down" / "lay_down_config_5_data_manipulation.yaml"

run(
    training_config_path=training_config_path,
    lay_down_config_path=lay_down_config_path,
)
