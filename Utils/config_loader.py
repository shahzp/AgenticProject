# config_loader.py
import json
import os

CONFIG_FILE = "config.json"


def load_rag_config(strategy_name: str) -> dict:
    """
    Loads config.json and merges global settings with the selected strategy block.
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file missing: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    global_cfg = data.get("global_settings", {})
    strategy_cfg = data.get("strategies", {}).get(strategy_name, {})

    if not strategy_cfg:
        raise ValueError(f"Strategy '{strategy_name}' not defined in config.json")

    # Merge dictionaries. Strategy-specific overrides global if keys match.
    merged_config = {**global_cfg, **strategy_cfg, "active_strategy": strategy_name}
    return merged_config
