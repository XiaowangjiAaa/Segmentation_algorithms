from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_default_config(model: str) -> Dict[str, Any]:
    """Return default config dict for the given model."""
    cfg_path = CONFIG_DIR / f"{model}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No default config for model {model}")
    return load_config(str(cfg_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model name")
    args = parser.parse_args()
    cfg = get_default_config(args.model)
    print(yaml.dump(cfg))
