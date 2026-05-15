import json
from pathlib import Path
from typing import Dict, Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_json(filename: str) -> Dict[str, Any]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_knowledge_base() -> Dict[str, Any]:
    return load_json("knowledge_base.json")


def load_historical_patterns() -> Dict[str, Any]:
    return load_json("historical_patterns.json")