"""Locations shared by the app and its analysis scripts."""

from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent
TEMPLATES_PATH = PAGES_DIR / "templates"
DATA_PATH = PAGES_DIR / "data" / "mindwareData"
SCRIPTS_PATH = PAGES_DIR / "scripts"
LOG_PATH = PAGES_DIR / "logs"


def project_names():
    return sorted(path.stem.removesuffix("_template") for path in TEMPLATES_PATH.glob("*_template.json"))
