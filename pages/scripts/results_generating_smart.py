"""Run the appropriate NeuroKit analysis for a project."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pages.data_manager import DataManager
from pages.paths import DATA_PATH, PAGES_DIR

logger = logging.getLogger(__name__)


def main(project):
    manager = DataManager(project)
    if manager.has_signal_data() and Path(manager.events_parquet_path).is_file():
        from pages.scripts.results_generating_fibro_update_polars_no_hdf5 import main as analyze
        return analyze(project)
    if (DATA_PATH / f"{project}_mindware.hdf5").is_file():
        return subprocess.run(
            [sys.executable, "-u", "-m", "pages.scripts.results_generating_fibro_update_polars", project],
            cwd=PAGES_DIR.parent,
        ).returncode
    logger.error("No signals and events are available for %s. Import recordings or migrate HDF5 first.", project)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main(args.project))
