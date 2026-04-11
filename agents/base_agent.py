"""
Base agent class. All trading bot agents inherit from this.

Provides:
- Timestamped file logger → logs/{name}.log
- Atomic JSON read/write (os.replace via .tmp file)
- Shared state read/write helpers
- Lifecycle markers: mark_running, mark_done, mark_error
"""

import json
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
SHARED_STATE_PATH = BASE_DIR / "shared_state.json"
CONFIG_PATH = BASE_DIR / "config.json"


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.config = self._load_config()
        self.logger = self._setup_logger()

    def _load_config(self) -> dict:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def _setup_logger(self) -> logging.Logger:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # File handler
            fh = logging.FileHandler(LOGS_DIR / f"{self.name}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)

        return logger

    # ------------------------------------------------------------------ #
    # JSON I/O                                                             #
    # ------------------------------------------------------------------ #

    def read_json(self, filepath: pathlib.Path) -> dict:
        """Read a JSON file. Returns {} if file is missing or corrupt."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write_json(self, filepath: pathlib.Path, data: dict) -> None:
        """Atomic write: write to .tmp then os.replace() to avoid partial reads."""
        filepath = pathlib.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)

    # ------------------------------------------------------------------ #
    # Shared state helpers                                                 #
    # ------------------------------------------------------------------ #

    def update_shared_state(self, key_path: str, value) -> None:
        """
        Read shared_state.json, set a nested key via dot-notation, write back.
        Example: update_shared_state("agents.market_data_agent.status", "running")
        """
        state = self.read_json(SHARED_STATE_PATH)
        keys = key_path.split(".")
        node = state
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        self.write_json(SHARED_STATE_PATH, state)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # Lifecycle markers                                                    #
    # ------------------------------------------------------------------ #

    def mark_running(self) -> None:
        self.update_shared_state(f"agents.{self.name}.status", "running")
        self.update_shared_state(f"agents.{self.name}.last_run", self._now_iso())
        self.update_shared_state(f"agents.{self.name}.last_error", None)
        self.logger.info("Starting")

    def mark_done(self) -> None:
        self.update_shared_state(f"agents.{self.name}.status", "idle")
        self.logger.info("Done")

    def mark_error(self, err: Exception) -> None:
        self.update_shared_state(f"agents.{self.name}.status", "error")
        self.update_shared_state(f"agents.{self.name}.last_error", str(err))
        self.logger.error(f"Error: {err}")

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run(self) -> bool:
        """Execute one cycle. Return True on success, False on failure."""
        ...
