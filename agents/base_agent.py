"""BaseAgent — Shared functionality for all trading bot agents.

Provides:
- Config loading from trading_bot/config.json  
- Directory constants (DATA_DIR, LOGS_DIR, etc.)
- Shared state management
- JSON read/write helpers with fallbacks
- Agent lifecycle (mark_running/done/error)
"""

import json
import logging
import os
import pathlib
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
CONFIG_PATH = BASE_DIR / "config.json"
SHARED_STATE_PATH = DATA_DIR / "shared_state.json"
IK_PATH = BASE_DIR / "institutional_knowledge.json"


class BaseAgent:
    _ik_cache: dict | None = None
    _shared_state_lock: threading.Lock = threading.Lock()

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self._ik = self.__class__._load_ik()

    @classmethod
    def _load_ik(cls) -> dict:
        if cls._ik_cache is None:
            try:
                import json as _json
                cls._ik_cache = _json.loads(IK_PATH.read_text(encoding="utf-8"))
            except Exception:
                cls._ik_cache = {"roles": {}, "assets": {}}
        return cls._ik_cache

    def get_asset_role(self, symbol: str) -> str:
        return self._ik.get("assets", {}).get(symbol, "unknown")

    def get_role_config(self, symbol: str) -> dict:
        role = self.get_asset_role(symbol)
        return self._ik.get("roles", {}).get(role, {})
        
        # Ensure directories exist
        for d in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"agent.{self.name}")
        if not logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh = logging.FileHandler(LOGS_DIR / f"{self.name}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Config load failed: {e}. Using minimal config.")
            return {}
    
    # ------------------------------------------------------------------
    # Data I/O with fallbacks
    def read_json(self, path: pathlib.Path) -> dict:
        """Read JSON with corrupt-file recovery via .bak fallback.

        Priority:
          1. path          — primary file, must be valid JSON
          2. path.bak      — last known-good snapshot written by write_json
          3. {}            — empty sentinel (logged as warning)
        """
        primary = pathlib.Path(path)
        backup  = primary.with_suffix(primary.suffix + ".bak")

        def _try_read(p: pathlib.Path) -> dict | None:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                self.logger.warning(f"JSON read failed {p}: {exc}")
                return None

        data = _try_read(primary)
        if data is not None:
            return data

        data = _try_read(backup)
        if data is not None:
            self.logger.warning(f"Primary {primary.name} corrupt — restored from .bak")
            return data

        return {}

    def write_json(self, path: pathlib.Path, data: Dict) -> bool:
        """Atomic write via tmp file; updates .bak on success."""
        path = pathlib.Path(path)
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_str, str(path))
            except Exception:
                try:
                    os.unlink(tmp_str)
                except OSError:
                    pass
                raise
            # Promote current file to backup after confirmed write
            try:
                import shutil
                shutil.copy2(str(path), str(backup))
            except OSError as exc:
                self.logger.warning(f"Could not update .bak for {path.name}: {exc}")
            return True
        except Exception as e:
            self.logger.error(f"JSON write failed {path}: {e}")
            return False
    
    def update_shared_state(self, key: str, value: Any) -> None:
        """Update specific shared_state.json field atomically (thread-safe)."""
        with self.__class__._shared_state_lock:
            try:
                state = self.read_json(SHARED_STATE_PATH) or {}
                if "data_freshness" not in state:
                    state["data_freshness"] = {}

                keys = key.split(".")
                d = state
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value

                self.write_json(SHARED_STATE_PATH, state)
            except Exception as e:
                self.logger.error(f"Shared state update failed {key}: {e}")
    
    # ------------------------------------------------------------------
    # Agent lifecycle
    def require_write(self, path: pathlib.Path, data: Dict) -> None:
        """Like write_json but raises RuntimeError on failure — use for critical outputs."""
        ok = self.write_json(path, data)
        if not ok:
            raise RuntimeError(f"Critical write failed: {path} — agent cannot mark_done safely")

    def mark_running(self) -> None:
        self.logger.info(f"[{self.name}] START")

    def mark_done(self) -> None:
        self.logger.info(f"[{self.name}] DONE")

    def mark_error(self, exc: Exception) -> None:
        self.logger.error(f"[{self.name}] ERROR: {exc}", exc_info=True)
