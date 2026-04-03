"""Runtime configuration for the standalone reviewAI service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"
# Local development fallback when REVIEWAI_MODEL_ROOT is not set explicitly.
DEFAULT_MODEL_ROOT = (PROJECT_ROOT.parent / "reviewAI-model").resolve()


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _parse_bool(raw_value: str | None, *, default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(raw_value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw_value is None or not raw_value.strip():
        return default
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _getenv_stripped(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    return stripped or None


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    host: str
    port: int
    log_level: str
    device: str
    batch_size: int
    model_root: Path
    manifest_path: Path
    allowed_origins: tuple[str, ...]
    eager_load_model: bool = False

    @classmethod
    def from_env(cls, dotenv_path: Path = DEFAULT_DOTENV_PATH) -> "Settings":
        _load_dotenv(dotenv_path)

        model_root = Path(_getenv_stripped("REVIEWAI_MODEL_ROOT") or str(DEFAULT_MODEL_ROOT)).expanduser()
        if not model_root.is_absolute():
            model_root = (PROJECT_ROOT / model_root).resolve()

        manifest_override = _getenv_stripped("REVIEWAI_MANIFEST_PATH")
        manifest_path = (
            Path(manifest_override).expanduser()
            if manifest_override
            else model_root / "artifacts" / "final_model_manifest.json"
        )
        if not manifest_path.is_absolute():
            manifest_path = (PROJECT_ROOT / manifest_path).resolve()

        return cls(
            app_name="reviewAI",
            host=os.getenv("REVIEWAI_HOST", "127.0.0.1"),
            port=int(os.getenv("REVIEWAI_PORT", "8008")),
            log_level=os.getenv("REVIEWAI_LOG_LEVEL", "INFO").upper(),
            device=os.getenv("REVIEWAI_DEVICE", "auto"),
            batch_size=int(os.getenv("REVIEWAI_BATCH_SIZE", "16")),
            model_root=model_root.resolve(),
            manifest_path=manifest_path.resolve(),
            allowed_origins=_parse_csv(
                os.getenv("REVIEWAI_ALLOWED_ORIGINS"),
                default=("*",),
            ),
            eager_load_model=_parse_bool(
                os.getenv("REVIEWAI_EAGER_LOAD_MODEL"),
                default=False,
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
