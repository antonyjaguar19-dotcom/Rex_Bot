"""
Claw Bot — startup config validation + disk preflight

Catches broken config at boot (second 1) instead of mid-pipeline (hour 2).
A typo in models.json used to surface as a cryptic KeyError halfway through
a render; now the bot refuses to start and says exactly which file and key
is wrong — like a render submit that lints the scene before farm dispatch.

Also exposes check_disk_space() for the per-job preflight: video pipelines
write GBs; running the drive to zero mid-render corrupts the output.
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("claw_bot.config_check")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "05_Config"

MODELS_PATH = CONFIG_DIR / "models.json"
STYLES_PATH = CONFIG_DIR / "styles.json"
RUNTIME_PATH = CONFIG_DIR / "runtime_settings.json"
SECRETS_PATH = CONFIG_DIR / "secrets.env"

MIN_FREE_GB_DEFAULT = 10.0


def _load_json(path: Path, errors: list) -> dict | None:
    if not path.exists():
        errors.append(f"{path.name}: file missing ({path})")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"{path.name}: not valid JSON — {e}")
        return None


def _check_backend_section(models: dict, section: str, errors: list) -> None:
    sec = models.get(section)
    if not isinstance(sec, dict):
        errors.append(f"models.json: missing '{section}' section")
        return
    active = sec.get("active")
    available = sec.get("available")
    if not isinstance(available, dict) or not available:
        errors.append(f"models.json: {section}.available must be a non-empty dict")
        return
    if active not in available:
        errors.append(
            f"models.json: {section}.active is '{active}' but available "
            f"backends are {sorted(available)}"
        )


def validate_configs() -> list[str]:
    """Return a list of fatal config problems (empty list = good to boot)."""
    errors: list[str] = []

    models = _load_json(MODELS_PATH, errors)
    if models is not None:
        for section in ("llm_backend", "image_backend", "video_backend"):
            _check_backend_section(models, section, errors)
        # audio_backend is optional (music-video pipeline); validate only if present.
        if models.get("audio_backend"):
            _check_backend_section(models, "audio_backend", errors)

    styles = _load_json(STYLES_PATH, errors)
    if styles is not None:
        available = styles.get("available")
        if not isinstance(available, dict) or not available:
            errors.append("styles.json: 'available' must be a non-empty dict")
        elif styles.get("default") not in available:
            errors.append(
                f"styles.json: default '{styles.get('default')}' not in "
                f"available styles {sorted(available)}"
            )

    # runtime_settings.json is optional, but if present it must parse —
    # otherwise every get_effective_* silently falls back to defaults.
    if RUNTIME_PATH.exists():
        _load_json(RUNTIME_PATH, errors)

    return errors


def warn_on_secrets_bom() -> bool:
    """True (and a log warning) when secrets.env starts with a UTF-8 BOM.

    PowerShell's `Out-File -Encoding utf8` writes a BOM, and python-dotenv
    then misreads the FIRST line — typically DISCORD_BOT_TOKEN vanishing.
    """
    try:
        if SECRETS_PATH.exists() and SECRETS_PATH.read_bytes()[:3] == b"\xef\xbb\xbf":
            log.warning(
                "secrets.env starts with a UTF-8 BOM — python-dotenv will drop "
                "its first line. Re-save it with a plain editor (no BOM)."
            )
            return True
    except Exception as e:
        log.warning(f"Could not check secrets.env for BOM: {e}")
    return False


def check_disk_space(min_gb: float = MIN_FREE_GB_DEFAULT) -> tuple[bool, float]:
    """(enough, free_gb) for the project drive."""
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024 ** 3)
    return free_gb >= min_gb, round(free_gb, 1)
