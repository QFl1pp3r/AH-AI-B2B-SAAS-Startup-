"""Settings from ah.toml (next to the repo or in the working directory) plus environment."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["local", "inprocess", "remote"]
BACKENDS: tuple[str, ...] = ("local", "inprocess", "remote")

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RemoteSettings:
    url: str = ""
    token_env: str = "AH_REMOTE_TOKEN"
    timeout_s: float = 900.0

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "")


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path(".ah"))
    tools_root: Path = REPO_ROOT / "tools"
    device: str = "auto"
    backends: dict[str, str] = field(default_factory=dict)
    remote: RemoteSettings = field(default_factory=RemoteSettings)

    def backend_for(self, tool: str) -> str:
        return self.backends.get(tool, "local")


def _find_config() -> Path | None:
    explicit = os.environ.get("AH_CONFIG")
    if explicit:
        return Path(explicit)
    for candidate in (Path.cwd() / "ah.toml", REPO_ROOT / "ah.toml"):
        if candidate.is_file():
            return candidate
    return None


def load_settings() -> Settings:
    settings = Settings()
    path = _find_config()
    if path is not None:
        raw = tomllib.loads(path.read_text())
        base = path.parent
        if "home" in raw:
            settings.home = base / raw["home"]
        if "tools_root" in raw:
            settings.tools_root = base / raw["tools_root"]
        settings.device = raw.get("device", settings.device)
        settings.backends = dict(raw.get("backends", {}))
        remote = raw.get("remote", {})
        settings.remote = RemoteSettings(
            url=remote.get("url", ""),
            token_env=remote.get("token_env", "AH_REMOTE_TOKEN"),
            timeout_s=float(remote.get("timeout_s", 900)),
        )
    if os.environ.get("AH_HOME"):
        settings.home = Path(os.environ["AH_HOME"])
    if os.environ.get("AH_REMOTE_URL"):
        settings.remote.url = os.environ["AH_REMOTE_URL"]
    for tool, backend in settings.backends.items():
        if backend not in BACKENDS:
            raise ValueError(f"ah.toml: backend for {tool!r} must be one of {BACKENDS}, got {backend!r}")
    return settings
