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


PROVIDERS: tuple[str, ...] = ("anthropic", "openai")
THINKING: tuple[str, ...] = ("", "enabled", "disabled")


@dataclass
class AgentSettings:
    """The planning LLM. "openai" means any OpenAI-compatible endpoint (OpenAI, OpenRouter, DeepSeek, Ollama, vLLM...)."""

    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    base_url: str = ""
    api_key_env: str = "ANTHROPIC_API_KEY"
    # OpenAI-compatible only (DeepSeek); empty leaves the provider default. Anthropic ignores both.
    thinking: str = ""
    reasoning_effort: str = ""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path(".ah"))
    tools_root: Path = REPO_ROOT / "tools"
    device: str = "auto"
    backends: dict[str, str] = field(default_factory=dict)
    remote: RemoteSettings = field(default_factory=RemoteSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

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
        agent = raw.get("agent", {})
        settings.agent = AgentSettings(**{k: str(v) for k, v in agent.items() if k in AgentSettings.__dataclass_fields__})
    for key in ("provider", "model", "base_url", "api_key_env", "thinking", "reasoning_effort"):
        if os.environ.get(f"AH_AGENT_{key.upper()}"):
            setattr(settings.agent, key, os.environ[f"AH_AGENT_{key.upper()}"])
    if settings.agent.provider not in PROVIDERS:
        raise ValueError(f"ah.toml: agent.provider must be one of {PROVIDERS}, got {settings.agent.provider!r}")
    if settings.agent.thinking not in THINKING:
        raise ValueError(f"ah.toml: agent.thinking must be one of {THINKING}, got {settings.agent.thinking!r}")
    if os.environ.get("AH_HOME"):
        settings.home = Path(os.environ["AH_HOME"])
    if os.environ.get("AH_REMOTE_URL"):
        settings.remote.url = os.environ["AH_REMOTE_URL"]
    for tool, backend in settings.backends.items():
        if backend not in BACKENDS:
            raise ValueError(f"ah.toml: backend for {tool!r} must be one of {BACKENDS}, got {backend!r}")
    return settings
