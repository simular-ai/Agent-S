"""Validated UI configuration, provider adapters, and private persistence."""

import base64
import json
import os
import threading
import secrets
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gui_agents.s3.core.openai_compatible import (
    LMSTUDIO_DEFAULT_URL,
    PROVIDER_PRESETS,
    is_local_url,
    normalize_base_url,
)

Provider = Literal[
    "openai",
    "lmstudio",
    "openai_compatible",
    "ollama",
    "vllm",
    "open_router",
    "anthropic",
    "gemini",
    "huggingface",
    "azure",
    "deepseek",
    "qwen",
    "codex",
]
Approval = Literal["auto", "manual", "dry_run"]
PROVIDERS = list(Provider.__args__)
DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "lmstudio": LMSTUDIO_DEFAULT_URL,
    "openai_compatible": LMSTUDIO_DEFAULT_URL,
    "ollama": "http://localhost:11434/v1",
    "open_router": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "vllm": "vLLM_API_KEY",
    "open_router": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
    "azure": "AZURE_OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "codex": "",
}
CODEX_DEFAULT_MODEL = "gpt-6-astra"
URL_ENVS = {
    "ollama": "OLLAMA_HOST",
    "vllm": "vLLM_ENDPOINT_URL",
    "huggingface": "HF_ENDPOINT_URL",
    "azure": "AZURE_OPENAI_ENDPOINT",
}


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    provider: Provider = "lmstudio"
    model: str = Field(default="", max_length=256)
    profile_id: str = Field(default="", max_length=64)
    profile_name: str = Field(default="", max_length=64)
    model_url: str = LMSTUDIO_DEFAULT_URL
    model_api_key: str = Field(default="", max_length=4096, repr=False)
    model_temperature: float = Field(default=0.0, ge=0, le=2)
    ground_provider: Provider = "lmstudio"
    ground_model: str = Field(default="", max_length=256)
    ground_url: str = LMSTUDIO_DEFAULT_URL
    ground_api_key: str = Field(default="", max_length=4096, repr=False)
    azure_api_version: str = ""
    ground_azure_api_version: str = ""
    grounding_width: int = Field(default=1920, ge=1, le=8192)
    grounding_height: int = Field(default=1080, ge=1, le=8192)
    max_steps: int = Field(default=15, ge=1, le=60)
    max_trajectory_length: int = Field(default=4, ge=1, le=30)
    enable_reflection: bool = True
    enable_local_env: bool = False
    approval: Approval = "dry_run"
    overlay: bool = False
    monitor: int = Field(default=1, ge=1, le=32)
    inference_timeout: int = Field(default=60, ge=5, le=300)
    task_timeout: int = Field(default=600, ge=10, le=3600)
    action_timeout: int = Field(default=60, ge=5, le=300)

    @field_validator("model_url", "ground_url")
    @classmethod
    def endpoint(cls, value):
        if value:
            normalize_base_url(value)
        return value.strip().rstrip("/")

    def public(self):
        data = self.model_dump(exclude={"model_api_key", "ground_api_key"})
        provider_env = KEY_ENVS.get(self.provider, "")
        ground_env = KEY_ENVS.get(self.ground_provider, "")
        data["model_api_key_configured"] = bool(
            self.model_api_key or (provider_env and os.getenv(provider_env))
        )
        data["ground_api_key_configured"] = bool(
            self.ground_api_key or (ground_env and os.getenv(ground_env))
        )
        return data


def state_directory():
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "AgentS"
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "agent-s"


def default_config_path():
    return Path(
        os.getenv("AGENT_S_UI_CONFIG", str(state_directory() / "ui-config.json"))
    )


def server_token(create=True):
    explicit = os.getenv("AGENT_S_UI_TOKEN")
    if explicit:
        return explicit
    path = default_config_path().with_name("ui-token")
    if not path.exists() and create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as file:
                if os.name != "nt":
                    os.chmod(path, 0o600)
                file.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def protect(value: str) -> str:
    if not value or os.name != "nt":
        return value
    import win32crypt

    return (
        "dpapi:"
        + base64.b64encode(
            win32crypt.CryptProtectData(
                value.encode(), "Agent S backend credential", None, None, None, 0
            )
        ).decode()
    )


def unprotect(value: str) -> str:
    if not value.startswith("dpapi:"):
        return value
    import win32crypt

    return win32crypt.CryptUnprotectData(
        base64.b64decode(value[6:]), None, None, None, 0
    )[1].decode()


class ConfigStore:
    SCHEMA_VERSION = 2

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else default_config_path()
        self.lock = threading.RLock()
        self.profiles: dict[str, AgentConfig] = {}
        self.profile_order: list[str] = []
        self.active_profile_id = ""
        if self.path.exists():
            self._load()
        if not self.profiles:
            profile = AgentConfig(
                profile_id="default", profile_name="Default"
            )
            self.profiles[profile.profile_id] = profile
            self.profile_order.append(profile.profile_id)
            self.active_profile_id = profile.profile_id
            self._save_locked()

    def _load(self):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "profiles" in raw:
            order = raw.get("profile_order") or list(raw["profiles"].keys())
            for profile_id in order:
                entry = raw["profiles"].get(profile_id)
                if not isinstance(entry, dict):
                    continue
                data = dict(entry.get("config", entry))
                for key in ("model_api_key", "ground_api_key"):
                    data[key] = unprotect(data.get(key, ""))
                data["profile_id"] = profile_id
                data["profile_name"] = entry.get("name", data.get("profile_name", profile_id))
                self.profiles[profile_id] = AgentConfig.model_validate(data)
                self.profile_order.append(profile_id)
            active = raw.get("active_profile_id", "")
            self.active_profile_id = (
                active if active in self.profiles else (self.profile_order[0] if self.profile_order else "")
            )
            return
        for key in ("model_api_key", "ground_api_key"):
            raw[key] = unprotect(raw.get(key, ""))
        raw["profile_id"] = "default"
        raw["profile_name"] = "Default"
        profile = AgentConfig.model_validate(raw)
        self.profiles[profile.profile_id] = profile
        self.profile_order = [profile.profile_id]
        self.active_profile_id = profile.profile_id
        self._save_locked()

    def _save_locked(self):
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "active_profile_id": self.active_profile_id,
            "profile_order": list(self.profile_order),
            "profiles": {},
        }
        for profile_id in self.profile_order:
            profile = self.profiles[profile_id]
            saved = profile.model_dump()
            for key in ("model_api_key", "ground_api_key"):
                saved[key] = protect(saved[key])
            payload["profiles"][profile_id] = {
                "name": profile.profile_name or profile_id,
                "config": saved,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as file:
                if os.name != "nt":
                    os.chmod(temp, 0o600)
                json.dump(payload, file, indent=2)
            temp.replace(self.path)
        finally:
            temp.unlink(missing_ok=True)

    def snapshot(self, profile_id=""):
        with self.lock:
            target = profile_id or self.active_profile_id
            return self.profiles[target].model_copy(deep=True)

    def list_profiles(self):
        with self.lock:
            return {
                "active_profile_id": self.active_profile_id,
                "profiles": [
                    {
                        "id": profile_id,
                        "name": self.profiles[profile_id].profile_name or profile_id,
                        "provider": self.profiles[profile_id].provider,
                        "model": self.profiles[profile_id].model,
                        "ground_provider": self.profiles[profile_id].ground_provider,
                        "ground_model": self.profiles[profile_id].ground_model,
                        "active": profile_id == self.active_profile_id,
                    }
                    for profile_id in self.profile_order
                ],
            }

    def _new_profile_id(self, name):
        base = "".join(
            ch.lower() if ch.isalnum() else "-" for ch in name.strip().lower()
        ).strip("-") or "profile"
        candidate, index = base[:48], 2
        while candidate in self.profiles:
            suffix = f"-{index}"
            candidate = (base[: 48 - len(suffix)] + suffix) if base else f"profile{suffix}"
            index += 1
        return candidate

    def create_profile(self, name="", source_id=""):
        with self.lock:
            clean = (name or "").strip()[:64] or "New profile"
            source = source_id or self.active_profile_id
            template = self.profiles[source].model_dump()
            profile_id = self._new_profile_id(clean)
            template.update(
                {"profile_id": profile_id, "profile_name": clean}
            )
            self.profiles[profile_id] = AgentConfig.model_validate(template)
            self.profile_order.append(profile_id)
            self.active_profile_id = profile_id
            self._save_locked()
            return {"active_profile_id": profile_id, "config": self.snapshot().public()}

    def duplicate_profile(self, profile_id=""):
        with self.lock:
            source = profile_id or self.active_profile_id
            name = f"{self.profiles[source].profile_name or source} copy"
            return self.create_profile(name, source)

    def rename_profile(self, profile_id, name):
        with self.lock:
            clean = (name or "").strip()
            if not clean:
                raise ValueError("Profile name cannot be empty")
            target = profile_id or self.active_profile_id
            data = self.profiles[target].model_dump()
            data["profile_name"] = clean[:64]
            self.profiles[target] = AgentConfig.model_validate(data)
            self._save_locked()
            return self.list_profiles()

    def delete_profile(self, profile_id):
        with self.lock:
            if len(self.profiles) <= 1:
                raise ValueError("At least one configuration profile is required")
            if profile_id not in self.profiles:
                raise ValueError("Unknown configuration profile")
            del self.profiles[profile_id]
            self.profile_order = [pid for pid in self.profile_order if pid != profile_id]
            if self.active_profile_id == profile_id:
                self.active_profile_id = self.profile_order[0]
            self._save_locked()
            return {
                "active_profile_id": self.active_profile_id,
                "profiles": self.list_profiles()["profiles"],
                "config": self.snapshot().public(),
            }

    def select_profile(self, profile_id):
        with self.lock:
            if profile_id not in self.profiles:
                raise ValueError("Unknown configuration profile")
            self.active_profile_id = profile_id
            self._save_locked()
            return {"active_profile_id": profile_id, "config": self.snapshot().public()}

    def update(self, changes: dict, profile_id=""):
        with self.lock:
            target = profile_id or self.active_profile_id
            data = self.profiles[target].model_dump()
            changes = {k: v for k, v in changes.items() if k not in ("profile_id", "profile_name")}
            for provider, url, key in (
                ("provider", "model_url", "model_api_key"),
                ("ground_provider", "ground_url", "ground_api_key"),
            ):
                if provider in changes and changes[provider] != data[provider]:
                    if changes[provider] == "codex":
                        data[url] = ""
                    else:
                        data[url] = DEFAULT_URLS.get(changes[provider], "")
                    data[key] = ""
                    if provider == "provider" and changes[provider] == "codex" and not changes.get("model"):
                        data["model"] = data.get("model") or CODEX_DEFAULT_MODEL
            data.update(changes)
            candidate = AgentConfig.model_validate(data)
            candidate.profile_id = target
            candidate.profile_name = self.profiles[target].profile_name
            self.profiles[target] = candidate
            self._save_locked()
            return candidate.public()


def backend(config: AgentConfig, aspect="planner", overrides=None):
    data = config.model_dump()
    ground = aspect == "grounding"
    provider = data["ground_provider" if ground else "provider"]
    url = data["ground_url" if ground else "model_url"]
    key = data["ground_api_key" if ground else "model_api_key"]
    model = data["ground_model" if ground else "model"]
    if provider == "codex" and not ground:
        from gui_agents.s3.core.codex_cli import codex_status

        model = (overrides or {}).get("model") or model or CODEX_DEFAULT_MODEL
        status = codex_status(model)
        if not status.get("signed_in"):
            raise ValueError(status.get("error") or "Codex is not signed in; use Login with Codex")
        return {
            "provider": "codex",
            "base_url": "",
            "api_key": "",
            "model": status.get("model") or model,
        }
    if overrides:
        if overrides.get("provider") and overrides["provider"] != provider:
            provider = overrides["provider"]
            url, key = "", ""
        explicit_url = overrides.get("base_url")
        if (
            explicit_url
            and url
            and urlsplit(explicit_url).netloc != urlsplit(url).netloc
            and not overrides.get("api_key")
        ):
            key = ""
        provider = overrides.get("provider") or provider
        url = overrides.get("base_url") or url
        key = overrides.get("api_key") or key
        model = overrides.get("model") or model
    url = (
        url
        or os.getenv(URL_ENVS.get(provider, ""), "")
        or DEFAULT_URLS.get(provider, "")
    )
    key = key or (os.getenv(KEY_ENVS.get(provider, ""), "") if KEY_ENVS.get(provider) else "")
    if provider == "azure":
        version = data[
            "ground_azure_api_version" if ground else "azure_api_version"
        ] or os.getenv("OPENAI_API_VERSION", "")
        if not url or not version:
            raise ValueError("Azure requires an endpoint and API version")
        # Validate without appending the generic /v1 path to Azure's resource root.
        normalize_base_url(url)
    else:
        version = ""
        if provider != "anthropic":
            if not url:
                raise ValueError(f"{provider} requires an endpoint URL")
            url = normalize_base_url(url)
    if not key and url and is_local_url(url) and provider != "anthropic":
        key = "ollama" if provider == "ollama" else "lm-studio"
    if not key:
        raise ValueError(
            f"{provider} requires an API key or its provider environment variable"
        )
    return {
        "provider": provider,
        "base_url": url,
        "api_key": key,
        "model": model,
        "api_version": version,
    }


def engine_params(config, aspect="planner"):
    resolved = backend(config, aspect)
    if resolved["provider"] == "codex":
        result = {
            "engine_type": "codex",
            "model": resolved["model"],
            "base_url": "",
            "api_key": "",
            "timeout": config.inference_timeout,
            "max_retries": 0,
        }
        if aspect == "planner":
            result["temperature"] = config.model_temperature
        return result
    result = {
        "engine_type": resolved["provider"],
        "model": resolved["model"],
        "base_url": resolved["base_url"],
        "api_key": resolved["api_key"],
        "timeout": config.inference_timeout,
        "max_retries": 0,
    }
    if aspect == "planner":
        result["temperature"] = config.model_temperature
    else:
        result.update(
            grounding_width=config.grounding_width,
            grounding_height=config.grounding_height,
        )
    if resolved["provider"] == "azure":
        result.update(
            azure_endpoint=resolved["base_url"], api_version=resolved["api_version"]
        )
    return result


def request_target(resolved, operation, model=""):
    if resolved["provider"] == "codex":
        raise ValueError("Codex models are served through the local Codex CLI")
    if resolved["provider"] == "anthropic":
        raise ValueError(
            "This endpoint supports OpenAI-compatible providers; enter Anthropic model IDs manually"
        )
    if resolved["provider"] == "azure":
        if operation == "models":
            raise ValueError(
                "Enter the Azure deployment ID manually; generic model discovery is not supported"
            )
        deployment = quote(model or resolved["model"], safe="")
        url = f"{resolved['base_url'].rstrip('/')}/openai/deployments/{deployment}/{operation}?api-version={quote(resolved['api_version'], safe='')}"
        headers = {"api-key": resolved["api_key"]}
    else:
        url = resolved["base_url"].rstrip("/") + "/" + operation
        headers = {"Authorization": f"Bearer {resolved['api_key']}"}
    return url, {"Content-Type": "application/json", **headers}


def presets():
    return {
        provider: {
            "base_url": DEFAULT_URLS.get(provider, ""),
            "api_key": "",
            "hint": PROVIDER_PRESETS.get(provider, {}).get(
                "hint", "Configure the provider endpoint and credentials."
            ),
        }
        for provider in PROVIDERS
    }
