import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from openai import AsyncOpenAI


Provider = Literal["openai", "alice", "gemini"]


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    api_key: str = field(repr=False)
    chat_base_url: Optional[str]
    embed_base_url: Optional[str]
    chat_model: str
    embed_model: str


def _env(name: str) -> str:
    v = os.getenv(name)
    return "" if v is None else str(v).strip()


def detect_provider() -> ProviderConfig:
    """Detect which OpenAI-compatible API endpoint to use.

    Selection rule (Gemini-first to match your current setup intent):
    - If GOOGLE_API_KEY is set -> use Gemini OpenAI-compatible endpoint (requires base_url)
    - Else if OPENAI_API_KEY is set -> use OpenAI (no base_url required)
    - Else if ALICE_OPENAI_KEY is set -> use ALICE (requires base_url)
    - Else -> raise

    Model override env vars (optional):
    - CHAT_MODEL (global override)
    - EMBED_MODEL (global override)
    - GEMINI_CHAT_MODEL / GEMINI_EMBED_MODEL
    - GEMINI_BASE_URL (override default base_url)
    - OPENAI_CHAT_MODEL / OPENAI_EMBED_MODEL
    - ALICE_CHAT_MODEL / ALICE_EMBED_MODEL
    """

    # Gemini (OpenAI-compatible) base_url recommended by Google
    default_gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    openai_key = _env("OPENAI_API_KEY")
    alice_key = _env("ALICE_OPENAI_KEY")
    google_key = _env("GOOGLE_API_KEY")

    forced = _env("LLM_PROVIDER").lower()
    if forced:
        if forced not in ("openai", "gemini", "alice"):
            raise RuntimeError("LLM_PROVIDER must be one of: openai, gemini, alice")
        # Enforce that the corresponding key exists.
        if forced == "openai" and not openai_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        if forced == "gemini" and not google_key:
            raise RuntimeError("LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set")
        if forced == "alice" and not alice_key:
            raise RuntimeError("LLM_PROVIDER=alice but ALICE_OPENAI_KEY is not set")

    # If multiple keys are present and no explicit selection is made, fail fast.
    key_count = int(bool(openai_key)) + int(bool(google_key)) + int(bool(alice_key))
    if key_count >= 2 and not forced:
        raise RuntimeError(
            "Multiple provider API keys are set (OPENAI_API_KEY/GOOGLE_API_KEY/ALICE_OPENAI_KEY). "
            "Set LLM_PROVIDER=openai|gemini|alice to choose explicitly."
        )

    if (forced == "gemini") or (not forced and google_key):
        base_url = _env("GEMINI_BASE_URL") or default_gemini_base_url
        chat_model = _env("GEMINI_CHAT_MODEL") or _env("CHAT_MODEL") or "gemini-2.5-flash-lite"
        embed_model = _env("GEMINI_EMBED_MODEL") or _env("EMBED_MODEL") or "gemini-embedding-001"

        # Gemini endpoint is OpenAI-compatible, but model IDs should be Gemini model names.
        # If a user accidentally kept an 'openai/' prefix, strip it to avoid 404s.
        if chat_model.startswith("openai/"):
            chat_model = chat_model[len("openai/"):]
        if embed_model.startswith("openai/"):
            embed_model = embed_model[len("openai/"):]

        return ProviderConfig(
            provider="gemini",
            api_key=google_key,
            chat_base_url=base_url,
            embed_base_url=base_url,
            chat_model=chat_model,
            embed_model=embed_model,
        )

    if (forced == "openai") or (not forced and openai_key):
        chat_model = _env("OPENAI_CHAT_MODEL") or _env("CHAT_MODEL") or "gpt-4o-mini"
        embed_model = _env("OPENAI_EMBED_MODEL") or _env("EMBED_MODEL") or "text-embedding-3-small"

        # OpenAI direct API expects model IDs like 'gpt-4o-mini' (no 'openai/' prefix).
        if chat_model.startswith("openai/"):
            chat_model = chat_model[len("openai/"):]

        return ProviderConfig(
            provider="openai",
            api_key=openai_key,
            chat_base_url=None,
            embed_base_url=None,
            chat_model=chat_model,
            embed_model=embed_model,
        )

    if (forced == "alice") or (not forced and alice_key):
        chat_base_url = _env("ALICE_CHAT_URL")
        embed_base_url = _env("ALICE_EMBED_URL")
        chat_model = _env("ALICE_CHAT_MODEL") or _env("CHAT_MODEL") or "openai/gpt-4o-mini"
        embed_model = _env("ALICE_EMBED_MODEL") or _env("EMBED_MODEL") or "text-embedding-3-small"
        return ProviderConfig(
            provider="alice",
            api_key=alice_key,
            chat_base_url=chat_base_url or None,
            embed_base_url=embed_base_url or None,
            chat_model=chat_model,
            embed_model=embed_model,
        )

    raise RuntimeError(
        "No API key found. Set GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY (OpenAI) or ALICE_OPENAI_KEY (ALICE) in your environment/.env."
    )


def create_async_chat_client(cfg: Optional[ProviderConfig] = None) -> AsyncOpenAI:
    cfg = cfg or detect_provider()
    if cfg.provider == "openai":
        return AsyncOpenAI(api_key=cfg.api_key)

    if cfg.provider == "gemini":
        if not cfg.chat_base_url:
            raise RuntimeError(
                "Gemini provider selected (GOOGLE_API_KEY is set) but chat base_url is missing. "
                "Set GEMINI_BASE_URL or use the default recommended endpoint."
            )
        return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.chat_base_url)

    if not cfg.chat_base_url:
        raise RuntimeError(
            "ALICE provider selected (ALICE_OPENAI_KEY is set) but ALICE_CHAT_URL is missing."
        )
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.chat_base_url)


def create_async_embed_client(cfg: Optional[ProviderConfig] = None) -> AsyncOpenAI:
    cfg = cfg or detect_provider()
    if cfg.provider == "openai":
        return AsyncOpenAI(api_key=cfg.api_key)

    if cfg.provider == "gemini":
        if not cfg.embed_base_url:
            raise RuntimeError(
                "Gemini provider selected (GOOGLE_API_KEY is set) but embed base_url is missing. "
                "Set GEMINI_BASE_URL or use the default recommended endpoint."
            )
        return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.embed_base_url)

    if not cfg.embed_base_url:
        raise RuntimeError(
            "ALICE provider selected (ALICE_OPENAI_KEY is set) but ALICE_EMBED_URL is missing."
        )
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.embed_base_url)
