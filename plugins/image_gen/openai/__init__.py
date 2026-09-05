"""OpenAI ``gpt-image-2`` at three quality tiers (virtual ids ``gpt-image-2-low/-medium/-high``);
base64 output → image cache. Selection: ``OPENAI_IMAGE_MODEL`` → ``image_gen.openai.model`` →
``image_gen.model`` → :data:`DEFAULT_MODEL`."""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from plugins.image_gen._common import (
    GPT_IMAGE_2_API_MODEL as API_MODEL, GPT_IMAGE_2_DEFAULT as DEFAULT_MODEL, GPT_IMAGE_2_TIERS,
    StaticImageGenProvider, collect_source_images, error_factory, import_openai,
    load_image_gen_config, materialize_image, openai_importable, prompt_required_error,
    resolve_static_model, size_for)

logger = logging.getLogger(__name__)

# Metadata for a model id that isn't one of the fixed gpt-image-2 tiers —
# i.e. a custom model on an OpenAI-compatible endpoint (see _resolve_base_url).
_CUSTOM_MODEL_META: Dict[str, Any] = {
    "display": "Provider-defined image model", "speed": "provider-defined",
    "strengths": "provider-defined", "price": "varies", "quality": None,
}


def _resolve_base_url() -> Optional[str]:
    """Resolve an optional OpenAI-compatible API base URL.

    Scoped to ``image_gen.openai.base_url`` only. No top-level
    ``image_gen.base_url`` and no ``OPENAI_BASE_URL`` env fallback: repo
    policy treats config.yaml as the single source of truth for endpoint
    URLs (``runtime_provider_backends.py``) and ``.env`` is secrets-only,
    so a base URL — not a secret — has no env-var path here.
    """
    cfg = load_image_gen_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    value = openai_cfg.get("base_url") if isinstance(openai_cfg, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip().rstrip("/")
    return None


def _resolve_api_key() -> str:
    """Resolve the key for OpenAI or an OpenAI-compatible image endpoint."""
    cfg = load_image_gen_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    key_env = openai_cfg.get("api_key_env") if isinstance(openai_cfg, dict) else None
    if isinstance(key_env, str) and key_env.strip():
        key = get_secret(key_env.strip()) or ""
        if key:
            return key

    key = get_secret("OPENAI_API_KEY") or ""
    if key:
        return key

    # Existing Hermes custom providers may keep their key in model config.
    # Reuse it in memory instead of duplicating a credential into another
    # .env — but ONLY when an explicit non-empty image endpoint base URL is
    # configured. Without a resolved base URL this key would otherwise be
    # sent straight to the default api.openai.com, leaking a credential
    # meant for a different (often OpenAI-compatible but non-OpenAI)
    # endpoint to OpenAI itself.
    if _resolve_base_url():
        try:
            from hermes_cli.config import load_config

            model_cfg = load_config().get("model")
            if isinstance(model_cfg, dict):
                key = str(model_cfg.get("api_key") or "").strip()
                if key:
                    return key
        except Exception as exc:
            logger.debug("Could not resolve image API key from model config: %s", exc)
    return ""


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide model + quality metadata.

    Custom passthrough (a model id outside the fixed gpt-image-2 tiers) is
    scoped to provider-owned keys only: ``OPENAI_IMAGE_MODEL`` env and
    ``image_gen.openai.model``. The shared top-level ``image_gen.model``
    key is picker-written and shared across providers (e.g. FAL reads
    unknown ids from it), so it keeps the ``resolve_static_model``
    known-ids-only rule and falls back to ``DEFAULT_MODEL`` for an
    unrecognized id there.
    """
    env_override = (os.environ.get("OPENAI_IMAGE_MODEL") or "").strip()
    if env_override:
        return env_override, GPT_IMAGE_2_TIERS.get(env_override, _CUSTOM_MODEL_META)

    cfg = load_image_gen_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            return candidate, GPT_IMAGE_2_TIERS.get(candidate, _CUSTOM_MODEL_META)

    return resolve_static_model(
        GPT_IMAGE_2_TIERS, DEFAULT_MODEL, env_var="OPENAI_IMAGE_MODEL", config_key="openai")


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load ``(data, filename)`` from a URL, data URI or local path; raises on IO/network error."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        resp = requests.get(ref, timeout=60)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = (header.split("image/", 1)[1].split(";", 1)[0] if "image/" in header else "") or "png"
        return base64.b64decode(b64), f"image.{ext}"
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    return data, os.path.basename(ref) or "image.png"


def _named_bytes_io(ref: str) -> io.BytesIO:
    """``images.edit()`` expects named file-like objects for correct multipart."""
    data, fname = _load_image_bytes(ref)
    bio = io.BytesIO(data)
    bio.name = fname
    return bio


class OpenAIImageGenProvider(StaticImageGenProvider):
    """OpenAI ``images.generate`` / ``images.edit`` backend — gpt-image-2."""

    provider_id = "openai"
    label = "OpenAI"
    models = GPT_IMAGE_2_TIERS
    default_model_id = DEFAULT_MODEL
    price = "varies"
    setup = dict(
        name="OpenAI", badge="paid",
        tag="gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing",
        key="OPENAI_API_KEY", prompt="OpenAI API key", url="https://platform.openai.com/api-keys")

    def is_available(self) -> bool:
        return bool(_resolve_api_key()) and openai_importable()

    def capabilities(self) -> Dict[str, Any]:
        # images.edit() accepts up to 16 source images.
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("openai", aspect)
        api_key = _resolve_api_key()
        if not api_key:
            return error_factory("openai", aspect)(
                "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                "Generation → OpenAI to configure, or `hermes setup` "
                "to add the key.",
                "auth_required")

        openai, err = import_openai("openai", aspect)
        if err:
            return err
        tier_id, meta = _resolve_model()
        size = size_for(aspect)
        sources = collect_source_images(image_url, reference_image_urls, limit=16)
        is_edit = bool(sources)
        fail = error_factory("openai", aspect, model=tier_id, prompt=prompt)
        base_url = _resolve_base_url()
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        # Known tiers all hit the single underlying API model (API_MODEL)
        # with a quality knob — gpt-image-2-low/medium/high are NOT valid
        # OpenAI model IDs. Only an unrecognized (provider-defined /
        # custom-endpoint) model id is sent to the API as-is.
        api_model = API_MODEL if tier_id in GPT_IMAGE_2_TIERS else tier_id
        request: Dict[str, Any] = dict(model=api_model, prompt=prompt, size=size, n=1)
        if meta.get("quality"):
            request["quality"] = meta["quality"]
        if is_edit:
            try:
                files = [_named_bytes_io(ref) for ref in sources]
            except Exception as exc:
                return fail(f"Could not load source image for editing: {exc}", "io_error")
            request["image"] = files if len(files) > 1 else files[0]
        verb, call = ("edit", client.images.edit) if is_edit else ("generation", client.images.generate)
        try:
            response = call(**request)
        except Exception as exc:
            logger.debug("OpenAI image %s failed", verb, exc_info=True)
            return fail(f"OpenAI image {'editing' if is_edit else 'generation'} failed: {exc}", "api_error")

        data = getattr(response, "data", None) or []
        if not data:
            return fail("OpenAI returned no image data", "empty_response")
        first = data[0]
        image_ref, err = materialize_image(
            getattr(first, "b64_json", None), getattr(first, "url", None),
            prefix=f"openai_{tier_id.replace('/', '_')}", label="OpenAI", provider="openai",
            model=tier_id, prompt=prompt, aspect=aspect, log=logger)
        if err:
            return err
        extra: Dict[str, Any] = {"size": size}
        if meta.get("quality"):
            extra["quality"] = meta["quality"]
        if getattr(first, "revised_prompt", None):
            extra["revised_prompt"] = first.revised_prompt
        return success_response(
            image=image_ref, model=tier_id, prompt=prompt, aspect_ratio=aspect, provider="openai",
            modality="image" if is_edit else "text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'ImageGenProvider': ('agent.image_gen_provider', 'ImageGenProvider'),
    'error_response': ('agent.image_gen_provider', 'error_response'),
    'normalize_reference_images': ('agent.image_gen_provider', 'normalize_reference_images'),
    'save_b64_image': ('agent.image_gen_provider', 'save_b64_image'),
    'save_url_image': ('agent.image_gen_provider', 'save_url_image'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
