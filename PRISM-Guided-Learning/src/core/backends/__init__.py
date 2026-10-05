"""LLM backends: one class per serving engine, all implementing `LLMBackend.execute_batch`."""
from config import LLMConfig
from core.backends.base import BackendInfo, LLMBackend

BACKENDS = ("ollama",)


def make_backend(config: LLMConfig) -> LLMBackend:
    """The backend named by `llm.backend`."""
    if config.backend == "ollama":
        from core.backends.ollama import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"llm.backend={config.backend!r} is not supported (allowed: {', '.join(BACKENDS)})")


__all__ = ["BACKENDS", "BackendInfo", "LLMBackend", "make_backend"]
