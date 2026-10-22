from __future__ import annotations

import sys
import types
from pathlib import Path


class ModelUnavailableError(RuntimeError):
    """Raised when DeepFilterNet or its pretrained weights cannot be used."""


class DeepFilterEngine:
    def __init__(self) -> None:
        self._model = None
        self._state = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            # DeepFilterNet 0.5.x imports a Torchaudio module removed in newer
            # releases. The object was moved to the top-level namespace.
            import torchaudio

            if "torchaudio.backend.common" not in sys.modules:
                compatibility_module = types.ModuleType("torchaudio.backend.common")
                # DeepFilterNet only uses this symbol as a type annotation.
                compatibility_module.AudioMetaData = object
                sys.modules["torchaudio.backend.common"] = compatibility_module

            from df.enhance import init_df

            self._model, self._state, *_ = init_df(log_file=None)
        except Exception as exc:
            raise ModelUnavailableError(
                "DeepFilterNet is unavailable: "
                f"{type(exc).__name__}. Install service dependencies and ensure the "
                "pretrained model can download on the first run."
            ) from exc

    def denoise(self, source: Path, destination: Path) -> None:
        self.load()
        try:
            from df.enhance import enhance, load_audio, save_audio

            audio, original_sample_rate = load_audio(source, sr=self._state.sr())
            enhanced = enhance(self._model, self._state, audio)
            save_audio(destination, enhanced, sr=self._state.sr())
        except Exception as exc:
            raise RuntimeError(f"DeepFilterNet failed for {source.name}") from exc
