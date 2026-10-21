"""Speech restoration with Sidon v0.1 (denoise, dereverb, bandwidth extension in one pass).

Follows the official demo (huggingface.co/spaces/sarulab-speech/sidon_demo_beta):
TorchScript feature predictor (w2v-BERT 2.0) + TorchScript vocoder, 16 kHz in,
48 kHz out, 96 s chunks with a one-frame feature cache between chunks.

Sidon is trained on speech. On singing or a mix with music it resynthesises
what it believes is speech, so run `separate` first and restore the vocal stem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ah import audio
from ah.contract import ToolOutput, ToolResult

REPO_ID = "sarulab-speech/sidon-v0.1"
PREPROCESSOR_ID = "facebook/w2v-bert-2.0"
IN_SR = 16_000
OUT_SR = 48_000
TAIL_PAD = 24_000  # samples at 16 kHz, as in the official demo
FRAME_PAD = 160
TRIM = 960  # samples at 48 kHz dropped from every decoded chunk


class Params(BaseModel, extra="forbid"):
    chunk_seconds: float = Field(default=96.0, gt=5, le=96)
    match_peak: bool = True  # Sidon outputs at 0.9 peak; put the input's peak back


class RestoreTool:
    name = "restore"

    def __init__(self) -> None:
        self.device = "cpu"
        self._models = None

    def load(self, device: str) -> None:
        self.device = device
        if self._models is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import SeamlessM4TFeatureExtractor

        suffix = "cuda" if device.startswith("cuda") else "cpu"
        fe_path = hf_hub_download(REPO_ID, f"feature_extractor_{suffix}.pt")
        decoder_path = hf_hub_download(REPO_ID, f"decoder_{suffix}.pt")
        self._models = {
            "preprocessor": SeamlessM4TFeatureExtractor.from_pretrained(PREPROCESSOR_ID),
            "fe": torch.jit.load(fe_path, map_location=device).to(device).eval(),
            "decoder": torch.jit.load(decoder_path, map_location=device).to(device).eval(),
        }

    def restore(self, mono: np.ndarray, sr: int, chunk_seconds: float) -> np.ndarray:
        import torch
        import torchaudio

        models = self._models
        peak = float(np.max(np.abs(mono)))
        if peak == 0:
            raise ValueError("input is digital silence")
        wave = torch.from_numpy(0.9 * mono / peak).float().view(1, -1)
        wave = torchaudio.functional.highpass_biquad(wave, sr, 50)
        wave16 = audio.resample(wave.view(-1).numpy(), sr, IN_SR)
        wave16 = np.pad(wave16, (0, TAIL_PAD))

        restored: list[torch.Tensor] = []
        cache = None
        with torch.inference_mode():
            for chunk in torch.from_numpy(wave16).split(int(IN_SR * chunk_seconds)):
                inputs = models["preprocessor"](
                    torch.nn.functional.pad(chunk, (FRAME_PAD, FRAME_PAD)).numpy(),
                    sampling_rate=IN_SR, return_tensors="pt",
                )
                feature = models["fe"](inputs["input_features"].to(self.device))["last_hidden_state"]
                if cache is not None:
                    feature = torch.cat([cache, feature], dim=1)
                restored.append(models["decoder"](feature.transpose(1, 2)).view(-1)[:-TRIM].cpu())
                cache = feature[:, -1:]
        target = int(OUT_SR / sr * mono.shape[0])
        return torch.cat(restored).numpy()[:target]

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        options = Params.model_validate(params)
        data, sr = audio.decode(inputs["audio"])
        source = audio.to_mono(data)
        result = self.restore(source, sr, options.chunk_seconds)
        if options.match_peak:
            result = result * (min(float(np.max(np.abs(source))), 0.99) / max(float(np.max(np.abs(result))), 1e-9))
        target = out_dir / "restored.wav"
        audio.write_wav(target, result.astype(np.float32), OUT_SR)
        warnings = ["Sidon outputs mono speech; stereo image and non-speech content are not preserved"]
        # Canonical artifacts are always 2-channel; only warn when the channels actually differ.
        if data.ndim == 2 and data.shape[1] > 1 and not np.allclose(data[:, :1], data[:, 1:], atol=1e-6):
            warnings.append("stereo input was downmixed to mono")
        return ToolResult(
            tool=self.name,
            outputs=[ToolOutput(role="restored", path=str(target), kind="audio", meta={"model": REPO_ID})],
            meta={"model": REPO_ID, "device": self.device, "model_sample_rate": OUT_SR},
            warnings=warnings,
        )
