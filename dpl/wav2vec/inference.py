from typing import Dict, List, Optional, Union

import numpy as np
import torch

from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor


class AudioFeatureExtractor:

    FPS: int = 50
    MAX_DURATION_SEC: float = 10.0

    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-base",
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)

        model = Wav2Vec2ForPreTraining.from_pretrained(checkpoint)
        self.model = model.to(self.device)
        self.model.eval()

        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)

    def encode(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        if not isinstance(waveform, (np.ndarray, torch.Tensor)):
            raise TypeError("'waveform' should be np.ndarray or torch.Tensor")

        features = self._encode(waveform, sample_rate)
        return {
            "wav2vec": features.squeeze(),
            "volume": self._compute_audio_volume(waveform, sample_rate),
        }

    def batch_encode(
        self,
        waveforms: List[np.ndarray],
        sample_rate: int,
    ) -> Dict[str, List[np.ndarray]]:
        if not isinstance(waveforms, list) or (
            waveforms and not isinstance(waveforms[0], np.ndarray)
        ):
            raise TypeError("'waveforms' should be list of np.ndarray")

        # Some magic.
        lengths = [int(len(wf) * self.FPS / sample_rate - 0.25) for wf in waveforms]

        features = self._encode(waveforms, sample_rate)
        features = [y[:length] for y, length in zip(features, lengths)]

        volume = [
            self._compute_audio_volume(wf, sample_rate)[:length]
            for wf, length in zip(waveforms, lengths)
        ]

        return {"wav2vec": features, "volume": volume}

    @torch.no_grad()
    def _encode(self, waveform: List[np.ndarray], sample_rate: int) -> np.ndarray:
        audio_features = []

        for wf in waveform:
            num_samples = int(sample_rate * self.MAX_DURATION_SEC)
            if len(wf) > num_samples:
                wf_list = [
                    wf[start : start + num_samples]
                    for start in range(0, len(wf), num_samples)
                ]
            else:
                wf_list = [wf]

            data = self.processor(
                wf_list,
                return_tensors="pt",
                padding="longest",
                sampling_rate=sample_rate,
            )

            input_values = data.input_values.to(self.device)
            output = self.model(input_values.detach())
            features = np.concatenate(
                output.projected_quantized_states.detach().cpu().numpy()
            )
            audio_features.append(features)

        return np.stack(audio_features)

    def _compute_audio_volume(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        if waveform.size == 0:
            return np.array([])

        amplitude = np.abs(waveform)
        samples_per_frame = int(np.floor(sample_rate / self.FPS))
        mean_amplitude = [
            np.mean(amplitude[index : index + samples_per_frame])
            for index in range(0, len(amplitude), samples_per_frame)
        ]
        return np.array(mean_amplitude)
