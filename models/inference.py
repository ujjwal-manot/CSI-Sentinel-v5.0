import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List, Union
from collections import deque
from pathlib import Path

from .wi_clip import WiCLIP
from .classifier import ActivityClassifier

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model: Union[WiCLIP, ActivityClassifier],
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
        activities: Optional[List[str]] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.activities = activities or []

        self._prediction_history = deque(maxlen=smoothing_window)
        self._score_history = deque(maxlen=smoothing_window)

        self._is_wiclip = isinstance(model, WiCLIP)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str = "wiclip",
        device: Optional[torch.device] = None,
        **kwargs
    ) -> "InferenceEngine":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if model_type == "wiclip":
            model = WiCLIP()
        else:
            num_classes = checkpoint.get("num_classes", 10)
            model = ActivityClassifier(num_classes=num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])

        activities = checkpoint.get("activities", [])

        return cls(model, device, activities=activities, **kwargs)

    def preprocess(self, spectrogram: np.ndarray) -> torch.Tensor:
        if isinstance(spectrogram, np.ndarray):
            tensor = torch.from_numpy(spectrogram).float()
        else:
            tensor = spectrogram.float()

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        return tensor

    def predict(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        return_all_scores: bool = True
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        tensor = self.preprocess(spectrogram)

        with torch.no_grad():
            if self._is_wiclip:
                activity, confidence, all_scores = self.model.predict(tensor)
            else:
                activity, confidence, all_scores = self.model.predict(tensor)

        self._prediction_history.append(activity)
        self._score_history.append(all_scores)

        if return_all_scores:
            return activity, confidence, all_scores
        return activity, confidence, None

    def predict_smoothed(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[str, float, Dict[str, float]]:
        _, _, all_scores = self.predict(spectrogram)

        if len(self._score_history) < 2:
            activity = max(all_scores, key=all_scores.get)
            return activity, all_scores[activity], all_scores

        smoothed_scores = {}
        for activity in all_scores.keys():
            scores = [h.get(activity, 0) for h in self._score_history]
            smoothed_scores[activity] = np.mean(scores)

        total = sum(smoothed_scores.values())
        smoothed_scores = {k: v / total for k, v in smoothed_scores.items()}

        predicted_activity = max(smoothed_scores, key=smoothed_scores.get)
        confidence = smoothed_scores[predicted_activity]

        return predicted_activity, confidence, smoothed_scores

    def predict_batch(
        self,
        spectrograms: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        if isinstance(spectrograms, np.ndarray):
            tensor = torch.from_numpy(spectrograms).float()
        else:
            tensor = spectrograms.float()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            if self._is_wiclip:
                rf_embeddings = self.model.encode_rf(tensor)
                text_embeddings = self.model._get_cached_text_embeddings(self.device)
                logit_scale = self.model.logit_scale.exp()
                similarity = logit_scale * rf_embeddings @ text_embeddings.t()
                probs = F.softmax(similarity, dim=-1)
            else:
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1)

        confidences, pred_indices = torch.max(probs, dim=-1)

        activities_list = self.activities or [str(i) for i in range(probs.shape[-1])]

        predictions = []
        all_scores = []

        for i in range(len(tensor)):
            pred_activity = activities_list[pred_indices[i].item()]
            predictions.append(pred_activity)

            scores = {}
            for j, act in enumerate(activities_list):
                scores[act] = probs[i, j].item()
            all_scores.append(scores)

        return predictions, confidences.tolist(), all_scores

    def is_confident(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold

    def reset_history(self) -> None:
        self._prediction_history.clear()
        self._score_history.clear()

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def set_smoothing_window(self, window: int) -> None:
        self.smoothing_window = window
        self._prediction_history = deque(maxlen=window)
        self._score_history = deque(maxlen=window)


class StreamingInference:
    def __init__(
        self,
        engine: InferenceEngine,
        buffer_size: int = 200,
        hop_size: int = 50,
        alert_cooldown: float = 10.0
    ):
        self.engine = engine
        self.buffer_size = buffer_size
        self.hop_size = hop_size
        self.alert_cooldown = alert_cooldown

        self._spectrogram_buffer = deque(maxlen=buffer_size)
        self._last_alert_time = 0.0
        self._alert_callbacks: List[Callable[[str, float, float], None]] = []

    def register_alert_callback(self, callback: Callable[[str, float, float], None]) -> None:
        self._alert_callbacks.append(callback)

    def process_frame(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        timestamp: float
    ) -> Optional[Tuple[str, float]]:
        self._spectrogram_buffer.append(spectrogram)

        if len(self._spectrogram_buffer) < self.buffer_size:
            return None

        if len(self._spectrogram_buffer) % self.hop_size != 0:
            return None

        activity, confidence, scores = self.engine.predict_smoothed(spectrogram)

        if activity == "fall" and confidence >= self.engine.confidence_threshold:
            if timestamp - self._last_alert_time >= self.alert_cooldown:
                self._last_alert_time = timestamp
                self._trigger_alert(activity, confidence, timestamp)

        return activity, confidence

    def _trigger_alert(self, activity: str, confidence: float, timestamp: float) -> None:
        for callback in self._alert_callbacks:
            try:
                callback(activity, confidence, timestamp)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def reset(self) -> None:
        self._spectrogram_buffer.clear()
        self.engine.reset_history()
        self._last_alert_time = 0.0
