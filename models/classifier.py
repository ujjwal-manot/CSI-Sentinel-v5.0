import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .rf_encoder import RFEncoder


class ActivityClassifier(nn.Module):
    def __init__(
        self,
        rf_encoder: Optional[RFEncoder] = None,
        num_classes: int = 10,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.rf_encoder = rf_encoder or RFEncoder(embedding_dim=embedding_dim)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        if freeze_encoder:
            for param in self.rf_encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._activity_names: List[str] = []

    def set_activity_names(self, names: List[str]) -> None:
        assert len(names) == self.num_classes
        self._activity_names = names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.rf_encoder(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)

            all_scores = {}
            for i in range(self.num_classes):
                name = self._activity_names[i] if i < len(self._activity_names) else str(i)
                all_scores[name] = probs[0, i].item()

            pred_name = self._activity_names[pred_idx[0].item()] if self._activity_names else str(pred_idx[0].item())

        return pred_name, confidence[0].item(), all_scores

    def predict_batch(self, x: torch.Tensor) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidences, pred_indices = torch.max(probs, dim=-1)

            pred_names = []
            for idx in pred_indices:
                name = self._activity_names[idx.item()] if idx.item() < len(self._activity_names) else str(idx.item())
                pred_names.append(name)

        return pred_names, confidences, probs


class TemporalActivityClassifier(nn.Module):
    def __init__(
        self,
        rf_encoder: Optional[RFEncoder] = None,
        num_classes: int = 10,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        sequence_length: int = 10
    ):
        super().__init__()

        self.rf_encoder = rf_encoder or RFEncoder(embedding_dim=embedding_dim)
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.temporal = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._activity_names: List[str] = []

    def set_activity_names(self, names: List[str]) -> None:
        self._activity_names = names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape

        x_flat = x.view(batch_size * seq_len, channels, height, width)
        features = self.rf_encoder(x_flat)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.temporal(features)

        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.classifier(context)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)

            all_scores = {}
            for i in range(self.num_classes):
                name = self._activity_names[i] if i < len(self._activity_names) else str(i)
                all_scores[name] = probs[0, i].item()

            pred_name = self._activity_names[pred_idx[0].item()] if self._activity_names else str(pred_idx[0].item())

        return pred_name, confidence[0].item(), all_scores


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        one_hot = torch.full_like(inputs, smooth_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)

        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1).mean()

        return loss
