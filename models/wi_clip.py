import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .rf_encoder import RFEncoder
from .text_encoder import TextEncoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WiCLIP(nn.Module):
    def __init__(
        self,
        rf_encoder: Optional[RFEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        projection_dim: int = 256,
        temperature: float = 0.07,
        rf_embedding_dim: int = 512,
        text_embedding_dim: int = 512
    ):
        super().__init__()

        self.rf_encoder = rf_encoder or RFEncoder(embedding_dim=rf_embedding_dim)
        self.text_encoder = text_encoder or TextEncoder(embedding_dim=text_embedding_dim)

        self.rf_projection = ProjectionHead(rf_embedding_dim, projection_dim)
        self.text_projection = ProjectionHead(text_embedding_dim, projection_dim)

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

        self._cached_text_embeddings: Optional[torch.Tensor] = None
        self._cached_activities: Optional[List[str]] = None

    def encode_rf(self, spectrograms: torch.Tensor) -> torch.Tensor:
        rf_features = self.rf_encoder(spectrograms)
        rf_embeddings = self.rf_projection(rf_features)
        rf_embeddings = F.normalize(rf_embeddings, p=2, dim=-1)
        return rf_embeddings

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        text_features = self.text_encoder(texts)
        text_embeddings = self.text_projection(text_features)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        return text_embeddings

    def forward(
        self,
        spectrograms: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rf_embeddings = self.encode_rf(spectrograms)

        if texts is not None:
            text_embeddings = self.encode_text(texts)
        else:
            text_embeddings = self._get_cached_text_embeddings(spectrograms.device)

        logit_scale = self.logit_scale.exp()
        logits_per_rf = logit_scale * rf_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_rf.t()

        return rf_embeddings, text_embeddings, logits_per_rf

    def _get_cached_text_embeddings(self, device: torch.device) -> torch.Tensor:
        activities = self.text_encoder.activity_list

        if self._cached_text_embeddings is None or self._cached_activities != activities:
            self._cached_activities = activities.copy()
            with torch.no_grad():
                self._cached_text_embeddings = self.encode_text(
                    self.text_encoder.get_all_activity_prompts()
                )

        return self._cached_text_embeddings.to(device)

    def predict(
        self,
        spectrograms: torch.Tensor,
        activities: Optional[List[str]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        self.eval()

        with torch.no_grad():
            rf_embeddings = self.encode_rf(spectrograms)

            if activities is not None:
                prompts = [self.text_encoder.get_activity_prompt(a) for a in activities]
                text_embeddings = self.encode_text(prompts)
            else:
                activities = self.text_encoder.activity_list
                text_embeddings = self._get_cached_text_embeddings(spectrograms.device)

            logit_scale = self.logit_scale.exp()
            similarity = logit_scale * rf_embeddings @ text_embeddings.t()
            probs = F.softmax(similarity, dim=-1)

            confidence, pred_idx = torch.max(probs, dim=-1)

            all_scores = {}
            for i, activity in enumerate(activities):
                all_scores[activity] = probs[0, i].item()

            predicted_activity = activities[pred_idx[0].item()]

        return predicted_activity, confidence[0].item(), all_scores

    def zero_shot_predict(
        self,
        spectrograms: torch.Tensor,
        text_queries: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        self.eval()

        with torch.no_grad():
            rf_embeddings = self.encode_rf(spectrograms)
            text_embeddings = self.encode_text(text_queries)

            logit_scale = self.logit_scale.exp()
            similarity = logit_scale * rf_embeddings @ text_embeddings.t()
            probs = F.softmax(similarity, dim=-1)

            confidence, pred_idx = torch.max(probs, dim=-1)

            all_scores = {}
            for i, query in enumerate(text_queries):
                all_scores[query] = probs[0, i].item()

            predicted_query = text_queries[pred_idx[0].item()]

        return predicted_query, confidence[0].item(), all_scores

    def clear_cache(self) -> None:
        self._cached_text_embeddings = None
        self._cached_activities = None


class WiCLIPLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        rf_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = rf_embeddings.shape[0]

        logits = logit_scale * rf_embeddings @ text_embeddings.t()

        if labels is None:
            labels = torch.arange(batch_size, device=rf_embeddings.device)

        loss_rf = F.cross_entropy(logits, labels, label_smoothing=self.smoothing)
        loss_text = F.cross_entropy(logits.t(), labels, label_smoothing=self.smoothing)

        loss = (loss_rf + loss_text) / 2

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        similarity = embeddings @ embeddings.t() / self.temperature

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.float().to(device)

        mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity)
        exp_sim.fill_diagonal_(0)

        pos_sum = (exp_sim * mask).sum(dim=1)
        neg_sum = (exp_sim * (1 - mask)).sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)
        loss = loss.mean()

        return loss
