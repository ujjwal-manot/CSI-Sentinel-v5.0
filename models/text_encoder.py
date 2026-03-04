import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 512,
        freeze: bool = True,
        use_projection: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze = freeze

        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)
            self.hidden_dim = self.text_model.config.hidden_size
            self.use_transformers = True
        except ImportError:
            self.tokenizer = None
            self.text_model = None
            self.hidden_dim = 512
            self.use_transformers = False
            self._init_simple_encoder()

        if freeze and self.text_model is not None:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.projection = nn.Identity()

        self._activity_prompts = {
            "walk": "a person walking slowly across the room",
            "run": "a person running fast through the space",
            "sit": "a person sitting down on a chair",
            "stand": "a person standing still in place",
            "fall": "a person falling down to the ground",
            "lie_down": "a person lying down on the floor",
            "wave": "a person waving their hand",
            "jump": "a person jumping up and down",
            "crouch": "a person crouching down low",
            "empty": "an empty room with no person present"
        }

    def _init_simple_encoder(self):
        self.simple_embedding = nn.Embedding(10000, self.hidden_dim)
        self.simple_lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True
        )

    def _simple_tokenize(self, texts: List[str]) -> torch.Tensor:
        max_len = 77
        tokens = torch.zeros(len(texts), max_len, dtype=torch.long)

        for i, text in enumerate(texts):
            words = text.lower().split()[:max_len]
            for j, word in enumerate(words):
                tokens[i, j] = hash(word) % 10000

        return tokens

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        if self.use_transformers and self.tokenizer is not None:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )

            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.set_grad_enabled(not self.freeze):
                outputs = self.text_model(**inputs)
                embeddings = outputs.pooler_output
        else:
            device = next(self.parameters()).device
            tokens = self._simple_tokenize(texts).to(device)
            embedded = self.simple_embedding(tokens)
            _, (hidden, _) = self.simple_lstm(embedded)
            embeddings = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        projected = self.projection(embeddings)
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized

    def encode_activities(self, activities: List[str]) -> torch.Tensor:
        prompts = [self._activity_prompts.get(act, f"a person performing {act}") for act in activities]
        return self.forward(prompts)

    def get_activity_embeddings(self, device: Optional[torch.device] = None) -> torch.Tensor:
        activities = list(self._activity_prompts.keys())
        embeddings = self.encode_activities(activities)

        if device is not None:
            embeddings = embeddings.to(device)

        return embeddings

    def add_activity_prompt(self, activity: str, prompt: str) -> None:
        self._activity_prompts[activity] = prompt

    @property
    def activity_list(self) -> List[str]:
        return list(self._activity_prompts.keys())


class LearnablePromptEncoder(nn.Module):
    def __init__(
        self,
        num_activities: int = 10,
        embedding_dim: int = 512,
        num_context_tokens: int = 4
    ):
        super().__init__()

        self.num_activities = num_activities
        self.embedding_dim = embedding_dim
        self.num_context_tokens = num_context_tokens

        self.context_embeddings = nn.Parameter(
            torch.randn(num_context_tokens, embedding_dim) * 0.02
        )

        self.activity_embeddings = nn.Parameter(
            torch.randn(num_activities, embedding_dim) * 0.02
        )

        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim * (num_context_tokens + 1), embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, activity_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if activity_indices is None:
            activity_indices = torch.arange(self.num_activities)

        device = self.context_embeddings.device
        activity_indices = activity_indices.to(device)

        batch_size = activity_indices.shape[0]
        context = self.context_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        activities = self.activity_embeddings[activity_indices].unsqueeze(1)

        combined = torch.cat([context, activities], dim=1)
        combined = combined.flatten(1)

        output = self.combiner(combined)
        output = F.normalize(output, p=2, dim=-1)

        return output
