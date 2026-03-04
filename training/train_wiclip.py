import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.wi_clip import WiCLIP, WiCLIPLoss
from models.rf_encoder import RFEncoder
from models.text_encoder import TextEncoder
from training.dataset import CSIDataset, create_dataloaders
from training.augmentations import CSIAugmentor
from utils.helpers import set_seed, get_device, AverageMeter, format_time
from utils.logger import get_logger


class WiCLIPTrainer:
    def __init__(
        self,
        model: WiCLIP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or get_device()

        self.model.to(self.device)

        self.criterion = WiCLIPLoss(smoothing=config.get("label_smoothing", 0.1))

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.98)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("warmup_epochs", 5),
            T_mult=2
        )

        self.scaler = GradScaler() if config.get("mixed_precision", True) else None

        self.num_epochs = config.get("num_epochs", 100)
        self.gradient_clip = config.get("gradient_clip", 1.0)
        self.save_dir = Path(config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("wiclip_trainer")
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (spectrograms, labels, activity_names) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    rf_emb, text_emb, logits = self.model(spectrograms, list(activity_names))
                    loss = self.criterion(rf_emb, text_emb, self.model.logit_scale.exp())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                rf_emb, text_emb, logits = self.model(spectrograms, list(activity_names))
                loss = self.criterion(rf_emb, text_emb, self.model.logit_scale.exp())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            loss_meter.update(loss.item(), spectrograms.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for spectrograms, labels, activity_names in tqdm(self.val_loader, desc="Validating"):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            rf_emb, text_emb, logits = self.model(spectrograms, list(activity_names))
            loss = self.criterion(rf_emb, text_emb, self.model.logit_scale.exp())

            loss_meter.update(loss.item(), spectrograms.size(0))

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return loss_meter.avg, accuracy

    def train(self) -> Dict[str, List[float]]:
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        self.save_checkpoint(self.num_epochs - 1, final=True)
        self._save_history()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "activities": self.model.text_encoder.activity_list
        }

        if is_best:
            path = self.save_dir / "best_model.pt"
        elif final:
            path = self.save_dir / "final_model.pt"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def _save_history(self) -> None:
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    set_seed(42)

    config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 100,
        "batch_size": 32,
        "warmup_epochs": 5,
        "gradient_clip": 1.0,
        "mixed_precision": True,
        "label_smoothing": 0.1,
        "save_dir": "checkpoints/wiclip",
        "data_dir": "data"
    }

    activities = ["walk", "run", "sit", "stand", "fall", "lie_down", "wave", "jump", "crouch", "empty"]

    augmentor = CSIAugmentor(time_mask_max=30, freq_mask_max=20, noise_std=0.02)

    train_loader, val_loader, _ = create_dataloaders(
        config["data_dir"],
        batch_size=config["batch_size"],
        transform=augmentor,
        activities=activities
    )

    rf_encoder = RFEncoder(input_channels=3, embedding_dim=512)
    text_encoder = TextEncoder(embedding_dim=512, freeze=True)
    model = WiCLIP(rf_encoder=rf_encoder, text_encoder=text_encoder, projection_dim=256)

    trainer = WiCLIPTrainer(model, train_loader, val_loader, config)
    history = trainer.train()

    print("Training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
