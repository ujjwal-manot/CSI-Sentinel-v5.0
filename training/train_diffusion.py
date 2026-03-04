import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.csi_diffusion import CSIDiffusion, DiffusionUNet
from training.dataset import CSIDataset
from utils.helpers import set_seed, get_device, AverageMeter
from utils.logger import get_logger


class DiffusionTrainer:
    def __init__(
        self,
        model: CSIDiffusion,
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

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
            eta_min=1e-6
        )

        self.scaler = GradScaler() if config.get("mixed_precision", True) else None

        self.num_epochs = config.get("num_epochs", 100)
        self.gradient_clip = config.get("gradient_clip", 1.0)
        self.save_dir = Path(config.get("save_dir", "checkpoints/diffusion"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("diffusion_trainer")
        self.best_val_loss = float("inf")
        self.ema_model = None
        self.ema_decay = config.get("ema_decay", 0.9999)

        if config.get("use_ema", True):
            self._init_ema()

    def _init_ema(self) -> None:
        self.ema_model = CSIDiffusion(
            unet=DiffusionUNet(
                in_channels=3,
                out_channels=3,
                num_classes=len(self.config.get("activities", [])) or 10
            ),
            num_timesteps=self.model.num_timesteps
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (spectrograms, labels, _) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    loss = self.model(spectrograms, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model(spectrograms, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            self._update_ema()
            loss_meter.update(loss.item(), spectrograms.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg

    @torch.no_grad()
    def validate(self) -> float:
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()
        loss_meter = AverageMeter()

        for spectrograms, labels, _ in tqdm(self.val_loader, desc="Validating"):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            batch_size = spectrograms.shape[0]
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=self.device)
            loss = model.p_losses(spectrograms, t, labels)

            loss_meter.update(loss.item(), spectrograms.size(0))

        return loss_meter.avg

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 16,
        class_label: Optional[int] = None,
        use_ddim: bool = True
    ) -> torch.Tensor:
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        if class_label is not None:
            class_labels = torch.full((num_samples,), class_label, device=self.device)
        else:
            num_classes = self.model.unet.num_classes or 10
            class_labels = torch.randint(0, num_classes, (num_samples,), device=self.device)

        if use_ddim:
            samples = model.ddim_sample(
                batch_size=num_samples,
                channels=3,
                height=128,
                width=128,
                class_labels=class_labels,
                num_inference_steps=50,
                device=self.device
            )
        else:
            samples = model.sample(
                batch_size=num_samples,
                channels=3,
                height=128,
                width=128,
                class_labels=class_labels,
                device=self.device
            )

        return samples

    def train(self) -> Dict[str, List[float]]:
        self.logger.info(f"Starting diffusion training for {self.num_epochs} epochs")

        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)
                samples = self.generate_samples(16)
                self._save_samples(samples, epoch)

        self.save_checkpoint(self.num_epochs - 1, final=True)
        self._save_history(history)

        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }

        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        if is_best:
            path = self.save_dir / "best_model.pt"
        elif final:
            path = self.save_dir / "final_model.pt"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)

    def _save_samples(self, samples: torch.Tensor, epoch: int) -> None:
        samples_dir = self.save_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        samples_np = samples.cpu().numpy()
        np.save(samples_dir / f"samples_epoch_{epoch + 1}.npy", samples_np)

    def _save_history(self, history: dict) -> None:
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)


def main():
    set_seed(42)

    config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 100,
        "batch_size": 16,
        "gradient_clip": 1.0,
        "mixed_precision": True,
        "use_ema": True,
        "ema_decay": 0.9999,
        "save_dir": "checkpoints/diffusion",
        "data_dir": "data",
        "activities": ["walk", "run", "sit", "stand", "fall", "lie_down", "wave", "jump", "crouch", "empty"]
    }

    train_dataset = CSIDataset(config["data_dir"], config["activities"], split="train")
    val_dataset = CSIDataset(config["data_dir"], config["activities"], split="val")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    unet = DiffusionUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_classes=len(config["activities"])
    )

    model = CSIDiffusion(unet=unet, num_timesteps=1000, beta_schedule="cosine")

    trainer = DiffusionTrainer(model, train_loader, val_loader, config)
    history = trainer.train()

    print("Diffusion training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
