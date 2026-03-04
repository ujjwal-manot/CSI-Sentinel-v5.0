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
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.classifier import ActivityClassifier, FocalLoss
from models.rf_encoder import RFEncoder
from training.dataset import CSIDataset, create_dataloaders
from training.augmentations import CSIAugmentor, MixUp
from utils.helpers import set_seed, get_device, AverageMeter
from utils.logger import get_logger


class ClassifierTrainer:
    def __init__(
        self,
        model: ActivityClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or get_device()

        self.model.to(self.device)

        loss_type = config.get("loss_type", "cross_entropy")
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get("learning_rate", 1e-4) * 10,
            epochs=config.get("num_epochs", 100),
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )

        self.scaler = GradScaler() if config.get("mixed_precision", True) else None
        self.mixup = MixUp(alpha=config.get("mixup_alpha", 0.2)) if config.get("use_mixup", False) else None

        self.num_epochs = config.get("num_epochs", 100)
        self.gradient_clip = config.get("gradient_clip", 1.0)
        self.save_dir = Path(config.get("save_dir", "checkpoints/classifier"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("classifier_trainer")
        self.best_val_acc = 0.0
        self.activities = config.get("activities", [])
        self.model.set_activity_names(self.activities)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for spectrograms, labels, _ in pbar:
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            if self.mixup is not None:
                spectrograms, labels_a, labels_b, lam = self.mixup(spectrograms, labels)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    logits = self.model(spectrograms)

                    if self.mixup is not None:
                        loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                    else:
                        loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(spectrograms)

                if self.mixup is not None:
                    loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                else:
                    loss = self.criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()

            _, predicted = logits.max(1)
            if self.mixup is None:
                correct = (predicted == labels).sum().item()
                acc_meter.update(correct / labels.size(0), labels.size(0))

            loss_meter.update(loss.item(), spectrograms.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for spectrograms, labels, _ in tqdm(self.val_loader, desc="Validating"):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels)

            loss_meter.update(loss.item(), spectrograms.size(0))

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return loss_meter.avg, accuracy

    @torch.no_grad()
    def test(self) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for spectrograms, labels, _ in tqdm(self.test_loader, desc="Testing"):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(spectrograms)
            probs = torch.softmax(logits, dim=-1)

            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        conf_matrix = confusion_matrix(all_labels, all_preds)

        report = classification_report(
            all_labels, all_preds,
            target_names=self.activities,
            output_dict=True
        )

        results = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report
        }

        return results

    def train(self) -> Dict[str, List[float]]:
        self.logger.info(f"Starting classifier training for {self.num_epochs} epochs")

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "learning_rate": []
        }

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["learning_rate"].append(current_lr)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)

        self.save_checkpoint(self.num_epochs - 1, final=True)

        test_results = self.test()
        self.logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")

        history["test_results"] = test_results
        self._save_history(history)

        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "activities": self.activities,
            "num_classes": len(self.activities)
        }

        if is_best:
            path = self.save_dir / "best_model.pt"
        elif final:
            path = self.save_dir / "final_model.pt"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)

    def _save_history(self, history: dict) -> None:
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)


def main():
    set_seed(42)

    config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 100,
        "batch_size": 32,
        "gradient_clip": 1.0,
        "mixed_precision": True,
        "use_mixup": True,
        "mixup_alpha": 0.2,
        "loss_type": "focal",
        "label_smoothing": 0.1,
        "save_dir": "checkpoints/classifier",
        "data_dir": "data",
        "activities": ["walk", "run", "sit", "stand", "fall", "lie_down", "wave", "jump", "crouch", "empty"]
    }

    augmentor = CSIAugmentor(time_mask_max=30, freq_mask_max=20, noise_std=0.02)

    train_loader, val_loader, test_loader = create_dataloaders(
        config["data_dir"],
        batch_size=config["batch_size"],
        transform=augmentor,
        activities=config["activities"]
    )

    rf_encoder = RFEncoder(input_channels=3, embedding_dim=512)
    model = ActivityClassifier(
        rf_encoder=rf_encoder,
        num_classes=len(config["activities"]),
        embedding_dim=512,
        dropout=0.3
    )

    trainer = ClassifierTrainer(model, train_loader, val_loader, test_loader, config)
    history = trainer.train()

    print("Classifier training complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")


if __name__ == "__main__":
    main()
