import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable
import json
import h5py


class CSIDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        activities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
        spectrogram_size: Tuple[int, int] = (128, 128)
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.spectrogram_size = spectrogram_size

        self.activities = activities or [
            "walk", "run", "sit", "stand", "fall",
            "lie_down", "wave", "jump", "crouch", "empty"
        ]
        self.activity_to_idx = {act: i for i, act in enumerate(self.activities)}

        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.data_dir / self.split

        if split_dir.exists():
            for activity in self.activities:
                activity_dir = split_dir / activity
                if activity_dir.exists():
                    for file_path in activity_dir.glob("*.npy"):
                        self.samples.append((str(file_path), self.activity_to_idx[activity]))

        if not self.samples:
            self._create_synthetic_samples()

    def _create_synthetic_samples(self) -> None:
        np.random.seed(42)
        num_samples_per_class = 100

        for activity in self.activities:
            idx = self.activity_to_idx[activity]
            for i in range(num_samples_per_class):
                self.samples.append((f"synthetic_{activity}_{i}", idx))

    def _generate_synthetic_spectrogram(self, activity_idx: int) -> np.ndarray:
        h, w = self.spectrogram_size
        spectrogram = np.random.randn(3, h, w).astype(np.float32) * 0.1

        freq_pattern = activity_idx / len(self.activities)
        for c in range(3):
            for i in range(h):
                spectrogram[c, i, :] += np.sin(2 * np.pi * freq_pattern * i / h + c) * 0.5

        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

        return spectrogram

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        file_path, label = self.samples[idx]

        if file_path.startswith("synthetic_"):
            spectrogram = self._generate_synthetic_spectrogram(label)
        else:
            spectrogram = np.load(file_path)

        if spectrogram.ndim == 2:
            spectrogram = np.stack([spectrogram] * 3, axis=0)
        elif spectrogram.shape[-1] == 3:
            spectrogram = np.transpose(spectrogram, (2, 0, 1))

        spectrogram = torch.from_numpy(spectrogram).float()

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        activity_name = self.activities[label]

        return spectrogram, label, activity_name


class CSIStreamDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        sequence_length: int = 10,
        transform: Optional[Callable] = None,
        activities: Optional[List[str]] = None
    ):
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.transform = transform

        self.activities = activities or [
            "walk", "run", "sit", "stand", "fall",
            "lie_down", "wave", "jump", "crouch", "empty"
        ]
        self.activity_to_idx = {act: i for i, act in enumerate(self.activities)}

        self.samples: List[Tuple[int, int, int]] = []
        self._index_sequences()

    def _index_sequences(self) -> None:
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                for activity in self.activities:
                    if activity in f:
                        group = f[activity]
                        num_samples = group['spectrograms'].shape[0]

                        for i in range(0, num_samples - self.sequence_length + 1):
                            self.samples.append((
                                self.activity_to_idx[activity],
                                i,
                                min(i + self.sequence_length, num_samples)
                            ))
        except FileNotFoundError:
            self._create_synthetic_samples()

    def _create_synthetic_samples(self) -> None:
        for activity in self.activities:
            idx = self.activity_to_idx[activity]
            for i in range(50):
                self.samples.append((idx, i * self.sequence_length, (i + 1) * self.sequence_length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        activity_idx, start_idx, end_idx = self.samples[idx]

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                activity = self.activities[activity_idx]
                spectrograms = f[activity]['spectrograms'][start_idx:end_idx]
        except (FileNotFoundError, KeyError):
            spectrograms = np.random.randn(self.sequence_length, 3, 128, 128).astype(np.float32) * 0.1

        spectrograms = torch.from_numpy(spectrograms).float()

        if self.transform is not None:
            transformed = []
            for i in range(spectrograms.shape[0]):
                transformed.append(self.transform(spectrograms[i]))
            spectrograms = torch.stack(transformed)

        return spectrograms, activity_idx


class CSIPairDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        activities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        num_pairs_per_activity: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_pairs_per_activity = num_pairs_per_activity

        self.activities = activities or [
            "walk", "run", "sit", "stand", "fall",
            "lie_down", "wave", "jump", "crouch", "empty"
        ]

        self.activity_samples: Dict[str, List[str]] = {act: [] for act in self.activities}
        self._load_samples()

        self.pairs: List[Tuple[str, str, int]] = []
        self._create_pairs()

    def _load_samples(self) -> None:
        for activity in self.activities:
            activity_dir = self.data_dir / activity
            if activity_dir.exists():
                for file_path in activity_dir.glob("*.npy"):
                    self.activity_samples[activity].append(str(file_path))

        for activity in self.activities:
            if not self.activity_samples[activity]:
                for i in range(50):
                    self.activity_samples[activity].append(f"synthetic_{activity}_{i}")

    def _create_pairs(self) -> None:
        for activity in self.activities:
            samples = self.activity_samples[activity]
            if len(samples) < 2:
                continue

            for _ in range(self.num_pairs_per_activity // 2):
                idx1, idx2 = np.random.choice(len(samples), 2, replace=False)
                self.pairs.append((samples[idx1], samples[idx2], 1))

            other_activities = [a for a in self.activities if a != activity]
            for _ in range(self.num_pairs_per_activity // 2):
                other_activity = np.random.choice(other_activities)
                other_samples = self.activity_samples[other_activity]
                if other_samples:
                    idx1 = np.random.randint(len(samples))
                    idx2 = np.random.randint(len(other_samples))
                    self.pairs.append((samples[idx1], other_samples[idx2], 0))

    def _load_spectrogram(self, path: str) -> np.ndarray:
        if path.startswith("synthetic_"):
            return np.random.randn(3, 128, 128).astype(np.float32) * 0.1
        return np.load(path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path1, path2, label = self.pairs[idx]

        spec1 = torch.from_numpy(self._load_spectrogram(path1)).float()
        spec2 = torch.from_numpy(self._load_spectrogram(path2)).float()

        if self.transform is not None:
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)

        return spec1, spec2, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    activities: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = CSIDataset(data_dir, activities, transform, split="train")
    val_dataset = CSIDataset(data_dir, activities, transform, split="val")
    test_dataset = CSIDataset(data_dir, activities, transform, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
