import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    udp_host: str = "0.0.0.0"
    udp_port: int = 5500
    buffer_size: int = 2048


@dataclass
class CSIConfig:
    num_subcarriers: int = 64
    num_antennas: int = 1
    sample_rate: int = 200
    window_size: int = 200
    hop_size: int = 50
    fft_size: int = 256


@dataclass
class DSPConfig:
    enable_phase_sanitization: bool = True
    enable_amplitude_calibration: bool = True
    hampel_window: int = 5
    hampel_threshold: float = 3.0
    butterworth_order: int = 4
    lowpass_cutoff: float = 80.0
    highpass_cutoff: float = 0.3


@dataclass
class SpectrogramConfig:
    n_fft: int = 256
    hop_length: int = 16
    n_mels: int = 128
    fmin: float = 0.1
    fmax: float = 100.0
    power: float = 2.0
    normalize: bool = True


@dataclass
class RFEncoderConfig:
    input_channels: int = 1
    base_channels: int = 64
    num_blocks: int = 4
    embedding_dim: int = 512
    dropout: float = 0.1


@dataclass
class TextEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    freeze: bool = True


@dataclass
class WiCLIPConfig:
    temperature: float = 0.07
    projection_dim: int = 256


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    unet_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    attention_resolutions: list = field(default_factory=lambda: [16, 8])


@dataclass
class ModelConfig:
    rf_encoder: RFEncoderConfig = field(default_factory=RFEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    wiclip: WiCLIPConfig = field(default_factory=WiCLIPConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


@dataclass
class AugmentationConfig:
    time_mask_max: int = 30
    freq_mask_max: int = 20
    noise_std: float = 0.02
    scale_range: tuple = (0.8, 1.2)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.7
    smoothing_window: int = 5
    alert_cooldown: float = 10.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    save_dir: str = "logs"
    tensorboard: bool = True


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.name = "CSI-Sentinel"
        self.version = "5.0.0"
        self.device_id = "sentinel-001"

        self.network = NetworkConfig()
        self.csi = CSIConfig()
        self.dsp = DSPConfig()
        self.spectrogram = SpectrogramConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.logging = LoggingConfig()
        self.activities = ["walk", "run", "sit", "stand", "fall", "lie_down", "wave", "jump", "crouch", "empty"]

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        self._update_from_dict(data)

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        if "system" in data:
            self.name = data["system"].get("name", self.name)
            self.version = data["system"].get("version", self.version)
            self.device_id = data["system"].get("device_id", self.device_id)

        if "network" in data:
            self.network = NetworkConfig(**data["network"])

        if "csi" in data:
            self.csi = CSIConfig(**data["csi"])

        if "dsp" in data:
            self.dsp = DSPConfig(**data["dsp"])

        if "spectrogram" in data:
            self.spectrogram = SpectrogramConfig(**data["spectrogram"])

        if "model" in data:
            model_data = data["model"]
            rf_enc = RFEncoderConfig(**model_data.get("rf_encoder", {}))
            text_enc = TextEncoderConfig(**model_data.get("text_encoder", {}))
            wiclip = WiCLIPConfig(**model_data.get("wiclip", {}))
            diff = DiffusionConfig(**model_data.get("diffusion", {}))
            self.model = ModelConfig(rf_encoder=rf_enc, text_encoder=text_enc, wiclip=wiclip, diffusion=diff)

        if "training" in data:
            train_data = data["training"].copy()
            aug_data = train_data.pop("augmentation", {})
            aug_config = AugmentationConfig(**aug_data)
            self.training = TrainingConfig(**train_data, augmentation=aug_config)

        if "inference" in data:
            self.inference = InferenceConfig(**data["inference"])

        if "logging" in data:
            self.logging = LoggingConfig(**data["logging"])

        if "activities" in data:
            self.activities = data["activities"]

    def save(self, config_path: str) -> None:
        data = {
            "system": {
                "name": self.name,
                "version": self.version,
                "device_id": self.device_id,
            },
            "network": self.network.__dict__,
            "csi": self.csi.__dict__,
            "dsp": self.dsp.__dict__,
            "spectrogram": self.spectrogram.__dict__,
            "model": {
                "rf_encoder": self.model.rf_encoder.__dict__,
                "text_encoder": self.model.text_encoder.__dict__,
                "wiclip": self.model.wiclip.__dict__,
                "diffusion": self.model.diffusion.__dict__,
            },
            "training": {
                **{k: v for k, v in self.training.__dict__.items() if k != "augmentation"},
                "augmentation": self.training.augmentation.__dict__,
            },
            "inference": self.inference.__dict__,
            "logging": self.logging.__dict__,
            "activities": self.activities,
        }

        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
