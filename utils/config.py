import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    pass


@dataclass(frozen=True)
class NetworkConfig:
    udp_host: str = "0.0.0.0"
    udp_port: int = 5500
    buffer_size: int = 2048

    def __post_init__(self) -> None:
        if not 1024 <= self.udp_port <= 65535:
            raise ConfigValidationError(f"udp_port must be 1024-65535, got {self.udp_port}")
        if self.buffer_size < 64:
            raise ConfigValidationError(f"buffer_size must be >= 64, got {self.buffer_size}")


@dataclass(frozen=True)
class CSIConfig:
    num_subcarriers: int = 64
    num_antennas: int = 1
    sample_rate: int = 200
    window_size: int = 200
    hop_size: int = 50
    fft_size: int = 256

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigValidationError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.window_size <= 0:
            raise ConfigValidationError(f"window_size must be positive, got {self.window_size}")
        if self.hop_size <= 0 or self.hop_size > self.window_size:
            raise ConfigValidationError(f"hop_size must be in (0, {self.window_size}], got {self.hop_size}")


@dataclass(frozen=True)
class DSPConfig:
    enable_phase_sanitization: bool = True
    enable_amplitude_calibration: bool = True
    hampel_window: int = 5
    hampel_threshold: float = 3.0
    butterworth_order: int = 4
    lowpass_cutoff: float = 80.0
    highpass_cutoff: float = 0.3

    def __post_init__(self) -> None:
        if self.hampel_window < 1:
            raise ConfigValidationError(f"hampel_window must be >= 1, got {self.hampel_window}")
        if self.lowpass_cutoff <= self.highpass_cutoff:
            raise ConfigValidationError("lowpass_cutoff must be > highpass_cutoff")


@dataclass(frozen=True)
class SpectrogramConfig:
    n_fft: int = 256
    hop_length: int = 16
    n_mels: int = 128
    fmin: float = 0.1
    fmax: float = 100.0
    power: float = 2.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or (self.n_fft & (self.n_fft - 1)) != 0:
            raise ConfigValidationError(f"n_fft must be positive power of 2, got {self.n_fft}")
        if self.fmax <= self.fmin:
            raise ConfigValidationError("fmax must be > fmin")


@dataclass(frozen=True)
class RFEncoderConfig:
    input_channels: int = 1
    base_channels: int = 64
    num_blocks: int = 4
    embedding_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigValidationError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    freeze: bool = True


@dataclass(frozen=True)
class WiCLIPConfig:
    temperature: float = 0.07
    projection_dim: int = 256

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ConfigValidationError(f"temperature must be positive, got {self.temperature}")


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    unet_channels: Tuple[int, ...] = (64, 128, 256, 512)
    attention_resolutions: Tuple[int, ...] = (16, 8)

    def __post_init__(self) -> None:
        if self.beta_end <= self.beta_start:
            raise ConfigValidationError("beta_end must be > beta_start")


@dataclass
class ModelConfig:
    rf_encoder: RFEncoderConfig = field(default_factory=RFEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    wiclip: WiCLIPConfig = field(default_factory=WiCLIPConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


@dataclass(frozen=True)
class AugmentationConfig:
    time_mask_max: int = 30
    freq_mask_max: int = 20
    noise_std: float = 0.02
    scale_range: Tuple[float, float] = (0.8, 1.2)

    def __post_init__(self) -> None:
        if self.scale_range[0] >= self.scale_range[1]:
            raise ConfigValidationError("scale_range[0] must be < scale_range[1]")


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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ConfigValidationError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass(frozen=True)
class InferenceConfig:
    confidence_threshold: float = 0.7
    smoothing_window: int = 5
    alert_cooldown: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigValidationError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    save_dir: str = "logs"
    tensorboard: bool = True

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ConfigValidationError(f"level must be one of {valid_levels}, got {self.level}")


class Config:
    VALID_ACTIVITIES = frozenset([
        "walk", "run", "sit", "stand", "fall", "lie_down",
        "wave", "jump", "crouch", "empty", "unknown"
    ])

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.name: str = "CSI-Sentinel"
        self.version: str = "5.0.0"
        self.device_id: str = "sentinel-001"

        self.network = NetworkConfig()
        self.csi = CSIConfig()
        self.dsp = DSPConfig()
        self.spectrogram = SpectrogramConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.logging = LoggingConfig()
        self._activities: List[str] = [
            "walk", "run", "sit", "stand", "fall",
            "lie_down", "wave", "jump", "crouch", "empty"
        ]

        if config_path:
            self.load(config_path)

    @property
    def activities(self) -> List[str]:
        return self._activities.copy()

    @activities.setter
    def activities(self, value: List[str]) -> None:
        if not value:
            raise ConfigValidationError("activities list cannot be empty")
        self._activities = list(value)

    @property
    def num_classes(self) -> int:
        return len(self._activities)

    def load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in config file: {e}") from e

        if data is None:
            raise ConfigValidationError("Config file is empty")

        self._update_from_dict(data)

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        if "system" in data:
            sys_data = data["system"]
            self.name = str(sys_data.get("name", self.name))
            self.version = str(sys_data.get("version", self.version))
            self.device_id = str(sys_data.get("device_id", self.device_id))

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
            wiclip_data = model_data.get("wiclip", {})
            wiclip = WiCLIPConfig(**wiclip_data)
            diff_data = model_data.get("diffusion", {})
            if "unet_channels" in diff_data:
                diff_data["unet_channels"] = tuple(diff_data["unet_channels"])
            if "attention_resolutions" in diff_data:
                diff_data["attention_resolutions"] = tuple(diff_data["attention_resolutions"])
            diff = DiffusionConfig(**diff_data)
            self.model = ModelConfig(rf_encoder=rf_enc, text_encoder=text_enc, wiclip=wiclip, diffusion=diff)

        if "training" in data:
            train_data = data["training"].copy()
            aug_data = train_data.pop("augmentation", {})
            if "scale_range" in aug_data:
                aug_data["scale_range"] = tuple(aug_data["scale_range"])
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
            "network": asdict(self.network),
            "csi": asdict(self.csi),
            "dsp": asdict(self.dsp),
            "spectrogram": asdict(self.spectrogram),
            "model": {
                "rf_encoder": asdict(self.model.rf_encoder),
                "text_encoder": asdict(self.model.text_encoder),
                "wiclip": asdict(self.model.wiclip),
                "diffusion": {
                    **asdict(self.model.diffusion),
                    "unet_channels": list(self.model.diffusion.unet_channels),
                    "attention_resolutions": list(self.model.diffusion.attention_resolutions),
                },
            },
            "training": {
                **{k: v for k, v in asdict(self.training).items() if k != "augmentation"},
                "augmentation": {
                    **asdict(self.training.augmentation),
                    "scale_range": list(self.training.augmentation.scale_range),
                },
            },
            "inference": asdict(self.inference),
            "logging": asdict(self.logging),
            "activities": self._activities,
        }

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except IOError as e:
            raise ConfigValidationError(f"Failed to save config: {e}") from e

    def validate(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "device_id": self.device_id,
            "network": asdict(self.network),
            "csi": asdict(self.csi),
            "dsp": asdict(self.dsp),
            "spectrogram": asdict(self.spectrogram),
            "inference": asdict(self.inference),
            "logging": asdict(self.logging),
            "activities": self._activities,
        }


_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reset_config() -> None:
    global _config_instance
    _config_instance = None
