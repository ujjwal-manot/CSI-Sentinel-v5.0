import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config, get_config
from utils.logger import setup_logger
from utils.helpers import set_seed, get_device


def run_server(config: Config):
    from server.realtime_processor import RealtimeProcessor
    from models.inference import InferenceEngine
    from models.wi_clip import WiCLIP

    logger = setup_logger("csi_server", config.logging.level)
    logger.info("Starting CSI-Sentinel Server...")

    processor = RealtimeProcessor(
        host=config.network.udp_host,
        port=config.network.udp_port,
        sample_rate=config.csi.sample_rate,
        window_size=config.csi.window_size,
        hop_size=config.csi.hop_size
    )

    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        logger.info(f"Loading model from {checkpoint_path}")
        engine = InferenceEngine.from_checkpoint(
            str(checkpoint_path),
            model_type="wiclip",
            confidence_threshold=config.inference.confidence_threshold,
            smoothing_window=config.inference.smoothing_window
        )
        processor.set_inference_model(engine.model)
    else:
        logger.warning("No trained model found. Running without inference.")

    def on_inference(result):
        logger.info(f"Activity: {result.activity} | Confidence: {result.confidence:.2f}")
        if result.activity == "fall" and result.confidence >= config.inference.confidence_threshold:
            logger.warning("FALL DETECTED!")

    processor.register_inference_callback(on_inference)

    logger.info(f"Listening on {config.network.udp_host}:{config.network.udp_port}")
    processor.start()

    try:
        import time
        while True:
            stats = processor.get_stats()
            logger.info(f"Packets: {stats['packets_received']} | Frames: {stats['frames_processed']}")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        processor.stop()


def run_train(config: Config, model_type: str):
    from training.dataset import create_dataloaders
    from training.augmentations import CSIAugmentor

    logger = setup_logger("csi_train", config.logging.level)
    logger.info(f"Starting training for {model_type}...")

    set_seed(42)
    device = get_device()
    logger.info(f"Using device: {device}")

    augmentor = CSIAugmentor(
        time_mask_max=config.training.augmentation.time_mask_max,
        freq_mask_max=config.training.augmentation.freq_mask_max,
        noise_std=config.training.augmentation.noise_std
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        "data",
        batch_size=config.training.batch_size,
        transform=augmentor,
        activities=config.activities
    )

    if model_type == "wiclip":
        from training.train_wiclip import WiCLIPTrainer
        from models.wi_clip import WiCLIP
        from models.rf_encoder import RFEncoder
        from models.text_encoder import TextEncoder

        rf_encoder = RFEncoder(
            input_channels=config.model.rf_encoder.input_channels,
            embedding_dim=config.model.rf_encoder.embedding_dim
        )
        text_encoder = TextEncoder(
            embedding_dim=config.model.text_encoder.embedding_dim,
            freeze=config.model.text_encoder.freeze
        )
        model = WiCLIP(
            rf_encoder=rf_encoder,
            text_encoder=text_encoder,
            projection_dim=config.model.wiclip.projection_dim
        )

        trainer = WiCLIPTrainer(
            model, train_loader, val_loader,
            config={
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "weight_decay": config.training.weight_decay,
                "gradient_clip": config.training.gradient_clip,
                "mixed_precision": config.training.mixed_precision,
                "save_dir": "checkpoints/wiclip"
            },
            device=device
        )
        trainer.train()

    elif model_type == "diffusion":
        from training.train_diffusion import DiffusionTrainer
        from models.csi_diffusion import CSIDiffusion, DiffusionUNet

        unet = DiffusionUNet(
            in_channels=3,
            out_channels=3,
            num_classes=len(config.activities)
        )
        model = CSIDiffusion(
            unet=unet,
            num_timesteps=config.model.diffusion.num_timesteps
        )

        trainer = DiffusionTrainer(
            model, train_loader, val_loader,
            config={
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "save_dir": "checkpoints/diffusion",
                "activities": config.activities
            },
            device=device
        )
        trainer.train()

    elif model_type == "classifier":
        from training.train_classifier import ClassifierTrainer
        from models.classifier import ActivityClassifier
        from models.rf_encoder import RFEncoder

        rf_encoder = RFEncoder(embedding_dim=config.model.rf_encoder.embedding_dim)
        model = ActivityClassifier(
            rf_encoder=rf_encoder,
            num_classes=len(config.activities)
        )

        trainer = ClassifierTrainer(
            model, train_loader, val_loader, test_loader,
            config={
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "save_dir": "checkpoints/classifier",
                "activities": config.activities
            },
            device=device
        )
        trainer.train()


def run_dashboard():
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/dashboard.py"])


def run_generate(config: Config, num_samples: int, activity: str):
    import torch
    from models.csi_diffusion import CSIDiffusion, DiffusionUNet

    logger = setup_logger("csi_generate", config.logging.level)
    logger.info(f"Generating {num_samples} samples for activity: {activity}")

    device = get_device()

    checkpoint_path = Path("checkpoints/diffusion/best_model.pt")
    if not checkpoint_path.exists():
        logger.error("No diffusion model found. Train one first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    unet = DiffusionUNet(in_channels=3, out_channels=3, num_classes=len(config.activities))
    model = CSIDiffusion(unet=unet)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    activity_idx = config.activities.index(activity) if activity in config.activities else 0
    class_labels = torch.full((num_samples,), activity_idx, device=device)

    logger.info("Generating samples using DDIM...")
    with torch.no_grad():
        samples = model.ddim_sample(
            batch_size=num_samples,
            channels=3,
            height=128,
            width=128,
            class_labels=class_labels,
            num_inference_steps=50,
            device=device
        )

    output_dir = Path("data/synthetic") / activity
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    for i, sample in enumerate(samples):
        np.save(output_dir / f"synthetic_{i:04d}.npy", sample.cpu().numpy())

    logger.info(f"Generated {num_samples} samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CSI-Sentinel v5.0")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    server_parser = subparsers.add_parser("server", help="Run the CSI server")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, choices=["wiclip", "diffusion", "classifier"], default="wiclip")

    dashboard_parser = subparsers.add_parser("dashboard", help="Run the Streamlit dashboard")

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    generate_parser.add_argument("--num-samples", type=int, default=100)
    generate_parser.add_argument("--activity", type=str, default="fall")

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = Config(str(config_path))
    else:
        config = Config()

    if args.command == "server":
        run_server(config)
    elif args.command == "train":
        run_train(config, args.model)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "generate":
        run_generate(config, args.num_samples, args.activity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
