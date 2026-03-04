import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


_loggers = {}


def setup_logger(
    name: str = "csi_sentinel",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "csi_sentinel") -> logging.Logger:
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]
