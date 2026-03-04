import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from logging.handlers import RotatingFileHandler
import threading


_loggers: Dict[str, logging.Logger] = {}
_lock = threading.Lock()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "csi_sentinel",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    use_colors: bool = True
) -> logging.Logger:
    with _lock:
        if name in _loggers:
            return _loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        logger.propagate = False

        base_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            if use_colors and sys.stdout.isatty():
                formatter = ColoredFormatter(fmt=base_format, datefmt=date_format)
            else:
                formatter = logging.Formatter(fmt=base_format, datefmt=date_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if log_dir:
            try:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_path / f"{name}_{timestamp}.log"

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_formatter = logging.Formatter(fmt=base_format, datefmt=date_format)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except (OSError, IOError) as e:
                logger.warning(f"Failed to create file handler: {e}")

        _loggers[name] = logger
        return logger


def get_logger(name: str = "csi_sentinel") -> logging.Logger:
    with _lock:
        if name not in _loggers:
            return setup_logger(name)
        return _loggers[name]


def set_log_level(name: str, level: str) -> None:
    with _lock:
        if name in _loggers:
            _loggers[name].setLevel(getattr(logging, level.upper(), logging.INFO))


def shutdown_logging() -> None:
    with _lock:
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
