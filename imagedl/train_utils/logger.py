"""Logging utils"""
import logging
import sys
from pathlib import Path


def setup_logger(job_dir: Path) -> None:
    """Setup logger"""
    fmt = '[%(levelname)s] %(asctime)s %(message)s'
    logging.basicConfig(filename=str(job_dir / 'train.log'),
                        level=logging.INFO, format=fmt)


def info(message: str) -> None:
    """Info message"""
    print(message)
    logging.info(message)


def error(message: str) -> None:
    """Error message"""
    print(message, file=sys.stderr)
    logging.error(message)
