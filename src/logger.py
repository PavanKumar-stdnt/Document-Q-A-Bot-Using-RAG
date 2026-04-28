# src/logger.py
"""
Centralised logging using loguru.
Import `logger` from here in every module.
"""

import sys
from loguru import logger

# Remove the default handler and add a clean one
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
    colorize=True,
)
logger.add(
    "rag_chatbot.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
)

__all__ = ["logger"]
