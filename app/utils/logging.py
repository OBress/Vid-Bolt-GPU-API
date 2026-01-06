"""Logging configuration and utilities."""

import logging
import sys
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        # Base message
        message = super().format(record)

        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            import json

            extras = json.dumps(extra_fields)
            message = f"{message} | {extras}"

        return message


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = StructuredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set third-party loggers to WARNING
    for logger_name in ["boto3", "botocore", "urllib3", "s3transfer"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """Log a message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    logger.log(level, message, extra=context)
