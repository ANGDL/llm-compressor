"""Shared human-readable logger for streaming PTQ."""

from loguru import logger

streaming_logger = logger.patch(
    lambda record: record.update(function="streaming")
)
