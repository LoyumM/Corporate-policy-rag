import logging
import json
import sys
from pathlib import Path
import numpy as np

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }

        if hasattr(record, "extra_data"):
            log_record.update(self._sanitize(record.extra_data))

        return json.dumps(log_record, default=str)

    def _sanitize(self, obj):
        """
        Recursively convert numpy types into native Python types.
        """
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]

        if isinstance(obj, np.generic):
            return obj.item()

        return obj


def setup_logger():
    logger = logging.getLogger("policy_rag")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger