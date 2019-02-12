import logging
import os
import datetime


def get_logger():
    log_dir = "log"
    log_level = os.environ.get("LOG_LEVEL") or "INFO"
    logger = logging.getLogger("complex-grad-norm")
    logger.setLevel(log_level)
    logger.handlers = []
    fh = logging.FileHandler(
        os.path.sep.join(
            [
                log_dir if os.path.isdir(log_dir) and os.access(log_dir, os.W_OK) else "/tmp",
                datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d"),
            ]
        )
    )
    fh.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
