# utils/logger.py
import logging
import os
from config import LOGS

def get_logger(name=__name__, logfile=None, level=logging.INFO):
    """
    Returns a logger that writes to console and optionally to a log file in LOGS/.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if logfile:
        os.makedirs(LOGS, exist_ok=True)
        fh = logging.FileHandler(os.path.join(LOGS, logfile))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
