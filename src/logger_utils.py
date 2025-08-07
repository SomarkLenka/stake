import logging
import sys
from pathlib import Path
import inspect
from typing import Optional
from logging.handlers import RotatingFileHandler

# Expose level constants
class logLev:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

def init_logger(
    name: Optional[str] = None,
    level: int = logLev.INFO,
    to_console: bool = True,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    is_orc: Optional[bool] = None,
    logs_dir: Path = Path.cwd() / "logs"
) -> logging.Logger:
    """
    Initializes and returns a logger that:
    - Writes to a rotating log file in `logs/` folder
    - Log file name based on whether the script is a worker or orchestrator
    - Keeps the last 3 logs using rotation on every run
    - Exposes `level.DEBUG`, `level.INFO`, etc., as importable constants
    """

    logger = logging.getLogger(name if name else "__main__")
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers

    # Determine script roles
    main_script = Path(sys.argv[0]).stem
    caller_script = Path(inspect.stack()[-1].filename).stem

    if is_orc is None:
        is_orc = main_script == caller_script

    log_file_base = f"{main_script if is_orc else caller_script}.log"

    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / log_file_base

    formatter = logging.Formatter(fmt)

    # Set up rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=1000000,  # Rotate immediately
        backupCount=2  # Keep 3 logs: .log, .log.1, .log.2
    )
    file_handler.doRollover()  # Force new log file each execution
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
