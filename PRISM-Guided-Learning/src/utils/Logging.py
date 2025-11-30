from config.Settings import LOGGING_PATH, EVAL_PATH
import os
import logging
from datetime import datetime
from typing import Optional


def create_run_directory(prefix: str = "run") -> str:
    """
    Create a timestamped directory for a parallel run.
    
    Args:
        prefix: Prefix for the directory name (e.g., 'RL_counterfactual', 'vanilla_LLM')
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = os.path.join(LOGGING_PATH, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logger(log_filename: str, run_dir: Optional[str] = None, 
                 include_timestamp: bool = True, console_output: bool = False, log_path = LOGGING_PATH):
    """
    Setup a logger with file and console handlers.
    
    Args:
        log_filename: Name for the log file (without extension)
        run_dir: Optional shared directory for the log file. If None, uses LOGGING_PATH directly.
        include_timestamp: Whether to include timestamp in filename (default True for backward compatibility)
        log_path: Base path for log files (default is LOGGING_PATH)
        
    Returns:
        Configured logger instance
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
        filename = f"{log_filename}_{timestamp}.log"
    else:
        filename = f"{log_filename}.log"
    
    if run_dir:
        log_path = os.path.join(run_dir, filename)
    else:
        log_path = os.path.join(log_path, filename)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Create unique logger name to avoid conflicts between workers
    logger_name = f"{log_filename}_{id(log_filename)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger