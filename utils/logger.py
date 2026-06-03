# utils/logger.py
import os
import sys
import logging
from datetime import datetime

def initialize_global_logging(config):
    """Sets up dual-destination logging (terminal + file) and global exception handling."""
    env_cfg = config.get("environment", {})
    output_dir = env_cfg.get("output_dir", "diagnostics_output")
    os.makedirs(output_dir, exist_ok=True)
    
    log_filename = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(output_dir, log_filename)
    
    # Configure root logger format matching enterprise standards
    log_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # --- READ LOGGING LEVEL FROM CONFIG ---
    pipeline_cfg = config.get("pipeline", {})
    yaml_level_str = pipeline_cfg.get("logging_level", "INFO").upper()
    
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    target_level = level_mapping.get(yaml_level_str, logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(target_level)
    
    # Stream Handler (Terminal Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (Disk Logging)
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Intercept unhandled exceptions globally
    def unhandled_exception_hook(exctype, value, traceback):
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Unhandled system exception encountered!", exc_info=(exctype, value, traceback))
        
    sys.excepthook = unhandled_exception_hook
    logging.info(f"Global pipeline logging initialized. Writing logs to: {log_filepath}")