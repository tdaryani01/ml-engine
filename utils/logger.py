# utils/logger.py
import os
import sys
import logging
from datetime import datetime
from config.schema import PipelineConfig

def initialize_global_logging(cfg: PipelineConfig) -> None:
    """
    Sets up dual-destination logging (terminal + file) and global exception handling
    using properties hydrated via the pipeline's strongly-typed configuration schema.
    """
    # 1. Enforce output target directory creation boundaries from the meta sub-config
    meta_cfg = cfg.meta
    os.makedirs(meta_cfg.output_dir, exist_ok=True)
    
    # 2. Build unique rolling timestamp filename layouts matching your conventions
    log_filename = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(meta_cfg.output_dir, log_filename)
    
    # 3. Configure root logger format matching enterprise standards
    log_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 4. Read log suppressions and target levels safely from config
    if meta_cfg.suppress_logging:
        logging.disable(logging.CRITICAL)
        return
        
    yaml_level_str = meta_cfg.logging_level.upper()
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
    root_logger.handlers.clear()  # Clear any default boilerplate tracking streams
    
    # Stream Handler (Terminal Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (Disk Logging)
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # 5. Intercept unhandled exceptions globally to maintain robust execution histories
    def unhandled_exception_hook(exctype, value, traceback):
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Unhandled system exception encountered!", exc_info=(exctype, value, traceback))
        
    sys.excepthook = unhandled_exception_hook
    logging.info(f"Global pipeline logging initialized. Writing logs to: {log_filepath}")