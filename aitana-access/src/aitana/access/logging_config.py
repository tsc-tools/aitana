"""
Logging configuration for the aitana package.

This module sets up logging to output to stdout/stderr without file handlers.
"""
import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configure logging for the aitana package.

    By default, INFO and DEBUG messages go to stdout,
    WARNING, ERROR, and CRITICAL go to stderr.

    Parameters:
    -----------
        level : int
            The logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = logging.getLogger('aitana')
    logger.setLevel(level)

    logger.propagate = False
    logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


def get_logger(name):
    """
    Get a logger for a specific module within aitana.

    Parameters:
    -----------
        name : str
            The name of the module (e.g., 'aitana.volcanoes.ruapehu')

    Returns:
    --------
        logging.Logger
            A configured logger instance
    """
    return logging.getLogger(name)
