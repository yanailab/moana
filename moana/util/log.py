import sys
import logging

"""Logging functions."""

def configure_logger(name, log_stream=sys.stdout, log_file=None,
                     log_level=logging.INFO, keep_old_handlers=False,
                     propagate=False):
    """Configures and returns a logger.

    This function serves to simplify the configuration of a logger that
    writes to a file and/or to a stream (e.g., stdout).

    Parameters
    ----------
    name: str
        The name of the logger. Typically set to ``__name__``.
    log_stream: a stream object, optional
        The stream to write log messages to. If ``None``, do not write to any
        stream. The default value is `sys.stdout`.
    log_file: str, optional
        The path of a file to write log messages to. If None, do not write to
        any file. The default value is ``None``.
    log_level: int, optional
        A logging level as `defined`__ in Python's logging module. The default
        value is `logging.INFO`.
    keep_old_handlers: bool, optional
        If set to ``True``, keep any pre-existing handlers that are attached to
        the logger. The default value is ``False``.
    propagate: bool, optional
        If set to ``True``, propagate the loggers messages to the parent
        logger. The default value is ``False``.

    Returns
    -------
    `logging.Logger`
        The logger.

    Notes
    -----
    Note that if ``log_stream`` and ``log_file`` are both ``None``, no handlers
    will be created.

    __ loglvl_

    .. _loglvl: https://docs.python.org/2/library/logging.html#logging-levels

    """
    # create a child logger
    logger = logging.getLogger(name)

    # set the logger's level
    logger.setLevel(log_level)

    # set the logger's propagation attribute
    logger.propagate = propagate

    if not keep_old_handlers:
        # remove previously attached handlers
        logger.handlers = []

    # create the formatter
    log_fmt = '[%(asctime)s] (%(name)s) %(levelname)s: %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_fmt, log_datefmt)

    # create and attach the handlers

    if log_stream is not None:
        # create a StreamHandler
        stream_handler = logging.StreamHandler(log_stream)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        # create a FileHandler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_stream is None and log_file is None:
        # "no handler" => use NullHandler
        logger.addHandler(logging.NullHandler())
      
    return logger


def get_logger(name='', log_stream=None, log_file=None,
               quiet=False, verbose=False):
    """Convenience function for getting a logger."""

    # configure root logger
    log_level = logging.INFO
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG

    if log_stream is None:
        log_stream = sys.stdout

    new_logger = configure_logger(name, log_stream=log_stream,
                                  log_file=log_file, log_level=log_level)

    return new_logger
