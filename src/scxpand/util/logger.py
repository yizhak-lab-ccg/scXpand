import logging
import sys

import structlog

# Configure structlog with colored console output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

# Configure standard library logging for structlog integration
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if name is None:
        return structlog.get_logger()
    return structlog.get_logger(name)


def configure_logging(
    show_timestamp: bool = False,
    show_log_level: bool = False,
    show_logger_name: bool = True,
    colors: bool = True,
) -> None:
    """Configure structlog output format.

    Args:
        show_timestamp: Whether to show timestamp prefix
        show_log_level: Whether to show log level prefix (e.g., [info])
        show_logger_name: Whether to show logger name suffix (e.g., [scxpand.util.general_util])
        colors: Whether to use colored output
    """
    processors = [
        structlog.stdlib.filter_by_level,
    ]

    if show_logger_name:
        processors.append(structlog.stdlib.add_logger_name)

    if show_log_level:
        processors.append(structlog.stdlib.add_log_level)

    processors.extend(
        [
            structlog.stdlib.PositionalArgumentsFormatter(),
        ]
    )

    if show_timestamp:
        processors.append(
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)
        )

    processors.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(
                colors=colors,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
