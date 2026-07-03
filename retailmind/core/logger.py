"""日志配置模块。

提供集中化的日志配置，所有核心模块通过 get_logger() 获取统一格式的 logger。
日志同时输出到控制台和文件（logs/retailmind.log），便于调试和审计。
"""

import logging
import os
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "retailmind.log"
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """配置 root logger（仅执行一次）。

    创建日志目录，添加 StreamHandler 和 FileHandler，
    设置统一的日志格式。日志级别通过环境变量 LOG_LEVEL 控制（默认 INFO）。
    """
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 日志目录创建失败时仅使用控制台输出
        pass

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(_LOG_FILE, encoding="utf-8"))
    except OSError:
        pass

    root = logging.getLogger("retailmind")
    root.setLevel(level)
    # 避免重复添加 handler（重新加载模块场景）
    if not root.handlers:
        for h in handlers:
            h.setFormatter(formatter)
            root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """获取统一命名的 logger。

    Args:
        name: 模块名称，通常传 __name__。

    Returns:
        配置好的 logging.Logger 实例，名称形如 "retailmind.core.xxx"。
    """
    _configure_root_logger()
    if not name.startswith("retailmind"):
        name = f"retailmind.{name}"
    return logging.getLogger(name)
