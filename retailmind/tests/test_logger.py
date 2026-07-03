"""logger 模块单元测试。

验证日志配置和 get_logger 行为。
"""
import logging
import os
from core.logger import get_logger, _configure_root_logger


class TestGetLogger:
    """get_logger 函数测试。"""

    def test_returns_logger_instance(self):
        """应返回 logging.Logger 实例。"""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefixed_with_retailmind(self):
        """logger 名称应以 retailmind 为前缀。"""
        logger = get_logger("core.foo")
        assert logger.name == "retailmind.core.foo"

    def test_get_logger_idempotent(self):
        """多次调用应返回同一个 logger 实例。"""
        l1 = get_logger("same.module")
        l2 = get_logger("same.module")
        assert l1 is l2

    def test_module_with_retailmind_prefix_not_doubled(self):
        """已带 retailmind 前缀的名称不应重复。"""
        logger = get_logger("retailmind.core.bar")
        assert logger.name == "retailmind.core.bar"


class TestLoggerConfiguration:
    """日志配置测试。"""

    def test_root_logger_has_handlers(self):
        """retailmind root logger 应已配置 handler。"""
        _configure_root_logger()
        root = logging.getLogger("retailmind")
        assert len(root.handlers) >= 1

    def test_configure_root_logger_idempotent(self):
        """_configured 标志应防止重复配置 handler。"""
        import core.logger as lg
        before = len(logging.getLogger("retailmind").handlers)
        lg._configured = True  # 模拟已配置
        lg._configure_root_logger()
        after = len(logging.getLogger("retailmind").handlers)
        assert before == after

    def test_log_level_respects_env_var(self, monkeypatch):
        """LOG_LEVEL 环境变量应控制日志级别。"""
        # 重置配置状态
        import core.logger as lg
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setattr(lg, "_configured", False)
        # 清除已有 handler
        root = logging.getLogger("retailmind")
        for h in list(root.handlers):
            root.removeHandler(h)
        lg._configure_root_logger()
        assert root.level == logging.DEBUG
