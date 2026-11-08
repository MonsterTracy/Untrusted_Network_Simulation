"""网络结果的最小区分；任何可能送达的请求都不能自动重发。"""


class NotExecuted(RuntimeError):
    """连接前失败或明确拒绝；只能在显式恢复时再次尝试。"""


class OutcomeUnknown(RuntimeError):
    """可能已经送达，保留证据并停止，不伪造预测。"""
