"""保留Source库存和外部导入位置；Runtime实现仅由werewolf.game_runtime维护。

本文件没有独立游戏CLI，也不复制装配、角色分配或执行循环。
"""
from werewolf.game_runtime import (
    PUBLIC_SPEECH_RUNTIME_PHASES, assign_agents, build_runtime, eval,
)
