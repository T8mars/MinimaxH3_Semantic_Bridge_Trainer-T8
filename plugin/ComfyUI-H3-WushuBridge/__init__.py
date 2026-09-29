"""ComfyUI-H3-WushuBridge：MiniMax H3 武打语义逻辑翻译桥。

节点注册在 ``MiniMax H3/Wushu Bridge`` 分类下，详见 README。
"""

from .wushu_bridge.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.1.0"
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
