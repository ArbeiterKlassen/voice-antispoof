"""RawNet2CAA 的 import shim（原为软链，Windows 检出对软链支持不佳故改为实体文件）。

score_group.py 用 importlib 以 "models.RawNet2CAA" 之名加载本模块，
此处把同目录上两级的 code/rawnet2_caa.py 重新导出。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rawnet2_caa import Model  # noqa: F401,E402
