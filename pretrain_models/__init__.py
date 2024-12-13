import sys
import os

# 获取当前文件所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录加入到Python解释器路径中
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
