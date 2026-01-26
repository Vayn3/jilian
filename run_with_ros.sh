#!/bin/bash
# 武汉话语音对话系统 ROS1 模式启动脚本

# 加载 ROS1 环境
source /opt/ros/noetic/setup.bash

# 加载 conda 环境
source ~/Software/anaconda3/bin/activate jilian

# 设置 Python 路径，确保能找到 ROS 的 Python 模块
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH

# 运行程序
python main.py "$@"
