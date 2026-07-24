#!/bin/bash
# 双击这个文件就能打开数据平台（macOS的".command"文件双击会开个终端窗口执行脚本，
# 这是macOS上最接近"双击exe"的东西——不需要用户自己敲命令行）。
cd "$(dirname "$0")"
../phase1/.venv/bin/streamlit run app.py
