import base64
import json
import os
import sys
import threading
import time

import requests

API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-vl:4b"
IMAGE_PATH = "/home/p9/Projects/embodiedrobot/visual_module/realtimeframe.jpg"

# 标记模型当前是否在回复中，避免堆积请求
_model_busy_lock = threading.Lock()
_model_busy = False


def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(callback_fn):
    global _model_busy
    # 如果已经有一次调用在进行，直接跳过这次，以只保留最新的一次
    with _model_busy_lock:
        if _model_busy:
            return
        _model_busy = True

    try:
        image_b64 = encode_image_to_base64(IMAGE_PATH)
    except FileNotFoundError:
        callback_fn("图片不存在，等待创建…\n")
        with _model_busy_lock:
            _model_busy = False
        return

    payload = {
        "model": MODEL_NAME,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "你在分析一组手术中的颅内图片，暗黄色是颅内背景，蓝色粉色明黄色分别代表病灶一号、二号、三号, 视野中可能出现不同的病灶组合。回答中不要出现颜色描述和背景描述，保持清晰简洁。"
            },
            {
                "role": "user",
                "content": "请描述病灶表现和病灶位置。",
                "images": [image_b64]
            }
        ]
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True)
        callback_fn("【检测到新图片 → 分析中…】\n")

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                delta = data.get("message", {}).get("content", "")
                if delta:
                    callback_fn(delta)

                if data.get("done", False):
                    callback_fn("\n\n【分析完成】\n\n")
                    break

            except Exception as e:
                callback_fn(f"[Error parsing stream]: {e}\n")
    finally:
        # 本轮回复结束，允许下一张最新图片触发分析
        with _model_busy_lock:
            _model_busy = False

def watch_image_file(callback_fn):
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(IMAGE_PATH)

            if last_mtime is None:
                last_mtime = mtime
            # 文件有更新且当前模型空闲时，启动一次新的分析
            if mtime != last_mtime:
                last_mtime = mtime
                threading.Thread(target=call_model, args=(callback_fn,), daemon=True).start()

        except FileNotFoundError:
            callback_fn("图片不存在，等待创建…\n")

        time.sleep(0.5)


def run_cli():
    print("=== CLI 模式：检测图片变化并输出模型回复 ===\n")
    def cli_print(text):
        print(text, end="")

    # 文件监听线程
    watch_image_file(cli_print)


def run_ui():
    from DialogWindow import DialogWindow
    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = DialogWindow()
    window.show()

    def ui_print(text):
        window.message_received.emit(text)

    threading.Thread(target=watch_image_file, args=(ui_print,), daemon=True).start()
    app.exec_()

# ------------------------
# 主入口：切换模式
# ------------------------
if __name__ == "__main__":
    MODE = "ui"   # 改成 "ui" 可切换 UI 模式
    if MODE == "ui":
        run_ui()
    else:
        run_cli()
