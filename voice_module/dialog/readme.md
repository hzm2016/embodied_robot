```
cd /home/p9/Projects/embodiedrobot/voice_module/dialog
conda activate py-xiaokai
python dialog.py
```
- 每0.5s查询`IMAGE_PATH`的图片是否更新，如果更新调用qwen3vl4b
- 点击窗口x关闭程序
- DialogWindow固定了窗口大小和气泡宽度

server log
```
docker logs xiaozhi-esp32-server -f
```
```
cd /home/p9/Projects/xiaokai/py-xiaokai/
conda activate py-xiaokai
python main.py
```