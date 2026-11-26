# 跌倒监测系统 v2.1（摄像头主判 + WS63 雷达辅助）

基于 v2 方案构建的可直接运行的 PC 端多模态跌倒监测系统：
- **摄像头主判**（`pc/camera_source.py`）：支持本机外接 USB 摄像头（默认索引 `1`，即 `HD video`）、WiFi RTSP/MJPEG 摄像头、本地视频文件；使用 `YOLOv8n-pose`（CUDA 加速）+ 三判据状态机（宽高比 `w/h`、躯干角、髋部垂直速度 + 关键点丢失兜底 + 1.2 s 抗反光抖动容忍 + 远景小框过滤与跌倒目标锁定）。
- **雷达接收与规则复用**（`fall_recog/` + `pc/udp_source.py` + `pc/serial_udp_bridge.py`）：UDP `9001` 端口接收 WS63 原始帧（支持 `[RADAR_SAMPLE]` 文本与 JSON），并附带串口转 UDP 桥接兜底脚本。
- **双模融合仲裁**（`pc/fusion.py`）：2 s 时间窗口双确认、`radar_veto` 否决记录、5 s 雷达超时自动 `DEGRADED` 降级单摄像头模式。
- **Web 监控面板**（`pc/server.py` + `pc/web/index.html`）：支持**标准模式（3s/8s）↔ 快速演示模式（1s/3s）**热切换、倒地计时进度条、**同步录制视频 + 雷达日志**自动生成 `manifest.csv`、**alarm / veto 自动抓拍留证**，电脑和手机同网段即可访问。

## 快速启动

直接双击 `start.bat` 即可一键启动（自动探测已装依赖的 Python，4 秒后自动打开监控页面），或在终端传参：

```powershell
# 1. 默认启动（外接 USB 摄像头 index 1 + UDP 9001 + Web 8000）
.\start.bat
# 或 PowerShell:
.\start.ps1

# 2. 快速演示模式启动（倒地 1s 疑似 / 3s 报警，也可在网页右上角随时切换）
.\start.bat --profile demo

# 3. 查看本机摄像头索引列表
.\start.bat --list-cameras

# 4. 切换为笔记本自带摄像头 (index 0)、WiFi RTSP/MJPEG 摄像头或本地视频
.\start.bat --source 0
.\start.bat --source "rtsp://admin:密码@192.168.1.60:554/stream2"
.\start.bat --source "http://192.168.43.1:8080/video"
.\start.bat --source "test_fall.mp4"

# 5. 同时弹出本地 OpenCV 预览窗口
.\start.bat --show-cv
```

启动后在浏览器打开 `http://127.0.0.1:8000`（手机连同一热点时，打开终端或页面顶栏显示的局域网 IP 地址）。

## WS63 板端接入（WiFi UDP 或串口兜底）

向 PC IP 的 UDP `9001` 端口发送任意一种格式即可：

```text
[RADAR_SAMPLE] gear=2 rng=1.85 vel=-1.20 snr=18.5 hm=1
```

或 JSON：

```json
{"gear": 2, "rng": 1.85, "vel": -1.20, "snr": 18.5, "human": 1}
```

如果 WS63 WiFi UDP 固件尚未烧录，可先用串口转 UDP 桥接脚本把板子串口日志转发到本地 `9001` 端口：

```powershell
python -m pc.serial_udp_bridge --list-ports
python -m pc.serial_udp_bridge --port COM3 --baud 115200
```

也可以不接板子直接模拟雷达跌倒包测试融合效果：

```powershell
python -m pc.udp_source --simulate-fall
```

## 同步采数与消融实验表（方案第 10 节）

1. 在监控网页右侧「同步采数录制」区选择场景（`真跌倒 / 躺沙发 / 坐下弯腰 / 正常走动 / 空场景`），点击**开始同步录制** → 完成动作后点**停止录制**。
2. 系统自动把原视频 `.mp4`、同时间轴雷达日志 `.jsonl` 存入 `datasets/recorded/` 并追加到 `datasets/recorded/manifest.csv`。
3. 采完 10–15 段后一键生成消融对比表：

```powershell
python -m pc.eval_ablation --manifest datasets/recorded/manifest.csv
```

## 测试

```powershell
python -m unittest discover -s tests -v
```
