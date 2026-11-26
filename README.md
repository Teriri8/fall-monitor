# 跌倒监测系统 v2（摄像头主判 + WS63 雷达辅助）

基于 v2 方案构建的可直接运行的 PC 端系统：
- **摄像头主判**（`pc/camera_source.py`）：支持本机外接 USB 摄像头（默认索引 `1`，即 `HD video`）、WiFi RTSP/MJPEG 摄像头、本地视频文件；使用 `YOLOv8n-pose`（RTX 3060 CUDA 加速）+ 三判据状态机（宽高比 `w/h`、躯干角、髋部垂直速度 + 关键点丢失兜底）。
- **雷达接收与规则复用**（`fall_recog/` + `pc/udp_source.py`）：UDP `9001` 端口接收 WS63 原始帧（支持 `[RADAR_SAMPLE]` 文本与 JSON），复用 `FallDetector` 状态机。
- **双模融合仲裁**（`pc/fusion.py`）：2 s 时间窗口确认、`radar_veto` 否决记录、5 s 雷达超时自动 `DEGRADED` 降级单摄像头模式。
- **Web 监控面板**（`pc/server.py` + `pc/web/index.html`）：FastAPI + SSE 实时推送 + MJPEG 视频流，兼容 9 字段事件协议，电脑和手机同一热点即可浏览器查看。

## 快速启动

直接双击 `start.bat` 即可一键启动（会自动在 4 秒后打开浏览器监控页面），或在终端使用命令行脚本（自动调用安装好 CUDA PyTorch / Ultralytics 的 Python 3.11）：

```powershell
# 1. 默认启动（外接 USB 摄像头 index 1 + UDP 9001 + Web 8000）
.\start.bat
# 或 PowerShell:
.\start.ps1

# 2. 查看本机摄像头索引列表
.\start.bat --list-cameras

# 3. 切换为笔记本自带摄像头 (index 0) 或外接 WiFi RTSP/MJPEG 摄像头
.\start.bat --source 0
.\start.bat --source "rtsp://admin:密码@192.168.1.60:554/stream2"
.\start.bat --source "http://192.168.43.1:8080/video"

# 4. 同时弹出本地 OpenCV 预览窗口
.\start.bat --show-cv
```

启动后在浏览器打开终端打印的地址（本机 `http://127.0.0.1:8000`，同网段手机打开局域网 IP 地址）。

## WS63 板端 UDP 上报格式

向 PC IP 的 UDP `9001` 端口发送任意一种格式即可：

```text
[RADAR_SAMPLE] gear=2 rng=1.85 vel=-1.20 snr=18.5 hm=1
```

或 JSON：

```json
{"gear": 2, "rng": 1.85, "vel": -1.20, "snr": 18.5, "human": 1}
```

可用以下命令不接板子直接模拟雷达跌倒包测试融合效果：

```powershell
C:\Users\蔡华升\AppData\Local\Programs\Python\Python311\python.exe -m pc.udp_source --simulate-fall
```

## 与 B 的代码合并说明

- 事件协议严格锁定在 `fall_recog/protocol.py` 的 9 个字段：`device_id, t_ms, frame_id, state, action, reason, rng, vel, human`。
- 替换 B 的模块时，只需保留相同接口：
  - `fall_recog/rules.py`：`FallDetector.update(sample, t_ms)`
  - `pc/server.py`：`POST /api/events`、`GET /api/state`、`GET /api/events/stream`

## 测试

```powershell
C:\Users\蔡华升\AppData\Local\Programs\Python\Python311\python.exe -m unittest discover -s tests -v
```
