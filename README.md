# 🌿 AI-Enabled Greenhouse Climatic Control System

> Real-time pest and disease detection + automated climate control using YOLOv8, ESP32-P4, and a Flask inference server — built as a final year project at Bells University of Technology, Nigeria.

---

## 🎯 Project Overview

This system monitors and controls greenhouse conditions autonomously using AI-powered computer vision and embedded sensor fusion. A camera captures plant images, sends them to a local inference server running three custom YOLOv8 models, and the results drive automated actuator responses (fans, pumps, alerts).

---

## 📊 Model Performance

| Model | Task | mAP@50 | Training Run |
|---|---|---|---|
| YOLOv8n | Plant Disease Detection | 60.5% | train5 |
| YOLOv8n | Fruit Detection | **96.8%** | train6 |
| YOLOv8n | Pest Detection | 72.0% | train7 |

---

## 🏗️ System Architecture
OV5647 Camera (MIPI CSI-2)
│
▼
ESP32-P4-NANO (Firmware — ESP-IDF)
├── DHT22        → Temperature & Humidity
├── BH1750       → Light Intensity
├── Soil Sensor  → Moisture Level
└── WiFi HTTP POST → Flask Inference Server
│
▼
YOLOv8 Detection Engine
(NVIDIA RTX 3050 Ti)
│
┌────────┴────────┐
Disease            Pest / Fruit
Model               Models
│
▼
SQLite Logging + Web Dashboard
│
▼
Relay-Controlled Actuators
(Fans · Water Pumps · Alerts)

---

## 🛠️ Tech Stack

**Firmware**
- ESP32-P4-NANO — ESP-IDF framework
- OV5647 MIPI CSI-2 camera (RGB565, 800×640)
- DHT22 (RMT driver), BH1750 (I2C), Capacitive Soil Sensor
- WiFi HTTP POST for image + sensor data transmission

**Inference Server**
- Python · Flask · Ultralytics YOLOv8
- NVIDIA RTX 3050 Ti (local, offline)
- SQLite for data logging

**Frontend**
- Web dashboard for live sensor readings and detection results

---

## 🌱 Features

- ✅ Real-time plant image capture and transmission
- ✅ Three specialized YOLOv8 detection models
- ✅ Live temperature, humidity, light, and soil moisture monitoring
- ✅ Automated actuator control via relay module
- ✅ Fully offline — no cloud dependency
- ✅ SQLite logging with web dashboard

---

## 📁 Repository Structure
├── firmware/          # ESP32-P4 ESP-IDF source code
├── inference/         # Flask server + YOLOv8 inference pipeline
├── models/            # Trained model weights (YOLOv8n)
├── dashboard/         # Web dashboard frontend
├── docs/              # System diagrams and project report excerpts
└── README.md

---

## 🚀 Getting Started

### Inference Server
```bash
git clone https://github.com/ted-cyber/ai-greenhouse-control-system
cd inference
pip install -r requirements.txt
python app.py
```

### Firmware
Built with ESP-IDF v5.x. See `firmware/README.md` for full setup instructions.

---

## 👤 Author

**Phillips Peace Temidayo**
Mechatronics Engineering · Bells University of Technology · 2025
[LinkedIn](https://www.linkedin.com/in/phillips-peace-007868263/) · [Upwork](https://upwork.com/freelancers/~0134f13272841cc124)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
