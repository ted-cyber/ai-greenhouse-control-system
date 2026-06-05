"""
GreenSense server.py  —  ESP32-P4 Greenhouse Monitor Backend
Receives raw RGB565 frames + sensor data from ESP32,
runs YOLO inference, serves dashboard API.

Endpoints:
  POST /analyze          <- ESP32 posts raw RGB565 frame + sensor headers
  GET  /api/latest       <- Latest sensor reading + detection
  GET  /api/history      <- Last N readings
  GET  /api/relays       <- Relay states
  POST /api/relays       <- Set relay state { "id": 0, "state": true }
  GET  /images/<name>    <- Serve captured images
  GET  /                 <- Dashboard web app
"""

import os, json, time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io, cv2
import numpy as np
import sqlite3
from contextlib import contextmanager

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────
IMAGE_DIR   = "captures"
DATA_FILE   = "history.json"
MODEL_PATH  = r"C:\GHMODEL\runs\detect\train7\weights\best.pt"
MAX_HISTORY = 200

# Frame dimensions sent by ESP32 (RGB565 = 2 bytes/pixel)
CAM_W = 800
CAM_H = 640
FRAME_SIZE_RGB565 = CAM_W * CAM_H * 2   # 1 024 000 bytes
FRAME_SIZE_BA81   = CAM_W * CAM_H * 1   #   512 000 bytes

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs("static",    exist_ok=True)
os.makedirs("templates", exist_ok=True)

DB_FILE = "greenhouse.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT,
                temp      REAL,
                humidity  REAL,
                lux       REAL,
                soil      INTEGER,
                status    TEXT,
                disease   INTEGER,
                pest      INTEGER,
                fruit     INTEGER,
                recommendation TEXT,
                image     TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relay_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT,
                relay_id  INTEGER,
                label     TEXT,
                state     INTEGER
            )""")
        conn.commit()

init_db()

# ── State ──────────────────────────────────────────────────────────────────
history = []
relays  = [
    {"id": 0, "label": "Water pump",  "state": False},
    {"id": 1, "label": "Exhaust fan", "state": False},
    {"id": 2, "label": "Grow lights", "state": False},
    {"id": 3, "label": "Heater",      "state": False},
]

def load_history():
    global history
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                history = json.load(f)[-MAX_HISTORY:]
        except Exception:
            history = []

def save_history():
    with open(DATA_FILE, "w") as f:
        json.dump(history[-MAX_HISTORY:], f)

load_history()

# ── Model ──────────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"[server] Model loaded: {MODEL_PATH}")
else:
    print(f"[server] WARNING: model not found at {MODEL_PATH}")

# ── Image decode ───────────────────────────────────────────────────────────
def decode_frame(img_bytes):
    size = len(img_bytes)
    print(f"[decode] received {size} bytes")

    # ① Try JSON with base64 image (ESP32 format)
    try:
        import base64
        payload = json.loads(img_bytes)
        raw = base64.b64decode(payload.get("image", ""))
        if len(raw) == FRAME_SIZE_RGB565:
            arr = np.frombuffer(raw, dtype=np.uint16).reshape((CAM_H, CAM_W))
            r = ((arr >> 11) & 0x1F) * 255 // 31
            g = ((arr >> 5)  & 0x3F) * 255 // 63
            b = ( arr        & 0x1F) * 255 // 31
            return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
    except Exception:
        pass

    # ② Try JPEG / PNG (manual uploads)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # ③ Raw RGB565
    if size == FRAME_SIZE_RGB565:
        arr = np.frombuffer(img_bytes, dtype=np.uint16).reshape((CAM_H, CAM_W))
        r = ((arr >> 11) & 0x1F) * 255 // 31
        g = ((arr >> 5)  & 0x3F) * 255 // 63
        b = ( arr        & 0x1F) * 255 // 31
        return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

    # ④ Raw BA81 Bayer
    if size == FRAME_SIZE_BA81:
        bayer = np.frombuffer(img_bytes, dtype=np.uint8).reshape((CAM_H, CAM_W))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

    print(f"[decode] FAILED: unrecognised format, size={size}")
    return None

def save_capture(img_bytes, prefix="cap"):
    """Decode frame and save as JPEG. Returns filename or None."""
    bgr = decode_frame(img_bytes)
    if bgr is None:
        return None
    name = f"{prefix}_{int(time.time())}.jpg"
    path = os.path.join(IMAGE_DIR, name)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return name

# ── Inference ──────────────────────────────────────────────────────────────
def run_inference(img_bytes):
    bgr = decode_frame(img_bytes)

    if bgr is None:
        return {
            "status": "error",
            "disease": 0, "pest": 0, "fruit": 0,
            "recommendation": "Could not decode image",
            "labels": [],
        }

    if model is None:
        return {
            "status": "healthy",
            "disease": 0, "pest": 0, "fruit": 0,
            "recommendation": "Model not loaded — visual OK",
            "labels": [],
        }

    results = model(bgr)

    disease = pest = fruit = 0
    labels  = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            name = model.names[cls].lower()
            labels.append(name)
            if any(k in name for k in ("leaf miner", "pest damage", "fruit borer")):
                disease += 1
            elif any(k in name for k in ("spider mite", "mealybug", "whitefly",
                                          "caterpiller", "thrip", "aphid")):
                pest += 1
            elif any(k in name for k in ("healthy tomato", "tomato flower")):
                fruit += 1
                
    if disease > 0:
        status = "warning"
        rec    = f"Disease detected: {', '.join(sorted(set(labels)))}. Inspect plants."
    elif pest > 0:
        status = "warning"
        rec    = f"Pest detected: {', '.join(sorted(set(labels)))}. Apply treatment."
    elif fruit > 0:
        status = "healthy"
        rec    = "Fruit/healthy growth detected — looking good!"
    else:
        status = "healthy"
        rec    = "All clear"

    return {
        "status": status,
        "disease": disease,
        "pest":    pest,
        "fruit":   fruit,
        "recommendation": rec,
        "labels": labels,
    }

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """ESP32 endpoint — raw frame body + sensor HTTP headers."""
    img_bytes = request.data
    if not img_bytes:
        return jsonify({"error": "no image data"}), 400

   # Sensor values from JSON body (ESP32 format)
    try:
        payload = json.loads(img_bytes)
        sens = payload.get("sensors", {})
        temp = sens.get("temperature")
        hum  = sens.get("humidity")
        lux  = sens.get("light_lux")
        soil = sens.get("soil_moisture")
    except Exception:
        temp = hum = lux = soil = None

    # Sanitise sentinel values (-999) to None so dashboard shows "—"
    if temp is not None and temp < -100: temp = None
    if hum  is not None and hum  < 0:   hum  = None
    if lux  is not None and lux  < 0:   lux  = None
    if soil is not None and soil < 0:   soil = None

    result   = run_inference(img_bytes)
    img_name = save_capture(img_bytes) or "no_image.jpg"

    entry = {
        "ts":       datetime.now().isoformat(timespec="seconds"),
        "temp":     temp,
        "humidity": hum,
        "lux":      lux,
        "soil":     soil,
        "image":    img_name,
        **result,
    }
    history.append(entry)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO detections
            (ts, temp, humidity, lux, soil, status, disease, pest, fruit, recommendation, image)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (entry["ts"], temp, hum, lux, soil,
             result["status"], result["disease"], result["pest"],
             result["fruit"], result["recommendation"], img_name))
        conn.commit()

    save_history()

    t_str = f"{temp:.1f}" if temp is not None else "?"
    h_str = f"{hum:.1f}"  if hum  is not None else "?"
    print(f"[{entry['ts']}] {result['status'].upper()} | "
          f"T={t_str}°C H={h_str}% soil={soil} lux={lux} | "
          f"{result['recommendation']}")

    return jsonify({
        "status":         result["status"],
        "disease":        result["disease"],
        "pest":           result["pest"],
        "fruit":          result["fruit"],
        "recommendation": result["recommendation"],
    })


@app.route("/api/scan", methods=["POST"])
def manual_scan():
    """Dashboard manual upload."""
    if "file" in request.files:
        img_bytes = request.files["file"].read()
    else:
        img_bytes = request.data

    if not img_bytes:
        return jsonify({"error": "no image"}), 400

    result   = run_inference(img_bytes)
    img_name = save_capture(img_bytes, prefix="scan") or "no_image.jpg"

    entry = {
        "ts":       datetime.now().isoformat(timespec="seconds"),
        "temp": None, "humidity": None, "lux": None, "soil": None,
        "image":    img_name,
        "manual":   True,
        **result,
    }
    history.append(entry)
    save_history()

    return jsonify({"image": img_name, **result})


@app.route("/api/latest")
def api_latest():
    if not history:
        return jsonify({})
    return jsonify(history[-1])


@app.route("/api/history")
def api_history():
    n = int(request.args.get("n", 50))
    return jsonify(history[-n:])


@app.route("/api/relays", methods=["GET"])
def get_relays():
    return jsonify(relays)


@app.route("/api/relays", methods=["POST"])
def set_relay():
    """
    Toggle a relay.  ESP32 will poll GET /api/relays every 10 s and
    act on any state=true entries (once relay control code is added).
    """
    data = request.json or {}
    rid  = data.get("id")
    if rid is None or not (0 <= rid < len(relays)):
        return jsonify({"error": "invalid id"}), 400
    relays[rid]["state"] = bool(data.get("state", False))
    label = relays[rid]["label"]
    state = "ON" if relays[rid]["state"] else "OFF"
    print(f"[relay] {label} → {state}")
    # ← INSERT STEP 4 HERE ←
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO relay_events (ts, relay_id, label, state)
            VALUES (?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"),
             rid, label, int(relays[rid]["state"])))
        conn.commit()
    # ← END STEP 4 ←
    return jsonify(relays[rid])


@app.route("/images/<path:name>")
def serve_image(name):
    return send_from_directory(IMAGE_DIR, name)


@app.route("/")
def dashboard():
    return send_file("templates/index.html")

@app.route("/api/db/detections")
def db_detections():
    n     = int(request.args.get("n", 100))
    since = request.args.get("since", "")
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        if since:
            rows = conn.execute(
                "SELECT * FROM detections WHERE ts >= ? ORDER BY id DESC LIMIT ?",
                (since, n)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
                (n,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/db/relay_events")
def db_relay_events():
    n = int(request.args.get("n", 50))
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM relay_events ORDER BY id DESC LIMIT ?",
            (n,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/db/summary")
def db_summary():
    with sqlite3.connect(DB_FILE) as conn:
        total    = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        warnings = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE status='warning'").fetchone()[0]
        avg_temp = conn.execute(
            "SELECT AVG(temp) FROM detections WHERE temp IS NOT NULL").fetchone()[0]
        avg_hum  = conn.execute(
            "SELECT AVG(humidity) FROM detections WHERE humidity IS NOT NULL").fetchone()[0]
    return jsonify({
        "total_detections": total,
        "warnings":         warnings,
        "avg_temp":         round(avg_temp, 1) if avg_temp else None,
        "avg_humidity":     round(avg_hum,  1) if avg_hum  else None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)