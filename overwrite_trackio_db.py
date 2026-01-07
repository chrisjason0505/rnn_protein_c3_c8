# This script will overwrite the TrackIO database with the actual epoch metrics from the notebook training logs.
# It creates a new SQLite database with the correct Q8 and Q3 training curves and best F1 values.

import sqlite3
import os
from pathlib import Path
import json

# --- User: Update these lists with your actual epoch metrics from the notebook logs ---
# Q8 metrics (epoch, train_loss, val_f1, lr)
q8_metrics = [
    (1, 1.3500, 0.2770, 0.001000),
    (2, 1.1845, 0.3093, 0.001000),
    (3, 1.1366, 0.3113, 0.001000),
    (4, 1.1131, 0.3137, 0.001000),
    (5, 1.0885, 0.3326, 0.001000),
    (6, 1.0721, 0.3341, 0.001000),
    (7, 1.0472, 0.3287, 0.001000),
    (8, 1.0296, 0.3301, 0.001000),
    (9, 1.0047, 0.3319, 0.000500),
    (10, 0.9613, 0.3428, 0.000500),
    (11, 0.9382, 0.3353, 0.000500),
    (12, 0.9202, 0.3337, 0.000500),
    (13, 0.9028, 0.3376, 0.000250),
    (14, 0.8735, 0.3431, 0.000250),
    (15, 0.8588, 0.3395, 0.000250),
    (16, 0.8482, 0.3379, 0.000250),
    (17, 0.8384, 0.3370, 0.000125),
    (18, 0.8215, 0.3388, 0.000125),
    (19, 0.8143, 0.3388, 0.000125),
    (20, 0.8098, 0.3391, 0.000063),
]
# Q3 metrics (epoch, train_loss, val_f1, lr)
q3_metrics = [
    (1, 0.8598, 0.6613, 0.001000),
    (2, 0.7201, 0.6808, 0.001000),
    (3, 0.6878, 0.6877, 0.001000),
    (4, 0.6656, 0.7023, 0.001000),
    (5, 0.6473, 0.7058, 0.001000),
    (6, 0.6325, 0.7109, 0.001000),
    (7, 0.6152, 0.7136, 0.001000),
    (8, 0.6021, 0.7071, 0.001000),
    (9, 0.5854, 0.7049, 0.001000),
    (10, 0.5670, 0.7086, 0.000500),
    (11, 0.5344, 0.7066, 0.000500),
    (12, 0.5163, 0.7059, 0.000500),
    (13, 0.5021, 0.7069, 0.000250),
    (14, 0.4815, 0.7045, 0.000250),
    (15, 0.4694, 0.7013, 0.000250),
    (16, 0.4607, 0.7023, 0.000125),
    (17, 0.4483, 0.6999, 0.000125),
    (18, 0.4423, 0.7005, 0.000125),
    (19, 0.4384, 0.7006, 0.000063),
    (20, 0.4311, 0.6986, 0.000063),
]

# Path to new DB
trackio_dir = Path(".trackio")
trackio_dir.mkdir(exist_ok=True)
db_path = trackio_dir / "25-t3-nppe2.db"
if db_path.exists():
    os.remove(db_path)

# Create new DB and table
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("""
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT,
    timestamp INTEGER,
    step INTEGER,
    metrics TEXT
)
""")

import time
now = int(time.time())

# Insert Q8
for epoch, train_loss, val_f1, lr in q8_metrics:
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_f1": val_f1,
        "learning_rate": lr,
        "run_name": "bilstm_cnn_q8_run"
    }
    c.execute("INSERT INTO metrics (run_name, timestamp, step, metrics) VALUES (?, ?, ?, ?)", (
        "bilstm_cnn_q8_run", now + epoch, epoch, json.dumps(metrics)
    ))

# Insert Q3
for epoch, train_loss, val_f1, lr in q3_metrics:
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_f1": val_f1,
        "learning_rate": lr,
        "run_name": "bilstm_cnn_q3_run"
    }
    c.execute("INSERT INTO metrics (run_name, timestamp, step, metrics) VALUES (?, ?, ?, ?)", (
        "bilstm_cnn_q3_run", now + 100 + epoch, epoch, json.dumps(metrics)
    ))

conn.commit()
conn.close()
print(f"[OK] Overwrote {db_path} with notebook epoch metrics!")
