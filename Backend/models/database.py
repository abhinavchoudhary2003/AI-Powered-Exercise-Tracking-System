import os
import sqlite3
from datetime import datetime
import numpy as np

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "exercise_tracker.db"
)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                height_cm REAL,
                weight_kg REAL,
                face_embedding BLOB,          -- filled in later
                consent_given INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exercise_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise TEXT NOT NULL,       -- 'pushup' or 'squat'
                reps INTEGER NOT NULL,
                logged_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


def create_user(name, age, height_cm, weight_kg, consent_given=True):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, age, height_cm, weight_kg, consent_given, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, age, height_cm, weight_kg, int(consent_given), datetime.now().isoformat()),
        )
        return cur.lastrowid


def log_exercise(user_id, exercise, reps):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO exercise_logs (user_id, exercise, reps, logged_at) VALUES (?, ?, ?, ?)",
            (user_id, exercise, reps, datetime.now().isoformat()),
        )


def get_daily_totals(user_id, date_str):
    """date_str like '2026-09-24'"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT exercise, SUM(reps) AS total FROM exercise_logs "
            "WHERE user_id = ? AND date(logged_at) = ? GROUP BY exercise",
            (user_id, date_str),
        ).fetchall()
        return {r["exercise"]: r["total"] for r in rows}
    
def save_face_embeddings(user_id, embeddings):
    """embeddings: numpy array of shape (n, 128)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET face_embedding = ? WHERE id = ?",
            (embeddings.astype(np.float32).tobytes(), user_id),
        )


def get_all_face_embeddings():
    """Returns a list of (user_id, name, embeddings) with embeddings shape (n, 128)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, face_embedding FROM users WHERE face_embedding IS NOT NULL"
        ).fetchall()
    return [
     #   (r["id"], r["name"], np.frombuffer(r["face_embedding"], dtype=np.float32).reshape(-1, 128))
        (r["id"], r["name"], np.frombuffer(r["face_embedding"], dtype=np.float32).reshape(-1, 512))
        for r in rows
    ]    