"""Tracker dedicato alla palla e interpolazione di brevi intervalli mancanti."""

import math  # Confronta gli spostamenti nelle coordinate originali del video.
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML


def crea_tracker(config_path, skip_frames=1):
    # Con frame saltati, 30 frame di buffer restano 30 frame del video originale.
    config = SimpleNamespace(**YAML.load(str(config_path)))
    tracker = BYTETracker(config, frame_rate=30)
    tracker.max_time_lost = max(1, math.ceil(config.track_buffer / skip_frames))
    return tracker


def boxes_da_righe(rows, shape):
    # ByteTrack richiede Boxes anche quando non ci sono rilevamenti.
    return Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), shape)


def centro(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def distanza(a, b):
    ax, ay = centro(a)
    bx, by = centro(b)
    return math.hypot(ax - bx, ay - by)


class BallTracker:
    """Segue una sola palla; separa l'identità logica dagli ID interni di ByteTrack."""

    def __init__(self, skip_frames=1):
        config = Path(__file__).resolve().parent.parent / "bytetrack_ball.yaml"
        self.tracker = crea_tracker(config, skip_frames)
        self.last = None  # Ultima osservazione reale accettata, mai una posizione prevista.
        self.ball_id = 0

    def movimento_valido(self, bbox, frame):
        if self.last is None:
            return True
        intervallo = frame - self.last["frame"]
        if intervallo <= 0:
            return False
        # Per buchi brevi controlla la velocità media; dopo oltre 5 frame manca ogni previsione affidabile.
        limite = 300 if intervallo - 1 > 5 else 200 * intervallo
        return distanza(bbox, self.last["bbox"]) <= limite

    def update(self, rows, frame, shape):
        # Il filtro precede ByteTrack: un salto impossibile non deve alterare il suo stato.
        candidates = [row for row in rows if self.movimento_valido(row[:4], frame)]
        if candidates:
            if self.last is None:
                chosen = max(candidates, key=lambda row: row[4])
            else:
                chosen = min(candidates, key=lambda row: (distanza(row[:4], self.last["bbox"]), -row[4]))
            candidates = [chosen]  # Le altre palle candidate non cambiano la traccia principale.
        tracks = self.tracker.update(boxes_da_righe(candidates, shape))
        if not candidates:
            return None
        row = candidates[0]
        tracker_id = int(tracks[0][4]) if len(tracks) else None
        # Una palla veloce può non sovrapporsi al vecchio box. Una vera osservazione sopra
        # 0.15 può riagganciarla; non pubblichiamo le sole previsioni del filtro di Kalman.
        if tracker_id is None and row[4] < self.tracker.args.new_track_thresh:
            return None  # I rilevamenti deboli richiedono l'associazione di ByteTrack.
        if self.last is None or frame - self.last["frame"] > self.tracker.args.track_buffer:
            self.ball_id += 1  # Dopo oltre 30 frame l'identità precedente non è garantita.
        observation = {"frame": frame, "id": self.ball_id, "tracker_id": tracker_id,
                       "bbox": [float(v) for v in row[:4]], "confidence": float(row[4]), "source": "yolo"}
        self.last = observation
        return observation


def interpola_palla(previous, current, frames):
    """Riempie solo buchi di 1–5 frame originali racchiusi da due osservazioni YOLO."""
    if previous is None or current is None:
        return {}
    gap = current["frame"] - previous["frame"] - 1
    if not 1 <= gap <= 5 or previous["id"] != current["id"]:
        return {}  # Non estrapola all'inizio, alla fine o attraverso perdite lunghe.
    interpolated = {}
    for frame in frames:
        if previous["frame"] < frame < current["frame"]:
            fraction = (frame - previous["frame"]) / (gap + 1)
            bbox = [a + (b - a) * fraction for a, b in zip(previous["bbox"], current["bbox"])]
            interpolated[frame] = {"frame": frame, "id": previous["id"], "tracker_id": None,
                                   "bbox": bbox, "confidence": min(previous["confidence"], current["confidence"]),
                                   "source": "interpolated"}
    return interpolated
