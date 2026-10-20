"""Tracker dedicato alla palla e interpolazione di brevi intervalli mancanti."""

import math  # Confronta gli spostamenti nelle coordinate originali del video.
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


def forma_da_palla(bbox, players):
    """Scarta box troppo grandi rispetto ai giocatori in campo (teste e orecchie del pubblico)."""
    if not players:
        return True
    # La palla misura circa un ottavo dell'altezza di un giocatore; anche mossa resta sotto 0.3.
    # Il rapporto larghezza/altezza non si usa: nel palleggio la sfocatura allunga la palla.
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1]) <= 0.3 * altezza_tipica(players)


def altezza_tipica(players):
    altezze = sorted(player[3] - player[1] for player in players)
    return altezze[len(altezze) // 2]  # La mediana ignora il pubblico in primo piano, molto più alto.


def vicino_a_giocatore(bbox, players):
    """Vero se il centro della palla cade in un giocatore allargato della sua altezza."""
    cx, cy = centro(bbox)
    limite = 2 * altezza_tipica(players) if players else 0
    for x1, y1, x2, y2 in (player[:4] for player in players):
        if y2 - y1 > limite:
            continue  # Spettatori in primo piano scambiati per giocatori.
        margine = y2 - y1
        if x1 - margine <= cx <= x2 + margine and y1 - margine <= cy <= y2 + margine:
            return True
    return False


class PuntiStatici:
    """Ricorda i punti in cui YOLO vede una "palla" ferma per secondi: spesso è un falso positivo."""

    def __init__(self, fps=30.0, min_hits=10, min_seconds=1.0, forget_seconds=10.0):
        self.min_hits = min_hits  # Osservazioni necessarie prima di dichiarare un punto statico.
        self.min_span = fps * min_seconds  # Intervallo minimo, in frame originali, fra prima e ultima osservazione.
        self.forget = fps * forget_seconds  # Un punto non più visto viene dimenticato.
        self.spots = []

    def _raggio(self, bbox):
        return max(12.0, 1.5 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))

    def osserva(self, rows, frame):
        self.spots = [spot for spot in self.spots if frame - spot["last"] <= self.forget]
        for row in rows:
            cx, cy = centro(row[:4])
            raggio = self._raggio(row[:4])
            spot = next((s for s in self.spots if math.hypot(cx - s["x"], cy - s["y"]) <= raggio), None)
            if spot is None:
                self.spots.append({"x": cx, "y": cy, "hits": 1, "first": frame, "last": frame})
            elif spot["last"] != frame:
                spot["hits"] += 1
                spot["last"] = frame
                # Media mobile: segue piccole vibrazioni della camera senza spostarsi con una palla vera.
                spot["x"] += (cx - spot["x"]) / spot["hits"]
                spot["y"] += (cy - spot["y"]) / spot["hits"]

    def statico(self, bbox):
        cx, cy = centro(bbox)
        raggio = self._raggio(bbox)
        return any(s["hits"] >= self.min_hits and s["last"] - s["first"] >= self.min_span
                   and math.hypot(cx - s["x"], cy - s["y"]) <= raggio for s in self.spots)


class BallTracker:
    """Segue una sola palla con un filtro a velocità costante e una finestra di ricerca che si allarga.

    ByteTrack associa per sovrapposizione dei rettangoli: una palla di 10-15 pixel che si muove
    veloce non si sovrappone quasi mai al frame precedente, quindi qui si usa la distanza dalla
    posizione prevista. Le osservazioni pubblicate sono sempre rilevamenti YOLO reali.
    """

    def __init__(self, skip_frames=1, fps=30.0, start_confidence=0.15, max_lost_seconds=1.0):
        self.skip_frames = skip_frames
        self.start_confidence = start_confidence  # Fiducia richiesta per iniziare o spostare la traccia.
        self.max_lost = max(1, round(fps * max_lost_seconds))  # Frame originali senza palla prima di ripartire.
        self.static = PuntiStatici(fps)
        self.last = None  # Ultima osservazione reale accettata, mai una posizione prevista.
        self.velocity = (0.0, 0.0)  # Pixel per frame originale.
        self.ball_id = 0
        self.weak = []  # Candidati deboli del frame elaborato precedente, per confermare un nuovo inizio.
        self.weak_frame = None

    def attivo(self, frame):
        return self.last is not None and frame - self.last["frame"] <= self.max_lost

    def previsione(self, frame):
        dt = frame - self.last["frame"]
        cx, cy = centro(self.last["bbox"])
        return cx + self.velocity[0] * dt, cy + self.velocity[1] * dt

    def raggio(self, frame, bbox):
        # Margine per la dimensione della palla più 10 pixel per ogni frame trascorso: dopo una
        # breve perdita la palla può aver cambiato direzione (passaggio, rimbalzo, tiro).
        dt = frame - self.last["frame"]
        lato = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        return max(30.0, 4 * lato) + 10.0 * dt

    def movimento_valido(self, bbox, frame):
        if not self.attivo(frame):
            return True
        if frame <= self.last["frame"]:
            return False
        px, py = self.previsione(frame)
        cx, cy = centro(bbox)
        return math.hypot(cx - px, cy - py) <= self.raggio(frame, bbox)

    def _accetta(self, row, frame, nuova_traccia):
        cx, cy = centro(row[:4])
        if nuova_traccia:
            self.ball_id += 1
            self.velocity = (0.0, 0.0)
        else:
            lx, ly = centro(self.last["bbox"])
            dt = frame - self.last["frame"]
            vx, vy = (cx - lx) / dt, (cy - ly) / dt
            # Media fra velocità precedente e misurata: riduce il tremolio del rettangolo.
            self.velocity = ((self.velocity[0] + vx) / 2, (self.velocity[1] + vy) / 2)
        self.last = {"frame": frame, "id": self.ball_id, "tracker_id": self.ball_id,
                     "bbox": [float(v) for v in row[:4]], "confidence": float(row[4]), "source": "yolo"}
        return self.last

    def update(self, rows, frame, shape=None, players=None):
        """Sceglie la palla fra i candidati; `players` sono i box dei giocatori dello stesso frame."""
        rows = [row for row in rows if forma_da_palla(row[:4], players)]
        self.static.osserva(rows, frame)
        # Loghi, teste del pubblico o palloni fermi a bordo campo restano nello stesso punto
        # per secondi; la palla in gioco quasi mai. Una palla tenuta ferma a lungo (per esempio
        # prima di un tiro libero) viene persa finché non si muove: è il prezzo del filtro.
        rows = [row for row in rows if not self.static.statico(row[:4])]
        if players:
            # Una nuova traccia nasce solo vicino a un giocatore: pubblico in primo piano, loghi e
            # pareti non possono più agganciare la palla. Una traccia già attiva segue anche i lanci.
            iniziali = [row for row in rows if vicino_a_giocatore(row[:4], players)]
        else:
            iniziali = rows  # Senza giocatori nel frame il controllo non è possibile.
        weak, weak_frame = self.weak, self.weak_frame
        self.weak, self.weak_frame = iniziali, frame
        if self.attivo(frame):
            vicini = [row for row in rows if self.movimento_valido(row[:4], frame)]
            px, py = self.previsione(frame)
            migliore_vicino = min(vicini, key=lambda row: math.hypot(centro(row[:4])[0] - px, centro(row[:4])[1] - py)
                                  / self.raggio(frame, row[:4]) - row[4], default=None)
            lontani = [row for row in iniziali if row not in vicini and row[4] >= self.start_confidence]
            migliore_lontano = max(lontani, key=lambda row: row[4], default=None)
            # Una palla sicura lontana batte un candidato vicino molto più debole: la traccia
            # potrebbe seguire un falso positivo.
            if migliore_lontano is not None and (migliore_vicino is None or migliore_lontano[4] >= migliore_vicino[4] + 0.2):
                return self._accetta(migliore_lontano, frame, True)
            if migliore_vicino is not None:
                return self._accetta(migliore_vicino, frame, False)
            return None
        forti = [row for row in iniziali if row[4] >= self.start_confidence]
        if forti:
            return self._accetta(max(forti, key=lambda row: row[4]), frame, True)
        # Due candidati deboli vicini in frame elaborati consecutivi bastano per ripartire.
        if weak_frame == frame - self.skip_frames:
            confermati = [row for row in iniziali if any(
                distanza(row[:4], old[:4]) <= max(30.0, 3 * (row[2] - row[0])) * self.skip_frames for old in weak)]
            if confermati:
                return self._accetta(max(confermati, key=lambda row: row[4]), frame, True)
        return None


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
