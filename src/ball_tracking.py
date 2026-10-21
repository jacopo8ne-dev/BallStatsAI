"""Tracker della palla con filtro di Kalman, memoria lunga dell'ID e interpolazione dei buchi."""

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


MAX_BUCO_INTERPOLATO = 29  # Frame originali mancanti riempiti con l'interpolazione (meno di 30).


class KalmanPalla:
    """Filtro di Kalman ad accelerazione costante: stato [x, y, vx, vy, ax, ay], misura [x, y].

    Le unità sono pixel e frame originali del video. Il rumore di processo è un "jerk" bianco:
    l'incertezza cresce col tempo trascorso senza osservazioni.
    """

    def __init__(self, x, y, misura_std=3.0, jerk_std=5.0):
        self.x = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        # Velocità e accelerazione iniziali sono ignote: varianza alta finché non arrivano misure.
        self.P = np.diag([misura_std ** 2] * 2 + [15.0 ** 2] * 2 + [3.0 ** 2] * 2)
        self.R = np.eye(2) * misura_std ** 2
        self.q = jerk_std ** 2
        self.H = np.zeros((2, 6))
        self.H[0, 0] = self.H[1, 1] = 1.0

    def _transizione(self, dt):
        F = np.eye(6)
        F[0, 2] = F[1, 3] = F[2, 4] = F[3, 5] = dt
        F[0, 4] = F[1, 5] = dt * dt / 2
        G = np.array([dt ** 3 / 6, dt ** 2 / 2, dt])  # Effetto di un jerk costante su posizione, velocità, accelerazione.
        Q = np.zeros((6, 6))
        for asse in (0, 1):
            indici = [asse, asse + 2, asse + 4]
            Q[np.ix_(indici, indici)] = np.outer(G, G) * self.q
        return F, Q

    def predict(self, dt):
        F, Q = self._transizione(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, cx, cy):
        y = np.array([cx, cy]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def mahalanobis2(self, cx, cy):
        """Distanza statistica al quadrato fra una misura e la posizione prevista."""
        y = np.array([cx, cy]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)

    def smorza(self, dt):
        # Senza osservazioni l'accelerazione stimata (rimbalzi, palleggio) porterebbe la stima fuori
        # dal campo in pochi frame: la azzeriamo e rallentiamo la palla del 30% per frame. Una palla
        # che sparisce è quasi sempre nascosta da un giocatore vicino all'ultima posizione vista.
        self.x[4:6] = 0.0
        self.x[2:4] *= 0.7 ** dt

    @property
    def posizione(self):
        return float(self.x[0]), float(self.x[1])


class BallTracker:
    """Segue una sola palla con un filtro di Kalman e una memoria lunga della sua identità.

    Le idee vengono da OC-SORT: l'associazione usa la posizione prevista e la sua incertezza, non la
    sovrapposizione dei rettangoli (una palla di 10-15 pixel veloce non si sovrappone quasi mai al
    frame precedente). Dopo un'occlusione la palla può essere ritrovata ovunque vicino a un giocatore
    e riprende lo stesso `id`: in partita c'è una sola palla. L'`id` cambia solo dopo più di
    `memoria_secondi` (150 frame a 30 FPS) senza nessuna osservazione.
    """

    GATE_CHI2 = 9.21  # 99% di una chi-quadro a 2 gradi di libertà.

    def __init__(self, skip_frames=1, fps=30.0, start_confidence=0.15, memoria_secondi=5.0,
                 stima_secondi=1.0):
        self.skip_frames = skip_frames
        self.start_confidence = start_confidence  # Fiducia richiesta per iniziare o ritrovare la traccia.
        self.track_buffer = max(1, round(fps * memoria_secondi))  # Frame originali di memoria dell'ID.
        self.max_stima = max(1, round(fps * stima_secondi))  # Oltre 30 frame (a 30 FPS) la stima non viene disegnata.
        self.static = PuntiStatici(fps)
        self.kalman = None
        self.kalman_frame = None  # Frame a cui si riferisce lo stato del filtro.
        self.last = None  # Ultima osservazione reale accettata, mai una posizione prevista.
        self.ball_id = 0
        self.weak = []  # Candidati deboli del frame elaborato precedente, per confermare un nuovo inizio.
        self.weak_frame = None

    def attivo(self, frame):
        """Vero finché l'identità della palla è in memoria (track_buffer)."""
        return self.last is not None and frame - self.last["frame"] <= self.track_buffer

    def _avanza(self, frame):
        if frame > self.kalman_frame:
            self.kalman.predict(frame - self.kalman_frame)
            self.kalman_frame = frame

    def raggio(self, frame, bbox):
        # Finestra minima intorno alla previsione, che si allarga di 15 pixel per frame perso: copre
        # i cambi di direzione che il filtro non può prevedere (passaggio, rimbalzo, tiro).
        perso = frame - self.last["frame"]
        lato = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        return max(30.0, 4 * lato) + 15.0 * (perso - 1)

    def movimento_valido(self, bbox, frame):
        if not self.attivo(frame):
            return True
        if frame <= self.last["frame"]:
            return False
        self._avanza(frame)
        px, py = self.kalman.posizione
        cx, cy = centro(bbox)
        euclidea = math.hypot(cx - px, cy - py)
        limite = self.raggio(frame, bbox)
        # Il gate statistico aiuta quando il filtro è sicuro; il limite in pixel evita che dopo una
        # lunga perdita l'incertezza enorme accetti qualsiasi punto dell'immagine.
        # Più la palla è veloce, più può cambiare direzione rispetto alla previsione (rimbalzo, tiro).
        spostamento = math.hypot(self.kalman.x[2], self.kalman.x[3]) * (frame - self.last["frame"])
        if euclidea <= max(30.0, 4 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]), 1.5 * spostamento):
            return True
        return self.kalman.mahalanobis2(cx, cy) <= self.GATE_CHI2 and euclidea <= limite

    def _accetta(self, row, frame, nuova_identita, nuovo_filtro):
        cx, cy = centro(row[:4])
        if nuova_identita:
            self.ball_id += 1
        if nuovo_filtro:
            self.kalman = KalmanPalla(cx, cy)
            self.kalman_frame = frame
        else:
            self._avanza(frame)
            self.kalman.update(cx, cy)
        self.last = {"frame": frame, "id": self.ball_id, "tracker_id": self.ball_id,
                     "bbox": [float(v) for v in row[:4]], "confidence": float(row[4]),
                     "source": "yolo", "estimated": False}
        return self.last

    def stima(self, frame, shape=None):
        """Posizione prevista dal filtro per un frame senza palla, oppure None.

        La stima esiste solo entro `max_stima` frame dall'ultima osservazione; dopo smette di essere
        disegnata, ma l'`id` resta in memoria fino a `track_buffer`.
        """
        if not self.attivo(frame) or not 0 < frame - self.last["frame"] <= self.max_stima:
            return None
        self._avanza(frame)
        cx, cy = self.kalman.posizione
        x1, y1, x2, y2 = self.last["bbox"]
        mezzo_w, mezzo_h = (x2 - x1) / 2, (y2 - y1) / 2
        if shape is not None:
            altezza, larghezza = shape[:2]
            cx = min(max(cx, mezzo_w), larghezza - mezzo_w)  # La stima resta dentro l'immagine.
            cy = min(max(cy, mezzo_h), altezza - mezzo_h)
        return {"frame": frame, "id": self.ball_id, "tracker_id": None,
                "bbox": [cx - mezzo_w, cy - mezzo_h, cx + mezzo_w, cy + mezzo_h],
                "confidence": 0.0, "source": "predicted", "estimated": True}

    def update(self, rows, frame, shape=None, players=None):
        """Sceglie la palla fra i candidati; `players` sono i box dei giocatori dello stesso frame."""
        rows = [row for row in rows if forma_da_palla(row[:4], players)]
        self.static.osserva(rows, frame)
        # Loghi, teste del pubblico o palloni fermi a bordo campo restano nello stesso punto
        # per secondi; la palla in gioco quasi mai. Una palla tenuta ferma a lungo (per esempio
        # prima di un tiro libero) viene persa finché non si muove: è il prezzo del filtro.
        rows = [row for row in rows if not self.static.statico(row[:4])]
        if players:
            # Una traccia nasce o viene ritrovata solo vicino a un giocatore: pubblico in primo piano,
            # loghi e pareti non possono agganciare la palla. La traccia attiva segue anche i lanci.
            iniziali = [row for row in rows if vicino_a_giocatore(row[:4], players)]
        else:
            iniziali = rows  # Senza giocatori nel frame il controllo non è possibile.
        weak, weak_frame = self.weak, self.weak_frame
        self.weak, self.weak_frame = iniziali, frame
        if self.attivo(frame):
            self._avanza(frame)
            px, py = self.kalman.posizione
            vicini = [row for row in rows if self.movimento_valido(row[:4], frame)]
            migliore_vicino = min(vicini, key=lambda row: math.hypot(centro(row[:4])[0] - px, centro(row[:4])[1] - py)
                                  / self.raggio(frame, row[:4]) - row[4], default=None)
            lontani = [row for row in iniziali if row not in vicini and row[4] >= self.start_confidence]
            migliore_lontano = max(lontani, key=lambda row: row[4], default=None)
            # Una palla sicura lontana batte un candidato vicino molto più debole. È la stessa palla
            # (passaggio lungo, uscita dall'occlusione): stesso id, ma il filtro riparte da lì.
            if migliore_lontano is not None and (migliore_vicino is None or migliore_lontano[4] >= migliore_vicino[4] + 0.2):
                return self._accetta(migliore_lontano, frame, False, True)
            if migliore_vicino is not None:
                return self._accetta(migliore_vicino, frame, False, False)
            self.kalman.smorza(self.skip_frames)
            return None
        forti = [row for row in iniziali if row[4] >= self.start_confidence]
        if forti:
            return self._accetta(max(forti, key=lambda row: row[4]), frame, True, True)
        # Due candidati deboli vicini in frame elaborati consecutivi bastano per ripartire.
        if weak_frame == frame - self.skip_frames:
            confermati = [row for row in iniziali if any(
                distanza(row[:4], old[:4]) <= max(30.0, 3 * (row[2] - row[0])) * self.skip_frames for old in weak)]
            if confermati:
                return self._accetta(max(confermati, key=lambda row: row[4]), frame, True, True)
        return None


def interpola_palla(previous, current, frames, max_gap=MAX_BUCO_INTERPOLATO):
    """Riempie buchi di 1–29 frame originali racchiusi da due osservazioni YOLO della stessa palla."""
    if previous is None or current is None:
        return {}
    gap = current["frame"] - previous["frame"] - 1
    if not 1 <= gap <= max_gap or previous["id"] != current["id"]:
        return {}  # Non estrapola all'inizio, alla fine o attraverso perdite lunghe.
    interpolated = {}
    for frame in frames:
        if previous["frame"] < frame < current["frame"]:
            fraction = (frame - previous["frame"]) / (gap + 1)
            bbox = [a + (b - a) * fraction for a, b in zip(previous["bbox"], current["bbox"])]
            interpolated[frame] = {"frame": frame, "id": previous["id"], "tracker_id": None,
                                   "bbox": bbox, "confidence": min(previous["confidence"], current["confidence"]),
                                   "source": "interpolated", "estimated": True}
    return interpolated
