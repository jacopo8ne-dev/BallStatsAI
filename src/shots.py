"""Stima i tiri da possesso recente, salita della palla e attraversamento del ferro."""

import math  # Misura distanze e verifica i parametri numerici.


def detect_hoop(model, frames, hoop_class_id):
    """Sceglie il canestro più sicuro nei frame iniziali, oppure restituisce None."""
    migliore = None  # Nessun canestro conosciuto all'inizio del video.
    fiducia_migliore = -1.0
    for frame in frames:
        risultati = model.predict(frame, classes=[hoop_class_id], conf=0.25, device="cpu", verbose=False)
        if not risultati or risultati[0].boxes is None:
            continue  # L'assenza di rilevamenti è normale, non un errore.
        altezza, larghezza = frame.shape[:2]
        for box in risultati[0].boxes:
            if int(box.cls.item()) != hoop_class_id:
                continue  # Considera esclusivamente la classe Hoop del modello personalizzato.
            coordinate = box.xyxy[0].tolist()
            fiducia = float(box.conf.item())
            if not all(math.isfinite(value) for value in coordinate) or not math.isfinite(fiducia):
                continue
            x1, y1, x2, y2 = map(int, coordinate)
            bbox = (max(0, x1), max(0, y1), min(larghezza, x2), min(altezza, y2))
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or fiducia < 0.25:
                continue  # Scarta rettangoli vuoti, fuori immagine o sotto la soglia.
            if fiducia > fiducia_migliore:
                migliore, fiducia_migliore = bbox, fiducia  # Conserva il risultato anche se nei frame seguenti manca Hoop.
    return migliore  # Questo rettangolo resta fisso per tutto il video con telecamera fissa.


def parse_hoop_bbox(value):
    """Legge x1,y1,x2,y2 e rifiuta rettangoli vuoti o coordinate negative."""
    try:
        bbox = tuple(int(coordinata.strip()) for coordinata in value.split(","))
    except ValueError as errore:
        raise ValueError("Il canestro richiede quattro interi: x1,y1,x2,y2.") from errore
    if len(bbox) != 4 or min(bbox) < 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Canestro non valido: usa 0 <= x1 < x2 e 0 <= y1 < y2.")
    return bbox


class ShotDetector:
    """Conserva lo stato di un solo tiro alla volta, senza modificare il tracking."""

    def __init__(self, hoop_bbox, fps, min_shot_speed=5.0):
        self.hoop = parse_hoop_bbox(",".join(str(value) for value in hoop_bbox))  # Valida anche le chiamate fuori da main.py.
        if not math.isfinite(min_shot_speed) or min_shot_speed <= 0:
            raise ValueError("--min-shot-speed deve essere un numero finito maggiore di zero.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Gli FPS devono essere finiti e maggiori di zero.")
        self.speed = min_shot_speed  # Velocità verso l'alto richiesta, in pixel/frame.
        self.memory = max(1, round(fps * 0.25))  # Ricorda il possessore per un quarto di secondo.
        self.max_gap = max(1, round(fps))  # Una perdita della palla oltre un secondo rende l'esito sconosciuto.
        self.max_flight = max(1, round(fps * 3))  # Non tiene aperto un candidato per oltre tre secondi.
        self.cooldown = max(1, round(fps * 0.5))  # Evita di contare subito una seconda volta la stessa azione.
        self.display_frames = max(1, round(fps * 2))  # Mostra l'esito per circa due secondi.
        self.previous = None  # Ultima posizione osservata: (frame, centro_x, centro_y).
        self.last_owner = None  # Ultimo possesso prima del rilascio: (frame, player_id).
        self.active = None  # Tiro in corso, ancora senza esito.
        self.entered = False  # Indica un ingresso discendente dal bordo superiore del canestro.
        self.occluded = False  # Una perdita della palla può impedire di valutare l'esito.
        self.last_seen = None  # Fotogramma dell'ultima osservazione durante il tiro.
        self.ready_at = 0  # Frame dal quale può iniziare un nuovo candidato.
        self.message = None  # Testo da mostrare sul video.
        self.message_until = -1  # Fine della visualizzazione dell'ultimo esito.
        self.shots = []  # Contiene solo i tiri per cui è stato stimato un esito.
        self.unresolved = 0  # Conta i candidati non classificabili, esclusi dal JSON booleano.

    def _finish(self, frame, made):
        if made is None:  # Una palla scomparsa non dimostra un errore al tiro.
            self.unresolved += 1
            esito = "ESITO NON OSSERVABILE"
        else:
            self.active["made"] = made  # Completa l'evento con un vero booleano.
            self.shots.append(self.active)  # Il campo frame resta quello del rilascio, non dell'esito.
            esito = "FATTO" if made else "SBAGLIATO"
        self.message = f"TIRO! ID: {self.active['player_id']} - {esito}"
        self.message_until = frame + self.display_frames
        self.active = None  # Libera lo stato per un tiro futuro.
        self.entered = False
        self.occluded = False
        self.last_owner = None  # Non riutilizza un possesso precedente al tiro appena terminato.
        self.ready_at = frame + self.cooldown

    def update(self, frame, balls, possession):
        """Riceve le tracce del frame corrente e restituisce l'eventuale etichetta del tiro."""
        x1, y1, x2, y2 = self.hoop
        hoop_x, hoop_y = (x1 + x2) / 2, (y1 + y2) / 2
        ball = max(balls, key=lambda item: item["confidence"]) if balls else None  # Come per il possesso, sceglie la palla più sicura.
        if self.active is not None:
            if frame - self.active["frame"] > self.max_flight or frame - self.last_seen > self.max_gap:
                self._finish(frame, None)  # Scadenza senza osservazioni sufficienti: non inventa un tiro sbagliato.
        if ball is None:
            self.previous = None  # Non misura velocità attraverso frame mancanti.
            self.entered = False  # Non conferma un canestro attraverso un'occlusione.
            if self.active is not None:
                self.occluded = True
            return self._label(frame)
        bx1, by1, bx2, by2 = ball["bbox"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2  # Usa il centro della palla, in pixel originali.
        if self.active is not None:
            self.last_seen = frame
            if self.previous is not None and self.previous[0] != frame - 1:
                self.entered = False
                self.occluded = True  # Anche indici non consecutivi interrompono l'osservazione.
        if self.previous is not None and self.previous[0] == frame - 1:
            _, px, py = self.previous
            dy = cy - py  # L'asse y cresce verso il basso: dy negativo indica salita.
            owner = self.last_owner
            toward = (cx - px) * (hoop_x - px) + dy * (hoop_y - py) > 0  # Il movimento deve avere una componente verso il canestro.
            horizontal = abs(cx - hoop_x) <= abs(px - hoop_x) + (x2 - x1) / 2  # Tollera piccole oscillazioni sotto il ferro.
            if (self.active is None and frame >= self.ready_at and owner is not None
                    and frame - owner[0] <= self.memory and -dy >= self.speed
                    and py > y2 and toward and horizontal):
                self.active = {"frame": frame - 1, "player_id": owner[1], "made": False,
                               "shot_x": float(px), "shot_y": float(py),
                               "distance": float(math.hypot(px - hoop_x, py - hoop_y))}
                self.last_seen = frame  # Memorizza il lancio una sola volta, all'inizio della salita.
                self.entered = False
                self.occluded = False
            if self.active is not None and dy > 0:  # Un canestro può essere confermato solo durante la discesa.
                if py <= y1 <= cy:
                    crossing_x = px + (cx - px) * (y1 - py) / dy  # Interpola l'incrocio fra due osservazioni consecutive.
                    self.entered = x1 <= crossing_x <= x2  # Richiede ingresso dall'alto, non dal fianco.
                    self.occluded = False  # Da sopra il ferro fino alla discesa l'osservazione è di nuovo continua.
                if py <= y2 <= cy:
                    crossing_x = px + (cx - px) * (y2 - py) / dy
                    made = bool(self.entered and x1 <= crossing_x <= x2)
                    self._finish(frame, None if self.occluded else made)  # Non chiama sbagliato un attraversamento nascosto.
                elif py > y2 and cy > y2:
                    self._finish(frame, None if self.occluded else False)  # Tiro corto: inizia a scendere senza raggiungere l'altezza del ferro.
            elif self.active is not None and dy < 0:
                self.entered = False  # Una nuova salita interrompe un attraversamento discendente.
        if self.active is None and frame >= self.ready_at and possession["player_id"] is not None:
            self.last_owner = (frame, possession["player_id"])  # Aggiorna il tiratore potenziale solo quando non c'è già un tiro in volo.
        self.previous = (frame, cx, cy)
        return self._label(frame)

    def _label(self, frame):
        if self.active is not None:
            return f"TIRO! ID: {self.active['player_id']} - IN CORSO"
        return self.message if frame <= self.message_until else None

    def finish_video(self, frame):
        if self.active is not None:
            self._finish(frame, None)  # Un video che termina a metà tiro non consente di stabilire fatto/sbagliato.
