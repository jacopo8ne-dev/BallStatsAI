"""Segue persone e palla, stima possesso e tiri fotogramma per fotogramma."""

import json  # Salva le tracce in un formato leggibile anche da altri programmi.
import math  # Controlla che la frequenza del video sia un numero valido.
from collections import deque  # Conserva pochi frame in attesa di una possibile interpolazione.
import tempfile  # Crea una cartella debug distinta per ogni analisi.
from pathlib import Path  # Gestisce cartelle e nomi dei file.

import cv2  # OpenCV legge, disegna e salva i fotogrammi del video.
from ultralytics.utils import ROOT
from ultralytics import YOLO  # YOLO riconosce gli oggetti nelle immagini.

from src.possession import calcola_possesso  # Collega la palla al giocatore più vicino nello stesso frame.
from src.ball_color import DEFAULT_HSV_RANGE, detect_ball_by_color, valida_range_hsv  # Recupera palle arancioni non trovate da YOLO.
from src.shots import ShotDetector, HoopTracker, parse_hoop_bbox
from src.ball_tracking import MAX_BUCO_INTERPOLATO, BallTracker, boxes_da_righe, crea_tracker, interpola_palla


def trova_classi_modello(names):
    """Ricava gli indici dai nomi, sia se Ultralytics restituisce un dizionario sia una lista."""
    elementi = names.items() if isinstance(names, dict) else enumerate(names)
    indici = {nome: int(indice) for indice, nome in elementi}  # Non presume un ordine delle classi.
    mancanti = [nome for nome in ("Player", "Ball", "Hoop") if nome not in indici]
    if mancanti:
        raise ValueError(f"Il modello deve contenere le classi Player, Ball e Hoop. Mancano: {', '.join(mancanti)}.")
    return {nome: indici[nome] for nome in ("Player", "Ball", "Hoop")}  # Le altre classi non sono usate.


def rileva_persone(input_path, output_path, model_path, confidence=0.1, tracks_path="output/tracks.json",
                   ball_tracks_path="output/ball_tracks.json", ball_confidence=0.05,
                   possession_path="output/possession.json", possession_threshold=150.0,
                   use_color_fallback=False, color_hsv_range=DEFAULT_HSV_RANGE,
                   shot_path="output/shots.json", min_shot_speed=5.0, hoop_confidence=0.3,
                   hoop_bbox=None, skip_frames=1, debug_dir="output/debug", imgsz=640):
    input_path = Path(input_path)  # Converte il percorso del video in un oggetto Path.
    output_path = Path(output_path)  # Converte il percorso del risultato.
    model_path = Path(model_path)  # Converte il percorso dei pesi del modello.
    tracks_path = Path(tracks_path)  # Sceglie il file dove salvare le posizioni con ID.
    ball_tracks_path = Path(ball_tracks_path)  # Tiene le posizioni della palla separate da quelle dei giocatori.
    possession_path = Path(possession_path)  # Sceglie dove salvare il possesso stimato.
    shot_path = Path(shot_path)  # Sceglie il file dei tiri, distinto dagli altri risultati.
    if not input_path.is_file():  # Controlla che il video esista prima di caricare il modello.
        raise FileNotFoundError(f"Video non trovato: {input_path}. Copialo nella cartella data/.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {model_path}. Copia il tuo best.pt in models/ oppure usa --model.")
    if input_path.resolve() == output_path.resolve():  # Protegge il video originale.
        raise ValueError("Il video di ingresso e quello di uscita devono avere nomi diversi.")
    if output_path.suffix.lower() != ".mp4":  # Usiamo il formato del codec scelto sotto.
        raise ValueError("Il file di uscita deve avere estensione .mp4.")
    if tracks_path.suffix.lower() != ".json":  # Rende riconoscibile il formato delle tracce.
        raise ValueError("Il file delle tracce deve avere estensione .json.")
    if tracks_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve()):  # Protegge video e modello dalla scrittura JSON.
        raise ValueError("Il file delle tracce deve essere diverso dal video originale, dal video di uscita e dal modello.")
    if not 0 < confidence <= 1:  # Controlla la soglia anche quando la funzione è importata.
        raise ValueError("La soglia di fiducia deve essere compresa tra 0 (escluso) e 1.")
    if ball_tracks_path.suffix.lower() != ".json":  # Richiede un file JSON anche per la palla.
        raise ValueError("Il file delle posizioni della palla deve avere estensione .json.")
    if ball_tracks_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve(), tracks_path.resolve()):
        raise ValueError("Il JSON della palla deve essere diverso da input, output, modello e JSON dei giocatori.")
    if not 0 < ball_confidence <= 1:  # Evita soglie non valide anche fuori dalla riga di comando.
        raise ValueError("La soglia della palla deve essere compresa tra 0 (escluso) e 1.")
    if not 0 < hoop_confidence <= 1:
        raise ValueError("--hoop-confidence deve essere maggiore di 0 e minore o uguale a 1.")
    if possession_path.suffix.lower() != ".json":  # Il possesso deve essere salvato in formato JSON.
        raise ValueError("Il file del possesso deve avere estensione .json.")
    if possession_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve(), tracks_path.resolve(), ball_tracks_path.resolve()):
        raise ValueError("Il JSON del possesso deve essere diverso da tutti gli altri file dell'analisi.")
    if not math.isfinite(possession_threshold) or possession_threshold <= 0:  # Controlla anche le chiamate dirette alla funzione.
        raise ValueError("La soglia del possesso deve essere un numero finito maggiore di zero.")
    valida_range_hsv(color_hsv_range)  # Controlla il range anche quando la funzione è chiamata senza main.py.
    if not math.isfinite(min_shot_speed) or min_shot_speed <= 0:
        raise ValueError("--min-shot-speed deve essere un numero finito maggiore di zero.")
    if shot_path.suffix.lower() != ".json":
        raise ValueError("Il file dei tiri deve avere estensione .json.")
    if shot_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve(), tracks_path.resolve(), ball_tracks_path.resolve(), possession_path.resolve()):
        raise ValueError("Il JSON dei tiri deve essere diverso da tutti gli altri file dell'analisi.")

    if not isinstance(skip_frames, int) or isinstance(skip_frames, bool) or skip_frames < 1:
        raise ValueError("--skip-frames deve essere un intero maggiore o uguale a 1.")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz < 32 or imgsz % 32:
        raise ValueError("--imgsz deve essere un multiplo di 32, per esempio 640, 960 o 1280.")
    if hoop_bbox is not None:
        hoop_bbox = parse_hoop_bbox(",".join(str(value) for value in hoop_bbox))

    video = cv2.VideoCapture(str(input_path))
    writer = None
    tracce, tracce_palla, possessi = [], [], []
    pending = deque()  # Ritarda l'uscita al massimo di 30 frame originali, senza caricare tutta la partita.
    tiri = None
    fotogrammi = 0
    elaborati = 0
    try:
        if not video.isOpened():
            raise ValueError("Impossibile aprire il video. Prova un file MP4 leggibile.")
        successo, frame = video.read()
        if not successo:
            raise ValueError("Il video non contiene fotogrammi leggibili.")
        fps = video.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("La frequenza FPS del video non è valida. Converti il video in MP4.")
        altezza, larghezza = frame.shape[:2]
        if hoop_bbox is not None and (hoop_bbox[2] > larghezza or hoop_bbox[3] > altezza):
            raise ValueError("Il canestro manuale deve essere contenuto nelle dimensioni del video.")
        model = YOLO(str(model_path))  # Una sola istanza, un solo passaggio per frame elaborato.
        classi = trova_classi_modello(model.names)
        tracker_giocatori = crea_tracker(ROOT / "cfg/trackers/bytetrack.yaml", skip_frames)
        tracker_palla = BallTracker(skip_frames, fps)
        tracker_canestro = HoopTracker(hoop_confidence, hoop_bbox)
        if hoop_bbox is not None:
            print(f"Canestro manuale: {hoop_bbox}. Ricerca automatica disattivata.")
        for path in (output_path, tracks_path, ball_tracks_path, possession_path, shot_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_run = Path(tempfile.mkdtemp(prefix=input_path.stem + "_", dir=debug_dir))
        print(f"Debug canestro: {debug_run.resolve()}")
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / skip_frames, (larghezza, altezza))
        if not writer.isOpened():
            raise RuntimeError("Impossibile creare il video di uscita. Controlla cartella e codec MP4.")
        while successo:
            processato = fotogrammi % skip_frames == 0
            hoops = []  # Rilevamenti attuali: non confonderli con il canestro conservato.
            if processato:
                richieste = [classi["Player"], classi["Ball"]]
                if hoop_bbox is None:
                    richieste.append(classi["Hoop"])
                # Nel debug mostriamo anche Hoop sotto soglia, fino al limite diagnostico 0.01.
                soglia = min(confidence, ball_confidence, hoop_confidence, 0.01 if fotogrammi < 150 else 1.0)
                results = model.predict(frame, classes=richieste, conf=soglia, imgsz=imgsz, device="cpu", verbose=False)
                groups = separa_rilevamenti(results, classi, frame.shape[:2])
                players = [row for row in groups["Player"] if row[4] >= confidence]
                balls = [row for row in groups["Ball"] if row[4] >= ball_confidence
                         and (row[2]-row[0]) * (row[3]-row[1]) <= 0.05 * larghezza * altezza]
                hoops = groups["Hoop"] if hoop_bbox is None else []
                canestro = tracker_canestro.update(hoops, fotogrammi)
                tracks = tracker_giocatori.update(boxes_da_righe(players, frame.shape[:2]))
                giocatori = [{"frame": fotogrammi, "id": int(track[4]),
                              "bbox": [int(value) for value in track[:4]]} for track in tracks]
                previous = tracker_palla.last
                palla = tracker_palla.update(balls, fotogrammi, frame.shape[:2], players)
                # Il nuovo punto permette di ricostruire SOLO i frame ancora in attesa.
                interpolated = interpola_palla(previous, palla, [item["frame"] for item in pending])
                for item in pending:
                    if item["frame"] in interpolated:
                        item["ball"] = interpolated[item["frame"]]
                if palla is None and use_color_fallback:
                    bbox_colore = detect_ball_by_color(frame, color_hsv_range)
                    if bbox_colore is not None and tracker_palla.movimento_valido(bbox_colore, fotogrammi):
                        palla = {"frame": fotogrammi, "id": None, "tracker_id": None, "bbox": bbox_colore,
                                 "confidence": 0.0, "source": "color", "estimated": False}  # Il colore non aggiorna il tracker né ancora l'interpolazione.
                if palla is None:
                    # Il filtro di Kalman stima dove si trova la palla nascosta. Se la palla riappare
                    # entro 29 frame, l'interpolazione fra le due osservazioni sostituirà la stima.
                    palla = tracker_palla.stima(fotogrammi, frame.shape[:2])
                pending.append({"frame": fotogrammi, "image": frame, "players": giocatori,
                                "ball": palla, "hoop": canestro})
                elaborati += 1
            if fotogrammi < 150:
                salva_debug_canestro(frame, fotogrammi, hoops, tracker_canestro.current,
                                     hoop_confidence, processato, debug_run)
            # Un punto futuro non può più riempire questo frame entro il limite di 29 mancanti.
            while pending and fotogrammi - pending[0]["frame"] > MAX_BUCO_INTERPOLATO:
                tiri = scrivi_frame(pending.popleft(), writer, tiri, fps, min_shot_speed, skip_frames,
                                   possession_threshold, tracce, tracce_palla, possessi)
            fotogrammi += 1
            if fotogrammi % 100 == 0:
                print(f"Letti {fotogrammi} fotogrammi, inferenze YOLO: {elaborati}...")
            successo, frame = video.read()
        while pending:  # A fine video non inventiamo il punto futuro: i buchi finali restano vuoti.
            tiri = scrivi_frame(pending.popleft(), writer, tiri, fps, min_shot_speed, skip_frames,
                               possession_threshold, tracce, tracce_palla, possessi)
    finally:
        video.release()
        if writer is not None:
            writer.release()
    if tiri is not None:
        tiri.finish_video(fotogrammi)
        print(f"Tiri: {len(tiri.shots)} con esito stimato, {tiri.unresolved} non classificabili.")
    else:
        print("Avviso: canestro mai trovato. shots.json sarà vuoto; prova --hoop-bbox.")
    for path, records in ((tracks_path, tracce), (ball_tracks_path, tracce_palla),
                          (possession_path, possessi), (shot_path, tiri.shots if tiri else [])):
        with path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, allow_nan=False)
    for source in ("yolo", "interpolated", "predicted", "color"):
        print(f"Palla {source}: {sum(item['source'] == source for item in tracce_palla)} posizioni.")
    print(f"Frame originali: {fotogrammi}; frame elaborati e salvati: {elaborati}.")
    return elaborati, len(tracce)


def separa_rilevamenti(results, classi, shape):
    """Divide l'unica risposta YOLO per classe e scarta coordinate non utilizzabili."""
    groups = {name: [] for name in classi}
    if not results or results[0].boxes is None:
        return groups
    altezza, larghezza = shape
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        for name, indice in classi.items():
            if class_id != indice:
                continue
            coords = box.xyxy[0].tolist()
            score = float(box.conf.item())
            if not all(math.isfinite(value) for value in coords + [score]):
                continue
            x1, y1, x2, y2 = coords
            if x1 < 0 or y1 < 0 or x2 > larghezza or y2 > altezza or x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
                continue
            groups[name].append([x1, y1, x2, y2, score, class_id])
    return groups


def salva_debug_canestro(image, frame, detections, hoop, threshold, processed, directory):
    """Salva il frame originale con candidati grezzi e riferimento conservato, ben distinti."""
    debug = image.copy()
    status = "INFERENZA" if processed else "SALTATO: nessuna inferenza"
    cv2.putText(debug, f"Frame {frame} - {status}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    for row in detections:
        x1, y1, x2, y2 = map(int, row[:4])
        color = (0,255,0) if row[4] >= threshold else (0,165,255)
        cv2.rectangle(debug, (x1,y1), (x2,y2), color, 2)
        cv2.putText(debug, f"Hoop {row[4]:.3f}" + (" SOTTO SOGLIA" if row[4] < threshold else " CANDIDATO"),
                    (x1,max(40,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if not detections:
        cv2.putText(debug, "Nessun Hoop corrente", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if hoop is not None:
        x1, y1, x2, y2 = hoop["bbox"]
        cv2.rectangle(debug, (x1,y1), (x2,y2), (255,255,0), 1)
        confidence = "manuale" if hoop["confidence"] is None else f"conf {hoop['confidence']:.3f}"
        cv2.putText(debug, f"Riferimento frame {hoop['frame']} {confidence}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    if not cv2.imwrite(str(directory / f"frame_{frame:06d}.jpg"), debug):
        raise OSError(f"Impossibile salvare il debug del frame {frame} in {directory}.")


def cerchio_tratteggiato(image, center, radius, color, segments=12):
    """OpenCV non ha cerchi tratteggiati: disegna archi alternati a spazi vuoti."""
    step = 360 / segments
    for index in range(segments):
        start = index * step
        cv2.ellipse(image, center, (radius, radius), 0, start, start + step / 2, color, 2)


def scrivi_frame(record, writer, tiri, fps, min_shot_speed, skip_frames, threshold, tracks, balls, possessions):
    """Disegna e calcola le statistiche dopo aver completato l'eventuale interpolazione."""
    frame = record["image"]
    numero = record["frame"]
    palle = [record["ball"]] if record["ball"] is not None else []
    possesso = calcola_possesso(record["players"], palle, numero, threshold)
    possessions.append(possesso)
    tracks.extend(record["players"])
    balls.extend(palle)
    for ball in palle:
        x1, y1, x2, y2 = map(int, ball["bbox"])
        center = ((x1+x2)//2, (y1+y2)//2)
        if ball.get("estimated"):
            # Posizione stimata, non vista da YOLO: cerchio arancione tratteggiato.
            cerchio_tratteggiato(frame, center, max(10, (max(x2-x1,y2-y1)+1)//2), (0,165,255))
            label = "PALLA INTERPOLATA" if ball["source"] == "interpolated" else "PALLA STIMATA"
            color = (0,165,255)
        else:
            cv2.circle(frame, center, max(1, (max(x2-x1,y2-y1)+1)//2), (0,0,255), 2)
            label, color = "PALLA", (0,0,255)
        cv2.putText(frame, label, (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for player in record["players"]:
        x1, y1, x2, y2 = player["bbox"]
        owner = player["id"] == possesso["player_id"]
        color = (0,255,255) if owner else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"ID: {player['id']}" + (" PALLA" if owner else ""),
                    (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if record["hoop"] is not None:
        bbox = record["hoop"]["bbox"]
        if tiri is None:
            tiri = ShotDetector(bbox, fps, min_shot_speed, frame_step=skip_frames)
        else:
            tiri.set_hoop(bbox, numero)  # Usa il riferimento valido per questo frame, non uno scoperto nel futuro.
        message = tiri.update(numero, palle, possesso)
        x1, y1, x2, y2 = bbox
        cv2.circle(frame, ((x1+x2)//2,(y1+y2)//2), max(1,round(math.hypot(x2-x1,y2-y1)/2)), (0,255,0), 2)
        if message:
            cv2.putText(frame, message, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    writer.write(frame)
    return tiri
