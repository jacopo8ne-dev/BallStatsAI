"""Segue persone e palla, stima possesso e tiri fotogramma per fotogramma."""

import json  # Salva le tracce in un formato leggibile anche da altri programmi.
import math  # Controlla che la frequenza del video sia un numero valido.
from pathlib import Path  # Gestisce cartelle e nomi dei file.

import cv2  # OpenCV legge, disegna e salva i fotogrammi del video.
from ultralytics import YOLO  # YOLO riconosce gli oggetti nelle immagini.

from src.possession import calcola_possesso  # Collega la palla al giocatore più vicino nello stesso frame.
from src.ball_color import DEFAULT_HSV_RANGE, detect_ball_by_color, valida_range_hsv  # Recupera palle arancioni non trovate da YOLO.
from src.shots import ShotDetector  # Riconosce salita e attraversamento discendente del canestro.


def rileva_persone(input_path, output_path, model_path, confidence=0.1, tracks_path="output/tracks.json",
                   ball_tracks_path="output/ball_tracks.json", ball_confidence=0.25,
                   possession_path="output/possession.json", possession_threshold=150.0,
                   use_color_fallback=True, color_hsv_range=DEFAULT_HSV_RANGE,
                   hoop_bbox=None, shot_path="output/shots.json", min_shot_speed=5.0):
    input_path = Path(input_path)  # Converte il percorso del video in un oggetto Path.
    output_path = Path(output_path)  # Converte il percorso del risultato.
    model_path = Path(model_path)  # Converte il percorso dei pesi del modello.
    tracks_path = Path(tracks_path)  # Sceglie il file dove salvare le posizioni con ID.
    ball_tracks_path = Path(ball_tracks_path)  # Tiene le posizioni della palla separate da quelle dei giocatori.
    possession_path = Path(possession_path)  # Sceglie dove salvare il possesso stimato.
    shot_path = Path(shot_path)  # Sceglie il file dei tiri, distinto dagli altri risultati.
    if not input_path.is_file():  # Controlla che il video esista prima di scaricare il modello.
        raise FileNotFoundError(f"Video non trovato: {input_path}. Copialo nella cartella data/.")
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
    if possession_path.suffix.lower() != ".json":  # Il possesso deve essere salvato in formato JSON.
        raise ValueError("Il file del possesso deve avere estensione .json.")
    if possession_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve(), tracks_path.resolve(), ball_tracks_path.resolve()):
        raise ValueError("Il JSON del possesso deve essere diverso da tutti gli altri file dell'analisi.")
    if not math.isfinite(possession_threshold) or possession_threshold <= 0:  # Controlla anche le chiamate dirette alla funzione.
        raise ValueError("La soglia del possesso deve essere un numero finito maggiore di zero.")
    valida_range_hsv(color_hsv_range)  # Controlla il range anche quando la funzione è chiamata senza main.py.
    if hoop_bbox is None:
        raise ValueError("Specifica il canestro con --hoop-bbox x1,y1,x2,y2.")
    if shot_path.suffix.lower() != ".json":
        raise ValueError("Il file dei tiri deve avere estensione .json.")
    if shot_path.resolve() in (input_path.resolve(), output_path.resolve(), model_path.resolve(), tracks_path.resolve(), ball_tracks_path.resolve(), possession_path.resolve()):
        raise ValueError("Il JSON dei tiri deve essere diverso da tutti gli altri file dell'analisi.")

    video = cv2.VideoCapture(str(input_path))  # Apre il video di ingresso.
    writer = None  # Il video di uscita sarà aperto dopo aver letto il primo fotogramma.
    fotogrammi = 0  # Conta i fotogrammi elaborati.
    rilevamenti = 0  # Conta le posizioni alle quali ByteTrack ha assegnato un ID.
    tracce = []  # Ogni elemento conterrà frame, ID e rettangolo di una persona.
    tracce_palla = []  # Ogni elemento conterrà frame, rettangolo e fiducia di una palla rilevata.
    possessi = []  # Conterrà esattamente un risultato per ogni fotogramma elaborato.
    try:  # Chiude sempre i file, anche quando si verifica un errore.
        if not video.isOpened():  # Verifica che OpenCV riesca ad aprire il formato.
            raise ValueError("Impossibile aprire il video. Prova un file MP4 leggibile.")
        successo, frame = video.read()  # Legge il primo fotogramma.
        if not successo:  # Rifiuta un video vuoto o non decodificabile.
            raise ValueError("Il video non contiene fotogrammi leggibili.")
        fps = video.get(cv2.CAP_PROP_FPS)  # Legge quanti fotogrammi vengono mostrati al secondo.
        if not math.isfinite(fps) or fps <= 0:  # Un valore mancante impedisce di preservare la velocità.
            raise ValueError("La frequenza FPS del video non è valida. Converti il video in MP4.")
        altezza, larghezza = frame.shape[:2]  # Legge le dimensioni delle immagini.
        tiri = ShotDetector(hoop_bbox, fps, min_shot_speed)  # Inizializza uno stato nuovo per ogni video.
        if tiri.hoop[2] > larghezza or tiri.hoop[3] > altezza:
            raise ValueError("Il rettangolo del canestro deve essere contenuto nelle dimensioni del video originale.")
        model_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella del modello.
        model = YOLO(str(model_path))  # Scarica automaticamente yolov8n.pt al primo utilizzo, se manca.
        # Carica gli stessi pesi in una seconda istanza: il rilevamento della palla non modifica lo stato di ByteTrack.
        modello_palla = YOLO(str(model_path))  # Non serve un modello nuovo o un addestramento aggiuntivo.
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella dei risultati.
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # Sceglie un codec per salvare un video MP4.
        writer = cv2.VideoWriter(str(output_path), codec, fps, (larghezza, altezza))  # Crea il video.
        if not writer.isOpened():  # Controlla che la destinazione sia scrivibile e il codec disponibile.
            raise RuntimeError("Impossibile creare il video di uscita. Controlla cartella e codec MP4.")
        tracks_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella delle tracce prima dell'analisi.
        ball_tracks_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara anche la cartella del JSON della palla.
        possession_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella del possesso prima di elaborare il video.
        shot_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella dei tiri prima dell'analisi.
        while successo:  # Elabora un fotogramma alla volta per contenere l'uso di memoria.
            giocatori_frame = []  # Azzera i giocatori da confrontare in questo fotogramma.
            palle_frame = []  # Non riutilizza una palla rilevata in un fotogramma passato.
            # persist=True conserva lo stesso tracker tra i fotogrammi invece di ricreare gli ID ogni volta.
            # La soglia bassa lascia a ByteTrack anche rilevamenti meno sicuri, utili per recuperare una traccia.
            risultato = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=confidence, device="cpu", verbose=False)[0]
            # La classe COCO 32 è "sports ball"; usa il fotogramma originale prima di disegnare le annotazioni.
            risultato_palla = modello_palla.predict(frame, classes=[32], conf=ball_confidence, device="cpu", verbose=False)[0]
            for box in risultato.boxes:  # Visita le persone restituite dal tracker.
                if box.id is None:  # Un rilevamento può non avere ancora un'identità confermata.
                    continue  # Non inventa ID e non esporta posizioni prive di identità.
                giocatore_id = int(box.id.item())  # Converte l'ID di ByteTrack in un normale intero Python.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Legge gli angoli del rettangolo.
                posizione = {"frame": fotogrammi, "id": giocatore_id, "bbox": [x1, y1, x2, y2]}  # Stesso formato delle tracce precedenti.
                giocatori_frame.append(posizione)  # Tiene i giocatori del frame disponibili per il possesso e il disegno.
                tracce.append(posizione)  # Conserva anche la traccia completa per il JSON dei giocatori.
                rilevamenti += 1  # Conta una posizione con ID esportata nel JSON.
            for box in risultato_palla.boxes:  # Salva tutte le palle rilevate, anche se nello stesso frame sono più di una.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Legge il rettangolo della palla in pixel.
                fiducia = float(box.conf.item())  # Converte la fiducia YOLO in un numero utilizzabile nel JSON.
                posizione = {"frame": fotogrammi, "bbox": [x1, y1, x2, y2], "confidence": fiducia}  # Stesso formato dei JSON precedenti.
                palle_frame.append(posizione)  # Conserva prima i risultati YOLO, senza disegnare sul frame originale.
            if not palle_frame and use_color_fallback:  # Usa il colore solo se nessuna palla supera --ball-confidence.
                bbox_colore = detect_ball_by_color(frame, color_hsv_range)  # Analizza il frame originale, senza cerchi o etichette.
                if bbox_colore is not None:
                    palle_frame.append({"frame": fotogrammi, "bbox": bbox_colore, "confidence": 0.0, "source": "color"})  # Il colore non fornisce una probabilità YOLO.
            for posizione in palle_frame:  # Disegna ed esporta allo stesso modo sia YOLO sia il fallback.
                x1, y1, x2, y2 = posizione["bbox"]  # Legge il rettangolo del metodo selezionato.
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calcola il centro del rettangolo.
                raggio = max(1, (max(x2 - x1, y2 - y1) + 1) // 2)  # Usa metà del lato più lungo come raggio.
                cv2.circle(frame, centro, raggio, (0, 0, 255), 2)  # OpenCV usa BGR: questo colore è rosso.
                cv2.putText(frame, "PALLA", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Etichetta la palla.
                tracce_palla.append(posizione)  # Offre al JSON e al possesso anche le palle recuperate dal colore.
            possesso = calcola_possesso(giocatori_frame, palle_frame, fotogrammi, possession_threshold)  # Confronta i centri solo in questo frame.
            possessi.append(possesso)  # Registra anche i frame con possessore null.
            for giocatore in giocatori_frame:  # Disegna i giocatori dopo aver stabilito chi ha il possesso.
                x1, y1, x2, y2 = giocatore["bbox"]  # Recupera il rettangolo senza modificare la traccia.
                ha_palla = giocatore["id"] == possesso["player_id"]  # Al massimo un giocatore è selezionato.
                colore = (0, 255, 255) if ha_palla else (0, 255, 0)  # Giallo per il possessore, verde per gli altri (BGR).
                etichetta = f"ID: {giocatore['id']}" + (" PALLA" if ha_palla else "")  # Aggiunge PALLA accanto all'ID.
                cv2.rectangle(frame, (x1, y1), (x2, y2), colore, 2)  # Evidenzia il possessore stimato.
                cv2.putText(frame, etichetta, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore, 2)  # Mostra l'etichetta corrispondente.
            messaggio_tiro = tiri.update(fotogrammi, palle_frame, possesso)  # Usa le stesse tracce esportate nei JSON.
            hx1, hy1, hx2, hy2 = tiri.hoop
            centro_canestro = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
            raggio_canestro = max(1, round(math.hypot(hx2 - hx1, hy2 - hy1) / 2))  # Cerchio che racchiude il rettangolo fornito.
            cv2.circle(frame, centro_canestro, raggio_canestro, (0, 255, 0), 2)  # Evidenzia sempre il canestro in verde.
            if messaggio_tiro:
                cv2.putText(frame, messaggio_tiro, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Mostra TIRO! e poi l'esito quando osservato.
            writer.write(frame)  # Aggiunge il fotogramma con i rettangoli al video di uscita.
            fotogrammi += 1  # Aggiorna il numero di fotogrammi elaborati.
            if fotogrammi % 100 == 0:  # Comunica l'avanzamento durante i video lunghi.
                print(f"Elaborati {fotogrammi} fotogrammi...")  # Mostra il progresso nel terminale.
            successo, frame = video.read()  # Legge il fotogramma successivo; termina alla fine del video.
    finally:  # Libera le risorse anche in caso di interruzione.
        video.release()  # Chiude il video di ingresso.
        if writer is not None:  # Chiude l'uscita soltanto se è stata creata.
            writer.release()  # Completa la scrittura del file MP4.
    tiri.finish_video(fotogrammi)  # Esclude i candidati rimasti in volo alla fine della registrazione.
    with shot_path.open("w", encoding="utf-8") as file:
        json.dump(tiri.shots, file, indent=2, allow_nan=False)  # Nessun tiro classificato produce la lista vuota [].
    print(f"Tiri: {len(tiri.shots)} con esito stimato, {tiri.unresolved} non classificabili (esclusi dal JSON).")
    with tracks_path.open("w", encoding="utf-8") as file:  # Scrive il JSON solo se l'analisi termina senza errori.
        json.dump(tracce, file, indent=2)  # Salva una lista di oggetti; senza persone viene salvata la lista vuota [].
    with ball_tracks_path.open("w", encoding="utf-8") as file:  # Salva la palla in un file indipendente.
        json.dump(tracce_palla, file, indent=2)  # Se la palla non viene mai rilevata, il JSON contiene [].
    with possession_path.open("w", encoding="utf-8") as file:  # Salva un record per ogni frame al termine dell'analisi.
        json.dump(possessi, file, indent=2, allow_nan=False)  # Usa solo numeri JSON validi, senza NaN o Infinity.
    recuperate = sum(posizione.get("source") == "color" for posizione in tracce_palla)  # Conta il contributo del fallback.
    print(f"Palla: {len(tracce_palla) - recuperate} rilevamenti YOLO, {recuperate} recuperati dal colore.")  # Aiuta a confrontare i metodi sul proprio video.
    return fotogrammi, rilevamenti  # Restituisce il riepilogo al programma principale.
