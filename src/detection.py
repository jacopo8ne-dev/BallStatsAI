"""Segue le persone con ByteTrack e rileva la palla con gli stessi pesi YOLO."""

import json  # Salva le tracce in un formato leggibile anche da altri programmi.
import math  # Controlla che la frequenza del video sia un numero valido.
from pathlib import Path  # Gestisce cartelle e nomi dei file.

import cv2  # OpenCV legge, disegna e salva i fotogrammi del video.
from ultralytics import YOLO  # YOLO riconosce gli oggetti nelle immagini.


def rileva_persone(input_path, output_path, model_path, confidence=0.1, tracks_path="output/tracks.json",
                   ball_tracks_path="output/ball_tracks.json", ball_confidence=0.25):
    input_path = Path(input_path)  # Converte il percorso del video in un oggetto Path.
    output_path = Path(output_path)  # Converte il percorso del risultato.
    model_path = Path(model_path)  # Converte il percorso dei pesi del modello.
    tracks_path = Path(tracks_path)  # Sceglie il file dove salvare le posizioni con ID.
    ball_tracks_path = Path(ball_tracks_path)  # Tiene le posizioni della palla separate da quelle dei giocatori.
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

    video = cv2.VideoCapture(str(input_path))  # Apre il video di ingresso.
    writer = None  # Il video di uscita sarà aperto dopo aver letto il primo fotogramma.
    fotogrammi = 0  # Conta i fotogrammi elaborati.
    rilevamenti = 0  # Conta le posizioni alle quali ByteTrack ha assegnato un ID.
    tracce = []  # Ogni elemento conterrà frame, ID e rettangolo di una persona.
    tracce_palla = []  # Ogni elemento conterrà frame, rettangolo e fiducia di una palla rilevata.
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
        while successo:  # Elabora un fotogramma alla volta per contenere l'uso di memoria.
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Disegna un rettangolo verde.
                cv2.putText(frame, f"ID: {giocatore_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Mostra lo stesso ID sui fotogrammi associati.
                tracce.append({"frame": fotogrammi, "id": giocatore_id, "bbox": [x1, y1, x2, y2]})  # I fotogrammi sono numerati da zero.
                rilevamenti += 1  # Conta una posizione con ID esportata nel JSON.
            for box in risultato_palla.boxes:  # Salva tutte le palle rilevate, anche se nello stesso frame sono più di una.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Legge il rettangolo della palla in pixel.
                fiducia = float(box.conf.item())  # Converte la fiducia YOLO in un numero utilizzabile nel JSON.
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calcola il centro del rettangolo.
                raggio = max(1, (max(x2 - x1, y2 - y1) + 1) // 2)  # Usa metà del lato più lungo come raggio.
                cv2.circle(frame, centro, raggio, (0, 0, 255), 2)  # OpenCV usa BGR: questo colore è rosso.
                cv2.putText(frame, "PALLA", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Etichetta la palla.
                tracce_palla.append({"frame": fotogrammi, "bbox": [x1, y1, x2, y2], "confidence": fiducia})  # Stesso indice frame dei giocatori, a partire da zero.
            writer.write(frame)  # Aggiunge il fotogramma con i rettangoli al video di uscita.
            fotogrammi += 1  # Aggiorna il numero di fotogrammi elaborati.
            if fotogrammi % 100 == 0:  # Comunica l'avanzamento durante i video lunghi.
                print(f"Elaborati {fotogrammi} fotogrammi...")  # Mostra il progresso nel terminale.
            successo, frame = video.read()  # Legge il fotogramma successivo; termina alla fine del video.
    finally:  # Libera le risorse anche in caso di interruzione.
        video.release()  # Chiude il video di ingresso.
        if writer is not None:  # Chiude l'uscita soltanto se è stata creata.
            writer.release()  # Completa la scrittura del file MP4.
    with tracks_path.open("w", encoding="utf-8") as file:  # Scrive il JSON solo se l'analisi termina senza errori.
        json.dump(tracce, file, indent=2)  # Salva una lista di oggetti; senza persone viene salvata la lista vuota [].
    with ball_tracks_path.open("w", encoding="utf-8") as file:  # Salva la palla in un file indipendente.
        json.dump(tracce_palla, file, indent=2)  # Se la palla non viene mai rilevata, il JSON contiene [].
    return fotogrammi, rilevamenti  # Restituisce il riepilogo al programma principale.
