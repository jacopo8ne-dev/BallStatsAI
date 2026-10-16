"""Legge un video, rileva le persone e disegna un rettangolo per ogni rilevamento."""

import math  # Controlla che la frequenza del video sia un numero valido.
from pathlib import Path  # Gestisce cartelle e nomi dei file.

import cv2  # OpenCV legge, disegna e salva i fotogrammi del video.
from ultralytics import YOLO  # YOLO riconosce gli oggetti nelle immagini.


def rileva_persone(input_path, output_path, model_path, confidence=0.35):
    input_path = Path(input_path)  # Converte il percorso del video in un oggetto Path.
    output_path = Path(output_path)  # Converte il percorso del risultato.
    model_path = Path(model_path)  # Converte il percorso dei pesi del modello.
    if not input_path.is_file():  # Controlla che il video esista prima di scaricare il modello.
        raise FileNotFoundError(f"Video non trovato: {input_path}. Copialo nella cartella data/.")
    if input_path.resolve() == output_path.resolve():  # Protegge il video originale.
        raise ValueError("Il video di ingresso e quello di uscita devono avere nomi diversi.")
    if output_path.suffix.lower() != ".mp4":  # Usiamo il formato del codec scelto sotto.
        raise ValueError("Il file di uscita deve avere estensione .mp4.")
    if not 0 < confidence <= 1:  # Controlla la soglia anche quando la funzione è importata.
        raise ValueError("La soglia di fiducia deve essere compresa tra 0 (escluso) e 1.")

    video = cv2.VideoCapture(str(input_path))  # Apre il video di ingresso.
    writer = None  # Il video di uscita sarà aperto dopo aver letto il primo fotogramma.
    fotogrammi = 0  # Conta i fotogrammi elaborati.
    rilevamenti = 0  # Conta le persone rilevate sommando tutti i fotogrammi.
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
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la cartella dei risultati.
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # Sceglie un codec per salvare un video MP4.
        writer = cv2.VideoWriter(str(output_path), codec, fps, (larghezza, altezza))  # Crea il video.
        if not writer.isOpened():  # Controlla che la destinazione sia scrivibile e il codec disponibile.
            raise RuntimeError("Impossibile creare il video di uscita. Controlla cartella e codec MP4.")
        while successo:  # Elabora un fotogramma alla volta per contenere l'uso di memoria.
            risultato = model.predict(frame, classes=[0], conf=confidence, device="cpu", verbose=False)[0]  # La classe 0 è 'persona'.
            for box in risultato.boxes:  # Visita ogni persona individuata in questo fotogramma.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Legge gli angoli del rettangolo.
                fiducia = float(box.conf[0])  # Legge la fiducia del modello in questo rilevamento.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Disegna un rettangolo verde.
                cv2.putText(frame, f"Persona {fiducia:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Aggiunge l'etichetta.
                rilevamenti += 1  # Conta questo rilevamento, senza attribuire un'identità alla persona.
            writer.write(frame)  # Aggiunge il fotogramma con i rettangoli al video di uscita.
            fotogrammi += 1  # Aggiorna il numero di fotogrammi elaborati.
            if fotogrammi % 100 == 0:  # Comunica l'avanzamento durante i video lunghi.
                print(f"Elaborati {fotogrammi} fotogrammi...")  # Mostra il progresso nel terminale.
            successo, frame = video.read()  # Legge il fotogramma successivo; termina alla fine del video.
    finally:  # Libera le risorse anche in caso di interruzione.
        video.release()  # Chiude il video di ingresso.
        if writer is not None:  # Chiude l'uscita soltanto se è stata creata.
            writer.release()  # Completa la scrittura del file MP4.
    return fotogrammi, rilevamenti  # Restituisce il riepilogo al programma principale.
