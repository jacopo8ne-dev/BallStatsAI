"""Stima il possesso dalla distanza fra i centri di giocatori e palla."""

import math  # Calcola le distanze in pixel e controlla che la soglia sia finita.


def calcola_possesso(giocatori, palle, frame, soglia=150.0):
    # Una soglia non valida deve essere rifiutata anche usando direttamente questa funzione.
    if not math.isfinite(soglia) or soglia <= 0:
        raise ValueError("La soglia del possesso deve essere un numero finito maggiore di zero.")
    # -1.0 significa che non è possibile misurare la distanza; non indica una distanza reale.
    possesso = {"frame": frame, "player_id": None, "ball_distance": -1.0, "confidence": 0.0}
    if not palle:  # Senza palla non manteniamo il possessore del fotogramma precedente.
        return possesso
    palla = max(palle, key=lambda elemento: elemento["confidence"])  # Seleziona la palla con maggiore fiducia YOLO.
    possesso["confidence"] = float(palla["confidence"])  # È la fiducia della palla, non una probabilità di possesso.
    if not giocatori:  # Non assegna la palla a una persona che il tracker non sta seguendo.
        return possesso
    x1, y1, x2, y2 = palla["bbox"]  # Legge il rettangolo della palla selezionata.
    centro_palla_x, centro_palla_y = (x1 + x2) / 2, (y1 + y2) / 2  # Conserva anche i mezzi pixel.
    distanza_minima = None  # Nessun giocatore è stato ancora confrontato.
    giocatore_vicino = None  # Memorizza l'ID del giocatore con il centro più vicino.
    for giocatore in giocatori:  # Confronta solo le posizioni dello stesso fotogramma.
        x1, y1, x2, y2 = giocatore["bbox"]  # Legge il rettangolo del corpo.
        centro_x, centro_y = (x1 + x2) / 2, (y1 + y2) / 2  # Calcola il centro del corpo.
        distanza = math.hypot(centro_x - centro_palla_x, centro_y - centro_palla_y)  # Distanza euclidea in pixel.
        if distanza_minima is None or distanza < distanza_minima:  # A parità di distanza mantiene il primo giocatore.
            distanza_minima = distanza  # Aggiorna la distanza più piccola.
            giocatore_vicino = giocatore["id"]  # Conserva l'ID corrispondente.
    possesso["ball_distance"] = float(distanza_minima)  # Salva la distanza anche se supera la soglia.
    if distanza_minima < soglia:  # Il confronto è stretto: una distanza uguale alla soglia non basta.
        possesso["player_id"] = giocatore_vicino  # Assegna un solo possessore a questo fotogramma.
    return possesso  # Restituisce sempre un record, anche quando il possessore è sconosciuto.
