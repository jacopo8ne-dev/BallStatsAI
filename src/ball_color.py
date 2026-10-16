"""Cerca una possibile palla arancione usando colore e forma, senza YOLO."""

import cv2  # Converte i colori e misura i contorni.
import numpy as np  # Crea gli intervalli HSV e il piccolo filtro per pulire la maschera.

DEFAULT_HSV_RANGE = ((5, 100, 100), (15, 255, 255))  # Limite inferiore e superiore per l'arancione.


def valida_range_hsv(hsv_range):
    # In OpenCV H va da 0 a 179; S e V vanno da 0 a 255.
    if len(hsv_range) != 2 or any(len(limite) != 3 for limite in hsv_range):
        raise ValueError("Il range HSV deve contenere due gruppi di tre interi.")
    inferiore, superiore = hsv_range  # Separa i due estremi dell'intervallo.
    for minimo, massimo, limite in zip(inferiore, superiore, (179, 255, 255)):
        if not isinstance(minimo, (int, np.integer)) or not isinstance(massimo, (int, np.integer)):
            raise ValueError("I valori HSV devono essere interi.")
        if not 0 <= minimo <= massimo <= limite:  # Impedisce intervalli invertiti o fuori scala.
            raise ValueError("Range HSV non valido: richiede minimo <= massimo, H in 0..179 e S/V in 0..255.")
    return np.array(inferiore, dtype=np.uint8), np.array(superiore, dtype=np.uint8)


def detect_ball_by_color(frame, hsv_range=DEFAULT_HSV_RANGE):
    """Restituisce [x1, y1, x2, y2] della forma arancione più circolare, oppure None."""
    inferiore, superiore = valida_range_hsv(hsv_range)  # Controlla il range prima di convertirlo in byte.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Separa tonalità, saturazione e luminosità.
    maschera = cv2.inRange(hsv, inferiore, superiore)  # I pixel del colore cercato diventano bianchi.
    filtro = np.ones((3, 3), dtype=np.uint8)  # Un piccolo filtro evita di eliminare palle abbastanza piccole.
    maschera = cv2.morphologyEx(maschera, cv2.MORPH_OPEN, filtro)  # Elimina puntini isolati.
    maschera = cv2.morphologyEx(maschera, cv2.MORPH_CLOSE, filtro)  # Riempie piccoli buchi nella forma.
    contorni, _ = cv2.findContours(maschera, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Trova le forme separate.
    migliore = None  # Nessuna forma valida è stata ancora trovata.
    punteggio_migliore = (0.0, 0.0)  # Confronta prima la circolarità e poi l'area.
    area_frame = frame.shape[0] * frame.shape[1]  # Permette di escludere grandi superfici arancioni.
    for contorno in contorni:
        area = cv2.contourArea(contorno)  # Misura l'area della forma in pixel quadrati.
        if area < 20 or area > area_frame * 0.05:  # Scarta rumore e oggetti che occupano oltre il 5% del frame.
            continue
        perimetro = cv2.arcLength(contorno, True)  # Misura il bordo della forma chiusa.
        if perimetro <= 0:  # Evita una divisione per zero.
            continue
        circolarita = 4 * np.pi * area / (perimetro * perimetro)  # Un cerchio ideale ha valore 1.
        x, y, larghezza, altezza = cv2.boundingRect(contorno)  # Trova il rettangolo che contiene la forma.
        rapporto = larghezza / altezza  # Una forma rotonda ha larghezza e altezza simili.
        vertici = cv2.approxPolyDP(contorno, 0.03 * perimetro, True)  # Distingue forme curve da rettangoli e triangoli.
        if circolarita < 0.75 or not 0.75 <= rapporto <= 1.33 or len(vertici) < 6:
            continue  # Scarta forme allungate o troppo squadrate, anche se sono arancioni.
        punteggio = (circolarita, area)  # A parità di circolarità preferisce la forma più grande.
        if punteggio > punteggio_migliore:
            punteggio_migliore = punteggio  # Conserva il miglior candidato del fotogramma.
            migliore = [int(x), int(y), int(x + larghezza), int(y + altezza)]  # Stesso formato dei rettangoli YOLO.
    return migliore  # Restituisce None se nessun candidato supera i filtri.
