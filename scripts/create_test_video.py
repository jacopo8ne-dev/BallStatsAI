"""Crea un breve video dimostrativo da una foto pubblica di Ultralytics."""

from pathlib import Path  # Gestisce i percorsi dei file.
from urllib.request import urlopen  # Scarica l'immagine di esempio tramite HTTPS.

import cv2  # Legge l'immagine e crea un video.


root = Path(__file__).resolve().parents[1]  # Trova la cartella principale del progetto.
image_path = root / "data" / "test_bus.jpg"  # Sceglie dove salvare la foto di esempio.
video_path = root / "data" / "test_persone.mp4"  # Sceglie dove salvare il video di test.
if video_path.exists():  # Evita di sostituire un video già presente.
    raise SystemExit(f"Il file esiste già: {video_path}. Rimuovilo se vuoi ricrearlo.")
image_path.parent.mkdir(parents=True, exist_ok=True)  # Crea la cartella data se manca.
if not image_path.exists():  # Riutilizza la foto se è già stata scaricata.
    url = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"  # Foto dimostrativa ufficiale.
    with urlopen(url, timeout=60) as risposta:  # Scarica mantenendo attiva la verifica HTTPS.
        image_path.write_bytes(risposta.read())  # Salva la foto nella cartella data.
frame = cv2.imread(str(image_path))  # Apre la foto appena scaricata.
if frame is None:  # Controlla che la foto sia leggibile.
    raise SystemExit("Immagine di test non leggibile.")
frame = cv2.resize(frame, (480, 640))  # Riduce la risoluzione per un test rapido sulla CPU.
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (480, 640))  # Prepara due secondi di video.
try:  # Assicura la chiusura del file anche in caso di errore.
    if not writer.isOpened():  # Controlla la disponibilità del codec.
        raise SystemExit("Impossibile creare il video di test.")
    for numero in range(12):  # Ripete la foto per dodici fotogrammi.
        writer.write(frame)  # Inserisce la foto nel video.
finally:  # Completa il file video.
    writer.release()  # Chiude il file di uscita.
print(f"Video dimostrativo creato: {video_path}")  # Indica il risultato.
