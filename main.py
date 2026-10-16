"""Avvia il rilevamento delle persone in un video di basket."""

import argparse  # Legge le opzioni scritte nel terminale.
from pathlib import Path  # Gestisce i percorsi dei file.

from src.detection import rileva_persone  # Importa la funzione che analizza il video.


def main():
    # Descriviamo il programma e le opzioni disponibili.
    parser = argparse.ArgumentParser(description="Rileva le persone con YOLOv8 e salva un video MP4.")
    parser.add_argument("--input", required=True, type=Path, help="Video da analizzare, es. data/partita.mp4")
    parser.add_argument("--output", type=Path, default=Path("output/partita_rilevata.mp4"), help="Video da salvare")
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"), help="Pesi del modello YOLOv8")
    parser.add_argument("--confidence", type=float, default=0.35, help="Soglia di fiducia tra 0 e 1")
    args = parser.parse_args()  # Legge i valori forniti dall'utente.
    if not 0 < args.confidence <= 1:  # Impedisce di usare una soglia non valida.
        parser.error("--confidence deve essere maggiore di 0 e minore o uguale a 1.")
    try:  # Mostra un messaggio leggibile in caso di problemi con video o modello.
        fotogrammi, rilevamenti = rileva_persone(args.input, args.output, args.model, args.confidence)
    except (OSError, ValueError, RuntimeError) as errore:
        parser.exit(1, f"Errore: {errore}\n")  # Termina con un codice che segnala il problema.
    print(f"Completato: {fotogrammi} fotogrammi, {rilevamenti} rilevamenti totali.")  # Riassume il lavoro.
    print("I rilevamenti ripetuti nei fotogrammi NON sono giocatori distinti.")  # Chiarisce il significato del conteggio.
    print(f"Video salvato in: {args.output.resolve()}")  # Indica dove trovare il risultato.


if __name__ == "__main__":  # Esegue il programma solo quando avviamo questo file.
    main()  # Avvia il programma.
