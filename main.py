"""Avvia il tracking delle persone e il rilevamento della palla."""

import argparse  # Legge le opzioni scritte nel terminale.
from pathlib import Path  # Gestisce i percorsi dei file.

from src.detection import rileva_persone  # Importa la funzione che analizza il video.


def main():
    # Descriviamo il programma e le opzioni disponibili.
    parser = argparse.ArgumentParser(description="Segue le persone con ByteTrack e rileva la palla con YOLOv8; salva video e JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Video da analizzare, es. data/partita.mp4")
    parser.add_argument("--output", type=Path, default=Path("output/partita_rilevata.mp4"), help="Video da salvare")
    parser.add_argument("--tracks", type=Path, default=Path("output/tracks.json"), help="File JSON con frame, ID e rettangoli")
    parser.add_argument("--ball-tracks", type=Path, default=Path("output/ball_tracks.json"), help="File JSON con le posizioni della palla")
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"), help="Pesi del modello YOLOv8")
    parser.add_argument("--confidence", type=float, default=0.1, help="Soglia YOLO tra 0 e 1 (default: 0.1, adatta a ByteTrack)")
    parser.add_argument("--ball-confidence", type=float, default=0.25, help="Soglia della palla tra 0 e 1 (default: 0.25)")
    args = parser.parse_args()  # Legge i valori forniti dall'utente.
    if not 0 < args.confidence <= 1:  # Impedisce di usare una soglia non valida.
        parser.error("--confidence deve essere maggiore di 0 e minore o uguale a 1.")
    if not 0 < args.ball_confidence <= 1:  # La palla ha una soglia separata da quella dei giocatori.
        parser.error("--ball-confidence deve essere maggiore di 0 e minore o uguale a 1.")
    try:  # Mostra un messaggio leggibile in caso di problemi con video o modello.
        fotogrammi, rilevamenti = rileva_persone(
            args.input, args.output, args.model, args.confidence, args.tracks,
            ball_tracks_path=args.ball_tracks, ball_confidence=args.ball_confidence,
        )  # Mantiene il tracking dei giocatori e aggiunge il rilevamento della palla.
    except (OSError, ValueError, RuntimeError) as errore:
        parser.exit(1, f"Errore: {errore}\n")  # Termina con un codice che segnala il problema.
    print(f"Completato: {fotogrammi} fotogrammi, {rilevamenti} posizioni con ID.")  # Conta le posizioni, non i giocatori distinti.
    print("Gli ID sono locali a questo video e possono cambiare dopo occlusioni o uscite dall'inquadratura.")  # Spiega il limite del tracking.
    print(f"Video salvato in: {args.output.resolve()}")  # Indica dove trovare il risultato.
    print(f"Tracce JSON salvate in: {args.tracks.resolve()}")  # Indica dove trovare i dati per le statistiche future.
    print(f"Posizioni della palla salvate in: {args.ball_tracks.resolve()}")  # Indica il JSON separato della palla.


if __name__ == "__main__":  # Esegue il programma solo quando avviamo questo file.
    main()  # Avvia il programma.
