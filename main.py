"""Avvia tracking, possesso e rilevamento dei tiri verso un canestro fisso."""

import argparse  # Legge le opzioni scritte nel terminale.
import math  # Verifica che la soglia del possesso sia un numero finito.
from pathlib import Path  # Gestisce i percorsi dei file.

from src.detection import rileva_persone  # Importa la funzione che analizza il video.
from src.ball_color import valida_range_hsv  # Controlla gli intervalli di colore scelti dall'utente.
from src.shots import parse_hoop_bbox  # Legge il rettangolo del ferro nelle coordinate del video.


def main():
    # Descriviamo il programma e le opzioni disponibili.
    parser = argparse.ArgumentParser(description="Segue giocatori e palla, stima possesso e tiri verso il canestro; salva video e JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Video da analizzare, es. data/partita.mp4")
    parser.add_argument("--hoop-bbox", required=True, type=parse_hoop_bbox, help='Coordinate del ferro nel primo frame: "x1,y1,x2,y2"')
    parser.add_argument("--shot-output", type=Path, default=Path("output/shots.json"), help="File JSON con i tiri e gli esiti stimati")
    parser.add_argument("--min-shot-speed", type=float, default=5.0, help="Velocità minima verso l'alto in pixel/frame (default: 5)")
    parser.add_argument("--output", type=Path, default=Path("output/partita_rilevata.mp4"), help="Video da salvare")
    parser.add_argument("--tracks", type=Path, default=Path("output/tracks.json"), help="File JSON con frame, ID e rettangoli")
    parser.add_argument("--ball-tracks", type=Path, default=Path("output/ball_tracks.json"), help="File JSON con le posizioni della palla")
    parser.add_argument("--possession-output", type=Path, default=Path("output/possession.json"), help="File JSON con il possesso per ogni frame")
    parser.add_argument("--possession-threshold", type=float, default=150.0, help="Distanza massima per il possesso in pixel (default: 150)")
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"), help="Pesi del modello YOLOv8")
    parser.add_argument("--confidence", type=float, default=0.1, help="Soglia YOLO tra 0 e 1 (default: 0.1, adatta a ByteTrack)")
    parser.add_argument("--ball-confidence", type=float, default=0.25, help="Soglia della palla tra 0 e 1 (default: 0.25)")
    parser.add_argument("--use-color-fallback", action=argparse.BooleanOptionalAction, default=True, help="Fallback arancione attivo per default; disattiva con --no-use-color-fallback")
    parser.add_argument("--color-hsv-range", type=int, nargs=6, default=[5, 100, 100, 15, 255, 255], metavar=("H_MIN", "S_MIN", "V_MIN", "H_MAX", "S_MAX", "V_MAX"), help="Sei valori HSV: minimo e massimo (OpenCV)")
    args = parser.parse_args()  # Legge i valori forniti dall'utente.
    if not 0 < args.confidence <= 1:  # Impedisce di usare una soglia non valida.
        parser.error("--confidence deve essere maggiore di 0 e minore o uguale a 1.")
    if not 0 < args.ball_confidence <= 1:  # La palla ha una soglia separata da quella dei giocatori.
        parser.error("--ball-confidence deve essere maggiore di 0 e minore o uguale a 1.")
    if not math.isfinite(args.possession_threshold) or args.possession_threshold <= 0:  # Rifiuta anche NaN e infinito.
        parser.error("--possession-threshold deve essere un numero finito maggiore di zero.")
    if not math.isfinite(args.min_shot_speed) or args.min_shot_speed <= 0:
        parser.error("--min-shot-speed deve essere un numero finito maggiore di zero.")
    try:  # Mostra un messaggio leggibile in caso di problemi con video o modello.
        hsv_range = (args.color_hsv_range[:3], args.color_hsv_range[3:])  # Divide i sei valori nei due limiti HSV.
        valida_range_hsv(hsv_range)  # Segnala subito un intervallo non valido.
        fotogrammi, rilevamenti = rileva_persone(
            args.input, args.output, args.model, args.confidence, args.tracks,
            ball_tracks_path=args.ball_tracks, ball_confidence=args.ball_confidence,
            possession_path=args.possession_output, possession_threshold=args.possession_threshold,
            use_color_fallback=args.use_color_fallback, color_hsv_range=hsv_range,
            hoop_bbox=args.hoop_bbox, shot_path=args.shot_output, min_shot_speed=args.min_shot_speed,
        )  # Mantiene le analisi precedenti e aggiunge i tiri verso il canestro indicato.
    except (OSError, ValueError, RuntimeError) as errore:
        parser.exit(1, f"Errore: {errore}\n")  # Termina con un codice che segnala il problema.
    print(f"Completato: {fotogrammi} fotogrammi, {rilevamenti} posizioni con ID.")  # Conta le posizioni, non i giocatori distinti.
    print("Gli ID sono locali a questo video e possono cambiare dopo occlusioni o uscite dall'inquadratura.")  # Spiega il limite del tracking.
    print(f"Video salvato in: {args.output.resolve()}")  # Indica dove trovare il risultato.
    print(f"Tracce JSON salvate in: {args.tracks.resolve()}")  # Indica dove trovare i dati per le statistiche future.
    print(f"Posizioni della palla salvate in: {args.ball_tracks.resolve()}")  # Indica il JSON separato della palla.
    print(f"Possesso stimato salvato in: {args.possession_output.resolve()}")  # Indica il JSON con un elemento per fotogramma.
    print(f"Tiri stimati salvati in: {args.shot_output.resolve()}")  # Indica il nuovo JSON dei tiri conclusi.


if __name__ == "__main__":  # Esegue il programma solo quando avviamo questo file.
    main()  # Avvia il programma.
