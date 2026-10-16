# BallStatsAI — primo passo: rilevare le persone

Questo progetto legge un video, usa **YOLOv8 Nano** di Ultralytics per trovare le
persone e salva un video MP4 con rettangoli verdi e una misura di fiducia.
Funziona sulla CPU: non serve una scheda grafica dedicata.

YOLO rileva tutte le persone visibili, compresi arbitri, spettatori e panchine.
In questo primo passo non distingue le squadre, non riconosce i singoli giocatori
e non calcola statistiche, possessi o canestri. La fiducia (es. `0.85`) indica
quanto il modello è sicuro del rilevamento, non la probabilità che sia un giocatore.

## Struttura

```text
BallStatsAI/
├── main.py                      # Programma da eseguire
├── requirements.txt             # Librerie necessarie
├── src/
│   ├── __init__.py
│   └── detection.py              # Lettura, rilevamento e salvataggio
├── scripts/
│   └── create_test_video.py      # Crea un piccolo video dimostrativo
├── data/                        # Inserisci qui i video originali
├── models/                      # Pesi YOLO scaricati automaticamente
└── output/                      # Video con rettangoli
```

I video, i modelli e l'ambiente virtuale sono esclusi da Git tramite `.gitignore`.
Conserva una copia dei tuoi video originali.

## 1. Installa Python e le librerie

Installa **Python 3.12** (versione verificata per questo progetto). Su Windows
seleziona “Add Python to PATH” durante l'installazione. Apri un terminale nella
cartella `BallStatsAI`, quella che contiene `main.py`.

Crea un ambiente virtuale: è una cartella che tiene separate le librerie di questo
progetto dalle altre installazioni di Python.

```bash
python -m venv .venv
```

Attivalo su Linux/macOS:

```bash
source .venv/bin/activate
```

Oppure su Windows, nel Prompt dei comandi (`cmd`):

```bat
.venv\Scripts\activate.bat
```

Installa le dipendenze:

```bash
python -m pip install -r requirements.txt
```

In questo ambiente cloud Linux sono state usate le versioni CPU di PyTorch,
installate **prima** di `requirements.txt` per evitare i pacchetti CUDA più grandi:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Il primo download delle librerie e del modello richiede Internet e spazio su disco.

## 2. Analizza la tua partita

Copia il video nella cartella `data/`, per esempio con il nome `partita.mp4`.
Esegui dalla cartella principale del progetto:

```bash
python main.py --input data/partita.mp4 --output output/partita_rilevata.mp4
```

Al primo avvio, Ultralytics scarica automaticamente `yolov8n.pt` in `models/` da
GitHub. I successivi avvii riutilizzano il file. Se lavori senza rete, devi prima
aver scaricato i pesi o indicare un modello YOLOv8 locale con `--model percorso/modello.pt`.

Aspetta il messaggio “Completato”, poi apri il file in `output/` con un lettore
video come VLC. Non appare una finestra durante l'elaborazione. Su CPU un video
lungo può richiedere tempo: inizia con una clip breve. Viene mostrato un messaggio
ogni 100 fotogrammi.

Il video salvato mantiene risoluzione e FPS del video di ingresso a frequenza
costante, **senza audio**. Per riprese a frequenza variabile, la temporizzazione
può differire. Un file di uscita esistente con lo stesso nome viene sovrascritto;
il programma impedisce di usare lo stesso percorso del video originale.

## 3. Prova rapida senza un video personale

```bash
python scripts/create_test_video.py
python main.py --input data/test_persone.mp4 --output output/test_rilevato.mp4
```

Lo script scarica la [foto dimostrativa di Ultralytics](https://github.com/ultralytics/assets/blob/main/im/bus.jpg)
e la ripete in 12 fotogrammi (2 secondi, 6 FPS, 480 × 640).
È una scena con persone e un autobus, non una partita: permette di controllare
download, lettura video, rilevamento e scrittura senza un video personale.
Le immagini sono ferme; questo test non verifica il comportamento su giocatori
in movimento. Il risultato atteso è un MP4 con rettangoli attorno alle persone.
Un numero di rilevamenti maggiore di 12 è normale: ogni persona viene contata
di nuovo in ciascun fotogramma. Non è il numero di giocatori distinti.

Verifica eseguita nell'ambiente cloud con Python 3.12: 12 fotogrammi elaborati,
48 rilevamenti complessivi. Il video risultante è stato riaperto e decodificato:
12 fotogrammi a 480 × 640 e 6 FPS, con annotazioni verdi in ogni fotogramma.
Sono stati verificati anche i messaggi per file mancante, soglia non valida,
formato di uscita errato e la protezione del video originale. Le dipendenze
superano il controllo `python -m pip check`.

## Opzioni e problemi comuni

Per vedere tutte le opzioni:

```bash
python main.py --help
```

Per accettare rilevamenti meno sicuri (con possibili falsi positivi):

```bash
python main.py --input data/partita.mp4 --confidence 0.25
```

- **Video non trovato:** verifica nome e percorso; usa le virgolette se contengono spazi.
- **Libreria non trovata:** attiva `.venv` e ripeti l'installazione.
- **Download del modello fallito:** verifica la connessione a GitHub e riprova.
- **Video non leggibile:** prova a convertirlo in MP4 con codec H.264 e FPS costanti.
- **Poche persone rilevate:** abbassa la soglia; persone piccole, nascoste o sfocate sono difficili da riconoscere.

Questo è un prototipo didattico. Prima di distribuire un prodotto, consulta i
[termini di licenza Ultralytics](https://www.ultralytics.com/license) per codice e modelli.
