# BallStatsAI — rilevamento e tracking delle persone

Questo progetto legge un video, usa **YOLOv8 Nano** di Ultralytics per trovare le
persone. **ByteTrack** collega i rilevamenti tra fotogrammi e assegna un ID
a ogni traccia. Il programma salva un video MP4 con rettangoli verdi ed etichette
come `ID: 3`, oltre a un file JSON con le posizioni.
Funziona sulla CPU: non serve una scheda grafica dedicata.

YOLO rileva tutte le persone visibili, compresi arbitri, spettatori e panchine.
Non distingue le squadre, non riconosce i nomi dei giocatori e non calcola ancora
statistiche, possessi o canestri. Un ID identifica una traccia nel video, non una
persona riconosciuta per nome o numero di maglia.

## Struttura

```text
BallStatsAI/
├── main.py                      # Programma da eseguire
├── requirements.txt             # Librerie necessarie
├── src/
│   ├── __init__.py
│   └── detection.py              # Lettura, ByteTrack e salvataggio video/JSON
├── scripts/
│   └── create_test_video.py      # Crea un piccolo video dimostrativo
├── data/                        # Inserisci qui i video originali
├── models/                      # Pesi YOLO scaricati automaticamente
└── output/                      # Video con ID e tracce JSON
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

Se avevi già installato il progetto, ripeti questo comando nell'ambiente attivo:
ora include anche `lap`, la libreria usata da ByteTrack per associare i rilevamenti.

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
python main.py --input data/partita.mp4 --output output/partita_rilevata.mp4 --tracks output/tracks.json
```

Il tracking è attivo automaticamente: non serve un'opzione per abilitarlo.
Il vecchio comando senza `--tracks` continua a funzionare e salva le tracce in
`output/tracks.json`. Per ogni nuova partita usa nomi di uscita diversi se vuoi
conservare i risultati precedenti: video e JSON esistenti vengono sovrascritti.

### Come funziona l'ID

YOLO trova le persone in ogni fotogramma. ByteTrack confronta posizioni e movimento
con i fotogrammi precedenti e prova a riutilizzare lo stesso ID. Nel codice,
`model.track(..., persist=True, tracker="bytetrack.yaml")` mantiene lo stato
del tracker fra le chiamate consecutive. Ogni analisi crea un nuovo modello e un
nuovo tracker: gli ID sono locali al video e non vanno confrontati fra partite.

L'identità non è garantita: incroci, occlusioni, rilevamenti mancanti e uscite
dall'inquadratura possono causare un nuovo ID o uno scambio. Il tracker non
riconosce il volto o la maglia. Se una persona non riceve ancora un ID, non viene
annotata né esportata fino all'assegnazione; nei fotogrammi senza persone il video
continua normalmente. Non vengono esportate posizioni inventate durante le assenze.

### Formato di `tracks.json`

Il JSON è una lista, con un elemento per ogni persona con ID in ogni fotogramma.
Esempio illustrativo:

```json
[
  {"frame": 0, "id": 3, "bbox": [100, 120, 180, 300]},
  {"frame": 1, "id": 3, "bbox": [102, 120, 182, 300]}
]
```

- `frame`: indice intero del fotogramma, **a partire da 0**.
- `id`: intero assegnato da ByteTrack, uguale all'etichetta nel video; può avere salti.
- `bbox`: `[x1, y1, x2, y2]` in pixel sul video originale; angolo superiore sinistro
  e angolo inferiore destro. L'origine `(0, 0)` è in alto a sinistra.

La stessa persona compare in più elementi: il totale delle posizioni non è il
numero di giocatori. Il JSON viene scritto al termine dell'analisi riuscita;
senza tracce contiene `[]`. Le tracce sono conservate in memoria fino al salvataggio:
video molto lunghi producono file più grandi e richiedono più RAM. Se interrompi
l'elaborazione, il video può essere parziale e il JSON non viene aggiornato;
un eventuale JSON precedente resta quello della vecchia analisi.

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
python main.py --input data/test_persone.mp4 --output output/test_rilevato.mp4 --tracks output/test_tracks.json
```

Se il video di prova esiste già, salta `create_test_video.py` e riutilizzalo.

Lo script scarica la [foto dimostrativa di Ultralytics](https://github.com/ultralytics/assets/blob/main/im/bus.jpg)
e la ripete in 12 fotogrammi (2 secondi, 6 FPS, 480 × 640).
È una scena con persone e un autobus, non una partita: permette di controllare
download, lettura video, rilevamento e scrittura senza un video personale.
Le immagini sono ferme; questo test non verifica il comportamento su giocatori
in movimento. Il risultato atteso è un MP4 con rettangoli ed ID attorno alle persone
e un JSON nel quale le stesse persone mantengono l'ID fra i fotogrammi.
Un numero di rilevamenti maggiore di 12 è normale: ogni persona viene contata
di nuovo in ciascun fotogramma. Non è il numero di giocatori distinti.

Verifiche eseguite con Python 3.12 e la soglia predefinita `0.1`:

- Video dimostrativo: 12 fotogrammi, 48 posizioni, quattro ID stabili; JSON
  verificato e MP4 riaperto con 12 fotogrammi annotati a 480 × 640 e 6 FPS.
- Prova sintetica di 27 fotogrammi: foto, tre fotogrammi neri e foto traslata
  di 5 pixel. I quattro ID sono stati recuperati dopo la breve assenza; nessuna
  posizione è stata esportata nei fotogrammi neri. Le etichette disegnate
  corrispondono agli ID del JSON.
- Nuova analisi di un video senza persone: tre fotogrammi elaborati e JSON `[]`,
  senza conservare le tracce dell'analisi precedente.
- Percorsi delle tracce non validi rifiutati; dipendenze verificate con `pip check`.

Queste prove controllano il funzionamento tecnico. Non misurano l'accuratezza
durante una partita reale, che va verificata sul proprio video.

## Opzioni e problemi comuni

Per vedere tutte le opzioni:

```bash
python main.py --help
```

La soglia YOLO predefinita è ora `0.1` (prima era `0.35`): ByteTrack può usare
anche rilevamenti meno sicuri per continuare tracce già esistenti. Non ogni
rilevamento debole diventa una nuova traccia: ByteTrack applica anche le proprie
soglie di associazione e conferma. Puoi alzare la soglia YOLO, ma rischi di perdere
persone poco visibili e interrompere le tracce:

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
