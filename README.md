# BallStatsAI — tracking, palla e possesso stimato

Questo progetto legge un video, usa **YOLOv8 Nano** di Ultralytics per trovare le
persone. **ByteTrack** collega i rilevamenti tra fotogrammi e assegna un ID
a ogni traccia. Il programma salva un video MP4 con rettangoli verdi ed etichette
come `ID: 3`, oltre a un file JSON con le posizioni.
La classe COCO **sports ball (32)** viene rilevata con gli stessi pesi YOLO:
nel video appare con un cerchio rosso e l'etichetta **PALLA**; le sue posizioni
sono salvate in un secondo JSON.
Se YOLO non trova la palla, un fallback OpenCV cerca una forma arancione circolare.
Il giocatore con il centro più vicino alla palla, entro una soglia configurabile,
viene evidenziato in **giallo** con `ID: 3 PALLA`. Un terzo JSON registra il
possesso stimato per ogni fotogramma.
Funziona sulla CPU: non serve una scheda grafica dedicata.

YOLO rileva tutte le persone visibili, compresi arbitri, spettatori e panchine.
Non distingue le squadre, non riconosce i nomi dei giocatori e non calcola ancora
statistiche avanzate o canestri. Il possesso è una stima basata sulla vicinanza,
non un riconoscimento certo del controllo della palla. Un ID identifica una traccia nel video, non una
persona riconosciuta per nome o numero di maglia.

## Struttura

```text
BallStatsAI/
├── main.py                      # Programma da eseguire
├── requirements.txt             # Librerie necessarie
├── src/
│   ├── __init__.py
│   ├── detection.py              # Analisi e salvataggio video/JSON
│   ├── ball_color.py             # Fallback arancione con HSV e contorni
│   └── possession.py             # Possesso basato sulla distanza fra i centri
├── scripts/
│   ├── create_test_video.py      # Crea un piccolo video dimostrativo
│   ├── test_possession.py        # Test del calcolo del possesso
│   └── test_ball_color.py        # Test del fallback colore
├── data/                        # Inserisci qui i video originali
├── models/                      # Pesi YOLO scaricati automaticamente
└── output/                      # Video e JSON di giocatori, palla e possesso
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
python main.py --input data/partita.mp4 --output output/partita_rilevata.mp4 --tracks output/tracks.json --ball-tracks output/ball_tracks.json --possession-output output/possession.json --possession-threshold 150
```

Tracking, rilevamento della palla e possesso sono attivi automaticamente.
Il vecchio comando continua a funzionare: senza opzioni aggiuntive salva i
giocatori in `output/tracks.json`, la palla in `output/ball_tracks.json` e il possesso
in `output/possession.json`, con soglia di 150 pixel.
Non servono nuove dipendenze o un nuovo modello. Per ogni nuova partita usa nomi di uscita diversi se vuoi
conservare i risultati precedenti: video e JSON esistenti vengono sovrascritti.

### Come viene stimato il possesso

Per ogni fotogramma il programma confronta il centro del rettangolo della palla
con il centro del corpo di ciascun giocatore con ID. Calcola la distanza euclidea
in pixel e seleziona il giocatore più vicino **solo se la distanza è inferiore**
alla soglia: esattamente 150 pixel non basta con soglia 150. A parità di distanza
mantiene il primo giocatore restituito dal tracker.

Se YOLO rileva più palle, per il possesso viene usata quella con la fiducia maggiore
(a parità, la prima restituita da YOLO). Il JSON della palla continua a conservarle
tutte. Il possessore ha rettangolo ed etichetta gialli con `PALLA` accanto all'ID;
gli altri giocatori restano verdi e la palla mantiene il cerchio rosso.

Il calcolo usa in memoria le stesse posizioni esportate nei due JSON, senza
modificare YOLO o ByteTrack. Per ottenere il nuovo video e il possesso, esegui
nuovamente l'analisi sul video originale; questo comando non legge JSON precedenti.
Non vengono mantenuti possessi tra frame senza palla o senza giocatori.

La soglia dipende dalla risoluzione e dalla prospettiva della telecamera: non
corrisponde a una distanza in metri. Durante un passaggio, un tiro o un rimbalzo
il giocatore più vicino potrebbe non avere davvero il controllo della palla.
Anche arbitri e spettatori rilevati come persone possono essere selezionati.
Questo passo non calcola ancora durata del possesso o statistiche di squadra.

### Formato di `possession.json`

Il file contiene **un elemento per ogni fotogramma**, numerato da zero. Esempio:

```json
[
  {"frame": 0, "player_id": 3, "ball_distance": 64.5, "confidence": 0.82},
  {"frame": 1, "player_id": null, "ball_distance": 180.0, "confidence": 0.76},
  {"frame": 2, "player_id": null, "ball_distance": -1.0, "confidence": 0.0}
]
```

- `player_id`: ID del possessore stimato, oppure `null` se nessuno è abbastanza
  vicino, la palla manca o non ci sono giocatori con ID.
- `ball_distance`: distanza in pixel dal giocatore più vicino, anche se supera
  la soglia. Vale **`-1.0`** quando non è misurabile perché manca palla o giocatore;
  è un valore convenzionale da escludere dai calcoli delle distanze.
- `confidence`: fiducia YOLO della palla scelta, **non probabilità di possesso**.
  Vale `0.0` se non c'è una palla rilevata **o se è stata recuperata dal colore**:
  il fallback non produce una probabilità. Un possessore può quindi essere
  assegnato anche con `confidence: 0.0`. Consulta `ball_tracks.json` per la fonte.
  Se manca il giocatore ma la palla è
  presente, conserva la fiducia della palla e usa distanza `-1.0`.

`--possession-threshold` accetta un numero finito maggiore di zero. Per esempio,
per essere più restrittivo senza cambiare le soglie di rilevamento:

```bash
python main.py --input data/partita.mp4 --possession-threshold 100
```

Il percorso `--possession-output` deve terminare in `.json` ed essere diverso da
quelli degli altri file dell'analisi. Anche questo JSON viene salvato soltanto
al termine dell'analisi riuscita: interrompendo lo script, un eventuale file
precedente non viene aggiornato.

### Come viene rilevata la palla

Il tracking dei giocatori continua a usare `classes=[0]` e ByteTrack come prima.
Una seconda istanza **degli stessi pesi YOLO** esegue `predict` con `classes=[32]`
sul fotogramma originale, prima delle annotazioni. Questo mantiene indipendente
lo stato del tracker dei giocatori. Non vengono scaricati pesi aggiuntivi, ma due
passaggi di inferenza per fotogramma richiedono più tempo e RAM.

Per la palla il programma disegna un cerchio rosso centrato sul rettangolo rilevato
e scrive `PALLA`. In questo passo si tratta di **rilevamento**, senza un ID della
palla, interpolazione o previsione della posizione quando non è visibile.
La classe “sports ball” include anche palloni di altri sport: non distingue
specificamente una palla da basket. Palle piccole, mosse o nascoste possono non
essere rilevate dal modello COCO; sono possibili anche falsi positivi.

La soglia della palla è separata da quella dei giocatori e vale `0.25` per default.
Per cercare anche palle meno sicure, con più possibili falsi positivi:

```bash
python main.py --input data/partita.mp4 --ball-confidence 0.15
```

### Fallback arancione: attivazione e range HSV

Il fallback è **attivo per default** (`--use-color-fallback`). Se nel frame YOLO
non restituisce nessuna palla sopra `--ball-confidence`, il programma esegue
`detect_ball_by_color(frame)` sul fotogramma originale, prima di disegnare
etichette e rettangoli. Anche una palla YOLO sotto soglia viene quindi sostituita
dal tentativo a colori. Se YOLO trova almeno una palla, il colore non viene usato
e non aggiunge duplicati. Il tracking dei giocatori resta invariato.

OpenCV converte il frame da BGR a HSV, isola l'arancione con una maschera, rimuove
piccoli puntini e cerca contorni rotondi. Scarta forme allungate, rettangoli e
triangoli, aree inferiori a 20 pixel quadrati e superiori al 5% del fotogramma.
Richiede circolarità di almeno `0.75` e rapporto larghezza/altezza fra `0.75` e
`1.33`. Sceglie la forma più circolare (a parità, la più grande), oppure restituisce
`None`. Palle molto piccole, sfocate, parzialmente nascoste o allungate dal movimento
possono essere scartate da questi filtri.

Per ripristinare il comportamento solo YOLO:

```bash
python main.py --input data/partita.mp4 --no-use-color-fallback
```

Non scrivere `--use-color-fallback False`: usa l'opzione negativa sopra.
Per il range personalizzato, passa **sei interi separati da spazi**, prima i tre
minimi e poi i tre massimi. Il valore predefinito è:

```bash
python main.py --input data/partita.mp4 --color-hsv-range 5 100 100 15 255 255
```

L'ordine è `H_MIN S_MIN V_MIN H_MAX S_MAX V_MAX`. Nella scala OpenCV:

- **H** (tonalità): da 0 a 179; il range predefinito `5..15` cerca l'arancione.
- **S** (saturazione): da 0 a 255; un minimo più alto esclude colori sbiaditi.
- **V** (luminosità): da 0 a 255; un minimo più basso include oggetti più scuri.

Se la palla non entra nella maschera per illuminazione o colore, prova un range
più ampio, ad esempio `--color-hsv-range 3 80 70 22 255 255`. Per limitare falsi
positivi, restringi H o aumenta i minimi S/V. Ogni minimo deve essere minore o
uguale al relativo massimo; intervalli invertiti o fuori scala sono rifiutati.

Le palle recuperate vengono mostrate con lo stesso **cerchio rosso e “PALLA”**,
esportate nel JSON e usate anche per il possesso. Il terminale riporta separatamente
quanti rilevamenti provengono da YOLO e quanti dal colore. Non viene interpolata
una traiettoria nei frame in cui entrambi i metodi falliscono.

Parquet, divise e altri oggetti arancioni rotondi possono essere scambiati per la
palla. Confronta i video con e senza fallback, usando nomi di uscita diversi:
più rilevamenti non significa necessariamente più rilevamenti corretti. Anche il
possesso può risentire di questi falsi positivi.

### Formato di `ball_tracks.json`

Esempio illustrativo:

```json
[
  {"frame": 0, "bbox": [300, 180, 320, 200], "confidence": 0.82},
  {"frame": 2, "bbox": [310, 190, 330, 210], "confidence": 0.76}
]
```

Gli indici `frame` partono da **0**, come nel JSON dei giocatori, e `bbox` contiene
coordinate intere `[x1, y1, x2, y2]` in pixel sul video originale. `confidence`
è un numero decimale con la fiducia YOLO, non una certezza che sia la palla da basket.
I record YOLO mantengono il formato precedente. Solo quelli recuperati dal colore
aggiungono `"source": "color"` e usano `confidence: 0.0`, senza inventare una
fiducia YOLO o una probabilità di correttezza. Esempio di fallback:

```json
{"frame": 3, "bbox": [300, 180, 320, 200], "confidence": 0.0, "source": "color"}
```

Per distinguere un recupero a colori dall'assenza di palla, controlla la presenza
del record e il campo `source`, non soltanto `confidence`.
Un fotogramma senza palla non produce elementi; se non viene rilevata nessuna palla
nel video, il file contiene `[]`. Se ci sono più rilevamenti nello stesso frame,
vengono tutti disegnati ed esportati: non viene scelto automaticamente il pallone
di gioco. Il JSON viene scritto al termine dell'analisi e conserva i rilevamenti
in memoria fino a quel momento, come quello dei giocatori. I due JSON devono
avere percorsi diversi; il programma rifiuta di sovrapporre il JSON della palla
agli altri file usati dall'analisi.

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
python main.py --input data/test_persone.mp4 --output output/test_rilevato.mp4 --tracks output/test_tracks.json --ball-tracks output/test_ball_tracks.json
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
La foto dell'autobus non contiene una palla: in modalità solo YOLO è adatta a
verificare i giocatori e il JSON vuoto della palla. Il fallback può invece
selezionare altri oggetti arancioni; per provare un rilevamento positivo serve una clip
con un pallone visibile.

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

Per la palla è stata verificata anche una clip di sei fotogrammi ottenuta dalla
foto COCO `000000000368.jpg`, presente nel campione
[COCO128 distribuito da Ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip).
La foto contiene un pallone da calcio: la classe COCO è la stessa “sports ball”.
YOLO ha rilevato il pallone in tutti e sei i fotogrammi; sono stati controllati
schema del JSON, cerchi ed etichette rosse e leggibilità del video risultante.
Questo prova il rilevamento della classe 32, senza misurare l'accuratezza su
palle da basket durante una partita.

Per il possesso sono passati otto test della logica (vicinanza, soglia esatta,
assenze, più palle e soglie non valide). Nella clip di sei fotogrammi sono stati
verificati distanze, JSON e possessore giallo; i JSON di giocatori e palla sono
identici a quelli della versione precedente. Una prova di otto fotogrammi con
due frame neri intermedi ha verificato l'azzeramento del possessore durante
l'assenza e la corrispondenza fra ID nel JSON, rettangolo giallo ed etichetta.

Per il fallback sono passati anche i sei test del colore (14 test totali con
quelli del possesso). Su una clip sintetica di sei fotogrammi con un disco
arancione in movimento, il confronto con lo stesso YOLO ha prodotto **0 rilevamenti
senza fallback e 6 con fallback**. Sono stati verificati centri dei rettangoli,
cerchi rossi nel video decodificato e record `source: "color"`. Nel video in cui
YOLO rileva già il pallone, il fallback non viene chiamato e i JSON di giocatori,
palla e possesso restano identici alla versione precedente. Queste prove non
stabiliscono quanto migliori il rilevamento nella propria partita.

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
- **JSON della palla vuoto:** YOLO non ha trovato “sports ball” sopra la soglia
  e il fallback non ha trovato una forma valida. Prova una clip più nitida,
  regola il range HSV oppure abbassa `--ball-confidence`.
- **Possesso spesso sconosciuto:** controlla prima i rilevamenti della palla e dei
  giocatori. Aumentare `--possession-threshold` può assegnare più frame, ma anche
  introdurre associazioni sbagliate.

Per verificare la logica del possesso senza scaricare modelli o video:

```bash
python -m unittest discover -s scripts -p test_possession.py
```

Per eseguire anche i test del colore (cerchio arancione, rumore, forme da escludere,
selezione del candidato e range HSV personalizzati/non validi):

```bash
python -m unittest discover -s scripts -p "test_*.py"
```

Questo è un prototipo didattico. Prima di distribuire un prodotto, consulta i
[termini di licenza Ultralytics](https://www.ultralytics.com/license) per codice e modelli.
