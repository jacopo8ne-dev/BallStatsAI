# BallStatsAI — modello personalizzato per partite di basket

Il progetto usa il tuo modello **`models/best.pt`** per riconoscere giocatori,
palla e canestro. ByteTrack assegna un ID ai soli **Player**; la palla viene
rilevata come **Ball** e il canestro come **Hoop**. Gli indici vengono ricavati
ogni volta da `model.names`: non serve conoscere l'ordine delle classi.

Le classi **Ref, Clock, Overlay e Scoreboard** vengono ignorate nel tracking e
nell'esportazione. Le classi Player, Ball e Hoop devono essere presenti con questi
nomi; se mancano, il programma mostra un messaggio che elenca quelle mancanti.
`model.names` può essere sia un dizionario sia una lista.

Il video risultante mostra rettangoli verdi e ID sui giocatori, un cerchio rosso
con `PALLA` sul pallone e un cerchio verde sul canestro automatico. Il possessore
stimato diventa giallo e i tiri vengono indicati con `TIRO!` ed esito stimato.
Funziona sulla CPU, senza una scheda grafica dedicata.

## Struttura

```text
BallStatsAI/
├── main.py                       # Comando per avviare l'analisi
├── requirements.txt              # Dipendenze Python
├── src/
│   ├── __init__.py
│   ├── detection.py              # Classi dinamiche, tracking, video e JSON
│   ├── ball_color.py             # Fallback arancione opzionale, spento per default
│   ├── possession.py             # Possesso basato sulla distanza
│   └── shots.py                  # Canestro automatico e tiri stimati
├── scripts/                      # Video dimostrativo e test
├── data/                         # Video originali
├── models/
│   └── best.pt                   # Pesi personalizzati, distribuiti con Git LFS
└── output/                       # Video annotato e quattro JSON
```

Video, nuovi modelli e ambiente virtuale sono esclusi da Git tramite `.gitignore`.
Il file `models/best.pt` già presente nel repository è gestito con **Git LFS**.
Conserva una copia dei video originali e del modello.

## 1. Installazione

Usa **Python 3.12**, la versione verificata per il codice del progetto. Su Windows
seleziona “Add Python to PATH” durante l'installazione. Apri il terminale nella
cartella che contiene `main.py`, poi crea l'ambiente virtuale:

```bash
python -m venv .venv
```

Attivalo su Linux/macOS:

```bash
source .venv/bin/activate
```

Oppure nel Prompt dei comandi Windows (`cmd`):

```bat
.venv\Scripts\activate.bat
```

Installa le librerie:

```bash
python -m pip install -r requirements.txt
```

Nel cloud Linux sono state usate le versioni CPU di PyTorch, installate prima di
`requirements.txt` per evitare il download dei pacchetti CUDA:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Per scaricare il modello dal repository, installa [Git LFS](https://git-lfs.com/)
e, dalla cartella del progetto, esegui:

```bash
git lfs install
git lfs pull
```

Il file completo occupa circa **173 MB**: un file di poche righe è solo il
puntatore LFS, non i pesi. In alternativa copia il tuo file completo in
**`models/best.pt`**. Lo script Python non scarica i pesi da solo.
Il modello previsto riconosce
Ball, Clock, Hoop, Overlay, Player, Ref e Scoreboard. L'ordine numerico può cambiare.
Il caricamento dei pesi richiede che il file sia compatibile con la versione
installata di Ultralytics; un file assente non viene sostituito con altri pesi.

## 2. Analizza una partita

Il comando minimo è:

```bash
python main.py --input video.mp4
```

Se hai copiato il video in `data/partita.mp4`, usa:

```bash
python main.py --input data/partita.mp4 --output output/partita_analizzata.mp4
```

Non servono coordinate manuali del canestro. Se il modello si trova altrove,
indica il percorso con `--model percorso/best.pt`. Usa virgolette per i percorsi
con spazi. L'analisi usa due istanze degli stessi pesi: una per ByteTrack sui
Player, una per Ball e per la ricerca iniziale di Hoop. Le rilevazioni della
palla e del canestro non modificano lo stato del tracker dei giocatori.

I risultati predefiniti sono:

- `output/partita_rilevata.mp4`: video annotato, salvo diversa scelta con `--output`.
- `output/tracks.json`: posizioni dei giocatori con ID.
- `output/ball_tracks.json`: posizioni della palla.
- `output/possession.json`: possesso stimato, un record per fotogramma.
- `output/shots.json`: tiri con esito classificabile.

Aspetta il messaggio “Completato”, poi apri il video con un lettore come VLC.
Non viene aperta una finestra durante l'elaborazione. Il programma mostra il
progresso ogni 100 fotogrammi; su CPU, una partita lunga può richiedere tempo.
Il video mantiene risoluzione e FPS dell'ingresso a frequenza costante, **senza
l'audio**. La temporizzazione può differire per video a frequenza variabile.

I file di uscita già presenti vengono sovrascritti: scegli nomi diversi per
conservare analisi precedenti. I JSON vengono scritti alla fine dell'analisi
riuscita. Interrompendo lo script il video può essere parziale e i vecchi JSON
possono restare quelli dell'analisi precedente. Le tracce restano in memoria
fino al salvataggio, quindi video lunghi richiedono più RAM.

## Canestro automatico

Il programma cerca la classe **Hoop** nei primi **300 fotogrammi**, circa 10 secondi
a 30 FPS, e **si ferma appena trova un canestro valido**. La ricerca avviene durante
l'elaborazione, un frame alla volta, senza accumulare immagini in memoria né saltare
fotogrammi. Può quindi trovare un canestro che compare solo al frame 50 o 100.

La soglia predefinita è **`--hoop-confidence 0.3`**, indipendente dalle soglie di
giocatori e palla. Puoi modificarla con un valore maggiore di 0 e minore o uguale a 1:

```bash
python main.py --input data/partita.mp4 --hoop-confidence 0.3
```

Abbassarla può recuperare canestri poco visibili, ma aumenta i falsi positivi.
Quando trova il canestro, il terminale mostra ad esempio **`Canestro trovato al frame 54!`**.
Gli indici partono da **0**, come nei JSON: la ricerca copre i frame da 0 a 299.
Il cerchio verde e il rilevamento dei tiri si attivano dal frame della scoperta;
i frame precedenti mantengono l'analisi di giocatori, palla e possesso, senza tiri retroattivi.

La telecamera deve essere fissa. Dopo questa ricerca, il rettangolo viene
**conservato per l'intera partita**: non viene cancellato se Hoop manca nei frame
successivi e non oscilla con i nuovi rilevamenti. Se sono visibili più canestri,
viene scelto quello con la fiducia più alta **nel primo frame valido**.
Controlla che il cerchio verde corrisponda al canestro desiderato.

Se nessun canestro viene trovato entro 300 frame, il programma stampa un avviso,
disattiva i tiri e continua a elaborare giocatori, palla e possesso. In quel caso non
disegna il cerchio del canestro e salva `shots.json` come `[]`. Non ricerca Hoop
più avanti nel video. Se la clip è più corta, cerca in tutti i frame disponibili
e stampa l'avviso alla fine se il canestro manca.

Un frame senza Player, Ball o altre classi non interrompe l'analisi. I risultati
assenti vengono gestiti come liste vuote; un canestro già individuato rimane
comunque disponibile.

## Tracking, palla e fallback arancione

ByteTrack usa `persist=True` e il solo indice di **Player**. Gli ID sono locali
al video e non identificano nomi, maglie o squadre. Occlusioni, incroci e mancate
rilevazioni possono causare nuovi ID o scambi. Un giocatore senza ID confermato
non viene esportato. I rilevamenti Ref non vengono trattati come giocatori,
ma un'errata classificazione del modello come Player può comunque creare una traccia.

La soglia dei Player è `--confidence 0.1`; quella della palla è separata,
**`--ball-confidence 0.50`**. Di default la palla viene cercata **solo con il modello
personalizzato `best.pt`**, usando l'indice di **Ball** ricavato da `model.names`,
senza usare la classe COCO `sports ball`.

I rettangoli Ball con area **superiore al 5% del frame** vengono scartati prima
del disegno, del salvataggio JSON e del calcolo di possesso e tiri. L'area è
`(x2-x1) * (y2-y1)`, confrontata con larghezza × altezza del video.
Una soglia di confidenza più bassa può recuperare oggetti poco visibili, ma introduce
più falsi positivi. Il comando con le impostazioni predefinite è:

```bash
python main.py --input data/partita.mp4 --ball-confidence 0.50
```

Il fallback OpenCV arancione è **disattivato per default**, per evitare che scarpe,
divise e altri oggetti arancioni vengano scambiati per il pallone dal filtro colore.
Se nessuna Ball supera i filtri, il frame resta senza palla: il programma continua
senza errori e non riutilizza posizioni precedenti. Eventuali falsi positivi del
modello YOLO restano possibili e vanno verificati sul video.

Per riattivare manualmente il fallback:

```bash
python main.py --input data/partita.mp4 --use-color-fallback
```

Quando è attivo, interviene solo se nessuna Ball supera i filtri YOLO e di dimensione.
Cerca sul frame originale una forma circolare nel range HSV
`[5,100,100]`–`[15,255,255]`. Scarta puntini, forme troppo squadrate o allungate,
aree inferiori a 20 pixel quadrati o superiori al 5% del frame. Richiede circolarità
almeno `0.75`, rapporto larghezza/altezza fra `0.75` e `1.33` e almeno sei vertici
nel contorno approssimato. Sceglie il candidato più circolare, oppure nessuno.

Per mantenerlo esplicitamente disattivato:

```bash
python main.py --input data/partita.mp4 --no-use-color-fallback
```

Per personalizzare l'HSV passa sei interi, in ordine
`H_MIN S_MIN V_MIN H_MAX S_MAX V_MAX`:

```bash
python main.py --input data/partita.mp4 --use-color-fallback --color-hsv-range 3 80 70 22 255 255
```

In OpenCV H va da 0 a 179, S e V da 0 a 255. Allarga H per includere più tonalità,
abbassa S per colori meno saturi o V per oggetti più scuri. Restringi gli intervalli
per ridurre i falsi positivi. Ogni minimo deve essere minore o uguale al massimo.
Usa `--no-use-color-fallback`, non `--use-color-fallback False`, per disattivarlo.

Con il fallback attivo, parquet, divise o altri oggetti arancioni possono essere
scambiati per la palla. YOLO ha sempre la priorità: il colore non aggiunge duplicati
nei frame con Ball che superano tutti i filtri.
Il terminale conta separatamente i rilevamenti YOLO e i recuperi dal colore.

## Possesso stimato

Il centro della palla viene confrontato con il centro del corpo dei Player con ID
nello stesso frame. Il più vicino ha il possesso se la distanza è **inferiore** a
`--possession-threshold 150` pixel. A parità viene mantenuto il primo giocatore.
Se ci sono più palle, viene scelta quella con la fiducia maggiore; a parità la
prima. Il possessore diventa giallo con `PALLA` accanto all'ID.

```bash
python main.py --input data/partita.mp4 --possession-threshold 100
```

Senza palla o giocatori il possessore è sconosciuto: non viene mantenuto quello
del frame precedente. La distanza in pixel dipende dalla prospettiva e dalla
risoluzione. Passaggi, tiri e rimbalzi possono associare la palla a un giocatore
che non la controlla davvero. Non vengono ancora calcolate durate o statistiche
di squadra.

## Tiri ed esiti stimati

Quando un canestro è stato riconosciuto, il rilevatore cerca una salita della
palla fra due osservazioni consecutive, di almeno `--min-shot-speed 5`
pixel/frame. Richiede un possessore recente (entro un quarto di secondo), partenza
sotto il canestro e una componente del movimento diretta verso il suo centro.
Conserva quel giocatore come tiratore e mostra `TIRO! ... IN CORSO`.

Stima **FATTO** se il centro della palla attraversa dall'alto verso il basso entrambi
i bordi orizzontali del rettangolo Hoop, con x all'interno. Fra due osservazioni
consecutive interpola linearmente l'attraversamento. Stima **SBAGLIATO** quando
la discesa passa sotto il canestro senza un attraversamento valido, oppure quando
un tiro corto torna a scendere senza raggiungere il ferro. L'esito rimane visibile
per circa due secondi; dopo un esito attende mezzo secondo prima di un nuovo candidato.

```bash
python main.py --input data/partita.mp4 --min-shot-speed 5 --shot-output output/shots.json
```

La velocità è in pixel/frame: risoluzione e FPS cambiano la sensibilità. Sono
accettati solo valori finiti maggiori di zero. L'analisi mantiene un solo tiro
in corso alla volta. Se la palla scompare per oltre un secondo, il candidato dura
oltre tre secondi o il video termina durante il tiro, l'esito resta non osservabile.
Anche un'occlusione nel passaggio al ferro impedisce la conferma. Questi candidati
sono contati nel terminale ed esclusi dal JSON booleano, senza inventare un errore
al tiro. Non vengono calcolate velocità attraverso frame mancanti.

Un passaggio in un rettangolo 2D non dimostra un canestro reale nello spazio.
La qualità dipende anche da cosa il tuo modello ha imparato a delimitare come Hoop:
un riquadro ampio attorno al tabellone rende meno precisa la regola di attraversamento.
Falsi rilevamenti, possessi errati e passaggi alti possono generare tiri o esiti
sbagliati. Verifica gli eventi sul video prima di usarli come statistiche.

## Formato dei JSON

Tutti i frame partono da **0** e tutte le coordinate sono in pixel sul video
originale. `bbox` è `[x1,y1,x2,y2]`, dall'angolo superiore sinistro a quello inferiore
destro. L'origine è in alto a sinistra. I percorsi dei JSON devono essere distinti
fra loro e dagli altri file dell'analisi.

`tracks.json`, un record per Player con ID in ciascun frame:

```json
[{"frame": 0, "id": 7, "bbox": [100,120,180,300]}]
```

`ball_tracks.json`, un record per Ball rilevata; nessun record nei frame senza palla:

```json
[
  {"frame": 0, "bbox": [300,180,320,200], "confidence": 0.82},
  {"frame": 1, "bbox": [310,190,330,210], "confidence": 0.0, "source": "color"}
]
```

La fiducia è quella di YOLO. Solo i recuperi dal colore aggiungono `source: "color"`
e usano `confidence: 0.0`: il metodo a colori non produce una probabilità.
Tutte le Ball che superano la soglia di confidenza e il filtro dimensionale vengono
esportate, anche se sono più di una. I record `source: "color"` compaiono solo
quando il fallback è stato attivato manualmente.

`possession.json`, esattamente un record per frame:

```json
[
  {"frame": 0, "player_id": 7, "ball_distance": 64.5, "confidence": 0.82},
  {"frame": 1, "player_id": null, "ball_distance": -1.0, "confidence": 0.0}
]
```

`player_id` è `null` se nessuno è abbastanza vicino o mancano palla o giocatori.
`ball_distance` conserva la distanza dal più vicino anche oltre la soglia; vale
`-1.0` se non misurabile. Escludi quel valore dai calcoli sulle distanze.
`confidence` è la fiducia della palla scelta, non una probabilità di possesso:
un possessore basato sul colore può avere `confidence: 0.0`.

`shots.json`, un record per tiro con esito classificabile:

```json
[{"frame": 120, "player_id": 7, "made": true, "shot_x": 420.0, "shot_y": 350.0, "distance": 265.2}]
```

`frame` è il rilascio stimato, cioè il primo dei due frame di salita. `shot_x` e
`shot_y` sono il centro della palla al rilascio; `distance` è la distanza da quel
punto al centro di Hoop. Sono pixel, non metri, e non classificano tiri da due o
tre punti. `made` è un esito stimato. Un canestro non riconosciuto o nessun tiro
classificabile producono `[]`.

Puoi cambiare i percorsi con `--tracks`, `--ball-tracks`, `--possession-output` e
`--shot-output`. Per tutte le opzioni:

```bash
python main.py --help
```

## Test e problemi comuni

Per eseguire i test della logica e dell'integrazione con rilevamenti controllati:

```bash
python -m unittest discover -s scripts -p "test_*.py" -v
```

I test coprono indici dinamici (dizionario/lista e ordine diverso), canestro che
compare ai frame 50, 100 e 299, arresto della ricerca entro 300 frame, soglia
configurabile, conservazione del riferimento, esclusione delle classi estranee,
video senza rilevamenti e continuità dei fotogrammi. Verificano JSON e video, oltre
a colore, possesso e tiri dopo la scoperta tardiva. Verificano inoltre che il
fallback resti spento per default e attivabile manualmente, e che il filtro
dimensionale scarti le Ball oltre il 5% del frame. I test di integrazione simulano
le risposte di YOLO: non misurano l'accuratezza dei tuoi pesi.

Il modello `best.pt` del repository è stato caricato nel cloud: contiene le sette
classi previste. Una prova completa su una clip dimostrativa di sei frame ha
generato il video leggibile e tutti e quattro i JSON, gestendo l'assenza di
rilevamenti senza errori. Questa clip non è una partita di basket: la qualità
dei rilevamenti e degli esiti va valutata su una clip della tua partita.

Lo script `scripts/create_test_video.py` resta disponibile per generare una breve
clip da una foto pubblica. La foto non è una partita: con il modello personalizzato
non sono garantiti rilevamenti Player, Ball o Hoop. Per valutare la qualità usa
una breve clip della tua partita con canestro visibile all'inizio.

- **Modello non trovato:** controlla `models/best.pt` o indica il percorso con `--model`.
- **Classi mancanti:** usa i pesi addestrati che contengono Player, Ball e Hoop con questi nomi.
- **Canestro non trovato:** controlla i primi 300 frame e le prestazioni del modello sulla classe Hoop; prova a regolare `--hoop-confidence`.
- **Frame senza oggetti:** è normale; il programma continua. Il fallback colore interviene solo se attivato manualmente.
- **JSON della palla vuoto:** nessuna Ball ha superato confidenza e filtro dimensionale. Verifica il video e il modello, oppure regola `--ball-confidence`.
- **Video non leggibile:** controlla il percorso o converti in MP4 con codec H.264 e FPS costanti.
- **Libreria non trovata:** attiva `.venv` e reinstalla `requirements.txt`.

Prima di distribuire un prodotto, consulta i [termini Ultralytics](https://www.ultralytics.com/license)
e quelli dei dati usati per addestrare il tuo modello.
