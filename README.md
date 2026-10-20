# BallStatsAI — modello personalizzato per partite di basket

Il progetto usa il tuo modello **`models/best.pt`** per riconoscere giocatori,
palla e canestro. Un tracker ByteTrack segue **Player** e un tracker dedicato segue **Ball**;
un riferimento aggiornato periodicamente segue **Hoop**. Gli indici vengono ricavati
ogni volta da `model.names`: non serve conoscere l'ordine delle classi.

Le classi **Ref, Clock, Overlay e Scoreboard** vengono ignorate nel tracking e
nell'esportazione. Le classi Player, Ball e Hoop devono essere presenti con questi
nomi; se mancano, il programma mostra un messaggio che elenca quelle mancanti.
`model.names` può essere sia un dizionario sia una lista.

Il video risultante mostra rettangoli verdi e ID sui giocatori, un cerchio rosso
con `PALLA` sul pallone e un cerchio verde sul canestro. Le posizioni ricostruite
sono etichettate `PALLA INTERPOLATA`. Il possessore
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
│   ├── ball_tracking.py          # Tracker della palla, punti statici e interpolazione
│   ├── ball_color.py             # Fallback arancione opzionale, spento per default
│   ├── possession.py             # Possesso basato sulla distanza
│   └── shots.py                  # Canestro periodico/manuale e tiri stimati
├── scripts/                      # Video dimostrativo e test
├── data/                         # Video originali
├── models/
│   └── best.pt                   # Pesi personalizzati, distribuiti con Git LFS
└── output/                       # Video annotato, quattro JSON e debug/ con immagini
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

Le coordinate manuali del canestro sono opzionali. Se il modello si trova altrove,
indica il percorso con `--model percorso/best.pt`. Usa virgolette nei percorsi con
spazi. Il programma carica **una sola istanza di YOLO** ed esegue **una sola inferenza
per frame elaborato**, per Player, Ball e Hoop insieme. Divide poi i risultati in
Python: ByteTrack giocatori e tracker della palla hanno stati indipendenti.

Per ridurre le inferenze, elabora un frame ogni due:

```bash
python main.py --input data/partita.mp4 --skip-frames 2
```

Il default è `--skip-frames 1` (tutti i frame). Con `2` vengono analizzati e salvati
i frame originali 0, 2, 4, … e gli FPS del video di uscita vengono dimezzati.
La durata resta quella della partita, salvo l'arrotondamento dell'ultimo frame.
I JSON conservano gli **indici originali**, non li rinumerano. Saltare frame riduce
la precisione sui movimenti rapidi e può far perdere la palla o un tiro.

I risultati predefiniti sono:

- `output/partita_rilevata.mp4`: video annotato, salvo diversa scelta con `--output`.
- `output/tracks.json`: posizioni dei giocatori con ID.
- `output/ball_tracks.json`: posizioni della palla.
- `output/possession.json`: possesso stimato, un record per fotogramma.
- `output/shots.json`: tiri con esito classificabile.
- `output/debug/<nome_video>_<identificativo>/`: immagini diagnostiche del canestro.
  Ogni analisi crea una sottocartella distinta, indicata nel terminale; puoi cambiarne
  la cartella principale con `--debug-dir percorso`.

Aspetta il messaggio “Completato”, poi apri il video con un lettore come VLC.
Non viene aperta una finestra durante l'elaborazione. Il programma mostra il
progresso ogni 100 fotogrammi; su CPU, una partita lunga può richiedere tempo.
Il video mantiene la risoluzione e usa gli FPS originali divisi per `--skip-frames`,
**senza l'audio**. La temporizzazione può differire per video a frequenza variabile.

I file di uscita già presenti vengono sovrascritti: scegli nomi diversi per
conservare analisi precedenti. I JSON vengono scritti alla fine dell'analisi
riuscita. Interrompendo lo script il video può essere parziale e i vecchi JSON
possono restare quelli dell'analisi precedente. Le tracce restano in memoria
fino al salvataggio, quindi video lunghi richiedono più RAM.

## Canestro automatico, debug e alternativa manuale

Il programma aggiorna la classe **Hoop** al frame 0 e poi **ogni 30 frame originali**,
per tutta la partita. Usa i risultati della stessa inferenza di giocatori e palla:
non esegue un secondo passaggio YOLO. Se hai saltato il frame previsto, usa il primo
frame elaborato successivo (per esempio 35 con `--skip-frames 7`).

La soglia è **`--hoop-confidence 0.3`**. Fra i candidati validi del frame di ricerca
sceglie quello con confidenza maggiore. Il terminale stampa `Canestro trovato al
frame 60!` alla prima scoperta e `Canestro aggiornato al frame 90!` successivamente.
Se non trova Hoop, **mantiene l'ultimo rettangolo**; se non ne ha mai trovato uno,
continua a cercare e lascia i tiri disattivati. Non esiste più il limite di 300 frame.

Il cerchio verde segue gli aggiornamenti. Fra due ricerche la posizione resta
quella precedente: movimenti rapidi della telecamera possono renderla temporaneamente
errata. In presenza di più canestri viene scelto quello più sicuro; non viene
identificato automaticamente il lato di attacco.

Il debug salva **una JPG per ognuno dei primi 150 frame originali** (o tutti se il
video è più corto), anche quando non c'è nessun canestro:

- Verde: candidato Hoop attuale sopra soglia; arancione: candidato sotto soglia.
- Ogni candidato riporta la confidenza restituita dal modello.
- Azzurro: riferimento utilizzato, con il frame dell'ultima scoperta e la sua confidenza.
- I frame saltati indicano esplicitamente che non è stata eseguita l'inferenza.

Per il debug, la soglia dell'inferenza unica scende a `0.01` nei primi 150 frame:
questo permette di vedere candidati scartati dalla soglia del canestro. Restano
attivi i filtri separati di ciascuna classe. Il debug non mostra rilevamenti sotto
questa soglia minima, né quelli soppressi internamente da YOLO. Un candidato verde
non sostituisce il riferimento fuori dalla cadenza di aggiornamento di 30 frame.

Se il modello non riconosce il canestro, puoi impostarlo manualmente:

```bash
python main.py --input data/partita.mp4 --hoop-bbox "900,180,960,210"
```

Sono coordinate di esempio: sostituiscile con `x1,y1,x2,y2` del ferro nel tuo video
originale, dentro la sua larghezza e altezza. L'origine è in alto a sinistra.
Con questa opzione **Hoop viene esclusa dall'inferenza e la ricerca automatica viene
saltata**. Il rettangolo resta fisso per tutta la partita: usa una telecamera fissa.
Le immagini debug mostrano il riferimento come `manuale`, senza inventare una confidenza.

## Tracker della palla e interpolazione

I giocatori mantengono il loro ByteTrack standard e la soglia `--confidence 0.1`.
Gli ID sono locali al video; occlusioni e incroci possono modificarli. Arbitri e
altre classi non vengono inviati al tracker dei giocatori.

La palla usa **un tracker dedicato senza ByteTrack**, in `src/ball_tracking.py`.
ByteTrack associa i rettangoli per sovrapposizione: una palla di 10–15 pixel che si
muove veloce non si sovrappone quasi mai a quella del frame precedente, quindi le
osservazioni deboli venivano scartate. Il nuovo tracker usa invece la **distanza
dalla posizione prevista**.

**Il default di `--ball-confidence` resta 0.05**: è la soglia minima dei candidati,
non quella per iniziare una traccia. Il nome della classe è sempre **Ball**, letto
dinamicamente da `best.pt`, non `sports ball` di COCO.

Il tracker segue **una sola palla principale**:

- Scarta box invalidi e box con area superiore al **5% del frame**.
- **Dimensione:** scarta candidati più grandi di **0.3 volte l'altezza tipica (mediana)
  dei giocatori** del frame: teste e orecchie del pubblico in primo piano.
- **Inizio:** serve un candidato di almeno **0.15**, oppure due candidati deboli
  vicini in due frame elaborati consecutivi. In entrambi i casi il candidato deve
  trovarsi **vicino a un giocatore** (dentro il suo box allargato della sua altezza);
  i box alti più del doppio della mediana, cioè il pubblico, non contano.
- **Previsione:** stima la velocità dalle ultime osservazioni e prevede dove sarà la
  palla. Accetta il candidato più vicino alla previsione, entro una finestra di
  `max(30, 4 × lato della palla)` pixel più **10 pixel per ogni frame trascorso**.
  Anche un candidato a 0.05 viene accettato se è nella finestra, anche lontano dai
  giocatori (passaggi e tiri).
- **Cambio di traccia:** un candidato fuori dalla finestra con almeno 0.15, vicino a un
  giocatore, sostituisce la traccia se è più sicuro di 0.2 rispetto al migliore vicino
  (o se non ci sono vicini). Il tracker non resta mai bloccato su una posizione vecchia.
- **Perdita:** dopo **1 secondo** (calcolato sugli FPS reali del video) senza palla la
  traccia termina; la successiva riparte da qualsiasi punto del campo con un nuovo `id`.
- **Punti statici:** se YOLO vede una "palla" nello stesso punto per almeno 10 frame
  distribuiti su almeno 1 secondo (un logo, un murale, una palla ferma a bordo campo),
  i candidati in quel punto vengono ignorati finché non smette di comparire per 10
  secondi. Il prezzo: una palla tenuta ferma a lungo, per esempio prima di un tiro
  libero, si perde finché non si muove.

L'`id` della palla identifica la continuità della traccia principale. Per compatibilità
il JSON contiene ancora `tracker_id`, ora uguale a `id` per le osservazioni YOLO.

Per buchi di **1–5 frame originali**, due osservazioni YOLO accettate della stessa
traccia permettono un'interpolazione lineare dei quattro angoli del box. La
scrittura del video attende fino a sei frame: così il cerchio, il JSON e il possesso
usano tutti la stessa posizione ricostruita. L'etichetta diventa `PALLA INTERPOLATA`
e il JSON usa `source: "interpolated"`. Con `--skip-frames`, vengono ricostruiti
solo i frame che verranno effettivamente salvati.

Non vengono inventate posizioni all'inizio, alla fine o nei buchi superiori a
cinque frame. Le posizioni previste senza osservazione non vengono esportate
come rilevamenti. Se la palla non viene trovata, l'analisi continua senza errori.

### Risoluzione di inferenza

Con `--imgsz` scegli il lato dell'immagine data a YOLO (default `640`, multiplo di 32).
Su riprese larghe la palla è di pochi pixel e a 640 YOLO spesso non la vede:

```bash
python main.py --input data/partita.mp4 --imgsz 1280
```

A 1280 l'inferenza sulla CPU è circa tre volte e mezza più lenta.

## Fallback arancione opzionale

Resta **spento per default**. Per attivarlo esplicitamente:

```bash
python main.py --input data/partita.mp4 --use-color-fallback
```

Interviene solo quando il tracker non restituisce una palla. Il colore non aggiorna
il tracker, non cambia l'ultima posizione YOLO e non è usato come estremo per
l'interpolazione. Un'interpolazione successiva ha priorità sul colore. Scarpa,
divisa o parquet arancioni possono ancora ingannarlo: per usare solo il modello,
non attivare questa opzione oppure usa `--no-use-color-fallback`.

Il range predefinito è `[5,100,100]`–`[15,255,255]`. Per personalizzarlo:

```bash
python main.py --input data/partita.mp4 --use-color-fallback --color-hsv-range 3 80 70 22 255 255
```

I sei valori sono `H_MIN S_MIN V_MIN H_MAX S_MAX V_MAX`: H va da 0 a 179, S e V da
0 a 255. Ogni minimo deve essere minore o uguale al massimo. Il metodo richiede
forme quasi circolari e scarta aree troppo piccole o superiori al 5% del frame.

## Possesso stimato

Il centro della palla viene confrontato con il centro del corpo dei Player con ID
nello stesso frame. Il più vicino ha il possesso se la distanza è **inferiore** a
`--possession-threshold 150` pixel. A parità viene mantenuto il primo giocatore.
La pipeline passa al possesso una sola palla principale, osservata oppure interpolata. Il possessore diventa giallo con `PALLA` accanto all'ID.

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
palla fra due osservazioni consecutive elaborate, di almeno `--min-shot-speed 5`
pixel/frame originale (anche con `--skip-frames`). Richiede un possessore recente (entro un quarto di secondo), partenza
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
al tiro. Le posizioni `interpolated` non vengono usate per confermare tiri o canestri: un
attraversamento stimato non dimostra un canestro. Con frame saltati le velocità sono
normalizzate per l'intervallo originale. Se il rettangolo del canestro cambia durante
un tiro, l'evento viene chiuso come non classificabile per evitare esiti basati su
due riferimenti diversi.

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

`tracks.json`, un record per Player con ID in ciascun frame elaborato:

```json
[{"frame": 0, "id": 7, "bbox": [100,120,180,300]}]
```

`ball_tracks.json`, al massimo una posizione della palla principale per frame elaborato:

```json
[
  {"frame": 0, "id": 1, "tracker_id": 1, "bbox": [300,180,320,200], "confidence": 0.82, "source": "yolo"},
  {"frame": 1, "id": 1, "tracker_id": null, "bbox": [310,190,330,210], "confidence": 0.75, "source": "interpolated"},
  {"frame": 2, "id": 1, "tracker_id": 1, "bbox": [320,200,340,220], "confidence": 0.75, "source": "yolo"}
]
```

Per `yolo`, la confidenza è quella dell'osservazione. Per `interpolated` è il minimo
fra le confidenze dei due estremi: **non è una nuova probabilità misurata**. Per
`color`, possibile solo se attivato, vale `0.0` e sia `id` sia `tracker_id` sono `null`.
I box possono contenere coordinate decimali, soprattutto dopo l'interpolazione.
Nessun record viene aggiunto per un frame senza osservazione né interpolazione valida.

`possession.json`, esattamente un record per frame elaborato (con indici originali):

```json
[
  {"frame": 0, "player_id": 7, "ball_distance": 64.5, "confidence": 0.82},
  {"frame": 1, "player_id": null, "ball_distance": -1.0, "confidence": 0.0}
]
```

`player_id` è `null` se nessuno è abbastanza vicino o mancano palla o giocatori.
`ball_distance` conserva la distanza dal più vicino anche oltre la soglia; vale
`-1.0` se non misurabile. Escludi quel valore dai calcoli sulle distanze.
`confidence` è la fiducia della palla scelta o ricostruita, non una probabilità di possesso:
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

I test usano ByteTrack reale (giocatori) con rilevamenti YOLO controllati. Coprono inferenza
unica, esclusione delle classi estranee, soglie basse, filtro dimensionale, previsione
della palla, finestra di ricerca, punti statici, ripartenza dopo una perdita, interpolazione e assenza del punto
futuro, video/JSON coerenti, canestro oltre il frame 300, aggiornamenti periodici,
modalità manuale, candidati sotto soglia nel debug e frame saltati. Il modello
reale `best.pt` è verificato separatamente su una clip dimostrativa; per misurare
l'accuratezza serve un video della tua partita.

Lo script `scripts/create_test_video.py` resta disponibile per generare una breve
clip da una foto pubblica. La foto non è una partita: con il modello personalizzato
non sono garantiti rilevamenti Player, Ball o Hoop. Per valutare la qualità usa
una breve clip della tua partita con canestro visibile all'inizio.

- **Modello non trovato:** controlla `models/best.pt` o indica il percorso con `--model`.
- **Classi mancanti:** usa i pesi addestrati che contengono Player, Ball e Hoop con questi nomi.
- **Canestro non trovato:** consulta `output/debug/`, regola `--hoop-confidence` oppure usa `--hoop-bbox`. La ricerca automatica continua ogni 30 frame.
- **Frame senza oggetti:** è normale; il programma continua. Il fallback colore interviene solo se attivato manualmente.
- **JSON della palla vuoto:** nessuna Ball ha superato i filtri di confidenza, dimensione, associazione e movimento. Verifica il modello, prova senza saltare frame e con `--imgsz 1280`.
- **Video non leggibile:** controlla il percorso o converti in MP4 con codec H.264 e FPS costanti.
- **Libreria non trovata:** attiva `.venv` e reinstalla `requirements.txt`.

Prima di distribuire un prodotto, consulta i [termini Ultralytics](https://www.ultralytics.com/license)
e quelli dei dati usati per addestrare il tuo modello.
