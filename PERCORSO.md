# PERCORSO — Come è nato il PPE Tracker (e le difficoltà lungo la strada)

Documento retrospettivo: ripercorre l'evoluzione del progetto **Visual Security — PPE Tracker**,
le scelte fatte, i vicoli ciechi e le difficoltà tuttora aperte. Ricostruito dalla cronologia git
(28 apr → 3 lug 2026) e dalle decisioni di progetto.

> Per l'uso quotidiano vedi [README.md](README.md); per l'architettura tecnica vedi [INFO.md](INFO.md).
> Questo file è il "diario di bordo".

---

## 0 · L'obiettivo

Rilevare in tempo reale se gli operatori in un cantiere indossano i DPI richiesti
(**casco, gilet alta visibilità, guanti, scarpe antinfortunistiche**) da un flusso video,
e alzare un alert sulle violazioni **persistenti** (non sui singoli frame rumorosi).

Il vincolo che ha condizionato *tutto*: **deve girare sul mio laptop** — CPU-only,
16 GB di RAM, GPU integrata Intel Iris Xe (niente CUDA). Nessun modello pesante ospitabile in locale.

---

## 1 · La linea temporale (dai commit)

| Data | Fase | Commit chiave |
|------|------|---------------|
| **28 apr** | Avvio: scaffolding, `analyzer` + `evaluator` | `cc06ce9`, `845f754` |
| **4–5 mag** | **VLM in cloud**: YOLO ONNX + Azure AI Vision + GPT-4o su Foundry | `73ee764`, `d8e43ac` |
| **5 mag** | **Dietrofront**: rimosso Azure AI Vision (latenza) | `33a056d` (`:fire:`) |
| **5–6 mag** | Verso il locale + primo tracker; giornata di crash | `16d03f7`, `22ac788`, `f02b787`, `955453c`, `75fb85e`, `47b3288` |
| **1 giu** | Pausa / refactor "da riprendere" | `2eff1d5` |
| **1 lug** | **Riscrittura da zero**, primo test verde | `5e88374`, `de800d5`, `63ad04f` |
| **3 lug** | Affidabilità: fix unicode, switch dataset, VLM reliability, identità persone + memoria PPE | `8425c32`, `0c76158`, `c758c78`, `78def24`, `30980e5` |

---

## 2 · Le difficoltà, una per una

### 2.1 — Il VLM in cloud (Azure) e la latenza

**Cosa è successo.** La prima idea era usare un VLM potente in cloud. `analyzer.py` conteneva
due backend Azure: `AzureAIVisionAnalyzer` (Azure AI Vision 4.0) e `FoundryGPT4oAnalyzer`
(GPT-4o deployato su Foundry, endpoint `foundry-multimodal.cognitiveservices.azure.com`).
Funzionava e la qualità era buona (`d8e43ac — tested YOLO + GPT4o`).

**Il problema.** Ogni chiamata era un round-trip di rete verso il cloud: **latenza non compatibile**
con un tracker che deve stare al passo col video. Su un sistema che processa molti frame, la rete
diventa il collo di bottiglia — e c'è anche il tema costi/privacy dei dati che escono dalla macchina.

**La decisione.** `33a056d — :fire: removed azure AI vision`: rimosse ~111 righe di codice Azure.
Da qui parte la ricerca di un modello **locale**. (Il remote Azure DevOps `origin_devops` resta come
traccia di quella fase.)

---

### 2.2 — Il pacchetto di inferenza ONNX per YOLO

**Cosa è successo.** L'inferenza YOLO non usa direttamente Ultralytics a runtime, ma un
**sottopacchetto ONNX dedicato** (`yolo/`, repo separato `github.com/AndreaDAmicis22/yolo`,
installato in editable e caricato via `sys.path`). Primo commit di test il 4 mag (`73ee764`).

**Il problema.** Era codice sviluppato tempo prima: solido, ma **ha richiesto il suo tempo** per
essere integrato e reso affidabile nella pipeline (backend ONNX Runtime, normalizzazione delle bbox
in formati diversi — xyxy / cxcywh / normalizzato / polygon, gestita poi da `_to_xyxy`).

**Stato.** Funziona e regge la pipeline. È uno dei pezzi *stabili* dell'architettura.

---

### 2.3 — Il dataset PPE

**Cosa è successo.** Serviva un dataset abbastanza ampio e ben etichettato per addestrare la YOLO
sulle 5 classi. Si è passati da un primo dataset Roboflow (`akfa-beqxl/safety-rd-v1`) a quello
attuale (`roboflow-universe-projects/personal-protective-equipment-combined-model`) —
commit `0c76158 — switch to current dataset`.

**Il problema (aperto).** Trovare un dataset **sufficientemente ampio e di qualità** è difficile, e
**quello attuale non è il massimo**. La conseguenza pratica si vede a valle: la YOLO è **debole su
oggetti piccoli** come guanti e scarpe, che spesso non vengono rilevati → il checker li marca come
"mancanti" → tanti potenziali falsi positivi. Gran parte dell'ingegneria a valle (memoria temporale
PPE, validazione VLM) esiste proprio per **compensare** questa debolezza del dato.

---

### 2.4 — Dove addestrare la YOLO

**Il problema (aperto).** L'addestramento (`training.py`, Ultralytics YOLO11 con augmentation estese
ed export ONNX + NMS) **non è praticabile sul laptop**: senza GPU CUDA il training è troppo lento.
Serve una macchina con GPU (cloud / Colab / workstation) e questo introduce attrito logistico: dove
girare, come portare avanti/indietro dataset e pesi, come rendere ripetibile il tutto.

**Stato.** La YOLO **non è ancora addestrata in modo definitivo** — è il nodo che ha spinto verso una
seconda verifica "intelligente" a valle (il VLM), invece di affidarsi solo alla detection.

---

### 2.5 — Costruire un tracker che *validi* l'output della YOLO

**Cosa è successo.** Non basta la detection frame-per-frame: serve trasformarla in **eventi di
violazione stabili**, tollerando detection perse, occlusioni e più persone in scena. Il 5–6 mag è
stata la fase più tribolata — i messaggi dei commit lo dicono da soli:
`16d03f7 trying to add real tracker` → `22ac788 trying to fix tracker` →
`f02b787 tracker works but crash` → `955453c tracker seems to work`.

**Come è stato risolto.** Dopo la riscrittura da zero (1 lug) la validazione è diventata una
**pipeline a stadi**, ognuno con una responsabilità precisa:

1. `PersonPPEChecker` — associa spazialmente ogni DPI alla persona (containment + IoU, bbox espansa
   per DPI che sporgono). *Stateless.*
2. `PersonTracker` (3 lug, `78def24`) — dà un'**identità stabile** a ogni persona (IoU tra frame,
   `track_id`) e una **memoria temporale dei PPE**: un guanto visto negli ultimi ~2s è considerato
   ancora presente anche se YOLO lo perde. È il filtro anti-violazioni-fantasma.
3. `VideoViolationTracker` — **sliding window**: conferma una violazione solo se persiste in
   ≥ N frame su M, poi cooldown. Elimina i falsi positivi transitori.
4. `LocalVLMValidator` — **escalation** solo sulle violazioni confermate.

**Stato.** Funziona end-to-end. Resta la taratura fine (memoria + persistenza si sommano: troppo
conservativi e su clip brevi non si conferma nulla).

---

### 2.6 — La scelta del modello VLM (la parte più complicata)

Questa è stata **la decisione più difficile del progetto**, un percorso a tappe di modelli scartati:

| Tentativo | Perché scartato |
|-----------|-----------------|
| **GPT-4o / Azure AI Vision** (cloud) | Latenza di rete non compatibile col real-time; dati fuori dalla macchina |
| **`gemma4:12b` via Ollama** | Modello inesistente (è `gemma3`), server non installato, e comunque un 12B è improponibile su CPU/16 GB |
| **moondream2** (2B, VQA) | Scaricato ma **crasha con `transformers` 5.7** (`all_tied_weights_keys`): il suo codice `trust_remote_code` è per la 4.x |
| **CLIP / DINOv2** (contrastivi) | Sbagliano le **negazioni** ("è *senza* casco?"); DINOv2 richiederebbe comunque un classificatore addestrato → di nuovo il problema del dataset |
| **✅ SmolVLM-500M** | **Scelto**: generativo (gestisce le negazioni), **nativo in `transformers`** (no codice remoto, compatibile 5.x), in-process (no server), zero-shot |

**L'ottimizzazione che ha reso praticabile il locale.** SmolVLM di default "splitta" l'immagine in
molti sub-crop → ~900 token → **~21 s/query** su questa CPU. Con `do_image_splitting=False` scende a
~99 token → **~2,5 s/query** (misurato), a parità di risposta sui crop piccoli. Inoltre il VLM gira
**su un thread in background** e **solo sulle violazioni già confermate** dal tracker, così non blocca
mai il loop dei frame.

**Il nodo ancora aperto: la latenza.** Anche a 2,5 s/query **non è "risolta"**, è resa *tollerabile*
dall'architettura (escalation-only + asincrono). La scelta è un compromesso figlio dell'hardware:
**non posso ospitare un VLM ampio in locale**. Il modello più accurato disponibile
(`HuggingFaceTB/SmolVLM2-2.2B-Instruct`, ~9 GB) è molto più lento su CPU; su questa macchina il 500M
è il punto di equilibrio praticabile.

---

### 2.7 — Il colpo di scena finale: la licenza di Ultralytics (lug 2026)

**Cosa è successo.** A progetto funzionante, un vincolo non tecnico ha imposto la revisione più
radicale: **Ultralytics (YOLOv8/11) è AGPL-3.0** — mettere il codice in produzione in azienda
avrebbe obbligato a **pubblicarne il sorgente**. Non solo la libreria: anche i **pesi addestrati**
con Ultralytics sono derivati AGPL. Tutta la filiera YOLO (training, pesi, inferenza ONNX) andava
sostituita.

**La svolta: detection open-vocabulary.** Al posto di un detector addestrato, due modelli
**Apache 2.0** nativi in `transformers` che rilevano le classi **da prompt testuali, zero-shot**:

| Backend | Latenza CPU (misurata) | Ruolo |
|---|---|---|
| **Grounding DINO base** (IDEA-Research) | ~22 s/frame | default, massima accuratezza |
| **OmDet-Turbo swin-tiny** | ~1,5 s/frame (15×) | alternativa real-time, qualità comparabile |

Il paradosso positivo: il vincolo di licenza ha **risolto anche il problema storico del dataset** —
i detector open-vocabulary non richiedono alcun training ("a person", "a hard hat", "a reflective
safety vest" bastano come prompt). Niente più dataset da trovare, niente più GPU per il training.
Sul video di test entrambi rilevano Person/Helmet/Vest in modo consistente dove la vecchia YOLO
(mal addestrata) era instabile.

**In più: le aree vietate (poi accantonate).** Per un periodo è entrato un secondo requisito —
rilevare operai dentro **zone proibite** con poligoni definiti in JSON + test geometrico del
**punto-piedi** (modulo `zone_monitor.py`). La feature è stata **rimossa (2026-07-24)**: nei video
reali non esistono zone definibili come ground truth e l'idea di dedurle da cartelli/segnaletica non
è realizzabile con un detector (non legge i cartelli né inferisce l'estensione dell'area). Al suo
posto è stato ampliato il set di controlli PPE: aggiunti **occhiali** (DPI richiesto) e **sigarette**
(item vietato — violazione se presente).

**Il prezzo.** La latenza su questa CPU resta il nodo: Grounding DINO a ~22 s/frame è utilizzabile
solo con `skip_frames` alto o in batch; OmDet-Turbo è il ripiego pratico. Su GPU entrambi scendono
sotto i 200 ms/frame — la scelta del backend è un flag (`--detector`).

---

## 3 · Stato attuale

- **Pipeline completa senza componenti AGPL**: detector open-vocabulary (Apache 2.0, zero-shot)
  → associazione persona↔oggetti (DPI richiesti + item vietati) → identità + memoria PPE
  → sliding window → video annotato + log JSON.
- **Controlli**: DPI richiesti (casco, gilet, occhiali, guanti, scarpe) + item vietati (sigarette).
- **Zero dipendenze da server esterni**: tutto in-process, tutto in locale, tutto Apache 2.0/BSD/MIT.
- **Zero training richiesto**: nessun dataset, nessuna GPU per addestrare.

## 4 · Nodi ancora aperti

1. **Latenza detection su CPU** — grounding-dino ~22 s/frame, omdet-turbo ~1,5 s/frame. Per il
   real-time vero serve una GPU (entrambi < 200 ms) o un lavoro di ottimizzazione (ONNX export,
   quantizzazione, risoluzione ridotta).
2. **Oggetti piccoli restano difficili** — Glove/Shoe/Glasses/Cigarette, anche per i detector
   zero-shot; la memoria PPE compensa i DPI, ma la taratura va rifinita sul campo.
3. **Taratura sensibilità** — memoria PPE + persistenza + skip_frames vanno bilanciate per il caso
   d'uso reale (sorveglianza continua vs clip brevi).

## 5 · Le lezioni

- **L'hardware è un requisito, non un dettaglio.** Il vincolo "CPU-only, 16 GB" ha ristretto lo
  spazio delle soluzioni più di qualsiasi altra cosa e andava messo al centro fin dall'inizio.
- **Il cloud non è gratis in latenza.** Un VLM potente in rete sembra la scorciatoia, ma per il
  real-time la latenza (e la privacy) la rendono impraticabile.
- **Meglio compensare un modello debole con l'architettura** (memoria temporale + persistenza +
  seconda opinione) che aspettare il modello perfetto — soprattutto quando il dato non c'è.
- **La compatibilità delle librerie conta**: moondream2 era la scelta "ovvia", ma il codice remoto
  incompatibile con `transformers` 5.x l'ha resa un vicolo cieco. Un modello **nativo** nella libreria
  è più robusto nel tempo.
- **La scelta del modello è iterativa**: cinque tentativi prima di SmolVLM. Normale — l'importante è
  che ogni scarto fosse motivato e documentato.
- **La licenza è un requisito di prodotto, non una nota a piè di pagina.** AGPL su Ultralytics ha
  invalidato l'intera filiera detection a progetto finito. Verificare le licenze (modello E pesi)
  *prima* di costruirci sopra.
- **A volte il vincolo apre la strada migliore**: l'obbligo di abbandonare YOLO ha portato ai
  detector open-vocabulary, che hanno eliminato d'un colpo il problema del dataset e del training.
