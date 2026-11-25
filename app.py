"""
Esploratore 3D delle opinioni estratte da main.py (Voice of Customer).

Legge results.csv (prodotto da main.py), calcola con Gemini Embedding 2 (via OpenRouter) l'embedding di
ogni etichetta distinta ("categoria > sottocategoria") e delle menzioni neutre, raggruppa
le etichette in cluster (scegliendo da solo quanti) e le mostra come "costellazioni" in
uno spazio 3D: ogni sfera è un'etichetta, grande quanto le sue menzioni e colorata come
il suo cluster. I nomi dei cluster li scrive il modello linguistico di llm.py.

Per adattarla a un altro tipo di recensioni modifica context.py: lo stesso contesto
usato da main.py viene dato anche al modello quando scrive i nomi dei cluster e le analisi
(le cui regole ed esempi sono anch'essi in context.py), mentre il breve CONTESTO_EMBEDDING
viene messo davanti a ogni testo prima dell'embedding.
Il modello che scrive i nomi e la chiave sono in llm.py (come per main.py).

File creati accanto ad app.py (si possono cancellare: vengono ricalcolati, a pagamento):
    cache_embeddings_<modello>_<dimensioni>.npz   embedding già calcolati
    cache_cluster_names.json                      nomi dei cluster scritti dal modello
    cache_ai_analyses.json                        analisi con l'AI già ottenute
File scaricabili dall'app:
    cluster_mentions.csv                          menzioni filtrate con il loro cluster
    ai_analysis_<selezione>.md                    un'analisi con l'AI

Serve:  pip install streamlit pandas numpy scikit-learn umap-learn plotly openai
e la stessa API key di OpenRouter di main.py (file keys.txt accanto ad app.py, oppure
variabile d'ambiente OPENROUTER_API_KEY).

Velocità: embedding, cluster, proiezione 3D e nomi vengono calcolati una volta e poi
riusati; spostare un cursore ricalcola solo ciò che dipende da quel cursore.

Analisi con l'AI: cliccando su un cluster o su una sua parte, in qualsiasi grafico, si apre un
popup che chiede se inviarne le menzioni al modello di llm.py; il modello restituisce una sintesi
per il management con criticità e raccomandazioni (vedi ANALISI CON L'AI).
Serve Streamlit 1.51 o più recente; il browser scarica Plotly.js da cdn.plot.ly (senza internet i
grafici funzionano lo stesso, ma senza clic).

Come si usa:
    1. Lancia:  python main.py          (crea results.csv)
    2. Lancia:  streamlit run app.py
"""

import colorsys
import hashlib
import html
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)

import llm
llm.SORGENTE = "app.py"
from context import (ANALISI_AI_REGOLE, ANALISI_AI_RUOLO, CONTESTO, CONTESTO_EMBEDDING,
                     NOMI_REGOLE)


# ============================================================
# IMPOSTAZIONI (modifica qui)
# ============================================================

FILE_RISULTATI = "results.csv"            # risultato finale di main.py

# --- Tuning della geometria UMAP 3D (filtro unico "UMAP tuning" da 0 a 100) ---
# Il cursore nella barra laterale regola simultaneamente i tre parametri di UMAP:
#   - 0   = struttura globale / nebulosa continua (vicinato ampio, punti distanziati, basso spread)
#   - 50  = valore predefinito consigliato (i tre valori *_PREDEFINITO / *_PREDEFINITA)
#   - 100 = costellazioni compatte / isole nette (vicinato stretto, punti raccolti, alto spread)
UMAP_TUNING_PREDEFINITO = 85

# 1) Vicinato UMAP (n_neighbors): numero di etichette vicine considerate per costruire la mappa
N_VICINI_TUNING_0 = 40
N_VICINI_PREDEFINITO = 15
N_VICINI_TUNING_100 = 6

# 2) Respiro interno ai cluster (min_dist): distanza minima tra due punti nella proiezione 3D
MIN_DIST_TUNING_0 = 0.40
MIN_DIST_PREDEFINITA = 0.18
MIN_DIST_TUNING_100 = 0.05

# 3) Separazione tra cluster (spread): scala con cui UMAP distanzia gruppi distinti (sempre > min_dist)
SPREAD_TUNING_0 = 1.3
SPREAD_PREDEFINITO = 2.2
SPREAD_TUNING_100 = 3.4

# --- Tuning del clustering HDBSCAN (filtro unico "HDBSCAN tuning" da 0 a 100) ---
# Il cursore nella barra laterale regola simultaneamente i tre parametri qui sotto:
#   - 0   = massima copertura / macro-temi ampi (pochi outlier, soglie più permissive)
#   - 50  = valore predefinito consigliato (i tre valori *_PREDEFINITA / *_MINIMA)
#   - 100 = massima precisione / micro-temi compatti (più outlier, soglie più rigorose)
SPAZIO_EMBEDDING = "Embedding completi (coseno)"
SPAZIO_UMAP = "Coordinate UMAP 3D (euclidea)"
SPAZIO_PREDEFINITO = SPAZIO_UMAP

PARAMETRI_HDBSCAN = [(campioni, selezione) for campioni in (1, 2) for selezione in ("eom", "leaf")]
HDBSCAN_TUNING_PREDEFINITO = 25

# 1) % massima di menzioni che finisce tra gli outlier (taglio della coda dei micro-cluster)
QUOTA_OUTLIER_MIN = 0
QUOTA_OUTLIER_PREDEFINITA = 10
QUOTA_OUTLIER_MAX = 20

# 2) Probabilità minima HDBSCAN (da 0 a 1): sotto questa soglia le etichette periferiche vengono staccate
PROBABILITA_MINIMA_MIN = 0.0
PROBABILITA_MINIMA_HDBSCAN = 0.35
PROBABILITA_MINIMA_MAX = 0.55

# 3) Distinzione minima (da -1 a 1): confronta la distanza media dai compagni di cluster (a)
#    con quella da tutte le altre etichette (b) tramite (b - a) / max(a, b)
DISTINZIONE_MINIMA_MIN = 0.10
DISTINZIONE_MINIMA = 0.20
DISTINZIONE_MINIMA_MAX = 0.30

# Dimensioni (px) delle sfere delle etichette nella mappa 3D (scalate sul massimo del dataset)
DIM_SFERA_MIN = 20       # etichetta meno citata
DIM_SFERA_MAX = 60      # etichetta più citata
ESPONENTE_SFERA = 0.65  # 0.5 = radice quadrata (più uniforme), 1.0 = proporzionale (differenze forti)

# Modello di embedding (via OpenRouter)
MODELLO_CLOUD = "google/gemini-embedding-2"
DIM_CLOUD = 3072   # dimensioni del vettore (3072 = il massimo; da 128 in su)
# Perché il massimo: le etichette sono poche centinaia, quindi vettori più lunghi non costano
# nulla di apprezzabile (memoria e calcoli restano minimi, e le API contano i token del testo,
# non le dimensioni). Dimensioni più basse servono quando i vettori sono milioni.
LOTTO_CLOUD = 30   # testi inviati in una sola chiamata
# Gemini Embedding 2 non ha il parametro task_type: il compito si scrive nel testo
PREFISSO_CLOUD = "task: clustering | query: "

# I nomi dei cluster li scrive il modello di llm.py (lo stesso di main.py)
FILE_NOMI = "cache_cluster_names.json" # nomi già ottenuti, per non richiederli ogni volta
ETICHETTE_PER_NOME = 10                # etichette (le più vicine al centro) mostrate al modello per ogni cluster
MENZIONI_PER_NOME = 30                 # menzioni neutre stratificate mostrate al modello per ogni cluster
RAGIONAMENTO_NOMI = False              # ragionamento del modello: True/False (vedi llm.py)

# Analisi con l'AI al clic su un cluster o su una sua parte
FILE_ANALISI = "cache_ai_analyses.json" # analisi già ottenute, per non richiederle due volte
PREFISSO_FILE_ANALISI = "ai_analysis_"  # nome dell'analisi scaricata: ai_analysis_<selezione>.md
FILE_ESPORTAZIONE = "cluster_mentions.csv"  # menzioni filtrate con il loro cluster (pulsante "Esporta CSV")
MAX_MENZIONI_ANALISI = 250             # tetto massimo di menzioni inviate (oltre: campione stratificato)
MAX_CARATTERI_MENZIONE = 3000           # le menzioni più lunghe vengono accorciate
RAGIONAMENTO_ANALISI = False            # ragionamento del modello (vedi llm.py)

# Vettore di ogni etichetta = mescolanza di due embedding:
#   - quello dell'etichetta stessa ("categoria > sottocategoria"): cosa SIGNIFICA;
#   - la media degli embedding delle sue menzioni neutre (colonna menzione_neutra di
#     results.csv, scritta dalla fase 1 di main.py): di cosa parlano DAVVERO i clienti.
# PESO_MENZIONI è la quota del secondo: 0 = solo etichette, 0.5 = metà e metà, 1 = solo menzioni.
# Si regola con il cursore nella barra laterale; questo è il valore di partenza.
# Se results.csv non ha le menzioni neutre (versione precedente di main.py), si usano le sole etichette.
PESO_MENZIONI = 0.35

SFONDO = "#050A1A"   # blu notte molto scuro (sfondo della mappa e dei grafici)
COLORI_SENTIMENT = {
    "Positivo": "#0EA372",        # verde smeraldo (leggermente scuro)
    "Negativo": "#D73753",        # rosso cremisi (leggermente scuro)
    "Neutro": "#94A3B8",          # grigio ardesia
    "Non deducibile": "#475569",  # grigio scuro (sentiment vuoto in results.csv)
}
# Scala continua per il saldo sentiment: -1 (tutto negativo) ... +1 (tutto positivo)
# Il centro è un grigio senza sfumature di blu (il grigio ardesia del Neutro, mescolato col verde,
# darebbe un celeste spento) e a metà strada ci sono un verde e un rosso chiari: così i saldi
# appena positivi sono verdi sbiaditi e quelli appena negativi rossi sbiaditi.
SCALA_SALDO = [[0.0, COLORI_SENTIMENT["Negativo"]], [0.25, "#E58A98"], [0.5, "#A3A3A3"],
               [0.75, "#86CFA3"], [1.0, COLORI_SENTIMENT["Positivo"]]]
COLORE_OUTLIER = "#64748B"
# Colori dei cluster: luminosità (OKLab, da 0 a 1) ammessa, per staccarsi dallo sfondo scuro
LUMINOSITA_CLUSTER = (0.62, 0.90)
# Centri dei cluster sulla mappa: sfere nel colore del cluster, più scuro, con il bordo del colore pieno
DIM_CENTRI = 8
SCURIMENTO_CENTRI = 0.5     # 1 = stesso colore del cluster, 0 = nero
# Stelle di sfondo, sparse in tutto lo spazio della scena: la scena è un cubo AMPIEZZA_UNIVERSO
# volte più grande dei dati, con la telecamera dentro, così le stelle riempiono tutta la vista
STELLE_DI_SFONDO = 1500     # 0 = nessuna stella
AMPIEZZA_UNIVERSO = 6
COLORE_STELLE = "#E2E8F0"
NOME_OUTLIER = "Outlier"

# Riquadri che compaiono passando il mouse sui grafici: grandi, scuri e con il testo bianco
HOVER_SFONDO = "rgba(8,13,28,0.96)"
HOVER_BORDO = "#64748B"          # nella mappa 3D il bordo prende il colore del cluster
HOVER_TESTO = "#FFFFFF"
HOVER_FONT = "Segoe UI, Inter, sans-serif"
HOVER_DIM = 16                   # testo normale (px)
HOVER_DIM_TITOLO = 21            # nome del cluster (o dell'etichetta) in testa al riquadro
HOVER_DIM_VOCE = 18              # righe chiave: etichetta, menzioni e sentiment
HOVER_DIM_NOTE = 14              # esempi di frammenti
HOVERLABEL = dict(bgcolor=HOVER_SFONDO, bordercolor=HOVER_BORDO, align="left",
                  font=dict(size=HOVER_DIM, color=HOVER_TESTO, family=HOVER_FONT))
EMOJI_SENTIMENT = {"Positivo": "🟢", "Negativo": "🔴", "Neutro": "⚪", "Non deducibile": "⚫"}

# Riquadri guida delle sezioni e metriche di qualità: sfondo blu notte, testo bianco
GUIDA_SFONDO = "#0E1A33"
GUIDA_BORDO = "#24395F"
GUIDA_ACCENTO = "#60A5FA"
GUIDA_TITOLETTI = "#93C5FD"

st.set_page_config(page_title="The Voice of Customers", page_icon="🎯", layout="wide")

# Le versioni recenti di Streamlit usano width="stretch" al posto di use_container_width
_versione = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
LARGO = {"width": "stretch"} if _versione >= (1, 50) else {"use_container_width": True}


# ============================================================
# LETTURA DEI DATI
# ============================================================

def etichetta_sentiment(valore):
    """Converte 1 / 0 / -1 / vuoto (formato di main.py) nell'etichetta a parole."""
    testo = str(valore).strip().capitalize()
    if testo in COLORI_SENTIMENT:
        return testo  # già a parole (vecchio formato)
    try:
        numero = int(float(testo))
    except ValueError:
        return "Non deducibile"
    return {1: "Positivo", 0: "Neutro", -1: "Negativo"}.get(numero, "Non deducibile")


@st.cache_data(show_spinner=False)
def carica_dati(percorso, ultima_modifica):
    """
    Legge i risultati di main.py. 'ultima_modifica' serve solo a far ricaricare
    i dati quando il file cambia (cioè dopo aver rilanciato main.py).
    """
    # sep=None riconosce da solo "," oppure ";"
    df = pd.read_csv(percorso, sep=None, engine="python", encoding="utf-8-sig",
                     dtype=str, keep_default_na=False)

    colonne = ["id_commento", "categoria", "sottocategoria", "frammento", "menzione_neutra", "sentiment"]
    for colonna in colonne:
        if colonna not in df:
            df[colonna] = ""
        df[colonna] = df[colonna].fillna("").astype(str).str.strip()
    df["categoria"] = df["categoria"].str.lower()
    df["sottocategoria"] = df["sottocategoria"].str.lower()
    df["sentiment"] = df["sentiment"].map(etichetta_sentiment)

    # Senza frammento non c'è niente da rappresentare
    con_frammento = df["frammento"] != ""
    return df.loc[con_frammento, colonne].reset_index(drop=True), int((~con_frammento).sum())


def etichetta(categoria, sottocategoria):
    """
    "funzionamento dell'app > stabilità"; se manca qualcosa, solo ciò che c'è.
    Le etichette sono già neutre (senza giudizio) e normalizzate dalla fase 2 di main.py,
    quindi i cluster che ne nascono sono puramente tematici.
    """
    return " > ".join(parte for parte in (categoria, sottocategoria) if parte)


def mescola_con_menzioni(vettori_etichette, vettori_neutre, etichetta_di, neutra_di, peso):
    """
    Vettore finale di ogni etichetta: (1 - peso) x vettore dell'etichetta + peso x media dei
    vettori delle sue menzioni neutre (ogni menzione conta una volta), riportato a lunghezza 1.
    'etichetta_di' e 'neutra_di' dicono, per ogni menzione, quale etichetta e quale menzione
    neutra (indice nei rispettivi vettori) le corrispondono. Le etichette senza menzioni
    neutre restano come sono.
    """
    somme = np.zeros_like(vettori_etichette)
    quante = np.zeros(len(vettori_etichette))
    np.add.at(somme, etichetta_di, vettori_neutre[neutra_di])
    np.add.at(quante, etichetta_di, 1)
    con_menzioni = quante > 0
    medie = somme[con_menzioni] / quante[con_menzioni, None]
    medie /= np.maximum(np.linalg.norm(medie, axis=1, keepdims=True), 1e-12)
    risultato = vettori_etichette.copy()
    risultato[con_menzioni] = (1 - peso) * vettori_etichette[con_menzioni] + peso * medie
    risultato /= np.maximum(np.linalg.norm(risultato, axis=1, keepdims=True), 1e-12)
    return risultato.astype(np.float32)


def testo_per_embedding(testo_etichetta):
    """
    Il testo di cui si calcola l'embedding: l'etichetta, preceduta dal breve contesto
    CONTESTO_EMBEDDING di context.py (se c'è). Esempio:
        "recensioni dell'app mobile Poste Italiane: accesso all'app > riconoscimento dell'impronta digitale"
    Il contesto lungo CONTESTO, quello dei prompt, NON entra qui.
    """
    contesto = CONTESTO_EMBEDDING.strip()
    return f"{contesto}: {testo_etichetta}" if contesto else testo_etichetta


# ============================================================
# EMBEDDING CON GEMINI EMBEDDING 2 (VIA OPENROUTER)
# ============================================================

def trova_chiave():
    """La chiave di llm.py (file keys.txt o OPENROUTER_API_KEY); se manca, la chiede nella barra laterale."""
    chiave = llm.leggi_chiave()
    if chiave:
        return chiave
    chiave = st.sidebar.text_input("API key di OpenRouter", type="password").strip()
    if not llm.chiave_valida(chiave):
        if chiave:
            st.sidebar.error("La chiave contiene caratteri non validi o è troppo corta.")
        st.info("☁️ Per usare il cloud incolla la tua API key di OpenRouter nella barra laterale.")
        st.stop()
    llm.salva_chiave(chiave)  # la prossima volta non serve incollarla (vale anche per main.py)
    return chiave


def embedding_cloud(testi, chiave, avvisa):
    """Un vettore per ogni testo, LOTTO_CLOUD testi per richiesta."""
    client = llm.crea_client(chiave)
    vettori = []
    for inizio in range(0, len(testi), LOTTO_CLOUD):
        lotto = [PREFISSO_CLOUD + t for t in testi[inizio:inizio + LOTTO_CLOUD]]
        nuovi = llm.embedding(client, MODELLO_CLOUD, lotto, DIM_CLOUD, operazione="embedding_testi")
        if len(nuovi) != len(lotto):
            # Controllo di sicurezza: se i vettori non sono uno per testo, li chiedo uno alla volta
            nuovi = [llm.embedding(client, MODELLO_CLOUD, [t], DIM_CLOUD, operazione="embedding_testi_singolo")[0] for t in lotto]
        vettori.extend(nuovi)
        avvisa(f"☁️ Embedding con {MODELLO_CLOUD}: {len(vettori)}/{len(testi)}", len(vettori) / len(testi))
    return vettori


# ============================================================
# EMBEDDING CON MEMORIA SU DISCO (E IN MEMORIA)
# ============================================================
# Ogni testo viene calcolato UNA volta sola e salvato in un file: riaprendo l'app
# (o dopo aver aggiunto recensioni) si calcolano solo i testi mai visti prima.
# Il file viene letto una volta sola: tra un clic e l'altro i vettori restano in memoria.

PERCORSO_EMBEDDING = f"cache_embeddings_{MODELLO_CLOUD.split('/')[-1]}_{DIM_CLOUD}.npz"


@st.cache_resource(show_spinner=False)
def memoria_embedding(percorso):
    """Testo -> vettore, letto dal file una volta per sessione del server."""
    try:
        with np.load(percorso) as f:
            return dict(zip(f["testi"].tolist(), f["vettori"]))
    except Exception:
        return {}  # file assente o rovinato: si calcola tutto


def calcola_embedding(testi, chiave):
    """Restituisce (matrice dei vettori normalizzati, numero di testi calcolati ora)."""
    memoria = memoria_embedding(PERCORSO_EMBEDDING)
    mancanti = [t for t in dict.fromkeys(testi) if t not in memoria]
    if mancanti:
        barra = st.progress(0.0, text=f"Calcolo l'embedding di {len(mancanti)} testi nuovi...")
        fatto = [0.0]

        def avvisa(messaggio, frazione=None):
            if frazione is not None:
                fatto[0] = frazione
            barra.progress(fatto[0], text=messaggio)

        nuovi = np.asarray(embedding_cloud(mancanti, chiave, avvisa), dtype=np.float32)
        # Normalizzo sempre: con vettori di lunghezza 1 il prodotto scalare è la similarità coseno
        nuovi /= np.maximum(np.linalg.norm(nuovi, axis=1, keepdims=True), 1e-12)

        memoria.update(zip(mancanti, nuovi))
        np.savez_compressed(PERCORSO_EMBEDDING, testi=np.array(list(memoria)),
                            vettori=np.vstack(list(memoria.values())))
        barra.empty()

    return np.vstack([memoria[t] for t in testi]), len(mancanti)


# ============================================================
# CLUSTER DELLE ETICHETTE (HDBSCAN, similarità del coseno)
# ============================================================
# COSA viene raggruppato: le ETICHETTE distinte ("categoria > sottocategoria"), ognuna
# pesata per il numero delle sue menzioni. Ogni menzione eredita il cluster della sua etichetta.
# (Raggruppare le menzioni una per una non avrebbe senso: quelle con la stessa etichetta
# hanno lo stesso identico vettore, e migliaia di punti sovrapposti falserebbero le misure.)
#
# Metodo: HDBSCAN sulla distanza 1 - similarità del coseno tra gli embedding. Cerca le zone
# in cui le etichette sono fitte; quelle in mezzo al vuoto restano fuori. Non serve indicare
# quanti cluster cercare.
# I parametri si scelgono da soli: si provano diverse combinazioni e vince quella con il
# miglior compromesso tra Silhouette, Calinski-Harabasz, Davies-Bouldin e copertura (quota
# di menzioni che finiscono in un cluster).
# Poi, per non raggruppare nessuno per forza (con soglie regolate dal cursore "HDBSCAN tuning"):
#   - un'etichetta "attaccata" a un cluster solo perché è la meno lontana
#     (probabilità di appartenenza sotto probabilita_minima) viene staccata;
#   - un'etichetta più vicina a un altro cluster che al proprio (silhouette negativa) viene staccata;
#   - un'etichetta che non è chiaramente più vicina al suo cluster che a tutte le altre etichette
#     (distinzione sotto distinzione_minima) viene staccata: è un secondo controllo sulle
#     etichette isolate finite nel cluster meno lontano;
#   - un'etichetta rimasta sola fa cluster a sé (un tema può anche avere una sola etichetta);
#   - infine si taglia la "coda": i cluster più piccoli finiscono tra gli outlier, partendo dal
#     più piccolo, finché le loro menzioni non superano, sommate, la quota derivata dal cursore
#     "HDBSCAN tuning". Ne deriva un minimo di menzioni per cluster: chi ne ha meno è outlier.
#
# SPAZIO DEL CLUSTERING (scelta nella barra laterale; predefinito: SPAZIO_PREDEFINITO)
#   - embedding completi (DIM_CLOUD dimensioni, distanza del coseno): il metodo descritto sopra;
#   - coordinate UMAP 3D (distanza euclidea): stesso metodo, ma sulle coordinate della mappa.
#     I cluster coincidono con i gruppi che si vedono nello spazio 3D, ma dipendono dal cursore
#     "UMAP tuning": spostarlo rifà la mappa e il clustering (e può chiedere nuovi nomi al modello).
#     Valori alti di "UMAP tuning" stringono i gruppi e di solito danno cluster più netti.

def _interpola_tuning(tuning, minimo, predefinito, massimo):
    """Scala linearmente tra minimo (0), predefinito (50) e massimo (100)."""
    if tuning <= 50:
        return minimo + (predefinito - minimo) * (tuning / 50)
    return predefinito + (massimo - predefinito) * ((tuning - 50) / 50)


def calcola_parametri_umap(tuning):
    """Converte il cursore unico (0..100) nei tre parametri coordinati di UMAP."""
    n_vicini = round(
        _interpola_tuning(tuning, N_VICINI_TUNING_0, N_VICINI_PREDEFINITO, N_VICINI_TUNING_100)
    )
    min_dist = round(
        _interpola_tuning(tuning, MIN_DIST_TUNING_0, MIN_DIST_PREDEFINITA, MIN_DIST_TUNING_100), 2
    )
    spread = round(
        _interpola_tuning(tuning, SPREAD_TUNING_0, SPREAD_PREDEFINITO, SPREAD_TUNING_100), 2
    )
    return int(n_vicini), float(min_dist), float(spread)


def calcola_parametri_tuning(tuning):
    """Converte il cursore unico (0..100) nei tre parametri coordinati di HDBSCAN."""
    quota_outlier = round(
        _interpola_tuning(tuning, QUOTA_OUTLIER_MIN, QUOTA_OUTLIER_PREDEFINITA, QUOTA_OUTLIER_MAX), 1
    )
    probabilita_minima = round(
        _interpola_tuning(tuning, PROBABILITA_MINIMA_MIN, PROBABILITA_MINIMA_HDBSCAN, PROBABILITA_MINIMA_MAX), 3
    )
    distinzione_minima = round(
        _interpola_tuning(tuning, DISTINZIONE_MINIMA_MIN, DISTINZIONE_MINIMA, DISTINZIONE_MINIMA_MAX), 3
    )
    return quota_outlier, probabilita_minima, distinzione_minima


def matrice_distanze(punti, metrica="coseno"):
    """
    Distanza tra ogni coppia di etichette:
      - "coseno":   1 - similarità del coseno (per gli embedding, vettori di lunghezza 1);
      - "euclidea": distanza in linea retta (per le coordinate UMAP 3D, dove la vicinanza
                    sulla mappa è proprio la distanza euclidea).
    """
    p = punti.astype(np.float64)
    if metrica == "euclidea":
        distanze = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    else:
        distanze = np.clip(1 - p @ p.T, 0, 2)
    np.fill_diagonal(distanze, 0)
    return distanze


def rifinisci(distanze, etichette, distinzione_minima=DISTINZIONE_MINIMA):
    """
    Stacca le etichette con silhouette negativa e quelle poco distinte (sotto distinzione_minima);
    i gruppi rimasti con un solo elemento diventano -1.
    """
    def segna_i_soli(e):
        _, inverso, quanti = np.unique(e, return_inverse=True, return_counts=True)
        return np.where((quanti[inverso] >= 2) & (e != -1), e, -1)

    etichette = segna_i_soli(np.array(etichette, dtype=np.int64))  # interi con segno: serve -1
    dentro = etichette != -1
    if len(set(etichette[dentro].tolist())) >= 2:
        silhouette = silhouette_samples(distanze[np.ix_(dentro, dentro)], etichette[dentro], metric="precomputed")
        etichette[np.where(dentro)[0][silhouette < 0]] = -1
        etichette = segna_i_soli(etichette)

    # Distinzione: ogni etichetta deve stare più vicina ai suoi compagni che al resto
    stacca = []
    for i in np.where(etichette != -1)[0]:
        compagni = (etichette == etichette[i])
        compagni[i] = False
        fuori = etichette != etichette[i]
        if not fuori.any():
            continue
        a, b = distanze[i, compagni].mean(), distanze[i, fuori].mean()
        if (b - a) / max(a, b, 1e-12) < distinzione_minima:
            stacca.append(i)
    etichette[stacca] = -1
    return segna_i_soli(etichette)


def misura_qualita(vettori, distanze, pesi, etichette):
    """Le tre misure classiche (sui gruppi di almeno due etichette) e la copertura. None se non calcolabili."""
    dentro = etichette != -1
    n_cluster = len(set(etichette[dentro].tolist()))
    if n_cluster < 2 or dentro.sum() <= n_cluster:
        return None
    v, e = vettori[dentro], etichette[dentro]
    return {
        "copertura": pesi[dentro].sum() / pesi.sum(),  # quota di menzioni dentro un gruppo
        # da -1 a 1, più alto è meglio (distanza = 1 - similarità del coseno)
        "silhouette": silhouette_score(distanze[np.ix_(dentro, dentro)], e, metric="precomputed"),
        # Queste due usano la distanza euclidea: sui vettori normalizzati (lunghezza 1)
        # ordina i punti esattamente come la similarità del coseno, quindi va bene;
        # sulle coordinate UMAP 3D è già la distanza usata per il clustering.
        "calinski_harabasz": calinski_harabasz_score(v, e),     # più alto è meglio
        "davies_bouldin": davies_bouldin_score(v, e),           # più basso è meglio
    }


def compromesso(silhouette, calinski_harabasz, davies_bouldin):
    """
    Un solo punteggio dalle tre misure, ognuna portata tra 0 e 1 su una scala FISSA
    (non relativa alle soluzioni provate: così differenze piccole restano piccole):
      - Silhouette, da -1 a 1:              (s + 1) / 2
      - Davies-Bouldin, da 0 in su (basso = meglio):  1 / (1 + db)
      - Calinski-Harabasz, senza massimo:    diviso per il più alto tra quelli provati
        (l'unica scala relativa, ma in proporzione: 9 contro 10 diventa 0,9 contro 1)
    Poi la media delle tre.
    """
    return ((silhouette + 1) / 2 + 1 / (1 + davies_bouldin) + calinski_harabasz / calinski_harabasz.max()) / 3


def taglia_la_coda(grezzi, pesi, quota_outlier):
    """
    Manda tra gli outlier i cluster più piccoli, finché le loro menzioni sommate non superano
    'quota_outlier' (frazione da 0 a 1) del totale. I cluster con lo stesso numero di menzioni
    restano o escono insieme. Restituisce i cluster aggiornati e il minimo di menzioni risultante.
    """
    numeri, inverso = np.unique(grezzi, return_inverse=True)
    menzioni = np.bincount(inverso, weights=pesi)
    budget = quota_outlier * pesi.sum()
    minimo = menzioni.min()
    for candidato in np.unique(menzioni):              # dal più piccolo al più grande
        if menzioni[menzioni < candidato].sum() <= budget:
            minimo = candidato                          # escludere chi sta sotto rientra nella quota
        else:
            break
    return np.where(menzioni[inverso] >= minimo, grezzi, -1), int(minimo)


def trova_cluster(vettori, pesi, testi_etichette, quota_outlier,
                  probabilita_minima=PROBABILITA_MINIMA_HDBSCAN,
                  distinzione_minima=DISTINZIONE_MINIMA, metrica="coseno"):
    """
    Raggruppa le etichette. 'vettori' sono gli embedding (metrica "coseno") oppure le
    coordinate UMAP 3D (metrica "euclidea"). Restituisce:
      - il numero del cluster di ogni etichetta (da 1 in ordine di menzioni; -1 = outlier)
      - i nomi provvisori dei cluster
      - il minimo di menzioni per cluster che deriva dalla quota di outlier
    """
    grezzi = cerca_gruppi(vettori, pesi, metrica, probabilita_minima, distinzione_minima)
    # La coda dei cluster più piccoli va tra gli outlier, entro la quota scelta
    grezzi, minimo = taglia_la_coda(grezzi, pesi, quota_outlier)
    cluster_id, nomi = numera_e_nomina(grezzi, pesi, testi_etichette)
    return cluster_id, nomi, minimo


@st.cache_data(show_spinner="Calcolo struttura HDBSCAN...")
def adatta_hdbscan(vettori, metrica="coseno"):
    """Calcola la matrice delle distanze e i fit HDBSCAN una sola volta per spazio."""
    distanze = matrice_distanze(vettori, metrica)
    n = len(vettori)
    risultati = []
    for campioni, selezione in PARAMETRI_HDBSCAN:
        modello = HDBSCAN(min_cluster_size=2, min_samples=min(campioni, n - 1), metric="precomputed",
                          cluster_selection_method=selezione, copy=True).fit(distanze)
        risultati.append((modello.labels_.copy(), modello.probabilities_.copy()))
    return distanze, risultati


def cerca_gruppi(vettori, pesi, metrica="coseno",
                 probabilita_minima=PROBABILITA_MINIMA_HDBSCAN,
                 distinzione_minima=DISTINZIONE_MINIMA):
    """Applica le soglie ai fit HDBSCAN già calcolati e sceglie il miglior compromesso."""
    distanze, risultati = adatta_hdbscan(vettori, metrica)

    misure, soluzioni = [], []
    for etichette_base, probabilita in risultati:
        grezzi_base = np.where(probabilita >= probabilita_minima, etichette_base, -1)
        gruppi = rifinisci(distanze, grezzi_base, distinzione_minima)
        qualita = misura_qualita(vettori, distanze, pesi, gruppi)
        if qualita:
            misure.append(qualita)
            soluzioni.append(gruppi)

    if soluzioni:
        m = {chiave: np.array([q[chiave] for q in misure]) for chiave in misure[0]}
        # Moltiplico per la copertura: altrimenti vincerebbe sempre chi tiene pochi
        # cluster perfetti e butta quasi tutto tra gli outlier.
        punteggio = compromesso(m["silhouette"], m["calinski_harabasz"], m["davies_bouldin"]) * m["copertura"]
        grezzi = soluzioni[int(np.argmax(punteggio))].copy()
    else:
        grezzi = np.full(len(vettori), -1)  # nessuna etichetta somiglia a un'altra

    # Ogni etichetta rimasta sola fa cluster a sé (la coda la taglia poi taglia_la_coda)
    soli = np.where(grezzi == -1)[0]
    grezzi[soli] = grezzi.max() + 1 + np.arange(len(soli))
    return grezzi


def numera_e_nomina(grezzi, pesi, testi_etichette):
    """
    Numera i cluster da 1 in ordine di menzioni (-1 = outlier) e dà a ognuno un nome
    provvisorio: la sottocategoria della sua etichetta più citata.
    """
    menzioni = {c: pesi[grezzi == c].sum() for c in set(grezzi.tolist()) - {-1}}
    cluster_id = np.full(len(grezzi), -1)
    nomi = {-1: NOME_OUTLIER}
    for numero, c in enumerate(sorted(menzioni, key=menzioni.get, reverse=True), 1):
        membri = np.where(grezzi == c)[0]
        rappresentante = membri[np.argmax(pesi[membri])]
        cluster_id[membri] = numero
        nomi[numero] = f"C{numero:02d} · {testi_etichette[rappresentante].split(' > ')[-1].upper()}"
    return cluster_id, nomi


def somiglianza_al_centro(vettori, pesi, cluster_id):
    """
    Per ogni etichetta, la similarità del coseno con il centroide del suo cluster
    (media dei vettori delle sue etichette, pesata per menzioni). Per gli outlier: NaN.
    """
    somiglianza = np.full(len(vettori), np.nan)
    for numero in set(cluster_id.tolist()) - {-1}:
        membri = cluster_id == numero
        centroide = np.average(vettori[membri], axis=0, weights=pesi[membri])
        somiglianza[membri] = vettori[membri] @ centroide / np.linalg.norm(centroide)
    return somiglianza


@st.cache_data(show_spinner="Proiezione UMAP...")
def proietta_umap(vettori, n_vicini, min_dist, spread, cluster_id=None):
    """
    Coordinate 3D per il grafico. Se riceve i cluster, UMAP li usa come "suggerimento"
    per tenere vicine le etichette dello stesso cluster (gli outlier, -1, restano liberi).
    """
    n = len(vettori)
    riduttore = umap.UMAP(
        n_neighbors=max(2, min(n_vicini, n - 1)),
        n_components=3,
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        random_state=42,
        init="spectral" if n > 15 else "random",  # con pochissimi punti "spectral" non funziona
    )
    return riduttore.fit_transform(vettori, y=cluster_id)


# ============================================================
# NOMI DEI CLUSTER SCRITTI DAL MODELLO LINGUISTICO
# ============================================================
# Il nome provvisorio di un cluster è la sottocategoria della sua etichetta più citata.
# Qui invece un modello linguistico legge sia le etichette più rappresentative del cluster
# (le più vicine al suo centroide, con il numero di menzioni) sia un campione stratificato
# e deterministico di menzioni neutre centrali, e scrive un nome che ne riassume il tema,
# sapendo di che recensioni si tratta (il contesto di context.py).
# Si fa UNA sola chiamata per tutti i cluster (così i nomi non si ripetono tra loro)
# e i nomi ottenuti restano su disco: un cluster già visto non viene richiesto di nuovo.

# Forma della risposta: {"nomi": [{"id": 1, "nome": "..."}, ...]}
SCHEMA_NOMI = {
    "type": "object",
    "properties": {
        "nomi": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "nome": {"type": "string"}},
                "required": ["id", "nome"],
            },
        }
    },
    "required": ["nomi"],
}

# Parte tecnica (cosa riceve il modello e forma della risposta); le regole sono in context.py
ISTRUZIONI_NOMI = llm.con_contesto("""Ricevi alcuni cluster tematici estratti da recensioni di clienti. Per ogni cluster vedi:
1. Le etichette più rappresentative ("categoria > sottocategoria"), ordinate dalla più vicina al centro del cluster, con il numero di opinioni.
2. Un elenco rappresentativo di menzioni neutre dei clienti appartenenti a quel cluster (stratificate sulle etichette e ordinate per centralità, con eventuale conteggio se ripetute).

Per ogni cluster scrivi un nome che riassuma il tema comune alle sue etichette e menzioni. Restituisci un oggetto JSON con la chiave "nomi": un elemento per cluster, con "id" (il numero del cluster) e "nome".

""" + NOMI_REGOLE)


def menzioni_stratificate_cluster(df_cluster, vet_etichette, pesi_etichette,
                                  vet_neutre, pos_neutra, n_max=MENZIONI_PER_NOME,
                                  soglia_simili=0.85):
    """
    Estrae fino a n_max menzioni neutre stratificando proporzionalmente sulle etichette
    del cluster e scegliendo, in ogni strato, le più centrali e non ridondanti.
    """
    if len(vet_neutre) == 0:
        return []

    centroide = np.average(vet_etichette, axis=0, weights=pesi_etichette)
    centroide /= max(np.linalg.norm(centroide), 1e-12)

    sub = df_cluster[df_cluster["menzione_neutra"] != ""]
    if sub.empty:
        return []

    # Strati = etichette del cluster, ordinate per numero di menzioni
    pesi_strati = sub["etichetta"].value_counts()
    tot_menzioni = pesi_strati.sum()

    # Allocazione proporzionale dei n_max slot (metodo largest remainder)
    quote_esatte = pesi_strati * (n_max / tot_menzioni)
    slot = np.floor(quote_esatte).astype(int)
    resti = quote_esatte - slot
    mancanti = int(n_max - slot.sum())
    if mancanti > 0:
        slot.iloc[np.argsort(-resti.to_numpy())[:mancanti]] += 1

    scelte, vettori_scelti = [], []

    # Primo giro: rispetta gli slot proporzionali di ogni etichetta
    for etich, k in slot.items():
        if k <= 0:
            continue
        conteggi = sub.loc[sub["etichetta"] == etich, "menzione_neutra"].value_counts()
        testi = conteggi.index.tolist()
        matrice = vet_neutre[[pos_neutra[t] for t in testi]]
        punteggi = (matrice @ centroide) * (1.0 + np.log1p(conteggi.to_numpy()))

        prese = 0
        for idx in np.argsort(-punteggi):
            v = matrice[idx]
            if vettori_scelti and np.max(np.vstack(vettori_scelti) @ v) > soglia_simili:
                continue
            c = int(conteggi.iloc[idx])
            scelte.append(f"{testi[idx]} ({c})" if c > 1 else testi[idx])
            vettori_scelti.append(v)
            prese += 1
            if prese >= k or len(scelte) >= n_max:
                break

    # Se il filtro anti-duplicati ha lasciato slot liberi, riempi con le migliori rimanenti del cluster
    if len(scelte) < n_max:
        gia_usate = {s.rsplit(" (", 1)[0] for s in scelte}
        conteggi_all = sub["menzione_neutra"].value_counts()
        testi_all = [t for t in conteggi_all.index if t not in gia_usate]
        if testi_all:
            matrice_all = vet_neutre[[pos_neutra[t] for t in testi_all]]
            c_all = np.array([conteggi_all[t] for t in testi_all])
            punteggi_all = (matrice_all @ centroide) * (1.0 + np.log1p(c_all))
            for idx in np.argsort(-punteggi_all):
                v = matrice_all[idx]
                if vettori_scelti and np.max(np.vstack(vettori_scelti) @ v) > soglia_simili:
                    continue
                c = int(c_all[idx])
                scelte.append(f"{testi_all[idx]} ({c})" if c > 1 else testi_all[idx])
                vettori_scelti.append(v)
                if len(scelte) >= n_max:
                    break

    return scelte


def nomi_dal_modello(mappa_etichette, nomi_calcolati, chiave, df_dati,
                     vettori_etichette, pesi_etichette, vettori_neutre, pos_neutra):
    """
    Restituisce i nomi dei cluster scritti dal modello (quelli provvisori dove non risponde).
    'mappa_etichette' ha una riga per etichetta, con le colonne etichetta, menzioni,
    cluster_id e somiglianza_al_centro.
    """
    memoria = {}
    if os.path.exists(FILE_NOMI):
        try:
            with open(FILE_NOMI, encoding="utf-8") as f:
                memoria = json.load(f)
        except Exception:
            memoria = {}

    # Per ogni cluster: le etichette più vicine al centroide e le menzioni neutre stratificate.
    # Questo testo, insieme al contesto, fa da "impronta" del cluster: se non cambiano,
    # il nome già ottenuto viene riusato senza chiamare di nuovo il modello.
    impronta_contesto = hashlib.md5(CONTESTO.strip().encode("utf-8")).hexdigest()[:8]
    elenchi = {}
    for numero, gruppo in mappa_etichette[mappa_etichette["cluster_id"] != -1].groupby("cluster_id"):
        if len(gruppo) < 2:
            continue  # una sola etichetta: il nome è l'etichetta stessa, inutile chiederlo
        piu_vicine = gruppo.sort_values("somiglianza_al_centro", ascending=False).head(ETICHETTE_PER_NOME)
        righe_etichette = "\n".join(f"- {r.etichetta} ({r.menzioni})" for r in piu_vicine.itertuples())

        mask_cluster = (mappa_etichette["cluster_id"] == numero).to_numpy()
        menzioni_top = menzioni_stratificate_cluster(
            df_dati[df_dati["etichetta"].isin(gruppo["etichetta"])],
            vettori_etichette[mask_cluster],
            pesi_etichette[mask_cluster],
            vettori_neutre,
            pos_neutra,
        )
        blocco = f"Etichette:\n{righe_etichette}"
        if menzioni_top:
            righe_menzioni = "\n".join(f"- {m}" for m in menzioni_top)
            blocco += f"\nMenzioni rappresentative:\n{righe_menzioni}"
        elenchi[numero] = blocco

    impronte = {n: f"[{impronta_contesto}]\n{testo}" for n, testo in elenchi.items()}

    mancanti = [n for n in elenchi if impronte[n] not in memoria]
    if mancanti:
        domanda = "\n\n".join(f"Cluster {n}:\n{elenchi[n]}" for n in mancanti)
        gia_usati = [memoria[impronte[n]] for n in elenchi if n not in mancanti]
        if gia_usati:
            domanda += "\n\nNomi già usati da altri cluster (non ripeterli): " + ", ".join(gia_usati)
        risposta = llm.chiedi(llm.crea_client(chiave), ISTRUZIONI_NOMI, domanda, SCHEMA_NOMI, RAGIONAMENTO_NOMI,
                              operazione="nomi_cluster", elementi=len(mancanti))
        for voce in risposta.get("nomi", []):
            nome = str(voce.get("nome", "")).strip().strip(".").lower()
            if voce.get("id") in mancanti and nome:
                memoria[impronte[voce["id"]]] = nome
        with open(FILE_NOMI, "w", encoding="utf-8") as f:
            json.dump(memoria, f, ensure_ascii=False, indent=1)

    nomi = dict(nomi_calcolati)
    for numero, impronta in impronte.items():
        if impronta in memoria:
            nomi[numero] = f"C{numero:02d} · {memoria[impronta].upper()}"
    return nomi


# ============================================================
# COSTRUZIONE DELLA MAPPA 3D
# ============================================================

def oklab(colori_hex):
    """
    Colori "#rrggbb" -> coordinate OKLab (L, a, b): uno spazio in cui la distanza tra due colori
    corrisponde bene a quanto l'occhio li vede diversi.
    """
    rgb = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in colori_hex])
    lineare = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(lineare @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                                      [0.5363325363, 0.6806995451, 0.2817188376],
                                      [0.0514459929, 0.1073969566, 0.6299787005]]))
    return lms @ np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                           [0.7936177850, -2.4285922050, 0.7827717662],
                           [-0.0040720468, 0.4505937099, -0.8086757660]])


def candidati_colore():
    """
    Colori vivaci ma non "neon" (saturazione fino all'80%), abbastanza chiari da staccarsi
    dallo sfondo scuro; una tonalità ogni 4°.
    """
    candidati = [
        "#{:02x}{:02x}{:02x}".format(*(round(255 * v) for v in colorsys.hls_to_rgb(h / 360, l, s)))
        for h in range(0, 360, 4) for l in (0.55, 0.63, 0.72) for s in (0.65, 0.8)
    ]
    luminosita = oklab(candidati)[:, 0]
    return [c for c, l in zip(candidati, luminosita) if LUMINOSITA_CLUSTER[0] <= l <= LUMINOSITA_CLUSTER[1]]


def colori_dei_cluster(numeri):
    """
    Un colore per ogni cluster, scelti uno alla volta perché siano i più diversi possibile:
    ogni nuovo colore è quello più lontano (in OKLab, cioè per l'occhio) da tutti quelli già
    assegnati, dallo sfondo e dal grigio degli outlier. Con pochi cluster i colori sono quindi
    nettamente diversi; con molti restano comunque i più distanti possibile.
    I cluster più grandi (numeri più bassi) scelgono per primi. Gli outlier sono grigi.
    """
    colori = {-1: COLORE_OUTLIER}
    cluster = sorted(n for n in numeri if n != -1)
    if not cluster:
        return colori
    candidati = candidati_colore()
    punti = oklab(candidati)
    # Distanza di ogni candidato dal colore più vicino tra quelli "occupati"
    vicinanza = np.linalg.norm(punti[:, None, :] - oklab([SFONDO, COLORE_OUTLIER])[None, :, :], axis=-1).min(axis=1)
    for numero in cluster:
        scelto = int(np.argmax(vicinanza))
        colori[numero] = candidati[scelto]
        vicinanza = np.minimum(vicinanza, np.linalg.norm(punti - punti[scelto], axis=1))
    return colori


def trasparente(colore_hex, opacita):
    """"#rrggbb" -> "rgba(r,g,b,opacità)"."""
    r, g, b = (int(colore_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{opacita})"


def piu_scuro(colore_hex, fattore):
    """Lo stesso colore, più scuro: fattore 1 = uguale, 0 = nero."""
    return "#{:02x}{:02x}{:02x}".format(*(round(int(colore_hex[i:i + 2], 16) * fattore) for i in (1, 3, 5)))


def accorcia(testo, massimo=110):
    return testo if len(testo) <= massimo else testo[:massimo - 1] + "…"


def in_grande(testo, dimensione):
    """Testo di un riquadro hover con un carattere di dimensione diversa da quella di base."""
    return f"<span style='font-size:{dimensione}px'>{testo}</span>"


SEPARATORE_HOVER = "<span style='color:#475569'>" + "─" * 30 + "</span>"


def riga_sentiment_hover(positivi, negativi, neutri, non_deducibili=0):
    """"🟢 5 positive · 🔴 3 negative · ⚪ 1 neutre", in grande."""
    parti = [f"{EMOJI_SENTIMENT['Positivo']} <b>{positivi}</b> positive",
             f"{EMOJI_SENTIMENT['Negativo']} <b>{negativi}</b> negative",
             f"{EMOJI_SENTIMENT['Neutro']} <b>{neutri}</b> neutre"]
    if non_deducibili:
        parti.append(f"{EMOJI_SENTIMENT['Non deducibile']} <b>{non_deducibili}</b> non deducibili")
    return in_grande(" · ".join(parti), HOVER_DIM_VOCE)


def esempi_hover(frammenti, quanti, massimo):
    """I primi frammenti, in corsivo e più piccoli, uno per riga."""
    return "<br>".join(in_grande(f"• <i>“{html.escape(accorcia(t, massimo))}”</i>", HOVER_DIM_NOTE)
                       for t in frammenti.head(quanti))


def hover_etichetta(r):
    """Riquadro di una sfera della mappa: cluster, etichetta, menzioni e sentiment ben in vista."""
    saldo = (r.positivi - r.negativi) / r.menzioni
    return "<br>".join([
        in_grande(f"<b>🪐 {html.escape(r.tema_cluster)}</b>", HOVER_DIM_TITOLO),
        SEPARATORE_HOVER,
        in_grande(f"🏷️​ <b>{html.escape(r.etichetta)}</b>", HOVER_DIM_VOCE),
        in_grande(f"🗣️ Menzioni: <b>{r.menzioni}</b>", HOVER_DIM_VOCE),
        riga_sentiment_hover(r.positivi, r.negativi, r.neutri, r.non_deducibili),
        f"⚖️ Saldo sentiment: <b>{saldo:+.0%}</b>",
        SEPARATORE_HOVER,
        r.esempi,
    ])


def punti_della_mappa(df):
    """Una sfera per ogni etichetta, grande quanto le sue menzioni (tra quelle filtrate)."""
    p = df.groupby("etichetta", sort=False).agg(
        px=("x", "first"), py=("y", "first"), pz=("z", "first"),
        cluster_id=("cluster_id", "first"), tema_cluster=("tema_cluster", "first"),
        menzioni=("sentiment", "size"),
        positivi=("sentiment", lambda s: (s == "Positivo").sum()),
        negativi=("sentiment", lambda s: (s == "Negativo").sum()),
        neutri=("sentiment", lambda s: (s == "Neutro").sum()),
        non_deducibili=("sentiment", lambda s: (s == "Non deducibile").sum()),
        esempi=("frammento", lambda s: esempi_hover(s, 3, 95)),
    ).reset_index()

    min_m, max_m = p["menzioni"].min(), p["menzioni"].max()
    if max_m > min_m:
        quota = ((p["menzioni"] - min_m) / (max_m - min_m)) ** ESPONENTE_SFERA
        p["dimensione"] = DIM_SFERA_MIN + quota * (DIM_SFERA_MAX - DIM_SFERA_MIN)
    else:
        p["dimensione"] = (DIM_SFERA_MIN + DIM_SFERA_MAX) / 2

    p["hover"] = [hover_etichetta(r) for r in p.itertuples()]
    return p


def stelle_di_sfondo(centro, semilato, quante):
    """
    Stelle sparse uniformemente nel cubo della scena, in tre gruppi di grandezza e luminosità
    diverse. Seme fisso: le stelle non "saltano" tra un clic e l'altro.
    """
    rng = np.random.default_rng(7)
    posizioni = centro + rng.uniform(-1, 1, (quante, 3)) * semilato
    gruppo = rng.choice(3, quante, p=[0.65, 0.27, 0.08])
    for numero, (dimensione, opacita) in enumerate([(0.7, 0.35), (1.0, 0.5), (1.5, 0.7)]):
        yield np.round(posizioni[gruppo == numero], 4), dimensione, opacita


def crea_mappa_3d(punti, colori):
    fig = go.Figure()
    colore_punti = punti["cluster_id"].map(colori)

    # Spazio della scena: un cubo centrato sui dati, grande AMPIEZZA_UNIVERSO volte i dati se ci
    # sono le stelle. La telecamera si avvicina nella stessa proporzione: i dati si vedono
    # come prima e la telecamera sta dentro al cielo stellato.
    tutte = punti[["px", "py", "pz"]].to_numpy()
    centro_dati = (tutte.min(axis=0) + tutte.max(axis=0)) / 2
    semilato_dati = max(float(np.ptp(tutte, axis=0).max()) / 2, 1e-6) * 1.08  # margine per le sfere
    scala = AMPIEZZA_UNIVERSO if STELLE_DI_SFONDO else 1
    semilato = semilato_dati * scala
    if STELLE_DI_SFONDO:
        for posizioni, dimensione, opacita in stelle_di_sfondo(centro_dati, semilato, STELLE_DI_SFONDO):
            fig.add_trace(go.Scatter3d(
                x=posizioni[:, 0], y=posizioni[:, 1], z=posizioni[:, 2], mode="markers",
                marker=dict(size=dimensione, color=COLORE_STELLE, opacity=opacita, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))

    # Filamenti che legano ogni etichetta al centro del suo cluster, nel colore del cluster
    centri = []
    for numero, gruppo in punti[punti["cluster_id"] != -1].groupby("cluster_id"):
        if len(gruppo) < 2:
            continue
        coordinate = gruppo[["px", "py", "pz"]].to_numpy()
        centro = coordinate.mean(axis=0)
        # Per ogni punto: centro -> punto -> interruzione (NaN stacca un filamento dall'altro)
        tratto = np.full((len(gruppo), 3, 3), np.nan)
        tratto[:, 0, :] = centro
        tratto[:, 1, :] = coordinate
        linee = tratto.reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=linee[:, 0], y=linee[:, 1], z=linee[:, 2], mode="lines",
            line=dict(color=trasparente(colori[numero], 0.45), width=2),
            hoverinfo="skip", showlegend=False,
        ))
        centri.append({
            "x": centro[0], "y": centro[1], "z": centro[2], "colore": colori[numero],
            "tema": gruppo["tema_cluster"].iloc[0],
            "hover": "<br>".join([
                in_grande(f"<b>🪐 {html.escape(gruppo['tema_cluster'].iloc[0])}</b>", HOVER_DIM_TITOLO),
                SEPARATORE_HOVER,
                in_grande(f"🏷️​ <b>{len(gruppo)}</b>", HOVER_DIM_VOCE),
                in_grande(f"🗣️ Menzioni: <b>{gruppo['menzioni'].sum()}</b>", HOVER_DIM_VOCE),
                riga_sentiment_hover(gruppo["positivi"].sum(), gruppo["negativi"].sum(),
                                     gruppo["neutri"].sum(), gruppo["non_deducibili"].sum()),
                f"⚖️ Saldo sentiment: <b>{(gruppo['positivi'].sum() - gruppo['negativi'].sum()) / gruppo['menzioni'].sum():+.0%}</b>",
            ]),
        })

    # Sfere: un alone morbido (finto "glow") e un nucleo nitido, nel colore del cluster
    fig.add_trace(go.Scatter3d(
        x=punti["px"], y=punti["py"], z=punti["pz"], mode="markers",
        marker=dict(size=punti["dimensione"] * 2.2, color=colore_punti, opacity=0.16, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=punti["px"], y=punti["py"], z=punti["pz"], mode="markers",
        marker=dict(size=punti["dimensione"], color=colore_punti, opacity=1.0,
                    line=dict(width=1.2, color="#0F172A")),
        text=punti["hover"], hovertemplate="%{text}<extra></extra>", showlegend=False,
        hoverlabel=dict(bordercolor=colore_punti.tolist()),   # bordo del riquadro nel colore del cluster
        # bersaglio del clic (vedi GRAFICI CLICCABILI): cluster ed etichetta di ogni sfera
        meta=[{"cluster": t, "etichetta": e} for t, e in zip(punti["tema_cluster"], punti["etichetta"])],
    ))

    # Centri dei cluster: sfere del colore del cluster, più scuro; il nome compare solo passandoci sopra con il mouse
    if centri:
        c = pd.DataFrame(centri)
        fig.add_trace(go.Scatter3d(
            x=c["x"], y=c["y"], z=c["z"], mode="markers",
            marker=dict(size=DIM_CENTRI, color=[piu_scuro(colore, SCURIMENTO_CENTRI) for colore in c["colore"]],
                        symbol="circle", opacity=1.0, line=dict(width=1.5, color=c["colore"].tolist())),
            text=c["hover"], hovertemplate="%{text}<extra></extra>", showlegend=False,
            hoverlabel=dict(bordercolor=c["colore"].tolist()),
            meta=[{"cluster": t} for t in c["tema"]],   # clic sul centro = tutto il cluster
        ))

    assi = [dict(visible=False, range=[float(centro_dati[i] - semilato), float(centro_dati[i] + semilato)])
            for i in range(3)]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SFONDO, plot_bgcolor=SFONDO,
        height=780, margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
        uirevision="mappa",  # mantiene l'inquadratura quando cambi i filtri
        hoverlabel=HOVERLABEL,
        scene=dict(bgcolor=SFONDO, xaxis=assi[0], yaxis=assi[1], zaxis=assi[2], aspectmode="cube",
                   camera=dict(eye=dict(x=1.4 / scala, y=1.4 / scala, z=0.9 / scala))),
    )
    return fig


# ============================================================
# PRIORITÀ IN 3D (menzioni x saldo, con la profondità che separa le sfere)
# ============================================================
# Ogni cluster è una sfera vera (superficie illuminata), grande quanto le sue menzioni e
# colorata come il suo saldo. Orizzontale = menzioni, verticale = saldo sentiment.
# La terza dimensione (profondità) NON ha significato: serve solo a spostare indietro o
# avanti le sfere che altrimenti si sovrapporrebbero. Chi non si sovrappone resta al centro.
# Le coordinate sono in "unità di scena", con le stesse proporzioni sui tre assi, così le
# sfere restano rotonde; le tacche degli assi riportano poi i valori veri.

LATO_MENZIONI, LATO_SALDO, LATO_PROFONDITA = 1.4, 1.0, 0.8   # lunghezza degli assi nella scena
RAGGIO_MAX_PRIORITA = 0.10      # raggio della sfera del cluster più citato
RAGGIO_MIN_PRIORITA = 0.028
SPAZIO_TRA_SFERE = 0.025        # distanza minima cercata tra due sfere
ANGOLO_PRIORITA = 38            # rotazione iniziale della vista, in gradi (0 = vista frontale)
ELEVAZIONE_PRIORITA = 18        # quanto la vista guarda dall'alto, in gradi
DISTANZA_PRIORITA = 1.7         # distanza della telecamera: più bassa = più zoomato
ALTEZZA_PRIORITA = 680          # altezza del grafico (px)


def profondita_senza_sovrapposizioni(x, z, raggi, livelli=33):
    """
    Sceglie la profondità di ogni sfera, dalla più grande: tra i livelli possibili prende quello
    che la tiene più lontana da quelle già piazzate. Se più livelli vanno bene, preferisce il centro.
    """
    candidati = np.linspace(0, LATO_PROFONDITA, livelli)
    centro = LATO_PROFONDITA / 2
    y = np.full(len(x), centro)
    piazzate = []
    for i in np.argsort(-raggi, kind="stable"):
        if piazzate:
            p = np.array(piazzate)
            distanze = np.sqrt((x[p] - x[i]) ** 2 + (z[p] - z[i]) ** 2 + (y[p][None, :] - candidati[:, None]) ** 2)
            margine = (distanze - (raggi[p] + raggi[i])).min(axis=1)
            punteggio = np.minimum(margine, SPAZIO_TRA_SFERE) - 1e-4 * np.abs(candidati - centro)
            y[i] = candidati[int(np.argmax(punteggio))]
        piazzate.append(i)
    return y


def tacche_menzioni(massimo, quante=5):
    """Tacche "tonde" (1, 2, 5, 10, 20, 50...) da 0 al massimo delle menzioni."""
    grezzo = massimo / quante
    esponente = 10 ** np.floor(np.log10(grezzo))
    passo = max(1, min(m * esponente for m in (1, 2, 5, 10) if m * esponente >= grezzo))
    return np.arange(0, massimo + passo / 2, passo)


def sfera(cx, cy, cz, raggio, n_u=30, n_v=18):
    """Coordinate (griglie 2D) di una sfera, per go.Surface."""
    u, v = np.linspace(0, 2 * np.pi, n_u), np.linspace(0, np.pi, n_v)
    return (cx + raggio * np.outer(np.cos(u), np.sin(v)),
            cy + raggio * np.outer(np.sin(u), np.sin(v)),
            cz + raggio * np.outer(np.ones(n_u), np.cos(v)))


def crea_priorita_3d(riepilogo):
    massimo = max(int(riepilogo["menzioni"].max()), 1)
    x = riepilogo["menzioni"].to_numpy() / massimo * LATO_MENZIONI
    z = (riepilogo["saldo"].to_numpy() + 1) / 2 * LATO_SALDO
    raggi = np.maximum(RAGGIO_MAX_PRIORITA * np.sqrt(riepilogo["menzioni"].to_numpy() / massimo), RAGGIO_MIN_PRIORITA)
    y = profondita_senza_sovrapposizioni(x, z, raggi)

    fig = go.Figure()
    bordo = RAGGIO_MAX_PRIORITA + 0.03
    gamma = {"x": [-bordo, LATO_MENZIONI + bordo], "y": [-bordo, LATO_PROFONDITA + bordo],
             "z": [-bordo, LATO_SALDO + bordo]}

    tacche_x = tacche_menzioni(massimo)

    # Saldo zero (tanti positivi quanti negativi): una griglia tratteggiata al posto di un piano
    # pieno, che starebbe davanti alle sfere e "ruberebbe" il passaggio del mouse
    px0, px1 = gamma["x"]
    py0, py1 = gamma["y"]
    zero = LATO_SALDO / 2
    segmenti = [[(px0, py0), (px1, py0)], [(px0, py1), (px1, py1)],
                [(px0, py0), (px0, py1)], [(px1, py0), (px1, py1)]]
    segmenti += [[(t, py0), (t, py1)] for t in tacche_x / massimo * LATO_MENZIONI]
    segmenti += [[(px0, t), (px1, t)] for t in np.linspace(0, LATO_PROFONDITA, 5)]
    griglia = np.array([[*a, zero, *b, zero] for a, b in segmenti])
    linee = np.full((len(griglia), 3, 3), np.nan)   # inizio -> fine -> interruzione
    linee[:, 0], linee[:, 1] = griglia[:, :3], griglia[:, 3:]
    linee = linee.reshape(-1, 3)
    fig.add_trace(go.Scatter3d(
        x=linee[:, 0], y=linee[:, 1], z=linee[:, 2], mode="lines",
        line=dict(color="#475569", width=2, dash="dot"), hoverinfo="skip", showlegend=False,
    ))

    nessun_rilievo = dict(highlight=False)
    for r, cx, cy, cz, raggio in zip(riepilogo.itertuples(), x, y, z, raggi):
        # Il testo del riquadro è scritto direttamente nel modello del riquadro (senza %{text}):
        # così compare per intero su qualsiasi punto della sfera
        hover = "<br>".join([
            in_grande(f"<b>🪐 {html.escape(r.tema_cluster)}</b>", HOVER_DIM_TITOLO),
            SEPARATORE_HOVER,
            in_grande(f"🗣️ Menzioni: <b>{r.menzioni}</b>", HOVER_DIM_VOCE),
            riga_sentiment_hover(r.positive, r.negative, r.neutre),
            in_grande(f"⚖️ Saldo sentiment: <b>{r.saldo:+.0%}</b>", HOVER_DIM_VOCE),
        ]).replace("%{", "% {")
        sx, sy, sz = sfera(cx, cy, cz, raggio)
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz, surfacecolor=np.full(sx.shape, r.saldo),
            colorscale=SCALA_SALDO, cmin=-1, cmax=1, showscale=False,
            hovertemplate=hover + "<extra></extra>",
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.35, roughness=0.45, fresnel=0.15),
            lightposition=dict(x=2000, y=-3000, z=4000),
            contours=dict(x=nessun_rilievo, y=nessun_rilievo, z=nessun_rilievo),
            meta={"cluster": r.tema_cluster},   # bersaglio del clic: tutto il cluster
        ))

    tacche_saldo = np.array([-1, -0.5, 0, 0.5, 1])
    stile_asse = dict(backgroundcolor=SFONDO, gridcolor="#1E293B", zerolinecolor="#1E293B",
                      showspikes=False, color="#94A3B8")
    lunghezze = {asse: gamma[asse][1] - gamma[asse][0] for asse in gamma}
    piu_lungo = max(lunghezze.values())
    angolo, elevazione = np.radians(ANGOLO_PRIORITA), np.radians(ELEVAZIONE_PRIORITA)
    distanza = DISTANZA_PRIORITA
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SFONDO, height=ALTEZZA_PRIORITA, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, uirevision="priorita",  # mantiene l'inquadratura quando cambi i filtri
        hoverlabel=HOVERLABEL,
        scene=dict(
            bgcolor=SFONDO, dragmode="turntable",  # si ruota trascinando, con il "su" sempre in alto
            xaxis=dict(stile_asse, title="Menzioni", range=gamma["x"],
                       tickvals=tacche_x / massimo * LATO_MENZIONI, ticktext=[str(int(t)) for t in tacche_x]),
            yaxis=dict(stile_asse, title="", range=gamma["y"], showticklabels=False),
            zaxis=dict(stile_asse, title="Saldo sentiment", range=gamma["z"],
                       tickvals=(tacche_saldo + 1) / 2 * LATO_SALDO,
                       ticktext=[f"{t:+.0%}" if t else "0%" for t in tacche_saldo]),
            aspectmode="manual",
            aspectratio={asse: lunghezze[asse] / piu_lungo for asse in lunghezze},
            camera=dict(eye=dict(x=distanza * np.cos(elevazione) * np.sin(angolo),
                                 y=-distanza * np.cos(elevazione) * np.cos(angolo),
                                 z=distanza * np.sin(elevazione))),
        ),
    )
    return fig


# ============================================================
# QUOTE DI SENTIMENT DI UN CLUSTER (torta + elenco delle etichette)
# ============================================================

ALTEZZA_QUOTE = 460       # altezza (px) della torta e del riquadro con l'elenco accanto
COLORE_TESTO = "#E2E8F0"
SPAZIO_FETTE = 3          # spessore (px) del filo di spazio tra le sezioni della torta (prima: 1.2)


def pallino(sentiment):
    """Un pallino nel colore del sentiment (HTML, va bene sia in Plotly sia in Streamlit)."""
    return f"<span style='color:{COLORI_SENTIMENT[sentiment]}'>●</span>"


def crea_torta_quote(sorgente):
    """
    Torta delle quote di sentiment, con ogni fetta divisa nelle etichette che la compongono
    (dalla più citata, in senso orario). Le percentuali per sentiment sono scritte fuori dalla
    torta; nome e dettagli di ogni etichetta compaiono solo passando il mouse sulla sua sezione.
    """
    totale = len(sorgente)
    fette = sorgente.groupby(["sentiment", "etichetta"]).agg(
        menzioni=("frammento", "size"),
        esempi=("frammento", lambda s: esempi_hover(s, 2, 90)),
    ).reset_index()
    posizione = {s: i for i, s in enumerate(COLORI_SENTIMENT)}
    fette["ordine"] = fette["sentiment"].map(posizione)
    fette = fette.sort_values(["ordine", "menzioni", "etichetta"], ascending=[True, False, True],
                              kind="stable").reset_index(drop=True)
    per_sentiment = fette.groupby("sentiment")["menzioni"].transform("sum")

    # Una sola scritta per sentiment ("Positivo 43%"), fuori dalla torta: va sulla sezione
    # che sta a metà della fetta del sentiment, così cade al centro della fetta
    fine = fette["menzioni"].cumsum()
    inizio = fine - fette["menzioni"]
    testi, posizioni = [""] * len(fette), ["none"] * len(fette)
    for sentiment, gruppo in fette.groupby("sentiment", sort=False):
        meta = (inizio[gruppo.index].min() + fine[gruppo.index].max()) / 2
        dentro = ((inizio[gruppo.index] <= meta) & (meta < fine[gruppo.index])).to_numpy()
        centrale = gruppo.index[dentro][0]
        testi[centrale] = f"<b>{sentiment}</b>  {gruppo['menzioni'].sum() / totale:.0%}"
        posizioni[centrale] = "outside"

    hover = [
        "<br>".join([
            in_grande(f"<b>🛰️ {html.escape(r.etichetta)}</b>", HOVER_DIM_TITOLO),
            SEPARATORE_HOVER,
            in_grande(f"{EMOJI_SENTIMENT[r.sentiment]} {r.sentiment}: <b>{r.menzioni}</b> menzioni", HOVER_DIM_VOCE),
            f"<b>{r.menzioni / tot_s:.0%}</b> della fetta «{r.sentiment}» · <b>{r.menzioni / totale:.0%}</b> del totale",
            SEPARATORE_HOVER,
            r.esempi,
        ])
        for r, tot_s in zip(fette.itertuples(), per_sentiment)
    ]

    fig = go.Figure(go.Pie(
        # le etichette di Plotly devono essere uniche: la stessa etichetta può stare in più sentiment
        labels=fette["sentiment"] + " | " + fette["etichetta"], values=fette["menzioni"],
        sort=False, direction="clockwise", rotation=0, hole=0.45,
        marker=dict(colors=fette["sentiment"].map(COLORI_SENTIMENT).tolist(),
                    line=dict(color=SFONDO, width=SPAZIO_FETTE)),   # bordo nel colore dello sfondo = spazio tra le sezioni
        text=testi, textinfo="text", textposition=posizioni,
        textfont=dict(size=14, color=COLORE_TESTO),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        meta=[{"cluster": sorgente["tema_cluster"].iloc[0], "etichetta": e, "sentiment": s}
              for e, s in zip(fette["etichetta"], fette["sentiment"])],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SFONDO, height=ALTEZZA_QUOTE, showlegend=False,
        margin=dict(l=90, r=90, t=50, b=50),
        hoverlabel=HOVERLABEL,
        annotations=[dict(text=f"<b>{totale}</b><br>menzioni", x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=16, color=COLORE_TESTO))],
    )
    return fig


def elenco_etichette_html(sorgente):
    """
    Le etichette della selezione, dalla più citata: nome e, per ogni sentiment presente,
    un pallino del suo colore con il numero di menzioni. Riquadro scuro, alto quanto la torta.
    """
    conti = sorgente.groupby(["etichetta", "sentiment"]).size().unstack(fill_value=0)
    conti = conti[[s for s in COLORI_SENTIMENT if s in conti.columns]]
    conti = conti.loc[conti.sum(axis=1).sort_values(ascending=False, kind="stable").index]

    righe = []
    for nome, riga in conti.iterrows():
        numeri = "&nbsp;&nbsp;&nbsp;".join(f"{pallino(s)}&nbsp;{n}" for s, n in riga.items() if n)
        righe.append(
            "<div style='display:flex;justify-content:space-between;align-items:baseline;gap:14px;"
            "padding:7px 2px;border-bottom:1px solid #1E293B'>"
            f"<span style='color:{COLORE_TESTO}'>{html.escape(nome)}</span>"
            f"<span style='white-space:nowrap;color:{COLORE_TESTO};font-variant-numeric:tabular-nums'>{numeri}</span>"
            "</div>"
        )
    # Tutto su una riga: in st.markdown le righe rientrate diventerebbero blocchi di codice
    return (
        f"<div style='background:{SFONDO};border-radius:8px;padding:14px 18px;height:{ALTEZZA_QUOTE}px;"
        "overflow-y:auto;font-size:0.92rem;box-sizing:border-box'>"
        f"<div style='color:#94A3B8;font-size:0.8rem;margin-bottom:4px'>{len(conti)} etichette · "
        "menzioni per sentiment</div>"
        + "".join(righe) + "</div>"
    )


# ============================================================
# GRAFICI CLICCABILI (componente personalizzato di Streamlit)
# ============================================================
# st.plotly_chart riconosce i clic solo su punti 2D, linee e barre: non sulla torta né sui
# grafici 3D. Questo piccolo componente (Streamlit "custom components v2", dalla versione 1.51)
# disegna la stessa figura con Plotly.js direttamente nella pagina e, a ogni clic, manda a
# Python il "bersaglio" della parte cliccata. Il bersaglio sta nel campo meta di ogni traccia:
#   - un dizionario, se vale per tutta la traccia (es. la sfera di un cluster in Priorità);
#   - un elenco con un dizionario per punto (es. le etichette della mappa, le fette della torta).
# Esempio: {"cluster": "C03 · METODI DI ACCESSO", "etichetta": "accesso all'app > riconoscimento del volto", "sentiment": "Negativo"}.
# Le tracce senza meta (stelle, filamenti, griglie) non si cliccano.
# Plotly.js si scarica una volta dal sito ufficiale (cdn.plot.ly), nella stessa versione della
# libreria Python. Se il download non riesce, i grafici tornano a st.plotly_chart (senza clic).

try:
    from plotly.offline import get_plotlyjs_version
    URL_PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
except Exception:
    URL_PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

JS_GRAFICO = r"""
function caricaPlotly(url) {
    if (window.Plotly && window.Plotly.react) return Promise.resolve(window.Plotly);
    if (!window.__vocPlotly) {
        window.__vocPlotly = new Promise((ok, ko) => {
            const script = document.createElement("script");
            script.src = url;
            script.async = true;
            script.onload = () => (window.Plotly ? ok(window.Plotly) : ko(new Error("Plotly.js non trovato")));
            script.onerror = () => { window.__vocPlotly = null; ko(new Error("download non riuscito: " + url)); };
            document.head.appendChild(script);
        });
    }
    return window.__vocPlotly;
}

export default function ({ parentElement, data, setTriggerValue }) {
    // La funzione può essere richiamata a ogni rerun: riuso lo stesso contenitore,
    // così Plotly.react aggiorna il grafico e conserva l'inquadratura (uirevision)
    let div = parentElement.querySelector(".voc-grafico");
    if (!div) {
        div = document.createElement("div");
        div.className = "voc-grafico";
        parentElement.appendChild(div);
    }
    div.style.width = "100%";
    div.style.height = data.altezza + "px";
    div._vocInvia = setTriggerValue;   // sempre quella dell'ultima chiamata

    caricaPlotly(data.plotly).then((Plotly) => {
        const figura = JSON.parse(data.figura);
        return Plotly.react(div, figura.data, figura.layout, { responsive: true, displaylogo: false })
            .then(() => {
                if (div._vocAscolta) return;   // i "listener" si aggiungono una volta sola
                div._vocAscolta = true;
                div.on("plotly_click", (evento) => {
                    const punto = evento && evento.points && evento.points[0];
                    if (!punto || !punto.data) return;
                    const meta = punto.data.meta;
                    const bersaglio = Array.isArray(meta) ? meta[punto.pointNumber] : meta;
                    if (bersaglio && typeof bersaglio === "object") {
                        div._vocInvia("clic", Object.assign({}, bersaglio, { istante: Date.now() }));
                    }
                });
                // Dentro una scheda (st.tabs) nascosta il grafico nasce stretto: si adatta quando compare
                new ResizeObserver(() => { if (div.offsetWidth) Plotly.Plots.resize(div); }).observe(div);
            });
    }).catch((errore) => {
        div.textContent = "";
        setTriggerValue("errore", String(errore));
    });
}
"""


@st.cache_resource
def componente_grafico():
    """Registrato una volta sola. Stili non isolati: Plotly vuole il DOM normale della pagina."""
    return st.components.v2.component("voc_grafico_cliccabile", html="<div class='voc-grafico'></div>",
                                      js=JS_GRAFICO, isolate_styles=False)


def grafico_cliccabile(fig, chiave):
    """Mostra la figura; restituisce il bersaglio cliccato (un dizionario) oppure None."""
    if st.session_state.get("grafici_senza_clic"):
        st.plotly_chart(fig, theme=None, key=chiave, **LARGO)
        return None
    altezza = int(fig.layout.height or 450)
    risultato = componente_grafico()(
        key=chiave, data={"figura": fig.to_json(), "altezza": altezza, "plotly": URL_PLOTLY_JS},
        height=altezza, on_clic_change=lambda: None, on_errore_change=lambda: None,
    )
    if getattr(risultato, "errore", None):
        # Plotly.js non scaricabile (es. senza internet): da qui in poi grafici normali, senza clic
        st.session_state["grafici_senza_clic"] = True
        st.session_state["errore_grafici"] = risultato.errore
        st.rerun()
    clic = getattr(risultato, "clic", None)
    return clic if isinstance(clic, dict) and clic.get("cluster") else None


# ============================================================
# ANALISI CON L'AI DI UN CLUSTER O DI UNA SUA PARTE
# ============================================================
# Cliccando su un grafico si apre un popup che mostra la selezione (cluster, etichetta,
# sentiment) e chiede conferma. Se si conferma, il modello di llm.py riceve:
#   - il contesto di context.py;
#   - le statistiche della selezione, calcolate qui (esatte: il modello non deve contare);
#   - le menzioni, raggruppate per etichetta, con il loro sentiment. Oltre MAX_MENZIONI_ANALISI
#     se ne manda un campione proporzionale per etichetta e sentiment (le statistiche restano esatte).
# Risponde con sintesi, punti di forza, criticità, raccomandazioni, spunti e citazioni. Le
# citazioni vengono controllate: restano solo quelle presenti davvero tra le menzioni.
# Le analisi restano su disco (FILE_ANALISI): la stessa selezione non si paga due volte.

SIMBOLI_SENTIMENT = {"Positivo": "+", "Negativo": "-", "Neutro": "=", "Non deducibile": "?"}
LIVELLI = ["alta", "media", "bassa"]
ORIZZONTI = ["subito", "breve termine", "medio termine"]

_testo = {"type": "string"}
SCHEMA_ANALISI_AI = {
    "type": "object",
    "properties": {
        "titolo": _testo,
        "sintesi": _testo,
        "punti_di_forza": {"type": "array", "items": {
            "type": "object", "properties": {"titolo": _testo, "dettaglio": _testo},
            "required": ["titolo", "dettaglio"]}},
        "criticita": {"type": "array", "items": {
            "type": "object",
            "properties": {"titolo": _testo, "dettaglio": _testo, "gravita": {"type": "string", "enum": LIVELLI}},
            "required": ["titolo", "dettaglio", "gravita"]}},
        "raccomandazioni": {"type": "array", "items": {
            "type": "object",
            "properties": {"azione": _testo, "motivazione": _testo,
                           "priorita": {"type": "string", "enum": LIVELLI},
                           "orizzonte": {"type": "string", "enum": ORIZZONTI}},
            "required": ["azione", "motivazione", "priorita", "orizzonte"]}},
        "spunti": {"type": "array", "items": _testo},
        "citazioni": {"type": "array", "items": _testo},
    },
    "required": ["titolo", "sintesi", "punti_di_forza", "criticita", "raccomandazioni", "spunti", "citazioni"],
}

# Ruolo, contenuto e regole dell'analisi sono in context.py; qui c'è la descrizione di cosa riceve il modello
ISTRUZIONI_ANALISI_AI = llm.con_contesto(ANALISI_AI_RUOLO + """ Ricevi una selezione di opinioni dei clienti (un cluster tematico o una sua parte): la descrizione della selezione, le statistiche già calcolate sui dati e le menzioni, raggruppate per etichetta ("categoria > sottocategoria"), ognuna con il suo sentiment: [+] positivo, [-] negativo, [=] neutro o misto, [?] non deducibile.

""" + ANALISI_AI_REGOLE)


def percentuale(quota, decimali=1, segno=False):
    """0.425 -> "42,5%" (con segno: "+42,5%")."""
    return f"{quota:{'+' if segno else ''}.{decimali}%}".replace(".", ",")


def conta_sentiment(dati):
    return {s: int((dati["sentiment"] == s).sum()) for s in COLORI_SENTIMENT}


def saldo_di(dati):
    conti = conta_sentiment(dati)
    return (conti["Positivo"] - conti["Negativo"]) / len(dati) if len(dati) else 0.0


def seleziona(dati, ambito):
    filtro = dati["tema_cluster"] == ambito["cluster"]
    if ambito.get("etichetta"):
        filtro &= dati["etichetta"] == ambito["etichetta"]
    if ambito.get("sentiment"):
        filtro &= dati["sentiment"] == ambito["sentiment"]
    return dati[filtro]


def descrivi_ambito(ambito):
    """Breve descrizione per l'elenco delle scelte nel popup."""
    if ambito.get("etichetta") and ambito.get("sentiment"):
        return f"Etichetta «{ambito['etichetta']}», solo sentiment {ambito['sentiment']}"
    if ambito.get("etichetta"):
        return f"Etichetta «{ambito['etichetta']}», tutti i sentiment"
    if ambito.get("sentiment"):
        return f"Tutto il cluster, solo sentiment {ambito['sentiment']}"
    return "Tutto il cluster"


def ambiti_di_analisi(bersaglio, dati):
    """
    Dalla parte cliccata, le selezioni che si possono analizzare: dalla più specifica al
    cluster intero. Quelle vuote e i doppioni (stesse menzioni) vengono tolti.
    """
    cluster, etichetta_scelta, sentiment_scelto = (bersaglio.get("cluster"), bersaglio.get("etichetta") or None,
                                                   bersaglio.get("sentiment") or None)
    candidati = []
    if etichetta_scelta and sentiment_scelto:
        candidati.append({"etichetta": etichetta_scelta, "sentiment": sentiment_scelto})
    if etichetta_scelta:
        candidati.append({"etichetta": etichetta_scelta})
    if sentiment_scelto:
        candidati.append({"sentiment": sentiment_scelto})
    candidati.append({})
    ambiti, visti = [], set()
    for candidato in candidati:
        ambito = {"cluster": cluster, **candidato}
        sel = seleziona(dati, ambito)
        impronta = frozenset(sel.index)
        if len(sel) and impronta not in visti:
            visti.add(impronta)
            ambiti.append(ambito)
    return ambiti


def campione(sel):
    """La selezione intera, o un campione proporzionale per etichetta e sentiment (seme fisso)."""
    if len(sel) <= MAX_MENZIONI_ANALISI:
        return sel, False
    quota = MAX_MENZIONI_ANALISI / len(sel)
    parti = [gruppo.sample(n=max(1, round(len(gruppo) * quota)), random_state=7)
             for _, gruppo in sel.groupby(["etichetta", "sentiment"])]
    return pd.concat(parti), True


def prepara_domanda(ambito, sel, dati, nota_filtri):
    """Il testo per il modello, il numero di menzioni inviate e se sono un campione."""
    conti = conta_sentiment(sel)
    n = len(sel)
    righe = ["SELEZIONE", f"- Cluster tematico: «{ambito['cluster']}»"]
    if ambito.get("etichetta"):
        righe.append(f"- Etichetta: «{ambito['etichetta']}»")
    if ambito.get("sentiment"):
        righe.append(f"- Solo le menzioni con sentiment: {ambito['sentiment']}")
    if nota_filtri:
        righe.append(f"- {nota_filtri}")

    righe += ["", "STATISTICHE (esatte, su tutte le menzioni della selezione)",
              f"- Menzioni: {n} ({percentuale(n / len(dati))} di tutte le menzioni analizzate)",
              "- Sentiment: " + ", ".join(f"{s.lower()} {c} ({percentuale(c / n)})" for s, c in conti.items() if c),
              f"- Saldo sentiment, (positive - negative) / menzioni: {percentuale(saldo_di(sel), segno=True)}"]
    if ambito.get("etichetta") or ambito.get("sentiment"):
        intero = dati[dati["tema_cluster"] == ambito["cluster"]]
        righe.append(f"- Per confronto, il cluster intero: {len(intero)} menzioni, "
                     f"saldo {percentuale(saldo_di(intero), segno=True)}")
    tabella = sel.groupby(["etichetta", "sentiment"]).size().unstack(fill_value=0)
    tabella = tabella.reindex(columns=list(COLORI_SENTIMENT), fill_value=0)
    ordine = tabella.sum(axis=1).sort_values(ascending=False, kind="stable").index
    righe.append("- Menzioni per etichetta (totale: positive / negative / neutre o miste / non deducibili):")
    righe += [f"  - {e}: {int(tabella.loc[e].sum())} ("
              + " / ".join(str(int(tabella.loc[e, s])) for s in COLORI_SENTIMENT) + ")" for e in ordine]

    inviate, campionate = campione(sel)
    titolo = (f"MENZIONI (campione proporzionale: {len(inviate)} su {n})" if campionate
              else f"MENZIONI ({n})")
    righe += ["", titolo]
    posizione = {s: i for i, s in enumerate(COLORI_SENTIMENT)}
    for e in ordine:
        gruppo = inviate[inviate["etichetta"] == e]
        if gruppo.empty:
            continue
        righe.append(f"## {e}")
        gruppo = gruppo.assign(_o=gruppo["sentiment"].map(posizione)).sort_values("_o", kind="stable")
        righe += [f"- [{SIMBOLI_SENTIMENT[s]}] {accorcia(' '.join(t.split()), MAX_CARATTERI_MENZIONE)}"
                  for t, s in zip(gruppo["frammento"], gruppo["sentiment"])]
    return "\n".join(righe), len(inviate), campionate


def impronta_analisi(domanda):
    return hashlib.md5(f"{llm.MODELLO}\n{ISTRUZIONI_ANALISI_AI}\n{domanda}".encode("utf-8")).hexdigest()


def memoria_analisi():
    try:
        with open(FILE_ANALISI, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def pulisci_analisi(risposta, sel):
    """Risposta del modello -> forma sicura; citazioni tenute solo se presenti tra le menzioni."""
    def testo(valore):
        return " ".join(str(valore or "").split())

    def elenco(valore):
        return valore if isinstance(valore, list) else []

    frammenti = list(zip(sel["frammento"], sel["sentiment"]))
    citazioni = []
    for c in elenco(risposta.get("citazioni"))[:3]:
        c = testo(c).strip("«»\"“” ").rstrip("…").strip()
        trovato = next((s for f, s in frammenti if c and c.lower() in " ".join(f.split()).lower()), None)
        if trovato:
            citazioni.append({"testo": c, "sentiment": str(trovato)})
    return {
        "titolo": testo(risposta.get("titolo")),
        "sintesi": testo(risposta.get("sintesi")),
        "punti_di_forza": [{"titolo": testo(p.get("titolo")), "dettaglio": testo(p.get("dettaglio"))}
                           for p in elenco(risposta.get("punti_di_forza"))[:5] if isinstance(p, dict)],
        "criticita": [{"titolo": testo(p.get("titolo")), "dettaglio": testo(p.get("dettaglio")),
                       "gravita": p.get("gravita") if p.get("gravita") in LIVELLI else "media"}
                      for p in elenco(risposta.get("criticita"))[:5] if isinstance(p, dict)],
        "raccomandazioni": [{"azione": testo(p.get("azione")), "motivazione": testo(p.get("motivazione")),
                             "priorita": p.get("priorita") if p.get("priorita") in LIVELLI else "media",
                             "orizzonte": p.get("orizzonte") if p.get("orizzonte") in ORIZZONTI else "breve termine"}
                            for p in elenco(risposta.get("raccomandazioni"))[:5] if isinstance(p, dict)],
        "spunti": [testo(s) for s in elenco(risposta.get("spunti"))[:3] if testo(s)],
        "citazioni": citazioni,
    }


def esegui_analisi(chiave, domanda, impronta, sel, n_inviate, campionate):
    risposta = llm.chiedi(llm.crea_client(chiave), ISTRUZIONI_ANALISI_AI, domanda,
                          SCHEMA_ANALISI_AI, RAGIONAMENTO_ANALISI,
                          operazione="analisi_ai", elementi=n_inviate)
    voce = {"risultato": pulisci_analisi(risposta, sel), "modello": llm.MODELLO,
            "quando": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
            "menzioni": len(sel), "inviate": n_inviate, "campione": campionate}
    memoria = memoria_analisi()
    memoria[impronta] = voce
    with open(FILE_ANALISI, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=1)
    return voce


# --- Aspetto del report (HTML con stili in linea): tema chiaro, sobrio, adatto alla stampa.
# Tutti i colori sono in TEMA_REPORT: cambiali lì per adattare il report allo stile aziendale.

TEMA_REPORT = {
    "font": "Inter, 'Segoe UI', Helvetica, Arial, sans-serif",
    "sfondo": "#FFFFFF",          # pannello del report
    "bordo": "#E2E8F0",           # bordi e linee di separazione
    "testo": "#0F172A",           # titoli e testo principale
    "testo_secondario": "#475569",
    "testo_tenue": "#64748B",     # titoletti di sezione, note
    "tessera": "#F8FAFC",         # sfondo dei riquadri con i numeri
    "accento": "#1E3A8A",         # blu istituzionale: sintesi e numeri delle raccomandazioni
    "accento_tenue": "#EFF4FF",
    "verde": "#047857",           # numeri positivi (più scuri dei colori dei grafici, per il contrasto su bianco)
    "rosso": "#B91C1C",
    "grigio": "#64748B",
}
T = TEMA_REPORT
VERDE, ROSSO = COLORI_SENTIMENT["Positivo"], COLORI_SENTIMENT["Negativo"]   # barre e bordi
GRIGIO = T["grigio"]
STILE_PANNELLO = (f"background:{T['sfondo']};border:1px solid {T['bordo']};border-radius:10px;"
                  f"padding:24px 28px;color:{T['testo']};line-height:1.55;font-family:{T['font']}")
STILE_TITOLETTO = (f"font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:{T['testo_tenue']};"
                   f"font-weight:600;margin:22px 0 8px;padding-bottom:4px;border-bottom:1px solid {T['bordo']}")
COLORI_LIVELLO = {"alta": ("#B91C1C", "#FEF2F2"), "media": ("#B45309", "#FFFBEB"),
                  "bassa": ("#475569", "#F1F5F9")}
esc = html.escape


def titoletto(testo, primo=False):
    return f"<div style='{STILE_TITOLETTO}{';margin-top:0' if primo else ''}'>{esc(testo)}</div>"


def distintivo(testo, livello):
    colore, sfondo = COLORI_LIVELLO[livello]
    return (f"<span style='display:inline-block;padding:1px 8px;border-radius:4px;font-size:.68rem;"
            f"font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:{colore};"
            f"background:{sfondo};border:1px solid {colore}33;white-space:nowrap'>{esc(testo)}</span>")


def colore_saldo(saldo):
    return T["verde"] if saldo > 0.05 else T["rosso"] if saldo < -0.05 else GRIGIO


def riquadri_numeri(sel):
    """Menzioni, quota positive e negative, saldo; sotto, una barra con le quote di sentiment."""
    n, conti, saldo = len(sel), conta_sentiment(sel), saldo_di(sel)
    tessere = [("Menzioni", str(n), T["testo"]),
               ("Positive", percentuale(conti["Positivo"] / n, 0), T["verde"]),
               ("Negative", percentuale(conti["Negativo"] / n, 0), T["rosso"]),
               ("Saldo", percentuale(saldo, 0, segno=True), colore_saldo(saldo))]
    griglia = "".join(
        f"<div style='background:{T['tessera']};border:1px solid {T['bordo']};border-radius:8px;padding:10px 14px'>"
        f"<div style='font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:{T['testo_tenue']}'>{etichetta_t}</div>"
        f"<div style='font-size:1.35rem;font-weight:650;color:{colore};font-variant-numeric:tabular-nums'>{valore}</div></div>"
        for etichetta_t, valore, colore in tessere)
    barra = "".join(f"<div title='{s}: {c}' style='width:{c / n:.4%};background:{COLORI_SENTIMENT[s]}'></div>"
                    for s, c in conti.items() if c)
    return (f"<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:16px'>{griglia}</div>"
            f"<div style='display:flex;height:6px;border-radius:3px;overflow:hidden;margin-top:12px;"
            f"background:{T['bordo']}'>{barra}</div>")


def intestazione_html(ambito, dimensione="1.05rem"):
    """Il "percorso" della selezione: cluster › etichetta › sentiment (senza emoji: report sobrio)."""
    parti = [esc(ambito["cluster"])]
    if ambito.get("etichetta"):
        parti.append(esc(ambito["etichetta"]))
    if ambito.get("sentiment"):
        parti.append(f"{pallino(ambito['sentiment'])} {esc(ambito['sentiment'])}")
    return (f"<div style='font-size:{dimensione};font-weight:500;color:{T['testo_secondario']}'>"
            + f" <span style='color:{T['testo_tenue']};padding:0 4px'>›</span> ".join(parti) + "</div>")


def anteprima_html(ambito, sel):
    return (f"<div style='{STILE_PANNELLO}'>{titoletto('Selezione', primo=True)}"
            f"{intestazione_html(ambito)}{riquadri_numeri(sel)}</div>")


def scheda_analisi_html(ambito, sel, voce):
    r = voce["risultato"]
    vuoto = f"<div style='color:{T['testo_tenue']};font-size:.9rem'>Nessuna evidenza nella selezione.</div>"

    def blocco(voci, colore, con_gravita=False):
        if not voci:
            return vuoto
        return "".join(
            f"<div style='border-left:3px solid {colore};padding:6px 12px;margin-bottom:10px;break-inside:avoid'>"
            f"<div style='display:flex;justify-content:space-between;gap:10px;align-items:baseline'>"
            f"<span style='font-weight:600;color:{T['testo']}'>{esc(v['titolo'])}</span>"
            + (distintivo(f"gravità {v['gravita']}", v["gravita"]) if con_gravita else "")
            + f"</div><div style='color:{T['testo_secondario']};font-size:.9rem;margin-top:2px'>{esc(v['dettaglio'])}</div></div>"
            for v in voci)

    raccomandazioni = "".join(
        f"<div style='display:flex;gap:14px;padding:12px 0;{'' if i == 1 else 'border-top:1px solid ' + T['bordo'] + ';'}break-inside:avoid'>"
        f"<div style='flex:0 0 26px;height:26px;border-radius:50%;background:{T['accento']};display:flex;"
        f"align-items:center;justify-content:center;font-size:.8rem;font-weight:650;color:#FFFFFF'>{i}</div>"
        f"<div style='flex:1'><div style='font-weight:600;color:{T['testo']}'>{esc(v['azione'])}</div>"
        f"<div style='color:{T['testo_secondario']};font-size:.9rem;margin:2px 0 6px'>{esc(v['motivazione'])}</div>"
        f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{distintivo('priorità ' + v['priorita'], v['priorita'])}"
        f"<span style='display:inline-block;padding:1px 8px;border-radius:4px;font-size:.68rem;font-weight:600;"
        f"letter-spacing:.04em;text-transform:uppercase;color:{T['testo_secondario']};border:1px solid {T['bordo']}'>"
        f"{esc(v['orizzonte'])}</span></div></div></div>"
        for i, v in enumerate(r["raccomandazioni"], 1)) or vuoto

    spunti = (f"<ul style='margin:0;padding-left:18px;color:{T['testo_secondario']}'>"
              + "".join(f"<li style='margin-bottom:4px'>{esc(s)}</li>" for s in r["spunti"]) + "</ul>"
              if r["spunti"] else vuoto)
    citazioni = "".join(
        f"<div style='border-left:3px solid {COLORI_SENTIMENT[c['sentiment']]};padding:4px 12px;margin-bottom:8px;"
        f"font-style:italic;color:{T['testo_secondario']};break-inside:avoid'>“{esc(c['testo'])}”</div>"
        for c in r["citazioni"]) or vuoto

    campione_testo = (f"un campione di {voce['inviate']} menzioni su {voce['menzioni']}" if voce["campione"]
                      else f"{voce['menzioni']} menzioni")
    due_colonne = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:4px 28px"
    return (
        f"<div style='{STILE_PANNELLO}'>"
        f"{titoletto('Sintesi per il management', primo=True)}"
        f"<div style='font-size:1.4rem;font-weight:700;color:{T['testo']};line-height:1.3;margin-top:10px'>{esc(r['titolo'])}</div>"
        f"<div style='margin-top:6px'>{intestazione_html(ambito, '.88rem')}</div>"
        f"{riquadri_numeri(sel)}"
        f"<div style='margin-top:18px;padding:12px 16px;border-left:3px solid {T['accento']};background:{T['accento_tenue']};"
        f"font-size:.98rem;color:{T['testo']}'>{esc(r['sintesi'])}</div>"
        f"<div style='{due_colonne}'>"
        f"<div>{titoletto('Punti di forza')}{blocco(r['punti_di_forza'], VERDE)}</div>"
        f"<div>{titoletto('Criticità')}{blocco(r['criticita'], ROSSO, True)}</div></div>"
        f"{titoletto('Raccomandazioni')}{raccomandazioni}"
        f"<div style='{due_colonne}'><div>{titoletto('Spunti da approfondire')}{spunti}</div>"
        f"<div>{titoletto('Voci dei clienti')}{citazioni}</div></div>"
        f"<div style='margin-top:20px;padding-top:10px;border-top:1px solid {T['bordo']};font-size:.75rem;color:{T['testo_tenue']}'>"
        f"Generata da {esc(voce['modello'])} il {esc(voce['quando'])} su {campione_testo}. "
        f"È una sintesi automatica: verifica i punti chiave sui commenti originali.</div></div>"
    )


def report_html(ambito, sel, voce):
    """Il report come pagina HTML a sé, sfondo bianco: si apre nel browser e si stampa in PDF (Ctrl+P)."""
    titolo = esc(voce["risultato"]["titolo"] or "Analisi")
    return (
        "<!DOCTYPE html><html lang='it'><head><meta charset='utf-8'>"
        f"<title>{titolo}</title><style>"
        f"body{{margin:0;background:#F1F5F9;font-family:{T['font']}}}"
        ".pagina{max-width:880px;margin:32px auto;padding:0 16px}"
        "*{-webkit-print-color-adjust:exact;print-color-adjust:exact}"
        "@page{size:A4;margin:14mm}"
        "@media print{body{background:#FFFFFF}.pagina{margin:0;max-width:none;padding:0}"
        ".pagina>div{border:none!important;padding:0!important}}"
        "</style></head><body><div class='pagina'>"
        f"{scheda_analisi_html(ambito, sel, voce)}</div></body></html>"
    )


def analisi_markdown(ambito, sel, voce):
    """La stessa analisi in Markdown, da scaricare."""
    r = voce["risultato"]
    percorso = " › ".join([ambito["cluster"]] + [ambito[k] for k in ("etichetta", "sentiment") if ambito.get(k)])
    conti = conta_sentiment(sel)
    righe = [f"# {r['titolo']}", "", f"**Selezione:** {percorso}  ",
             f"**Menzioni:** {len(sel)} · positive {conti['Positivo']} · negative {conti['Negativo']} · "
             f"saldo {percentuale(saldo_di(sel), segno=True)}", "", "## Sintesi", r["sintesi"], "",
             "## Punti di forza"]
    righe += [f"- **{p['titolo']}**: {p['dettaglio']}" for p in r["punti_di_forza"]] or ["- (nessuno)"]
    righe += ["", "## Criticità"]
    righe += [f"- **{p['titolo']}** (gravità {p['gravita']}): {p['dettaglio']}" for p in r["criticita"]] or ["- (nessuna)"]
    righe += ["", "## Raccomandazioni"]
    righe += [f"{i}. **{p['azione']}** — priorità {p['priorita']}, {p['orizzonte']}. {p['motivazione']}"
              for i, p in enumerate(r["raccomandazioni"], 1)]
    righe += ["", "## Spunti da approfondire"] + [f"- {s}" for s in r["spunti"]]
    righe += ["", "## Voci dei clienti"] + [f"> {c['testo']} ({c['sentiment'].lower()})" for c in r["citazioni"]]
    righe += ["", f"_Generata da {voce['modello']} il {voce['quando']}._"]
    return "\n".join(righe)


@st.dialog("✨ Analisi con l'AI", width="large")
def finestra_analisi(bersaglio, dati, nota_filtri, chiave):
    ambiti = ambiti_di_analisi(bersaglio, dati)
    if not ambiti:
        st.warning("Nessuna menzione per questa selezione con i filtri attuali.")
        return
    if len(ambiti) > 1:
        scelta = st.radio(
            "Cosa vuoi analizzare?", range(len(ambiti)), horizontal=False,
            format_func=lambda i: f"{descrivi_ambito(ambiti[i])} · {len(seleziona(dati, ambiti[i]))} menzioni",
            key=f"ambito_{bersaglio.get('istante')}",
        )
    else:
        scelta = 0
    ambito = ambiti[scelta]
    sel = seleziona(dati, ambito)
    domanda, n_inviate, campionate = prepara_domanda(ambito, sel, dati, nota_filtri)
    impronta = impronta_analisi(domanda)
    voce = memoria_analisi().get(impronta)

    rigenera = st.session_state.pop(f"rigenera_{impronta}", False)
    if voce is None or rigenera:
        st.html(anteprima_html(ambito, sel))
        conferma = st.empty()
        if not rigenera:   # "Rigenera" vale già come conferma
            with conferma.container():
                invio = (f"un campione proporzionale di {n_inviate} menzioni su {len(sel)}" if campionate
                         else f"{'tutte le ' if len(sel) > 1 else 'la '}{len(sel)} menzion{'i' if len(sel) > 1 else 'e'}")
                st.markdown(f"#### Vuoi inviare {invio} al modello?")
                st.caption(f"{llm.MODELLO} riceve il contesto di `context.py`, le statistiche esatte della "
                           "selezione e le menzioni. Risponde con una sintesi per il management, i punti di forza, "
                           "le criticità e le raccomandazioni. Il risultato resta salvato: riaprire la stessa "
                           "selezione non costa nulla.")
                c1, c2 = st.columns(2)
                analizza = c1.button("✨ Analizza con l'AI", type="primary", key=f"vai_{impronta}", **LARGO)
                if c2.button("Annulla", key=f"annulla_{impronta}", **LARGO):
                    st.rerun()   # chiude il popup
            if not analizza:
                return
        conferma.empty()
        try:
            with st.spinner(f"✨ Il modello sta leggendo {n_inviate} menzioni..."):
                voce = esegui_analisi(chiave, domanda, impronta, sel, n_inviate, campionate)
        except Exception as errore:
            st.error(f"Analisi non riuscita: {errore}")
            st.button("Riprova", key=f"riprova_{impronta}")   # il clic ridisegna il popup con la conferma
            return
        st.rerun(scope="fragment")   # ridisegna il popup con il risultato (ora salvato)

    st.html(scheda_analisi_html(ambito, sel, voce))
    c1, c2, c3 = st.columns(3)
    nome_file = PREFISSO_FILE_ANALISI + "_".join(
        "".join(c if c.isalnum() else "-" for c in str(ambito[k])).strip("-")[:30]
        for k in ("cluster", "etichetta", "sentiment") if ambito.get(k))
    c1.download_button("🖨️ Scarica il report (HTML → PDF)", data=report_html(ambito, sel, voce),
                       file_name=nome_file + ".html", mime="text/html", on_click="ignore",
                       key=f"scarica_html_{impronta}", help="Aprilo nel browser e stampalo in PDF (Ctrl+P)", **LARGO)
    c2.download_button("📄 Scarica l'analisi (Markdown)", data=analisi_markdown(ambito, sel, voce),
                       file_name=nome_file + ".md", mime="text/markdown", on_click="ignore", key=f"scarica_{impronta}",
                       **LARGO)
    if c3.button("🔄 Rigenera l'analisi", key=f"rigenera_btn_{impronta}", **LARGO):
        st.session_state[f"rigenera_{impronta}"] = True
        st.rerun(scope="fragment")


# ============================================================
# GUIDE DELLE SEZIONI E QUALITÀ DEL CLUSTERING
# ============================================================

def riquadro_guida(titolo, scopo, composizione, uso):
    """
    Riquadro informativo di una sezione: scopo, composizione e come leggerla (testo HTML).
    È espandibile e parte chiuso, con in vista solo il titolo. Si usa <details> dell'HTML e non
    st.expander perché alcune guide stanno già dentro un expander, e Streamlit non li annida.
    """
    colonne = "".join(
        f"<div><div style='font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;"
        f"color:{GUIDA_TITOLETTI};margin-bottom:5px'>{nome}</div>"
        f"<div style='font-size:1rem;line-height:1.55;color:#FFFFFF'>{testo}</div></div>"
        for nome, testo in (("🎯 Scopo", scopo), ("🧩 Composizione", composizione),
                            ("🧭 Come leggerla e usarla", uso)))
    st.html(
        f"<details style='background:{GUIDA_SFONDO};border:1px solid {GUIDA_BORDO};border-left:5px solid {GUIDA_ACCENTO};"
        f"border-radius:10px;padding:10px 20px;margin:4px 0 12px;font-family:{HOVER_FONT}'>"
        f"<summary style='cursor:pointer;font-size:1.08rem;font-weight:700;color:#FFFFFF;padding:2px 0'>"
        f"ℹ️ {titolo} <span style='font-size:.82rem;font-weight:500;color:{GUIDA_TITOLETTI};margin-left:6px'>"
        "· guida, clicca per aprire</span></summary>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px 26px;"
        f"margin:12px 0 6px'>{colonne}</div></details>"
    )


VERDE_CHIARO, GIALLO, ARANCIO, ROSSO_CHIARO = "#34D399", "#FBBF24", "#FB923C", "#F87171"


def giudizio_silhouette(s):
    """Soglie indicative di uso comune (Kaufman e Rousseeuw)."""
    if s >= 0.5:
        return "struttura forte", VERDE_CHIARO
    if s >= 0.25:
        return "struttura discreta", GIALLO
    if s >= 0:
        return "struttura debole", ARANCIO
    return "nessuna struttura", ROSSO_CHIARO


def giudizio_davies_bouldin(db):
    if db <= 0.6:
        return "ottima separazione", VERDE_CHIARO
    if db <= 1.0:
        return "buona separazione", GIALLO
    if db <= 1.5:
        return "separazione discreta", ARANCIO
    return "cluster sovrapposti", ROSSO_CHIARO


def giudizio_copertura(c):
    if c >= 0.9:
        return "alta", VERDE_CHIARO
    if c >= 0.75:
        return "media", GIALLO
    return "bassa: molti outlier", ARANCIO


def decimale(valore, cifre=2):
    return f"{valore:.{cifre}f}".replace(".", ",")


def tessere_qualita_html(tessere):
    """Riquadri con le metriche: (nome, valore, giudizio o None, colore del giudizio, spiegazione)."""
    celle = "".join(
        f"<div style='background:#0B1428;border:1px solid {GUIDA_BORDO};border-radius:10px;padding:12px 14px'>"
        f"<div style='font-size:.78rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;"
        f"color:{GUIDA_TITOLETTI}'>{nome}</div>"
        f"<div style='font-size:1.75rem;font-weight:700;color:#FFFFFF;font-variant-numeric:tabular-nums;"
        f"margin:2px 0 4px'>{valore}</div>"
        + (f"<span style='display:inline-block;padding:1px 8px;border-radius:5px;font-size:.8rem;font-weight:700;"
           f"color:{colore};border:1px solid {colore}'>{giudizio}</span>" if giudizio else "")
        + f"<div style='font-size:.9rem;color:#FFFFFF;margin-top:7px;line-height:1.45'>{spiegazione}</div></div>"
        for nome, valore, giudizio, colore, spiegazione in tessere)
    return ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;"
            f"font-family:{HOVER_FONT};margin-bottom:10px'>{celle}</div>")


def qualita_per_cluster(distanze, cluster_id, pesi, mappa):
    """
    Una riga per cluster: etichette, menzioni, silhouette media delle sue etichette (nello spazio
    del clustering) e coesione semantica (somiglianza media al centro sugli embedding, pesata per
    menzioni). I cluster di una sola etichetta non si possono misurare: valori vuoti.
    """
    dentro = cluster_id != -1
    silhouette = np.full(len(cluster_id), np.nan)
    n_cluster = len(set(cluster_id[dentro].tolist()))
    if n_cluster >= 2 and dentro.sum() > n_cluster:
        silhouette[dentro] = silhouette_samples(distanze[np.ix_(dentro, dentro)], cluster_id[dentro],
                                                metric="precomputed")
    somiglianza = mappa["somiglianza_al_centro"].to_numpy()
    righe = []
    for numero in sorted(set(cluster_id[dentro].tolist())):
        m = cluster_id == numero
        solo = m.sum() < 2
        righe.append({
            "cluster": mappa.loc[m, "tema_cluster"].iloc[0],
            "etichette": int(m.sum()),
            "menzioni": int(pesi[m].sum()),
            "silhouette": np.nan if solo else float(silhouette[m].mean()),
            "coesione": np.nan if solo else float(np.average(somiglianza[m], weights=pesi[m])),
        })
    return pd.DataFrame(righe, columns=["cluster", "etichette", "menzioni", "silhouette", "coesione"])


# ============================================================
# PAGINA
# ============================================================

st.title("🗣️​ Voice of Customer: costellazioni semantiche")

# --- 1. Dati -------------------------------------------------
file_dati = FILE_RISULTATI
if not os.path.exists(file_dati):
    st.error(f"⚠️ File `{FILE_RISULTATI}` non trovato! Esegui prima `python main.py`.")
    st.stop()
df, n_senza_frammento = carica_dati(file_dati, os.path.getmtime(file_dati))
df["etichetta"] = [etichetta(c, s) for c, s in zip(df["categoria"], df["sottocategoria"])]
n_senza_etichetta = int((df["etichetta"] == "").sum())
df = df[df["etichetta"] != ""].reset_index(drop=True)

# Le etichette distinte, con il numero di menzioni: sono loro i punti da raggruppare
conteggio = df["etichetta"].value_counts().sort_index()
testi_etichette, pesi = conteggio.index.tolist(), conteggio.to_numpy()
if len(testi_etichette) < 5:
    st.warning(f"In `{file_dati}` ci sono solo {len(testi_etichette)} etichette distinte: troppo poche per una mappa.")
    st.stop()

# --- 2. Embedding ------------------------------------------
chiave = trova_chiave()

try:
    # Le menzioni neutre (se ci sono) si calcolano insieme alle etichette, in un solo passaggio.
    # Si calcolano sempre, qualunque sia il peso: così il cursore qui sotto lavora sui vettori
    # già pronti e spostarlo non richiede nuovi embedding.
    con_neutra = df[df["menzione_neutra"] != ""]
    testi_neutre = sorted(set(con_neutra["menzione_neutra"]))
    tutti, n_calcolati = calcola_embedding(
        [testo_per_embedding(t) for t in testi_etichette + testi_neutre], chiave
    )
except ImportError as errore:
    st.error(f"Manca una libreria ({errore}). Installa con: `pip install openai`")
    st.stop()
except Exception as errore:
    st.error(f"Calcolo degli embedding non riuscito: {errore}")
    st.stop()

st.sidebar.header("🔢​ Embedding")
peso_menzioni = st.sidebar.slider(
    "Peso delle menzioni", 0, 100, round(PESO_MENZIONI * 100), step=5, format="%d%%",
    disabled=not testi_neutre,
    help="Il vettore di ogni etichetta mescola due cose: il significato dell'etichetta stessa "
         "(«categoria > sottocategoria») e la media delle sue menzioni, cioè di cosa parlano "
         "davvero i clienti.  \n"
         "**0%** = solo le etichette: i cluster seguono i nomi scelti dalla fase 2.  \n"
         "**100%** = solo le menzioni: due etichette con nomi diversi ma usate per le stesse cose "
         "finiscono vicine.  \n"
         "Spostarlo rifà clustering e mappa (e può chiedere nuovi nomi dei cluster al modello).",
) / 100
if not testi_neutre:
    peso_menzioni = 0.0  # senza menzioni neutre il cursore resta disattivato: solo etichette

vettori = tutti[:len(testi_etichette)]
vettori_neutre = tutti[len(testi_etichette):]
posizione_neutra = {t: i for i, t in enumerate(testi_neutre)}

if testi_neutre and peso_menzioni > 0:
    posizione_etichetta = {t: i for i, t in enumerate(testi_etichette)}
    vettori = mescola_con_menzioni(
        vettori, vettori_neutre,
        con_neutra["etichetta"].map(posizione_etichetta).to_numpy(),
        con_neutra["menzione_neutra"].map(posizione_neutra).to_numpy(),
        peso_menzioni,
    )

st.sidebar.caption(
    f"**Gemini Embedding 2** (3072 dim.)  \n"
    f"  \n**· {len(testi_etichette)}** etichette e **{len(testi_neutre)}** menzioni distinte · "
    + ("" if testi_neutre
       else "**Menzioni:** assenti in `results.csv`, si usano le sole etichette  \n")
)

# --- 3. Geometria e cluster ----------------------------------
st.sidebar.header("🌌​ Geometria dello spazio 3D")
tuning_umap = st.sidebar.slider(
    "UMAP tuning", 0, 100, UMAP_TUNING_PREDEFINITO, step=5,
    help=(
        "Regola in modo coordinato la geometria della proiezione 3D bilanciando tre parametri insieme "
        "(vicinato `n_neighbors`, respiro interno `min_dist` e separazione `spread`):\n\n"
        "- **0 · Struttura globale:** conserva le relazioni macro tra temi, distribuendo le sfere in modo "
        f"fluido e uniforme (`n_neighbors` {N_VICINI_TUNING_0}, `min_dist` {MIN_DIST_TUNING_0:.2f}, "
        f"`spread` {SPREAD_TUNING_0:.1f}).\n"
        "- **50 · Bilanciato (consigliato):** compromesso ottimale tra visione d'insieme e leggibilità "
        f"dei singoli gruppi (`n_neighbors` {N_VICINI_PREDEFINITO}, `min_dist` {MIN_DIST_PREDEFINITA:.2f}, "
        f"`spread` {SPREAD_PREDEFINITO:.1f}).\n"
        "- **100 · Costellazioni compatte:** privilegia il dettaglio locale, impacchettando i gruppi e "
        f"aumentando la distanza tra cluster diversi (`n_neighbors` {N_VICINI_TUNING_100}, "
        f"`min_dist` {MIN_DIST_TUNING_100:.2f}, `spread` {SPREAD_TUNING_100:.1f})."
    ),
)
n_vicini, min_dist, spread = calcola_parametri_umap(tuning_umap)
st.sidebar.caption(
    f"Vicini **{n_vicini}** · min_dist **{min_dist:.2f}** · spread **{spread:.1f}**"
)

st.sidebar.header("🪐 Cluster")
spazio_umap = st.sidebar.radio(
    "Spazio del clustering", [SPAZIO_EMBEDDING, SPAZIO_UMAP],
    index=[SPAZIO_EMBEDDING, SPAZIO_UMAP].index(SPAZIO_PREDEFINITO),
    help="**Embedding completi**: i cluster nascono dal significato delle etichette, in tutte le "
         f"{DIM_CLOUD} dimensioni; la mappa 3D è solo una proiezione e può avvicinare o separare "
         "etichette in modo diverso.  \n"
         "**Coordinate UMAP 3D**: i cluster sono i gruppi che vedi sulla mappa. Dipendono dal "
         "cursore **UMAP tuning** qui sopra: spostarlo rifà il clustering (e può chiedere nuovi "
         "nomi al modello). Valori alti di UMAP tuning danno gruppi più netti.",
) == SPAZIO_UMAP
tuning_hdbscan = st.sidebar.slider(
    "HDBSCAN tuning", 0, 100, HDBSCAN_TUNING_PREDEFINITO, step=5,
    help=(
        "Regola in modo coordinato la granularità dei cluster bilanciando tre soglie insieme "
        "(quota massima di outlier, probabilità minima di appartenenza HDBSCAN e distinzione minima dal resto):\n\n"
        "- **0 · Macro-temi (alta copertura):** incorpora quasi tutte le etichette in temi ampi "
        f"(outlier max {QUOTA_OUTLIER_MIN}%, prob. HDBSCAN {PROBABILITA_MINIMA_MIN:.2f}, "
        f"distinzione {DISTINZIONE_MINIMA_MIN:.2f}).\n"
        "- **50 · Bilanciato (consigliato):** compromesso ottimale tra copertura e coerenza semantica "
        f"(outlier max {QUOTA_OUTLIER_PREDEFINITA}%, prob. HDBSCAN {PROBABILITA_MINIMA_HDBSCAN:.2f}, "
        f"distinzione {DISTINZIONE_MINIMA:.2f}).\n"
        "- **100 · Micro-temi (alta precisione):** tiene solo cluster molto compatti e isola i casi di confine "
        f"(outlier max {QUOTA_OUTLIER_MAX}%, prob. HDBSCAN {PROBABILITA_MINIMA_MAX:.2f}, "
        f"distinzione {DISTINZIONE_MINIMA_MAX:.2f})."
    ),
)
quota_outlier, probabilita_minima, distinzione_minima = calcola_parametri_tuning(tuning_hdbscan)
info_outlier = st.sidebar.empty()  # qui sotto, a calcolo finito, parametri attivi e quota effettiva
nascondi_outlier = st.sidebar.checkbox("Nascondi outlier", value=True)
allinea_mappa = st.sidebar.checkbox(
    "🧲 Avvicina sulla mappa le etichette dello stesso cluster", value=False,
    disabled=spazio_umap,
    help="La mappa 3D è una proiezione: senza questa opzione due etichette dello stesso "
         "cluster possono apparire lontane anche se sono simili. Non serve (ed è disattivata) "
         "quando il clustering è fatto sulle coordinate UMAP 3D: lì i cluster sono già i gruppi della mappa.",
) and not spazio_umap


if spazio_umap:
    # Prima la mappa (senza suggerimenti: i cluster non ci sono ancora), poi i cluster sulle sue coordinate
    coordinate = proietta_umap(vettori, n_vicini, min_dist, spread)
    cluster_id, nomi_cluster, minimo_menzioni = trova_cluster(
        coordinate, pesi, testi_etichette, quota_outlier / 100,
        probabilita_minima=probabilita_minima,
        distinzione_minima=distinzione_minima,
        metrica="euclidea",
    )
else:
    cluster_id, nomi_cluster, minimo_menzioni = trova_cluster(
        vettori, pesi, testi_etichette, quota_outlier / 100,
        probabilita_minima=probabilita_minima,
        distinzione_minima=distinzione_minima,
        metrica="coseno",
    )
    coordinate = proietta_umap(
        vettori, n_vicini, min_dist, spread,
        cluster_id if allinea_mappa and (cluster_id != -1).any() else None,
    )
fuori = cluster_id == -1
info_outlier.caption(
    f"Outlier max **{quota_outlier:.0f}%** · Prob. **{probabilita_minima:.2f}** · Dist. **{distinzione_minima:.2f}**  \n"
    f"Outlier effettivi: **{pesi[fuori].sum() / pesi.sum():.1%}** delle menzioni ({int(fuori.sum())} etichette)"
)
mappa_etichette = pd.DataFrame({
    "etichetta": testi_etichette, "menzioni": pesi,
    "x": coordinate[:, 0], "y": coordinate[:, 1], "z": coordinate[:, 2],
    "cluster_id": cluster_id, "somiglianza_al_centro": somiglianza_al_centro(vettori, pesi, cluster_id),
})

if (cluster_id != -1).any():
    try:
        with st.spinner("✨ Il modello sta dando un nome ai cluster..."):
            nomi_cluster = nomi_dal_modello(
                mappa_etichette, nomi_cluster, chiave, df,
                vettori, pesi, vettori_neutre, posizione_neutra,
            )
    except Exception as errore:
        st.sidebar.warning(f"Nomi dal modello non disponibili ({errore}): uso quelli provvisori.")
mappa_etichette["tema_cluster"] = mappa_etichette["cluster_id"].map(nomi_cluster)
colori = colori_dei_cluster(set(cluster_id.tolist()))

# Qualità del clustering, nello spazio in cui è stato fatto (stesse distanze, già in cache)
vettori_clustering, metrica_clustering = (coordinate, "euclidea") if spazio_umap else (vettori, "coseno")
distanze_clustering = adatta_hdbscan(vettori_clustering, metrica_clustering)[0]
qualita = misura_qualita(vettori_clustering, distanze_clustering, pesi, cluster_id)
qualita_cluster = qualita_per_cluster(distanze_clustering, cluster_id, pesi, mappa_etichette)

# Ogni menzione eredita posizione, cluster e somiglianza della sua etichetta
df = df.merge(mappa_etichette.drop(columns="menzioni"), on="etichetta")

# --- 4. Filtri -----------------------------------------------
st.sidebar.header("🔍 Filtri")
sentiment_presenti = [s for s in COLORI_SENTIMENT if (df["sentiment"] == s).any()]
sentiment_scelti = st.sidebar.multiselect("Sentiment:", sentiment_presenti, default=sentiment_presenti)

temi = sorted(t for t in df["tema_cluster"].unique() if not (nascondi_outlier and t == NOME_OUTLIER))
cluster_scelti = st.sidebar.multiselect("Cluster:", temi, default=temi)
filtro = df["sentiment"].isin(sentiment_scelti) & df["tema_cluster"].isin(cluster_scelti)
df_filtrato = df[filtro]


# --- 5. Numeri chiave ----------------------------------------
tot = len(df_filtrato)
n_pos = int((df_filtrato["sentiment"] == "Positivo").sum())
n_neg = int((df_filtrato["sentiment"] == "Negativo").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Menzioni", tot)
c2.metric("Cluster", df_filtrato.loc[df_filtrato["cluster_id"] != -1, "cluster_id"].nunique())
c3.metric("🟢 Positive", n_pos)
c4.metric("🔴 Negative", n_neg)
c5.metric("Saldo sentiment", f"{(n_pos - n_neg) / tot:+.0%}" if tot else "–",
          help="(positive − negative) / menzioni: da −100% a +100%")
esclusi = []
if n_senza_frammento:
    esclusi.append(f"{n_senza_frammento} righe senza frammento")
if n_senza_etichetta:
    esclusi.append(f"{n_senza_etichetta} frammenti senza categoria né sottocategoria")
if esclusi:
    st.caption(f"ℹ️ Non sono in mappa: {'; '.join(esclusi)}.")

with st.expander("​🔭​ Composizione dei cluster (quali etichette contiene ognuno)"):
    riquadro_guida(
        "Composizione dei cluster",
        "Vedere quali etichette sono finite in ogni cluster e quanto sono centrali rispetto al suo tema.",
        "Una riga per etichetta: cluster, etichetta, menzioni totali (senza filtri) e <b>somiglianza al "
        "centro</b> (coseno sugli embedding): <b>1</b> = l'etichetta coincide con il tema del cluster, "
        "valori bassi = etichetta ai margini. Gli outlier non hanno un centro.",
        "Clicca sull'intestazione di una colonna per ordinare. Se molte etichette di un cluster hanno "
        "somiglianza bassa, il cluster è eterogeneo: prova ad alzare <b>HDBSCAN tuning</b> nella barra laterale.",
    )
    st.dataframe(
        mappa_etichette.sort_values(["cluster_id", "menzioni"], ascending=[True, False])
        [["tema_cluster", "etichetta", "menzioni", "somiglianza_al_centro"]]
        .style.format({"somiglianza_al_centro": "{:.2f}"}, na_rep="–"),
        hide_index=True, **LARGO,
    )

with st.expander("📐 Qualità del clustering (quanto sono netti e coerenti i cluster)"):
    nome_spazio = ("coordinate UMAP 3D, distanza euclidea" if spazio_umap
                   else f"embedding completi ({DIM_CLOUD} dim.), distanza del coseno")
    riquadro_guida(
        "Qualità del clustering",
        "Capire se i cluster sono gruppi davvero distinti e coerenti o divisioni poco nette, e confrontare "
        "tra loro impostazioni diverse dei cursori della barra laterale.",
        f"Metriche calcolate su <b>tutte le etichette</b> (i filtri non contano) nello spazio del clustering: "
        f"<b>{nome_spazio}</b>. Gli outlier sono esclusi; ogni etichetta conta una volta, tranne nella "
        "copertura, che conta le menzioni. Sotto, le stesse misure cluster per cluster.",
        "Guarda soprattutto <b>Silhouette</b> (più alta è meglio) e <b>Davies-Bouldin</b> (più basso è meglio), "
        "insieme alla <b>copertura</b>: una configurazione è migliore se alza la prima e abbassa il secondo "
        "senza far crollare la copertura. I giudizi sono soglie indicative, non verdetti.",
    )
    if not qualita:
        st.info("Servono almeno due cluster con più di un'etichetta per calcolare le metriche.")
    else:
        tessere = [
            ("Copertura", percentuale(qualita["copertura"]), *giudizio_copertura(qualita["copertura"]),
             "Quota di menzioni che finisce in un cluster; il resto è tra gli outlier."),
            ("Silhouette", decimale(qualita["silhouette"]), *giudizio_silhouette(qualita["silhouette"]),
             "Da −1 a 1: quanto ogni etichetta è più vicina al suo cluster che al più vicino degli altri. "
             "Più alta è meglio."),
            ("Davies-Bouldin", decimale(qualita["davies_bouldin"]),
             *giudizio_davies_bouldin(qualita["davies_bouldin"]),
             "Da 0 in su: dispersione interna dei cluster rispetto alla distanza tra loro. Più basso è meglio."),
            ("Calinski-Harabasz", f"{qualita['calinski_harabasz']:,.0f}".replace(",", "."), None, None,
             "Separazione tra cluster rispetto alla dispersione interna. Non ha una scala assoluta: serve solo "
             "a confrontare impostazioni diverse sugli stessi dati. Più alto è meglio."),
        ]

    ))
    
        st.html(tessere_qualita_html(tessere))
        st.markdown("**Cluster per cluster**")
        st.dataframe(
            qualita_cluster, hide_index=True, **LARGO,
            column_config={
                "cluster": st.column_config.TextColumn("Cluster"),
                "etichette": st.column_config.NumberColumn("Etichette"),
                "menzioni": st.column_config.NumberColumn("Menzioni"),
                "silhouette": st.column_config.NumberColumn(
                    "Silhouette media", format="%.2f",
                    help="Media della silhouette delle etichette del cluster: bassa o negativa = cluster "
                         "poco separato dagli altri. Vuota per i cluster di una sola etichetta."),
                "coesione": st.column_config.ProgressColumn(
                    "Coesione semantica", min_value=0.0, max_value=1.0, format="%.2f",
                    help="Somiglianza media (coseno, sugli embedding) delle etichette al centro del cluster, "
                         "pesata per menzioni: vicina a 1 = tema compatto."),
            },
        )

if tot == 0:
    st.warning("Nessuna menzione con i filtri scelti.")
    st.stop()

# --- 6. Mappa 3D ---------------------------------------------
if st.session_state.get("grafici_senza_clic"):
    st.warning(f"Plotly.js non si è potuto scaricare ({st.session_state.get('errore_grafici')}): "
               "i grafici funzionano, ma senza il clic per l'analisi con l'AI.")
riquadro_guida(
    "Mappa 3D delle costellazioni",
    "Mostrare a colpo d'occhio di quali temi parlano i clienti, quanto pesa ognuno e quali sono vicini "
    "per significato.",
    "Ogni sfera è un'<b>etichetta</b> («categoria › sottocategoria»): la grandezza indica le menzioni, il colore "
    "il cluster. I filamenti la legano alla <b>sfera scura</b> al centro del suo cluster; le sfere grigie sono "
    "<b>outlier</b> (temi isolati). La posizione viene dagli embedding proiettati in 3D (UMAP): vicino = simile.",
    "Trascina per ruotare, usa la rotella per lo zoom; i pulsanti in alto a destra riportano la vista iniziale. "
    "Passa il mouse su una sfera per cluster, etichetta, menzioni e sentiment, sulla sfera scura per il totale "
    "del cluster. <b>Clicca</b> su una sfera per l'analisi con l'AI ✨. Conta chi sta vicino a chi: le distanze "
    "tra costellazioni lontane sono solo indicative.",
)
clic = grafico_cliccabile(crea_mappa_3d(punti_della_mappa(df_filtrato), colori), "mappa_3d")

# --- 7. Grafici per cluster ----------------------------------
st.subheader("📊 I cluster a confronto")
riepilogo = df_filtrato.groupby("tema_cluster").agg(
    menzioni=("sentiment", "size"),
    positive=("sentiment", lambda s: (s == "Positivo").sum()),
    negative=("sentiment", lambda s: (s == "Negativo").sum()),
    neutre=("sentiment", lambda s: (s == "Neutro").sum()),
).reset_index()
riepilogo["saldo"] = (riepilogo["positive"] - riepilogo["negative"]) / riepilogo["menzioni"]

scheda_priorita, scheda_bilancio, scheda_quote = st.tabs(["🎯 Priorità", "📊 Bilancio", "🍩 Quote"])

with scheda_priorita:
    riquadro_guida(
        "Priorità di intervento",
        "Capire su quali temi intervenire prima, incrociando quanto se ne parla e con che tono.",
        "Ogni sfera è un <b>cluster</b>: in orizzontale le menzioni, in verticale il <b>saldo sentiment</b> "
        "((positive − negative) / menzioni, da −100% a +100%). Grandezza = menzioni, colore = saldo (rosso "
        "negativo, verde positivo). La griglia tratteggiata è il saldo zero; la profondità serve solo a non "
        "sovrapporre le sfere e non ha significato.",
        "<b>In basso a destra</b>: temi molto citati e negativi, la priorità più alta. <b>In alto a destra</b>: "
        "punti di forza da difendere. <b>A sinistra</b>: temi di nicchia. Trascina per ruotare, passa il mouse "
        "per i dettagli, <b>clicca</b> su una sfera per l'analisi del cluster con l'AI ✨.",
    )
    clic = grafico_cliccabile(crea_priorita_3d(riepilogo), "priorita_3d") or clic

with scheda_bilancio:
    barre = df_filtrato.groupby(["tema_cluster", "sentiment"]).size().reset_index(name="menzioni")
    # Le tre quote mostrate passando il mouse su una porzione di barra
    barre["quota_cluster"] = barre["menzioni"] / barre.groupby("tema_cluster")["menzioni"].transform("sum")
    barre["quota_sentiment"] = barre["menzioni"] / barre.groupby("sentiment")["menzioni"].transform("sum")
    barre["quota_totale"] = barre["menzioni"] / tot
    ordine = riepilogo.sort_values("menzioni")["tema_cluster"].tolist()  # dal basso: il più citato in alto
    riquadro_guida(
        "Bilancio del sentiment per cluster",
        "Confrontare sulla stessa scala il volume e la composizione del sentiment di tutti i cluster.",
        "Una barra per cluster, dal più citato in alto: la lunghezza è il numero di menzioni, i colori le "
        "menzioni positive, negative, neutre e non deducibili.",
        "Guarda insieme la <b>lunghezza</b> (quanto pesa il tema) e la <b>quota di rosso</b> (quanto è critico). "
        "Passa il mouse su una porzione per le quote nel cluster e sul totale; clicca su una voce della legenda "
        "per nascondere un sentiment; <b>clicca</b> su una porzione per analizzare con l'AI quel sentiment del "
        "cluster ✨.",
    )
    fig = go.Figure()
    for sentiment in COLORI_SENTIMENT:   # una traccia per sentiment, impilate
        parte = barre[barre["sentiment"] == sentiment]
        if parte.empty:
            continue
        fig.add_trace(go.Bar(
            x=parte["menzioni"], y=parte["tema_cluster"], orientation="h", name=sentiment,
            marker_color=COLORI_SENTIMENT[sentiment],
            customdata=parte[["quota_cluster", "quota_sentiment", "quota_totale"]].to_numpy(),
            hovertemplate=(
                in_grande("<b>🪐 %{y}</b>", HOVER_DIM_TITOLO) + "<br>" + SEPARATORE_HOVER + "<br>"
                + in_grande(f"{EMOJI_SENTIMENT[sentiment]} {sentiment}: <b>%{{x}}</b> menzioni", HOVER_DIM_VOCE)
                + "<br>Quota nel cluster: <b>%{customdata[0]:.1%}</b><br>"
                + f"Quota sul totale «{sentiment}»: <b>%{{customdata[1]:.1%}}</b><br>"
                + "Quota sul totale complessivo: <b>%{customdata[2]:.1%}</b>"
                + "<extra></extra>"
            ),
            meta=[{"cluster": t, "sentiment": sentiment} for t in parte["tema_cluster"]],  # bersaglio del clic
        ))
    fig.update_layout(
        barmode="stack", bargap=0.35, template="plotly_dark", height=max(320, 36 * len(ordine) + 120),
        xaxis_title="", yaxis=dict(title="", categoryorder="array", categoryarray=ordine),
        legend_title_text="", legend_traceorder="normal", paper_bgcolor=SFONDO, plot_bgcolor=SFONDO,
        hoverlabel=HOVERLABEL,
    )
    clic = grafico_cliccabile(fig, "bilancio") or clic

with scheda_quote:
    # Cluster in ordine di numero (C01 per primo), outlier in fondo; si parte dal primo
    riquadro_guida(
        "Quote di sentiment di un cluster",
        "Scendere nel dettaglio di un singolo cluster: quali etichette generano i giudizi positivi e quali i negativi.",
        "Una ciambella con le quote di sentiment del cluster scelto; ogni fetta è divisa nelle etichette che "
        "la compongono (dalla più citata, in senso orario). Accanto, l'elenco delle etichette con le menzioni "
        "per sentiment.",
        "Scegli il cluster dal menu. Le sezioni più ampie dentro la fetta rossa sono le cause principali delle "
        "critiche, quelle nella verde i motivi di soddisfazione. Passa il mouse per etichetta, numeri ed esempi; "
        "<b>clicca</b> su una sezione per l'analisi con l'AI ✨.",
    )
    numero_tema = df_filtrato.groupby("tema_cluster")["cluster_id"].first()
    temi_in_ordine = sorted(numero_tema.index, key=lambda t: (numero_tema[t] == -1, numero_tema[t]))
    fuoco = st.selectbox("Mostra le quote per:", temi_in_ordine, index=0)
    sorgente = df_filtrato[df_filtrato["tema_cluster"] == fuoco]
    col_torta, col_elenco = st.columns([3, 2], gap="medium")
    with col_torta:
        clic = grafico_cliccabile(crea_torta_quote(sorgente), "torta_quote") or clic
    with col_elenco:
        st.markdown(elenco_etichette_html(sorgente), unsafe_allow_html=True)

# --- 8. Citazioni ed esportazione ----------------------------
st.divider()
st.subheader("💬 I commenti dei clienti")
riquadro_guida(
    "Commenti ed esportazione",
    "Leggere i commenti originali dietro ai numeri e portarli fuori dall'app.",
    "Le menzioni del cluster scelto, con i filtri attivi, dalla più vicina al tema del cluster alla più "
    "marginale: id del commento, categoria, sottocategoria, frammento e sentiment.",
    "Scegli il cluster dal menu; clicca sull'intestazione di una colonna per ordinare, usa la lente in alto a "
    "destra della tabella per cercare. <b>Esporta CSV</b> scarica tutte le menzioni filtrate con il loro "
    "cluster, pronte per Excel (separatore «;»).",
)
COLONNE_TABELLA = ["id_commento", "categoria", "sottocategoria", "frammento", "sentiment"]
col_scelta, col_bottone = st.columns([3, 1], vertical_alignment="bottom")
with col_scelta:
    cluster_da_leggere = st.selectbox("🔎 Scegli un cluster per leggere i commenti:",
                                      riepilogo.sort_values("menzioni", ascending=False)["tema_cluster"])
with col_bottone:
    st.download_button(
        "📥 Esporta CSV per Excel",
        data=df_filtrato[COLONNE_TABELLA + ["tema_cluster"]].to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name=FILE_ESPORTAZIONE, mime="text/csv", **LARGO,
    )
st.dataframe(
    df_filtrato.loc[df_filtrato["tema_cluster"] == cluster_da_leggere]
    .sort_values("somiglianza_al_centro", ascending=False)[COLONNE_TABELLA],
    hide_index=True, **LARGO,
)

# --- 9. Popup dell'analisi con l'AI ---------------------------
# Un clic su un grafico fa ripartire la pagina con il bersaglio cliccato: qui si apre il popup
if clic:
    esclusi_sentiment = [s for s in sentiment_presenti if s not in sentiment_scelti]
    nota_filtri = (f"Filtro attivo nell'app: escluse le menzioni con sentiment {', '.join(esclusi_sentiment)}"
                   if esclusi_sentiment else "")
    finestra_analisi(clic, df_filtrato, nota_filtri, chiave)
