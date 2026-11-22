"""
Collegamento al modello linguistico e agli embedding, tramite OpenRouter.
Lo usano main.py (fasi 1 e 2) e app.py (embedding, nomi dei cluster e analisi AI).

Serve:  pip install openai
e una API key di OpenRouter nel file keys.txt (oppure nella variabile d'ambiente
OPENROUTER_API_KEY).

PROMPT CACHING (automatico con OpenAI su OpenRouter)
Se l'inizio di una richiesta è identico a quello di una richiesta recente, il provider lo
rilegge dalla cache e lo fa pagare molto meno. Funziona solo sul PREFISSO, quindi le
richieste sono costruite sempre così:
    istruzioni fisse  ->  parti che cambiano di rado  ->  parte nuova (le recensioni)
riepilogo_uso() mostra quanti token sono arrivati dalla cache e quanto si è speso.
Ogni chiamata (chat ed embedding) viene inoltre registrata in FILE_LOG_API (log_chiamate_ai.csv).
"""

import csv
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime

from context import CONTESTO


# ============================================================
# IMPOSTAZIONI (modifica qui)
# ============================================================

INDIRIZZO = "https://openrouter.ai/api/v1"

# Modelli linguistici (OpenAI su OpenRouter, in precisione nativa senza quantizzazione):
# - MODELLO: usato nella fase 1 di main.py (estrazione frammenti) e in app.py
# - MODELLO_FASE2: usato nella fase 2 di main.py (consolidamento tassonomico con ragionamento)
MODELLO = "openai/gpt-6-luna"
MODELLO_FASE2 = "openai/gpt-6-luna-pro"

# Provider OpenAI su OpenRouter in ordine di preferenza
PROVIDER = {
    "order": ["OpenAI", "Azure"],
    "allow_fallbacks": False,
}

# File con la API key di OpenRouter, nella stessa cartella di questo file.
# Deve contenere solo la chiave. NON condividerlo e non caricarlo su GitHub.
FILE_CHIAVE = "keys.txt"

# Registro CSV dettagliato di tutte le chiamate ai modelli (chat ed embedding)
FILE_LOG_API = "log_chiamate_ai.csv"

# Tentativi in caso di errori passeggeri (troppe richieste, provider sovraccarico, attesa
# scaduta...): la libreria openai riprova da sola, aspettando un po' di più ogni volta
TENTATIVI = 10

# Attesa massima di una risposta, in secondi. Il valore predefinito della libreria openai è
# 10 minuti: troppo poco per la fase 2 di main.py, che in una sola risposta assegna TUTTE le
# etichette (con il ragionamento acceso può durare parecchi minuti). Scaduta l'attesa, la
# richiesta riparte da capo e i token già generati si pagano lo stesso.
# 60 minuti bastano anche per una risposta con ragionamento lungo.
ATTESA_MASSIMA = 60 * 60

# Token massimi di ogni risposta, ragionamento compreso. Si pagano solo quelli usati davvero.
# GPT-6 Luna e GPT-6 Luna Pro supportano fino a 128.000 token di output su OpenRouter.
# - MAX_TOKEN_RISPOSTA: fase 1 e app.py (gruppi piccoli, ragionamento spento: 32.768 basta);
# - MAX_TOKEN_RISPOSTA_FASE2: fase 2, che con il ragionamento acceso assegna TUTTE le etichette
#   in una sola risposta: con un limite basso rischia di essere troncata e di fallire del tutto.
# Se cambi modello, ricontrolla: https://openrouter.ai/api/v1/models/<modello>/endpoints
MAX_TOKEN_RISPOSTA = 64000
MAX_TOKEN_RISPOSTA_FASE2 = 128000


# ============================================================
# API KEY E CLIENT
# ============================================================

CARTELLA_BASE = os.path.dirname(os.path.abspath(__file__))
PERCORSO_CHIAVE = os.path.join(CARTELLA_BASE, FILE_CHIAVE)
PERCORSO_LOG_API = os.path.join(CARTELLA_BASE, FILE_LOG_API)
SORGENTE = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""


def chiave_valida(chiave):
    """Una chiave valida ha almeno 20 caratteri: solo lettere, numeri, "-" e "_"."""
    return len(chiave) >= 20 and all(c.isascii() and (c.isalnum() or c in "-_") for c in chiave)


def leggi_chiave():
    """La chiave dal file FILE_CHIAVE o, in mancanza, dalla variabile OPENROUTER_API_KEY ("" se non c'è)."""
    if os.path.isfile(PERCORSO_CHIAVE):
        # utf-8-sig ignora il carattere invisibile che il Blocco note a volte mette all'inizio
        with open(PERCORSO_CHIAVE, encoding="utf-8-sig") as f:
            chiave = f.read().strip()
        if chiave_valida(chiave):
            return chiave
        print(f"ATTENZIONE: la chiave nel file {FILE_CHIAVE} non è valida, controlla il file.")
    chiave = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return chiave if chiave_valida(chiave) else ""


def salva_chiave(chiave):
    """Salva la chiave nel file, così la prossima volta non serve incollarla (vale per main.py e app.py)."""
    with open(PERCORSO_CHIAVE, "w", encoding="utf-8") as f:
        f.write(chiave + "\n")


def mostra_nuovi_tentativi():
    """Fa stampare a schermo i nuovi tentativi della libreria openai, che altrimenti sono silenziosi."""
    import logging
    registro = logging.getLogger("openai")
    if not registro.handlers:
        gestore = logging.StreamHandler()
        gestore.setFormatter(logging.Formatter("   (openai) %(message)s"))
        registro.addHandler(gestore)
        registro.setLevel(logging.INFO)   # "Retrying request to ... in ... seconds"


def crea_client(chiave):
    from openai import OpenAI
    mostra_nuovi_tentativi()
    return OpenAI(base_url=INDIRIZZO, api_key=chiave, max_retries=TENTATIVI, timeout=ATTESA_MASSIMA)


def con_contesto(istruzioni):
    """Mette in testa alle istruzioni il contesto scritto in context.py (se c'è)."""
    if not CONTESTO.strip():
        return istruzioni
    return f"CONTESTO\n{CONTESTO.strip()}\n\n{istruzioni}"


def errore_bloccante(errore):
    """True se riprovare è inutile: chiave sbagliata (401), credito finito (402), accesso negato (403)."""
    return getattr(errore, "status_code", None) in (401, 402, 403)


# ============================================================
# TELEMETRIA E LOG CSV DELLE CHIAMATE AI
# ============================================================

COLONNE_LOG_API = [
    "timestamp",
    "sorgente",
    "operazione",
    "tipo",
    "modello",
    "modello_risposta",
    "provider",
    "id_richiesta",
    "modo",
    "esito",
    "finish_reason",
    "elementi_inviati",
    "caratteri_input",
    "durata_sec",
    "token_output_sec",
    "token_input",
    "token_cache_letti",
    "quota_cache",
    "token_cache_scritti",
    "token_output",
    "token_ragionamento",
    "token_totali",
    "costo_usd",
]
_lock_log = threading.Lock()


def estrai_metriche_uso(u):
    """Estrae token (input, cache, output, reasoning, totali) e costo in USD dall'oggetto usage."""
    if not u:
        return {
            "token_input": None,
            "token_cache_letti": None,
            "quota_cache": None,
            "token_cache_scritti": None,
            "token_output": None,
            "token_ragionamento": None,
            "token_totali": None,
            "costo_usd": None,
        }
    dettagli_input = getattr(u, "prompt_tokens_details", None)
    dettagli_output = getattr(u, "completion_tokens_details", None)

    t_in = getattr(u, "prompt_tokens", None)
    t_cache_letti = getattr(dettagli_input, "cached_tokens", None)
    t_cache_scritti = getattr(dettagli_input, "cache_write_tokens", None)
    t_out = getattr(u, "completion_tokens", None)
    t_reas = getattr(dettagli_output, "reasoning_tokens", None)
    t_tot = getattr(u, "total_tokens", None)
    if t_tot is None and t_in is not None:
        t_tot = t_in + (t_out or 0)
    costo = getattr(u, "cost", None)
    quota_cache = round(t_cache_letti / t_in, 4) if (t_in and t_cache_letti is not None) else None

    return {
        "token_input": t_in,
        "token_cache_letti": t_cache_letti,
        "quota_cache": quota_cache,
        "token_cache_scritti": t_cache_scritti,
        "token_output": t_out,
        "token_ragionamento": t_reas,
        "token_totali": t_tot,
        "costo_usd": f"{float(costo):.6f}" if isinstance(costo, (int, float)) else costo,
    }


def scrivi_log_api(riga):
    """Appende una riga a FILE_LOG_API (creando l'intestazione se il file non esiste)."""
    with _lock_log:
        scrivi_intestazione = not os.path.exists(PERCORSO_LOG_API) or os.path.getsize(PERCORSO_LOG_API) == 0
        with open(PERCORSO_LOG_API, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLONNE_LOG_API)
            if scrivi_intestazione:
                writer.writeheader()
            writer.writerow({k: "" if riga.get(k) is None else riga.get(k) for k in COLONNE_LOG_API})


# ============================================================
# DIALOGO CON IL MODELLO
# ============================================================

uso = Counter()           # token e spesa delle chiamate chat nella sessione corrente
uso_embedding = Counter() # token e spesa delle chiamate embedding nella sessione corrente
serviti_da = Counter()      # quante richieste ha servito ogni provider
ultima_risposta = {}       # l'ultima risposta così com'è arrivata (main.py la scrive nel file di debug)


def rigido(schema):
    """Aggiunge "additionalProperties": false a ogni oggetto dello schema (lo chiede la modalità strict)."""
    if isinstance(schema, dict):
        schema = {k: rigido(v) for k, v in schema.items()}
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
    elif isinstance(schema, list):
        schema = [rigido(v) for v in schema]
    return schema


def leggi_json(testo):
    """Il JSON della risposta, anche se il modello l'ha messo tra ```json ... ``` o con testo intorno."""
    try:
        return json.loads(testo)
    except json.JSONDecodeError as primo_errore:
        decoder = json.JSONDecoder()
        indice = 0
        while (inizio := testo.find("{", indice)) != -1:
            try:
                oggetto, _ = decoder.raw_decode(testo, inizio)
                if isinstance(oggetto, dict):
                    return oggetto
            except json.JSONDecodeError:
                pass
            indice = inizio + 1
        raise primo_errore


# Modi di fare la richiesta, dal più completo al più semplice: (formato della risposta,
# parametro di ragionamento sì/no, limite di token sì/no). Non tutti i provider accettano tutti
# i parametri: se OpenRouter scarta il provider per un parametro ("Filter by Parameters" o 400
# sui parametri strutturati/ragionamento), si passa al modo successivo e si continua col primo che funziona.
MODI = [
    ("json_schema", True, True),
    ("json_schema", False, True),
    ("json_object", True, True),
    ("json_object", False, True),
    (None, True, True),
    (None, False, True),
    (None, False, False),
]
# Il modo in uso, ricordato SEPARATAMENTE per ogni modello: se il modello della fase 1 rifiuta
# un parametro, il modello della fase 2 non deve perderlo anche lui (es. il ragionamento).
_modo_per_modello = {}
_modo = 0  # l'ultimo modo usato (solo informativo)


def descrivi_modo(modo):
    formato, con_ragionamento, con_limite = modo
    return (f"formato {formato or 'libero'}, "
            f"{'con' if con_ragionamento else 'senza'} parametro di ragionamento, "
            f"{'con' if con_limite else 'senza'} limite di token")


def parametri_rifiutati(errore):
    stato = getattr(errore, "status_code", None)
    messaggio = str(errore).lower()
    if stato == 404 and "filter by parameters" in messaggio:
        return True
    if stato == 400 and any(k in messaggio for k in (
        "response_format", "json_schema", "json_object", "strict", "anyof", "reasoning"
    )):
        return True
    return False


def chiedi(client, istruzioni, testo, schema, ragionamento=None, operazione="", elementi=None, modello=None,
           max_token=None):
    """
    Manda istruzioni + testo al modello e restituisce la risposta come dizionario.
    - 'schema' è la forma del JSON da restituire;
    - 'ragionamento': True (acceso), False (spento) o None (predefinito del modello);
    - 'operazione' e 'elementi': metadati scritti in FILE_LOG_API;
    - 'modello': slug del modello da usare (se None, usa MODELLO);
    - 'max_token': token massimi della risposta, ragionamento compreso (se None, MAX_TOKEN_RISPOSTA).
    Le istruzioni vanno nel messaggio di sistema, che è l'inizio di ogni richiesta: tenerle
    identiche tra una chiamata e l'altra è ciò che fa scattare la cache.
    """
    global _modo
    modello_attivo = modello or MODELLO
    limite = max_token or MAX_TOKEN_RISPOSTA
    modo = _modo_per_modello.get(modello_attivo, 0)
    ultima_risposta.clear()
    while True:
        try:
            risposta = invia(client, istruzioni, testo, schema, ragionamento, MODI[modo],
                             operazione=operazione, elementi=elementi, modello=modello_attivo,
                             max_token=limite)
            break
        except Exception as errore:
            if not parametri_rifiutati(errore) or modo == len(MODI) - 1:
                raise
            modo += 1
            _modo_per_modello[modello_attivo] = modo
            print(f"   ({modello_attivo} non accetta tutti i parametri: provo con {descrivi_modo(MODI[modo])})")
            if ragionamento is not None and not MODI[modo][1]:
                print(f"   ATTENZIONE: da ora il parametro di ragionamento non viene inviato a {modello_attivo}: "
                      f"vale il comportamento predefinito del modello (controlla token_ragionamento in {FILE_LOG_API})")
    _modo = modo

    scelta = risposta.choices[0] if risposta.choices else None
    ultima_risposta.update({
        "modello": modello_attivo,
        "provider": getattr(risposta, "provider", None) or "?",
        "modo": descrivi_modo(MODI[modo]),
        "finish_reason": scelta.finish_reason if scelta else None,
        "token_input": getattr(risposta.usage, "prompt_tokens", None),
        "token_output": getattr(risposta.usage, "completion_tokens", None),
        "testo": scelta.message.content if scelta else None,
    })
    if not scelta:
        raise RuntimeError("risposta vuota dal provider")
    if scelta.finish_reason == "length":
        raise RuntimeError(f"risposta troncata: superato il limite di {limite} token di risposta "
                           f"(MAX_TOKEN_RISPOSTA o MAX_TOKEN_RISPOSTA_FASE2 in llm.py)")
    if not scelta.message.content:
        raise RuntimeError(f"risposta vuota (motivo: {scelta.finish_reason})")
    return leggi_json(scelta.message.content)


def invia(client, istruzioni, testo, schema, ragionamento, modo, operazione="", elementi=None, modello=None,
          max_token=None):
    """Una richiesta a OpenRouter nel modo indicato (vedi MODI)."""
    modello_attivo = modello or MODELLO
    formato, con_ragionamento, con_limite = modo
    parametri = {}
    if formato == "json_schema":
        parametri["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "risposta", "strict": True, "schema": rigido(schema)}}
    else:
        # Lo schema non è imposto dal provider: lo descrivo in fondo alle istruzioni
        # (sempre uguale per ogni fase, quindi la cache continua a funzionare)
        istruzioni += ("\n\nFORMATO DELLA RISPOSTA\nRispondi solo con un oggetto JSON conforme a "
                       "questo JSON Schema, senza altro testo:\n" + json.dumps(schema, ensure_ascii=False))
        if formato == "json_object":
            parametri["response_format"] = {"type": "json_object"}
    extra = {"provider": PROVIDER}
    if ragionamento is not None and con_ragionamento:
        # exclude: il testo del ragionamento non serve, quindi non lo faccio rimandare
        extra["reasoning"] = {"enabled": ragionamento, "exclude": True}
    if con_limite:
        parametri["max_tokens"] = max_token or MAX_TOKEN_RISPOSTA

    caratteri_in = len(istruzioni) + len(testo)
    modo_desc = descrivi_modo(modo)
    t0 = time.perf_counter()
    try:
        risposta = client.chat.completions.create(
            model=modello_attivo,
            messages=[{"role": "system", "content": istruzioni}, {"role": "user", "content": testo}],
            extra_body=extra,
            **parametri,
        )
    except Exception as errore:
        durata = round(time.perf_counter() - t0, 3)
        scrivi_log_api({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sorgente": SORGENTE,
            "operazione": operazione,
            "tipo": "chat",
            "modello": modello_attivo,
            "modo": modo_desc,
            "esito": f"errore: {type(errore).__name__}: {errore}",
            "elementi_inviati": elementi,
            "caratteri_input": caratteri_in,
            "durata_sec": durata,
        })
        raise

    durata = round(time.perf_counter() - t0, 3)
    registra_uso(risposta.usage)
    provider = getattr(risposta, "provider", None) or "?"
    serviti_da[provider] += 1

    scelta = risposta.choices[0] if risposta.choices else None
    finish_reason = scelta.finish_reason if scelta else None
    metriche = estrai_metriche_uso(risposta.usage)
    t_out = metriche["token_output"]
    tok_sec = round(t_out / durata, 1) if (t_out and durata > 0) else None
    esito = "ok" if (scelta and scelta.message.content and finish_reason != "length") else f"anomalo ({finish_reason})"

    scrivi_log_api({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sorgente": SORGENTE,
        "operazione": operazione,
        "tipo": "chat",
        "modello": modello_attivo,
        "modello_risposta": getattr(risposta, "model", None),
        "provider": provider,
        "id_richiesta": getattr(risposta, "id", None),
        "modo": modo_desc,
        "esito": esito,
        "finish_reason": finish_reason,
        "elementi_inviati": elementi,
        "caratteri_input": caratteri_in,
        "durata_sec": durata,
        "token_output_sec": tok_sec,
        **metriche,
    })
    return risposta


def registra_uso(u, tipo="chat"):
    if not u:
        return
    target = uso if tipo == "chat" else uso_embedding
    dettagli_input = getattr(u, "prompt_tokens_details", None)
    dettagli_output = getattr(u, "completion_tokens_details", None)
    target["richieste"] += 1
    target["input"] += u.prompt_tokens or 0                         # comprende quelli dalla cache
    target["input_da_cache"] += getattr(dettagli_input, "cached_tokens", 0) or 0
    target["output"] += getattr(u, "completion_tokens", 0) or 0       # comprende il ragionamento
    target["ragionamento"] += getattr(dettagli_output, "reasoning_tokens", 0) or 0
    target["spesa"] += getattr(u, "cost", 0) or 0                     # in dollari, calcolata da OpenRouter


def riepilogo_uso():
    """Una riga con richieste, token, quota letta dalla cache e spesa."""
    parti = []
    if uso["richieste"]:
        quota = uso["input_da_cache"] / uso["input"] if uso["input"] else 0
        numero = lambda n: f"{n:,}".replace(",", ".")
        parti.append(
            f"{uso['richieste']} richieste · input {numero(uso['input'])} token, di cui "
            f"{numero(uso['input_da_cache'])} dalla cache ({quota:.0%}) · output {numero(uso['output'])}, "
            f"di cui ragionamento {numero(uso['ragionamento'])} · spesa ${uso['spesa']:.4f} · "
            f"provider: {', '.join(f'{p} {n}' for p, n in serviti_da.most_common())}"
        )
    if uso_embedding["richieste"]:
        numero = lambda n: f"{n:,}".replace(",", ".")
        parti.append(
            f"embedding: {uso_embedding['richieste']} richieste · input {numero(uso_embedding['input'])} token · "
            f"spesa ${uso_embedding['spesa']:.4f}"
        )
    return " | ".join(parti) if parti else "Nessuna richiesta al modello linguistico."


# ============================================================
# EMBEDDING
# ============================================================

def embedding(client, modello, testi, dimensioni, operazione="embedding"):
    """Un vettore per ogni testo (una sola richiesta), con telemetria salvata in FILE_LOG_API."""
    caratteri_in = sum(len(t) for t in testi)
    modo_desc = f"embedding ({dimensioni} dim)"
    t0 = time.perf_counter()
    try:
        risposta = client.embeddings.create(model=modello, input=testi, dimensions=dimensioni,
                                            encoding_format="float")
    except Exception as errore:
        durata = round(time.perf_counter() - t0, 3)
        scrivi_log_api({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sorgente": SORGENTE,
            "operazione": operazione,
            "tipo": "embedding",
            "modello": modello,
            "modo": modo_desc,
            "esito": f"errore: {type(errore).__name__}: {errore}",
            "elementi_inviati": len(testi),
            "caratteri_input": caratteri_in,
            "durata_sec": durata,
        })
        raise

    durata = round(time.perf_counter() - t0, 3)
    registra_uso(risposta.usage, tipo="embedding")
    provider = getattr(risposta, "provider", None) or "?"
    metriche = estrai_metriche_uso(risposta.usage)

    scrivi_log_api({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sorgente": SORGENTE,
        "operazione": operazione,
        "tipo": "embedding",
        "modello": modello,
        "modello_risposta": getattr(risposta, "model", None),
        "provider": provider,
        "id_richiesta": getattr(risposta, "id", None),
        "modo": modo_desc,
        "esito": "ok",
        "elementi_inviati": len(testi),
        "caratteri_input": caratteri_in,
        "durata_sec": durata,
        **metriche,
    })
    return [voce.embedding for voce in sorted(risposta.data, key=lambda v: v.index)]