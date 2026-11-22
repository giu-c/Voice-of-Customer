"""
TUTTO CIÒ CHE SI ADATTA AL TIPO DI RECENSIONI

È l'unico file da modificare per usare la pipeline con un altro tipo di commenti. Contiene
il contesto, la guida alle etichette e le regole, i consigli e gli esempi di tutti i prompt:
  1. CONTESTO             in testa a tutti i prompt (main.py fasi 1 e 2, app.py)
  2. CONTESTO_EMBEDDING   davanti a ogni testo prima dell'embedding (app.py)
  3. ETICHETTE            guida a categoria e sottocategoria (fase 1)
  4. FASE 1  (main.py)    dividere le recensioni in frammenti e classificarli
  5. FASE 2  (main.py)    rendere coerenti le etichette
  6. NOMI DEI CLUSTER     (app.py)
  7. ANALISI CON L'AI     al clic su un cluster o su una sua parte (app.py)

Nel codice restano solo le parti tecniche dei prompt: la descrizione di ciò che il programma
manda al modello e la forma della risposta. Le impostazioni numeriche (modello, dimensione dei
gruppi, soglie...) restano nella sezione IMPOSTAZIONI di main.py, app.py e llm.py.
"""

# ============================================================
# 1. CONTESTO DELLE RECENSIONI (in testa a tutti i prompt)
# ============================================================

CONTESTO = """\
Recensioni dell'app mobile Poste Italiane, scaricate dal Google Play Store.
Gli utenti valutano l'app e anche i servizi di Poste Italiane in generale.
"""

# ============================================================
# 2. CONTESTO PER GLI EMBEDDING (app.py)
# ============================================================

CONTESTO_EMBEDDING = "recensioni dell'app mobile Poste Italiane e dei relativi servizi"


# ============================================================
# 3. ETICHETTE: CATEGORIE E SOTTOCATEGORIE (fase 1 di main.py)
# ============================================================

# Facoltativo: se il report deve avere sezioni fisse, elencale qui 
CATEGORIE_OBBLIGATORIE = {}

GUIDA_ETICHETTE = """COME SCEGLIERE LE ETICHETTE:
- La categoria dovrebbe rappresentare un topic generale, in 1-4 parole.
- La sottocategoria dovrebbe rappresentare un sotto-topic più specifico, in 1-4 parole.
"""

def guida_etichette():
    """La guida inserita nel prompt della fase 1, preceduta dalle categorie ammesse (se ci sono)."""
    if not CATEGORIE_OBBLIGATORIE:
        return GUIDA_ETICHETTE
    ammesse = "\n".join(f"- {nome}: {descrizione}" for nome, descrizione in CATEGORIE_OBBLIGATORIE.items())
    return ("CATEGORIE AMMESSE (usa solo queste; se il frammento non rientra in nessuna, lascia la categoria vuota)\n"
            + ammesse + "\n\n" + GUIDA_ETICHETTE)


# ============================================================
# 4. FASE 1 DI main.py: DIVIDERE LE RECENSIONI IN FRAMMENTI E CLASSIFICARLI
# ============================================================

FASE1_REGOLE = """Sei un esperto di aspect based sentiment analysis. Isola ogni aspetto o valutazione e fai in modo che sia interpretabile anche senza il resto della recensione: deve quindi far capire da solo di cosa si parla e/o quale giudizio l'utente dia. 
Questi frammenti alimenteranno sistemi di supporto alle decisioni basati su analisi di sentiment, topic modeling e analytics: scarta solo ciò che è sconclusionato o non contiene opinioni. 
Se un frammento è ambiguo, scrivi la menzione neutra senza giudizio. Tieni presente che il modello non deve fare inferenze: se un frammento non dice nulla di chiaro, non puoi inventare il soggetto o il giudizio.
Considera che le recensioni possono contenere ironie, sarcasmo, frasi ambigue e giudizi contraddittori: il tuo compito è estrarre i frammenti con un senso chiaro, senza inventare nulla.

COME DIVIDERE
- Copia ogni frammento esattamente dalla sua recensione.
- Dividi per argomento. Tieni nel frammento tutto ciò che serve a capirlo e interpretarlo.
- Ogni opinione presente nella recensione deve diventare un frammento.
- Ogni frammento è un pezzo continuo e letterale della recensione.
- Una recensione di una sola parola o di sole emoji con un giudizio chiaro NON va scartata: è un frammento di giudizio complessivo. Scarta solo testi senza senso o senza alcun giudizio.
- Una recensione senza opinioni ha "frammenti": [].
- Problemi diversi vanno in frammenti diversi, anche se sono nella stessa frase. 
- Lo stesso aspetto commentato in più parti della recensione resta un solo frammento.
- Categoria e sottocategoria non devono mai menzionare il fatto che ci sia un problema o un giudizio positivo o negativo. 
- La sottocategoria deve essere più specifica della categoria ma allo stesso tempo non deve mai risultare eccessivamente specifica.
- Per la sottocategoria evita ripetizioni rispetto alla categoria.
- Evita il più possibile etichette contenitore, troppo vaghe e generiche.


PER OGNI FRAMMENTO, in quest'ordine
- "testo": il frammento copiato;
- "aspetto": ciò che viene valutato, con i fatti specifici e senza parole di giudizio (3-10 parole);
- "categoria": la categoria (1-4 parole) e "sottocategoria": la sottocategoria (1-4 parole).
- "sentiment": 1 positivo, 0 neutro, misto o semplice richiesta, -1 negativo, null se non deducibile; conta il senso reale, ironie comprese. Se qualcosa non è chiaro lascia il campo vuoto ("") o null: MAI forzare o inventare."""

# ============================================================
# 5. FASE 2 DI main.py: RENDERE COERENTI LE ETICHETTE
# ============================================================

FASE2_REGOLE = """Hai a disposizione tutte le etichette generate dalla precedente fase di labeling: 
hai il compito di analizzarle al fine di eliminare i doppioni, unendo le etichette che descrivono lo stesso aspetto. 
L'obiettivo è ottenere un elenco finale di voci coerenti, chiare e utili per una topic modeling di qualità, 
evitando dunque etichette eccessivamente generiche e allo stesso tempo troppo specifiche. Unisci solo etichette che descrivono lo stesso aspetto; aspetti diversi restano voci diverse anche se affini.
L'elenco finale deve essere strutturato come segue: al massimo {max_voci} voci, ognuna con almeno {min_menzioni} menzioni. 
Ogni voce finale deve poter essere descritta con un solo concetto, senza ad esempio: concetto x "e" concetto y.
Sfrutta al massimo la tua capacità logico-interpretativa e lo spazio di manovra concessoti, entro le regole.

RISPOSTA
- "etichette_finali": le voci del report, ognuna con un numero "n", una categoria "categoria" (1-4 parole) e una sottocategoria "sottocategoria" (1-4 parole).
- "assegnazioni": OGNI etichetta dell'elenco assegnata a una voce, come {"id": numero dell'etichetta, "n": numero della voce}."""

# Esempio per la fase 2 (main.py lo importa: lascialo anche se vuoto)
FASE2_ESEMPIO = ""

# ============================================================
# 6. NOMI DEI CLUSTER (app.py)
# ============================================================

NOMI_REGOLE = """REGOLE:
1. Il nome ha da 1 a 4 parole, in italiano, minuscolo, senza punteggiatura.
2. Deve descrivere ciò che accomuna le etichette del cluster e/o i relativi frammenti.
3. Dai più importanza alle etichette con più opinioni e a quelle in cima all'elenco e/o i frammenti simili più informativi e frequenti.
4. Indica il tema, non il giudizio, ad esempio: "assistenza clienti" e non "assistenza scadente".
5. Ogni cluster deve avere un nome chiaro, esplicativo e discriminante, possibilmente composto da un solo concetto cercando di limitare nomi composti con concetto x e concetto y.
"""


# ============================================================
# 7. ANALISI CON L'AI DI UN CLUSTER O DI UNA SUA PARTE (app.py)
# ============================================================

ANALISI_AI_RUOLO = "Sei un analista e devi fornire un report per il management."

ANALISI_AI_REGOLE = """Scrivi un'analisi breve e concreta, basata SOLO sui dati ricevuti.
- titolo: il messaggio principale, 5-10 parole;
- sintesi: al massimo 80 parole su cosa dicono i clienti e cosa pesa di più;
- punti di forza e criticità: ognuno con la sua evidenza (quante menzioni, cosa dicono); la gravità dipende da quante menzioni e da quanto pesa per il cliente;
- raccomandazioni: dalla più importante, azioni concrete all'imperativo (max 15 parole), ognuna legata a una criticità o a un punto di forza; niente consigli generici;
- spunti: ipotesi o dati da verificare;
- citazioni: frammenti copiati esattamente.
Usa i numeri delle statistiche fornite. Se una sezione non ha evidenze, lasciala vuota. Italiano, tono asciutto.
"""


if __name__ == "__main__":
    import main
    print("=" * 30, "FASE 1", "=" * 30)
    print(main.ISTRUZIONI_ANALISI)
    print("\n" + "=" * 30, "FASE 2", "=" * 30)
    print(main.istruzioni_normalizzazione())