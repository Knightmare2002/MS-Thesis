# Direzione A supervisionata — procedura separata v1

## Stato e installazione
Pacchetto locale additivo: nessun file H3 esistente viene modificato e nessun commit remoto viene creato.
Copia damage3d/supervised_direction_a.py nel package damage3d del repository e gli altri file nella root/configs.
Dipendenze: NumPy, SciPy e moduli damage3d gia presenti. Non richiede nuova inferenza PyTorch.
Interfacce lette da GitHub: experiment_data.py (blob 2d81eda483c07d1305de0f069feaf7ce0ee3d66f),
detectability.py (blob 9a3c11b58152ebd59d2f91a76b7d5a83440692a2),
observations.py (blob 5ba1ff7391943aa69126a16a39bdbb9cc90184fd).
Le modifiche locali non pubblicate non sono state ispezionate.

## Protocollo
- rho e offset del posterior fissati alla prevalenza della calibrazione. base_rate diverso da rho viene rifiutato.
- C_fit: selezione supervisionata g50/beta/alpha con NLL pesata per area.
- C_cal: sole soglie F1, parametri congelati.
- T: valutazione separata, nessuna selezione. Preflight non esporta statistiche delle label T.
- La funzione regions esistente legge l'intero NPZ per validazione strutturale/buffer; non significa che le label T siano usate nel fit.
- surface esclusa esplicitamente nella config iniziale; nessuna esclusione silenziosa di ulteriori classi.
- Tutte le viste dei punti campionati vengono conservate; nessuna pseudo-referenza o soglia di hit del detector.
- min_views=1 e min_regions=3 sono scelte iniziali modificabili PRIMA della run. Per confronti con T9 allineare min_views/supporto.

## Controlli prima del primo lancio
Verifica visiva GT/proiezione, esaustivita delle maschere valid, qualita dei negativi, scala della ricostruzione,
pesi area e mascheramento degli sconosciuti. Il software non puo certificarli.
Config: g50_grid usa SOURCE units / MAP pixel, NON automaticamente millimetri.
La griglia inclusa e un esempio operativo, non una misura fisica o un intervallo scientificamente validato.
Fissala sui soli dati di sviluppo prima del test; non adattarla a T.
La verifica provenance rilegge gli hash dell'intero log: puo costare I/O, ma non esegue proiezione/inferenza.

## Esecuzione dalla root del repository (venv attivo)
python tests_supervised_a.py
python -m damage3d.supervised_direction_a --help

.\run_supervised_a.ps1 -Run "PERCORSO_T7" -Regions "PERCORSO_regions_agree.npz" -BaseParams "PERCORSO_base_params.json" -Buffer 1.478 -Output "outputs\supervised_a_v1"

Buffer 1.478 e valido SOLO se corrisponde al buffer gia dichiarato e alla tua scala; non e un default universale.
Il runner si ferma al primo errore. Se il preflight fallisce il JSON viene comunque salvato.
Dopo un preflight fallito scegli una nuova directory alla successiva run; non sovrascrivere esperimenti congelati.

## Valutazione finale: comando separato
Solo dopo aver congelato il protocollo e deciso che non farai ulteriori selezioni su T:
.\run_supervised_a.ps1 -Run "PERCORSO_T7" -Regions "PERCORSO_regions_agree.npz" -BaseParams "PERCORSO_base_params.json" -Buffer 1.478 -Output "outputs\supervised_a_v1" -Evaluate

## Output
preflight.json: conteggi GT noti/sconosciuti e supporto dopo osservazioni valide, per regione e classe C_fit/C_cal;
range GSD e controlli provenance, schema, rho/calibrazione, buffer, duplicati nell'iteratore.
fit.json: configurazioni migliori e NLL di tutta la griglia censored; source_index del campione e provenance.
frozen.json: parametri invariati + soglie C_cal per ciascun metodo, hash del fit.
eval_T.json: metriche globali e per regione AP step-integrata con tie, NLL, Brier, F1, IoU e copertura per area.
Tutti gli output rifiutano sovrascrittura. Modificare config/base_params/GT/run invalida gli artifact precedenti.

## Campionamento e costo
Fit: campionamento uniforme senza rimpiazzo di max 2000 punti per regione C_fit, indipendente dalle label;
pesi area corretti con N_regione/n_campione. Non viene imposto un bilanciamento artificiale 50/50.
Il campione puo perdere classi rare: in quel caso il fit fallisce chiaramente. Aumenta sample_per_region in una nuova config.
La cache conserva tutte le osservazioni dei punti selezionati fino al cap esplicito (3 milioni); non tronca le viste.
Freeze/evaluate scorrono invece tutti i punti del rispettivo split e accumulano array per punto, non tutto il log in RAM.
Nessuna stima di tempi/consumi sul tuo ponte e disponibile. La verifica hash viene ripetuta per sicurezza.

## Baseline e comparabilita
mean e max: nessun parametro di fusione appreso, soglia propria C_cal.
llr: alpha proprio scelto su C_fit.
pi_mean: g50/beta propri scelti su C_fit.
pi_llr: g50/beta/alpha propri scelti su C_fit.
censored: g50/beta/alpha propri scelti su C_fit.
Stessi punti, pesi e min_views per tutti. alpha grid uguale per metodi con tempering.
Le baseline logiche sono ricalcolate in questo protocollo: non affermare identita con i vecchi output T9,
che potrebbero avere budget di calibrazione o supporto diversi. Non include angle/range/distance di T9.
Controlla search in fit.json per ottimi al bordo, loss piatte e configurazioni equivalenti.
Nessun bootstrap, test di significativita o asse mediale/tolerance IoU e implementato qui.
I risultati per regione sono forniti per consentire una successiva analisi paired/bootstrap.

## Non passare questi JSON alla pipeline H3 originale
Sono artifact scoped del nuovo modulo, NON fusion_params.json compatibili con tutta la vecchia CLI.
FusionModel originale richiede parametri censored per tutte le classi: questo modulo calcola solo le classi attive,
senza inventare parametri surface. Non esporta PLY o stati multilabel sul ponte intero.
Per quella fase serve una successiva integrazione esplicita del class scope nel percorso di inferenza/export.
Non eseguire resolvability_fit o tempering_fit tra fit e freeze: cambierebbero il protocollo.

## Validazione realmente effettuata
9 test passati: limiti di censura, Jensen, AP con tie, stabilita NLL, input invalido,
campionamento, equivalenza scoring diretto/a blocchi e catena sintetica fit/freeze/evaluate con invarianza rispetto alle label T.
In questo ambiente gli import upstream sono stati sostituiti da stub; non e stato clonato/eseguito il repository reale.
Esegui quindi tests_supervised_a.py nel tuo venv prima di usare dati reali.
Nessuna verifica sui tuoi file locali, nessun risultato sul ponte e nessuna validazione scientifica sono implicati.

## Interpretazione scientifica
La formula censurata rimane invariata; i parametri sono appresi supervisionatamente sulla fusione 3D.
Non sostenere che pi sia stata misurata come probabilita fisica di risolvibilita.
Il contributo da verificare e la trasformazione dell'evidenza dipendente da classe/risoluzione,
non la grid search, la calibrazione o la fusione multi-vista in generale.
