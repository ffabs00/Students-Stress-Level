
# Utilizzo di librerie Python per l'Intelligenza Artificiale Spiegabile su dati tabulari
Questo progetto mira a classificare lo stress degli studenti utilizzando algoritmi di machine learning e tecniche di eXplainable Artificial Intelligence (XAI). In particolare, utilizza i valori SHAP (SHapley Additive exPlanations) per interpretare le predizioni dei modelli e identificare le caratteristiche più rilevanti. Vengono impiegati diversi algoritmi di classificazione e metodi di selezione delle caratteristiche per confrontare le loro prestazioni e la trasparenza delle predizioni.


## Contenuto delle cartelle principali

analysis/: Contiene l'analisi del dataset utilizzato nel progetto.

dataset/: Contiene il file CSV del dataset utilizzato per addestrare e testare i modelli.

models/: Questa cartella include il file main.py e i file .py di ogni classificatore contenente le funzioni per l'addestramento dei modelli di classificazione, la generazione delle matrici di confusione, l'applicazione dei metodi di feature selection e l'interpretazione dei modelli tramite SHAP.

results/: Contiene una sottocartella per ogni classificatore contenente le immagini delle matrici di confusione e dei grafici SHAP generati per ogni classificatore. 
## Requisiti
Le principali librerie Python utilizzate in questo progetto sono:
* numpy
* shap
* scikit-learn
* pandas
* matplotlib
* seaborn
* xgboost
* shap
* ipython

Puoi installare queste librerie tramite pip utilizzando il comando:

                    pip install -r requirements.txt
## Utilizzo
Per replicare o estendere questo progetto, seguire i seguenti passaggi:

* Dataset: Assicurarsi di avere il file CSV contenente il dataset su cui addestrare i modelli nella cartella dataset/.

* Ambiente di sviluppo: Creare un ambiente virtuale Python e installare le dipendenze necessarie elencate nel file requirements.txt.

* Esecuzione: Utilizzare il file main.py nella cartella models/ per addestrare i modelli, generare le matrici di confusione e i grafici SHAP per ogni classificatore.

* Analisi: Visualizzare i file contenuti nella cartella results/ per vedere i grafici SHAP e le matrici generati per ogni classificatore