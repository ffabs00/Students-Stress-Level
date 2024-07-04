import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from dataset.data import read_df
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import seaborn as sns

data_path = os.path.join(project_root, 'dataset', 'StressLevelDataset.csv')
df = read_df(data_path)


def XGBoost(x_train, x_test, y_train, y_test, seed):

    # XGBoost
    model = XGBClassifier(random_state=seed)
    model.fit(x_train, y_train)

    print("\n\n\n---------------------------------------------- RISULTATI XGBOOST ----------------------------------------------\n\n\n")

    #matrice di confusione
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    cmap = sns.diverging_palette(220, 20, n=256)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Salva matrice in formato PNG
    save_path = os.path.join(project_root, 'results', 'XGBoost', 'cm_xgb.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    plt.close()


    print("\n","REPORT DI CLASSIFICAZIONE XGBOOST", "\n")
    print(classification_report(y_test, y_pred))



    # Metodi di feature selection
    feature_selection_methods = {
        "Chi-Squared": SelectKBest(score_func=chi2, k=5),  
        "Forward Selection": SequentialFeatureSelector(XGBClassifier(), n_features_to_select=5, direction='forward'),
        "Backward Selection": SequentialFeatureSelector(XGBClassifier(), n_features_to_select=5, direction='backward'),
        "Exhaustive Search": SequentialFeatureSelector(XGBClassifier(), n_features_to_select=5),
        "ANOVA F-value": SelectKBest(score_func=f_classif, k=5),
        "Mutual Information": SelectKBest(score_func=mutual_info_classif, k=5),
        "Tree-based Feature Selection": SelectFromModel(RandomForestClassifier(n_estimators=100))
    }

    # Addestramento e valutazione dei modelli con i metodi di feature selection
    for method, selector in feature_selection_methods.items():
        # Applica il metodo di feature selection
        X_train_selected = selector.fit_transform(x_train, y_train)
        X_test_selected = selector.transform(x_test)
        
        # Addestra un classificatore
        XGBoost = XGBClassifier()  
        XGBoost.fit(X_train_selected, y_train)

        # Ottieni gli indici delle feature selezionate
        selected_indices = selector.get_support(indices=True)
        
        # Ottieni i nomi delle feature selezionate
        selected_features = df.columns[selected_indices]

        # Valuta il modello
        y_pred = XGBoost.predict(X_test_selected)
        print(f"\nClassification Report for {method}:\n")
        print(classification_report(y_test, y_pred),"\n")
        print(f"Selected features for {method}: {selected_features.tolist()} \n")



    def f(x):
        return model.predict_proba(x)[:, 2]
    med = x_train.median().values.reshape((1, x_train.shape[1]))
    explainer = shap.Explainer(f, med)
    shap_values = explainer(x_test.iloc[0:1000, :])

    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)


    # Salva il grafico SHAP in formato PNG
    shap_path = os.path.join(project_root, 'results', 'XGBoost', 'shap_xgb.png')
    os.makedirs(os.path.dirname(shap_path), exist_ok=True)
    plt.savefig(shap_path, format='png')
    plt.close()


    # Seleziona le prime 5 feature più importanti
    shap_values_array = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': x_train.columns,
        'importance': shap_values_array
    })
    important_features = importance_df.sort_values(by='importance', ascending=False).head(5)
    important_feature_names = important_features['feature'].tolist()

    # Filtra il dataset per mantenere solo le 5 feature più importanti
    x_train_selected = x_train[important_feature_names]
    x_test_selected = x_test[important_feature_names]

    # Allena nuovamente il modello con le feature selezionate
    model_selected = XGBClassifier()
    model_selected.fit(x_train_selected, y_train)

    y_pred_selected = model_selected.predict(x_test_selected)
    print("\n", "REPORT DI CLASSIFICAZIONE CON SHAP:", "\n")
    print(classification_report(y_test, y_pred_selected))

    print("FEATURE SELEZIONATE CON SHAP:", important_feature_names)
