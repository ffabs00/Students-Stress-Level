import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import shap
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from dataset.data import read_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector


data_path = os.path.join(project_root, 'dataset', 'StressLevelDataset.csv')
df = read_df(data_path)

def DecisionTree(x_train, x_test, y_train, y_test, seed):

    # decision tree
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(x_train, y_train)

    print("\n\n\n---------------------------------------------- RISULTATI DECISION TREE ----------------------------------------------\n\n\n")

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
    save_path = os.path.join(project_root, 'results', 'Decision Tree', 'cm_dt.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    plt.close()

    #report di classificazione
    print("\n","REPORT DI CLASSIFICAZIONE DECISION TREE", "\n")
    print(classification_report(y_test, y_pred))

    # Metodi di feature selection
    feature_selection_methods = {
        "Chi-Squared": SelectKBest(score_func=chi2, k=5),  
        "Forward Selection": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5, direction='forward'),
        "Backward Selection": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5, direction='backward'),
        "Exhaustive Search": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5),
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
        DecisionTree = DecisionTreeClassifier()  
        DecisionTree.fit(X_train_selected, y_train)

        # Ottieni gli indici delle feature selezionate
        selected_indices = selector.get_support(indices=True)
        
        # Ottieni i nomi delle feature selezionate
        selected_features = df.columns[selected_indices]
        
        # Valuta il modello
        y_pred = DecisionTree.predict(X_test_selected)
        print(f"\nREPORT DI CLASSIFICAZIONE CON {method}:","\n")
        print(classification_report(y_test, y_pred),"\n")
        print(f"FEATURE SELEZIONATE CON {method}: {selected_features.tolist()} \n")


    background_data = shap.kmeans(x_train, 100)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(x_test)

    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)


    # Salva il grafico SHAP in formato PNG
    shap_path = os.path.join(project_root, 'results', 'Decision Tree', 'shap_dt.png')
    os.makedirs(os.path.dirname(shap_path), exist_ok=True)
    plt.savefig(shap_path, format='png')
    plt.close()


    # Seleziona le prime 5 feature più importanti
    importance_df = pd.DataFrame({
        'feature': x_train.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    })
    important_features = importance_df.sort_values(by='importance', ascending=False).head(5)
    important_feature_names = important_features['feature'].tolist()



    # Filtra il dataset per mantenere solo le 5 feature più importanti
    x_train_selected = x_train[important_feature_names]
    x_test_selected = x_test[important_feature_names]

    # Allena nuovamente il modello con le feature selezionate
    model_selected = DecisionTreeClassifier()
    model_selected.fit(x_train_selected, y_train)

    y_pred_selected = model_selected.predict(x_test_selected)
    print("\n", "REPORT DI CLASSIFICAZIONE CON SHAP:", "\n")
    print(classification_report(y_test, y_pred_selected))

    print("FEATURE SELEZIONATE CON SHAP", important_feature_names)