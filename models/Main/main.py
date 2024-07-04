import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.append(project_root)

import numpy as np
from models.RandomForest.random_forest import RandomForest
from models.LogisticRegression.logistic_regression import LogRegression
from models.DecisionTree.decision_tree import DecisionTree
from models.XGBoost.XGBoost import XGBoost
from models.knn.knn import KNN
from dataset.data import read_df
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')  # Usa un backend non interattivo

data_path = os.path.join(project_root, 'dataset', 'StressLevelDataset.csv')
df = read_df(data_path)


seed = 123
np.random.seed(seed)
X = df.drop(columns=['stress_level'])
Y = df['stress_level']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed)

# Modelli
models = {
    "XGBoost": XGBoost(x_train, x_test, y_train, y_test, seed),
    "Logistic Regression": LogRegression(x_train, x_test, y_train, y_test, seed),
    "XGBoost": XGBoost(x_train, x_test, y_train, y_test, seed),
    "Random Forest": RandomForest(x_train, x_test, y_train, y_test, seed),
    "Decision Tree": DecisionTree(x_train, x_test, y_train, y_test, seed),
    "K-Nearest Neighbors": KNN(x_train, x_test, y_train, y_test)
}

print("\nGrafici SHAP e matrici di confusione salvati correttamente")
