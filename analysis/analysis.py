import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from dataset.data import read_df
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path='dataset/StressLevelDataset.csv'
df=read_df(data_path)

#distribuzione dei dati
print("DISTRIBUZIONE DEI DATI")
print(df.describe(),"\n")

#info sui dati
print("TIPO DI DATI")
print(df.info(),"\n")

#numero di campioni per ogni tipo
print("NUMERO DI CAMPIONI PER OGNI LIVELLO DI STRESS")
print(df['stress_level'].value_counts(),"\n")

print(df.head(),"\n")

#correlazione delle altre feature allo stress
correl = df.corr()
plt.figure(figsize = (8,8))
sns.heatmap(correl.iloc[:-1,-1:], annot = True, cmap = sns.color_palette("coolwarm", as_cmap=True))

save_path = os.path.join(os.path.dirname(__file__), 'correlazione.png')
plt.savefig(save_path)
print("Grafico di correlazione salvato", "\n")

