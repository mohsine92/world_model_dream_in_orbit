"""
   Résume les fichiers power.csv : période, pas de temps, trous, statistiques par circuit.
   Sert à repérer les circuits qui dominent l'erreur (RMSE en unités brutes) avant de modéliser.
   Usage : python3 src/inspect_power.py kit_etudiants/annee1/power.csv kit_etudiants/annee2/power.csv
"""

import sys
import numpy as np
import pandas as pd

for path in sys.argv[1:]:
    d = pd.read_csv(path)
    cols = [c for c in d.columns if c != "ut_ms"]
    print(f"\n===== {path}")
    print("lignes :", len(d), "| circuits :", len(cols))
    print("période :", pd.to_datetime(d.ut_ms.min(), unit="ms"), "->", pd.to_datetime(d.ut_ms.max(), unit="ms"))
    print("pas médian (s) :", d.ut_ms.diff().median() / 1000, "| plus gros trou (h) :", round(d.ut_ms.diff().max() / 3.6e6, 1))
    print("NaN total :", int(d[cols].isna().sum().sum()))
    print("colonnes :", cols)
    s = d[cols].describe().T[["mean", "std", "min", "max"]].round(4)
    print(s.to_string())
    h = d.copy(); h["ut_ms"] = (h.ut_ms // 3_600_000) * 3_600_000
    hh = h.groupby("ut_ms").mean()
    print("heures :", len(hh), "| écart-type horaire moyen des circuits :", round(float(hh[cols].std().mean()), 4))
    print("RMSE d'une prédiction constante (moyenne globale du fichier) :",
          round(float(np.sqrt(((hh[cols] - hh[cols].mean()) ** 2).to_numpy().mean())), 4))