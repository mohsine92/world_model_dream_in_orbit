"""Baselines sur annee1 -> annee2 (même protocole que l'évaluation finale).
  A) constante = moyenne des 7 premiers jours réels, par circuit
  B) gradient boosting (sklearn) par circuit, à partir des features de contexte (sans mémoire)
Usage (depuis la racine du projet) :
    python3 src/baseline.py --train kit_etudiants/annee1 --val kit_etudiants/annee2
Besoin : pip install scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_features

H = 3_600_000
JOURS_INIT = 7


def power_horaire(power_csv):
    raw = pd.read_csv(power_csv)
    debut = raw["ut_ms"].min()
    raw["h"] = (raw["ut_ms"] // H) * H
    return raw, raw.groupby("h").mean().drop(columns="ut_ms"), debut


def rmse(pred, vrai):
    m = ~np.isnan(vrai)
    return float(np.sqrt(np.mean((pred[m] - vrai[m]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    a = ap.parse_args()
    tr, va = Path(a.train), Path(a.val)

    _, y_tr, _ = power_horaire(tr / "power.csv")
    raw_va, y_va, debut_va = power_horaire(va / "power.csv")
    circuits = list(y_tr.columns)

    # même découpe que make_validation.py : 7 jours réels d'init, puis tout le reste à simuler
    fin_init = debut_va + JOURS_INIT * 24 * H
    premiere = ((fin_init // H) + 1) * H
    y_va_test = y_va[y_va.index >= premiere]

    # A) constante
    init = raw_va[raw_va["ut_ms"] < fin_init][circuits].mean()
    pred_a = np.tile(init.to_numpy(), (len(y_va_test), 1))
    print(f"A) constante (7 jours d'init)        RMSE = {rmse(pred_a, y_va_test.to_numpy()):.4f}")

    # B) gradient boosting sur le contexte
    # grille explicite = heures où l'on a de la consommation (ltdata s'arrête parfois ~23 h avant power)
    X_tr, prefixes = build_features(tr, t0=y_tr.index.min(), t1=y_tr.index.max())
    X_va, _ = build_features(va, prefixes=prefixes, t0=y_va.index.min(), t1=y_va.index.max())
    X_tr = X_tr.loc[X_tr.index.intersection(y_tr.index)]
    Y_tr = y_tr.loc[X_tr.index]
    X_te = X_va.loc[y_va_test.index]
    pred_b = np.zeros((len(X_te), len(circuits)))
    for j, c in enumerate(circuits):
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
        mdl.fit(X_tr.to_numpy(), Y_tr[c].to_numpy())
        pred_b[:, j] = mdl.predict(X_te.to_numpy())
    vrai = y_va_test.to_numpy()
    print(f"B) gradient boosting sur le contexte  RMSE = {rmse(pred_b, vrai):.4f}")

    # contribution de chaque circuit à l'erreur (part du MSE total)
    mse_c = np.nanmean((pred_b - vrai) ** 2, axis=0)
    mse_a = np.nanmean((pred_a - vrai) ** 2, axis=0)
    t = pd.DataFrame({"mse_const": mse_a, "mse_gbm": mse_c, "part_erreur_gbm_%": 100 * mse_c / mse_c.sum()},
                     index=circuits).sort_values("mse_gbm", ascending=False)
    print("\nCircuits qui pèsent le plus dans l'erreur (B) :")
    print(t.head(10).round(5).to_string())


if __name__ == "__main__":
    main()