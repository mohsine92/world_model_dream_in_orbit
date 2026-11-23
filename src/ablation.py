"""Ablations du gradient boosting (annee1 -> annee2) : qu'apportent chaque groupe de features et la « mémoire » ?
  - on retire un groupe de features à la fois (lt_, saaf_, ftl_, ev_/orb_, dmop_)
  - on ajoute des moyennes glissantes du passé du CONTEXTE (3 h, 12 h, 48 h) = mémoire de l'historique du planning
Les circuits quasi constants (écart-type horaire < --min_std en train) sont prédits par la constante (7 jours d'init).
Usage (racine du projet) :
    python3 src/ablation.py --train kit_etudiants/annee1 --val kit_etudiants/annee2
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from baseline import JOURS_INIT, H, power_horaire, rmse
from features import build_features

ROLL_COLS_PREF = ("ftl_", "saaf_", "ev_umbra", "ev_penumbra", "ev_occ_mars", "dmop_total")
FENETRES = [3, 12, 48]


def ajoute_memoire(X):
    base = [c for c in X.columns if c.startswith(ROLL_COLS_PREF)]
    out = {}
    for w in FENETRES:
        r = X[base].rolling(w, min_periods=1).mean()
        for c in base:
            out[f"{c}_m{w}h"] = r[c].to_numpy()
    return pd.concat([X, pd.DataFrame(out, index=X.index)], axis=1)


def garde(X, retire):
    return X[[c for c in X.columns if not c.startswith(tuple(retire))]] if retire else X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--min_std", type=float, default=0.01)
    a = ap.parse_args()
    tr, va = Path(a.train), Path(a.val)

    _, y_tr, _ = power_horaire(tr / "power.csv")
    raw_va, y_va, debut_va = power_horaire(va / "power.csv")
    circuits = list(y_tr.columns)
    fin_init = debut_va + JOURS_INIT * 24 * H
    y_te = y_va[y_va.index >= ((fin_init // H) + 1) * H]
    vrai = y_te.to_numpy()
    init = raw_va[raw_va["ut_ms"] < fin_init][circuits].mean().to_numpy()
    modelises = [j for j, c in enumerate(circuits) if y_tr[c].std() >= a.min_std]
    print(f"{len(modelises)} circuits modélisés sur {len(circuits)} (les autres : constante)")

    X_tr0, prefixes = build_features(tr, t0=y_tr.index.min(), t1=y_tr.index.max())
    X_va0, _ = build_features(va, prefixes=prefixes, t0=y_va.index.min(), t1=y_va.index.max())

    configs = [
        ("tout", [], False),
        ("sans lt_ (tendances journalières)", ["lt_"], False),
        ("sans saaf_ (angles Soleil)", ["saaf_"], False),
        ("sans ftl_ (pointage/comms)", ["ftl_"], False),
        ("sans ev_/orb_ (éclipses, orbite)", ["ev_", "orb_"], False),
        ("sans dmop_ (commandes)", ["dmop_"], False),
        ("tout + mémoire du contexte", [], True),
        ("sans lt_ + mémoire du contexte", ["lt_"], True),
    ]
    res = []
    for nom, retire, mem in configs:
        Xtr, Xva = X_tr0, X_va0
        if mem:
            Xtr, Xva = ajoute_memoire(Xtr), ajoute_memoire(Xva)
        Xtr, Xva = garde(Xtr, retire), garde(Xva, retire)
        idx = Xtr.index.intersection(y_tr.index)
        Xa, Ya = Xtr.loc[idx].to_numpy(), y_tr.loc[idx]
        Xt = Xva.loc[y_te.index].to_numpy()
        pred = np.tile(init, (len(y_te), 1))
        for j in modelises:
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
            m.fit(Xa, Ya[circuits[j]].to_numpy())
            pred[:, j] = m.predict(Xt)
        r = rmse(pred, vrai)
        r2 = rmse(pred[:, [circuits.index("NPWD2551"), circuits.index("NPWD2451")]],
                  vrai[:, [circuits.index("NPWD2551"), circuits.index("NPWD2451")]])
        res.append((nom, r, r2))
        print(f"{nom:<38} RMSE = {r:.4f}   (2551+2451 seuls : {r2:.4f})", flush=True)

    ref = res[0][1]
    print("\nRésumé (écart vs « tout », négatif = mieux) :")
    for nom, r, _ in res:
        print(f"  {nom:<38} {r:.4f}  ({r - ref:+.4f})")


if __name__ == "__main__":
    main()