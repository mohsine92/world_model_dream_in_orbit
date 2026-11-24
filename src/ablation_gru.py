"""Ablations du GRU, validées dans les DEUX sens : annee1 -> annee2 et annee2 -> annee1.

Chaque variante retire une famille de features ou change la mémoire / la taille du réseau, puis mesure la RMSE
(même protocole que l'évaluation : 7 jours d'init exclus, vérité horaire). Un changement n'est crédible que s'il
améliore les deux sens d'au moins ~0,003 (en dessous, c'est du bruit de validation).

    python3 src/ablation_gru.py                     # toutes les variantes, 2 graines par modèle (~20-25 min)
    python3 src/ablation_gru.py --seeds 1           # plus rapide, plus bruité
    python3 src/ablation_gru.py --seulement 0 2 3   # seulement certaines variantes (la référence, n°0, tourne toujours)

Résultats enregistrés au fur et à mesure dans reports/ablation_gru_results.txt.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from gru_model import H, charge_segment, entraine, predit

RACINE = Path(__file__).resolve().parent.parent
CLES = ["NPWD2551", "NPWD2451"]
BASE = ["dmop_"]
PARAMS = dict(hidden=64, layers=1, dropout=0.1, longueur=336, pas=24, batch=32, epochs=30, lr=2e-3, wd=1e-5)
VARIANTES = [
    ("référence (sans dmop_)", BASE, {}),
    ("avec dmop_ (commandes)", [], {}),
    ("sans lt_ (tendances jour)", BASE + ["lt_"], {}),
    ("sans saaf_ (angles Soleil)", BASE + ["saaf_"], {}),
    ("sans ftl_ (pointage/comms)", BASE + ["ftl_"], {}),
    ("sans ev_/orb_ (éclipses, orbite)", BASE + ["ev_", "orb_"], {}),
    ("mémoire courte (24 h)", BASE, dict(longueur=24, pas=12)),
    ("mémoire longue (30 jours)", BASE, dict(longueur=720, pas=48)),
    ("réseau plus grand (128)", BASE, dict(hidden=128)),
]


def prepare(brut, retire):
    """Standardisation et segment d'entraînement à partir des features brutes (cache)."""
    X, Y = brut
    cols = [c for c in X.columns if not c.startswith(tuple(retire))] if retire else list(X.columns)
    xv = X[cols].to_numpy().astype(np.float64)
    mu, sd = xv.mean(0), xv.std(0)
    sd[sd < 1e-8] = 1.0
    y = Y.to_numpy()
    seg = [(((xv - mu) / sd).astype(np.float32), np.nan_to_num(y).astype(np.float32), (~np.isnan(y)).astype(np.float32))]
    return seg, dict(mu=mu, sd=sd, base=np.nanmean(y, axis=0), columns=cols, circuits=list(Y.columns))


def evalue(brut_train, brut_test, retire, params, seeds):
    seg, meta = prepare(brut_train, retire)
    Xt, Yt = brut_test
    Xs = ((Xt[meta["columns"]].to_numpy() - meta["mu"]) / meta["sd"]).astype(np.float32)
    garde = np.asarray(Xt.index >= Xt.index[0] + 169 * H)    # 7 jours d'init + l'heure partielle, comme make_validation.py
    vrai = Yt.to_numpy()[garde]
    preds = []
    for k in range(seeds):
        m = entraine(seg, meta, params, seed=k, log=lambda *_: None)
        preds.append(predit(m, Xs, params["longueur"] // 2, params["longueur"] // 2)[garde])
    err2 = (np.mean(preds, axis=0) - vrai) ** 2
    cles = [meta["circuits"].index(c) for c in CLES if c in meta["circuits"]]
    return float(np.sqrt(np.nanmean(err2))), float(np.sqrt(np.nanmean(err2[:, cles]))) if cles else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annee1", default=str(RACINE / "kit_etudiants" / "annee1"))
    ap.add_argument("--annee2", default=str(RACINE / "kit_etudiants" / "annee2"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seulement", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(RACINE / "reports" / "ablation_gru_results.txt"))
    a = ap.parse_args()
    idx = [0] + [i for i in (a.seulement if a.seulement is not None else range(1, len(VARIANTES))) if i != 0]
    print(f"{len(idx)} variantes x 2 sens x {a.seeds} graines — compte environ {len(idx) * 2 * a.seeds * 0.7:.0f} minutes\n")

    print("Calcul des features (une seule fois)...", flush=True)
    brut, prefixes = {}, None
    for nom, d in (("annee1", a.annee1), ("annee2", a.annee2)):
        X, Y, prefixes = charge_segment(d, prefixes, [])
        brut[nom] = (X, Y)

    entete = f"{'variante':<34}{'1->2':>8}{'2->1':>8}{'moyenne':>9}{'vs réf.':>9}{'2551+2451':>11}"
    lignes = [f"Ablations du GRU ({a.seeds} graines par modèle) — RMSE ; 1->2 = entraîné sur annee1, testé sur annee2",
              "Un écart vs la référence n'est crédible que s'il va dans le même sens dans les deux colonnes et dépasse ~0,003.",
              "", entete]
    print(entete)
    ref = None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for i in idx:
        nom, retire, over = VARIANTES[i]
        p = {**PARAMS, **over}
        t0 = time.time()
        r12, k12 = evalue(brut["annee1"], brut["annee2"], retire, p, a.seeds)
        r21, k21 = evalue(brut["annee2"], brut["annee1"], retire, p, a.seeds)
        moy = (r12 + r21) / 2
        ref = moy if ref is None else ref
        ligne = f"{i}. {nom:<31}{r12:8.4f}{r21:8.4f}{moy:9.4f}{moy - ref:+9.4f}{(k12 + k21) / 2:11.4f}"
        print(ligne + f"   ({time.time() - t0:.0f}s)", flush=True)
        lignes.append(ligne)
        Path(a.out).write_text("\n".join(lignes) + "\n")
    print(f"\nRésultats enregistrés dans {a.out}")


if __name__ == "__main__":
    main()