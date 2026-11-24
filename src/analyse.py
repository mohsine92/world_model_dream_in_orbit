"""Analyse d'une simulation : courbe de dérive, « rêve contre réalité » avec incertitude, erreurs par circuit.

    python3 predict.py --test_dir validation/test --weights weights/gru_val.pth --out pred.csv --out_std pred_std.csv
    python3 src/analyse.py --pred pred.csv --std pred_std.csv --verite validation/verite_horaire.csv

Produit dans reports/ : derive.png, reve_vs_realite.png, erreurs_par_circuit.csv, analyse_resume.txt
Incertitude = désaccord entre les réseaux de l'ensemble (elle ne voit pas les erreurs que tous les réseaux partagent).
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLES = ["NPWD2551", "NPWD2451"]   # les deux circuits qui dominent l'erreur
JOUR = 86_400_000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--verite", required=True)
    ap.add_argument("--std", default=None)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lignes = []

    def dit(msg=""):
        print(msg)
        lignes.append(msg)

    v = pd.read_csv(a.verite).set_index("ut_ms")
    p = pd.read_csv(a.pred).set_index("ut_ms")[v.columns]
    s = pd.read_csv(a.std).set_index("ut_ms")[v.columns] if a.std else None
    idx = v.index.intersection(p.index)
    v, p = v.loc[idx], p.loc[idx]
    s = s.loc[idx] if s is not None else None
    err = (p - v)                       # NaN là où la vérité n'existe pas
    e = err.to_numpy()
    dit(f"RMSE globale = {np.sqrt(np.nanmean(e ** 2)):.4f}  ({int((~np.isnan(e)).sum())} valeurs notées)")

    # ---- erreurs par circuit
    mse = (err ** 2).mean()
    tab = pd.DataFrame({"mse": mse, "part_erreur_%": 100 * mse / mse.sum()}).sort_values("mse", ascending=False)
    tab.to_csv(out / "erreurs_par_circuit.csv")
    dit("\nCircuits qui pèsent le plus dans l'erreur :")
    dit(tab.head(8).round(5).to_string())

    # ---- courbe de dérive : RMSE par bloc de 30 jours depuis le début de la simulation
    jours = np.asarray(idx - idx.min()) / JOUR
    bloc = (jours // 30).astype(int)
    ec = err[[c for c in CLES if c in err.columns]].to_numpy()
    pts = []
    for b in sorted(set(bloc)):
        m = bloc == b
        if np.isnan(e[m]).all():
            continue
        pts.append((b * 30 + 15, np.sqrt(np.nanmean(e[m] ** 2)), np.sqrt(np.nanmean(ec[m] ** 2))))
    pts = np.array(pts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(pts[:, 0], pts[:, 1], "o-", color="tab:blue")
    ax[0].set_title("RMSE globale par bloc de 30 jours")
    ax[1].plot(pts[:, 0], pts[:, 2], "o-", color="tab:red")
    ax[1].set_title("RMSE sur NPWD2551 + NPWD2451")
    for x in ax:
        x.set_xlabel("jours depuis le début de la simulation")
        x.set_ylabel("RMSE")
        x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "derive.png", dpi=130)
    plt.close(fig)
    tiers = max(1, len(pts) // 3)
    dit(f"\nDérive : RMSE du 1er tiers de la période = {pts[:tiers, 1].mean():.4f}, du dernier tiers = {pts[-tiers:, 1].mean():.4f}")

    # ---- rêve contre réalité (+ incertitude)
    cles = [c for c in CLES if c in v.columns]
    dates = pd.to_datetime(idx, unit="ms")
    fig, axes = plt.subplots(len(cles), 2, figsize=(13, 3.6 * len(cles)), squeeze=False)
    milieu = len(idx) // 2
    for i, c in enumerate(cles):
        d = pd.DataFrame({"v": v[c].to_numpy(), "p": p[c].to_numpy()}, index=dates)
        if s is not None:
            d["s"] = s[c].to_numpy()
        jour = d.resample("D").mean()
        zoom = d.iloc[milieu:milieu + 336]
        for ax, df, titre in ((axes[i, 0], jour, f"{c} — moyenne journalière, toute la période"),
                              (axes[i, 1], zoom, f"{c} — zoom : 14 jours, heure par heure")):
            ax.plot(df.index, df["v"], color="black", lw=1, label="réalité")
            ax.plot(df.index, df["p"], color="tab:orange", lw=1, label="rêve du modèle")
            if s is not None:
                ax.fill_between(df.index, df["p"] - 2 * df["s"], df["p"] + 2 * df["s"], color="tab:orange", alpha=.25,
                                label="± 2 écarts-types (ensemble)")
            ax.set_title(titre, fontsize=10)
            ax.grid(alpha=.3)
        axes[i, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reve_vs_realite.png", dpi=130)
    plt.close(fig)

    # ---- l'incertitude est-elle fiable ?
    if s is not None:
        sv = s.to_numpy()
        ok = ~np.isnan(e)
        dit(f"\nIncertitude : écart-type moyen de l'ensemble = {np.mean(sv[ok]):.4f}, erreur absolue moyenne = {np.mean(np.abs(e[ok])):.4f}")
        dit(f"Part des mesures dans ± 2 écarts-types : {100 * np.mean(np.abs(e[ok]) <= 2 * sv[ok]):.1f} %  (≈ 95 % si l'incertitude était bien calibrée)")
    dit(f"\nFigures : {out / 'derive.png'}, {out / 'reve_vs_realite.png'}")
    (out / "analyse_resume.txt").write_text("\n".join(lignes) + "\n")


if __name__ == "__main__":
    main()