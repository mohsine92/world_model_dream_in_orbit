"""
Construit la grille horaire de features (contexte) pour Dream in Orbit.

Utilise UNIQUEMENT les 5 fichiers de contexte du kit : dmop, evtf, ftl, saaf, ltdata.
Aucune donnée de power.csv : ces features sont connues à l'avance (planning),
donc utilisables pour simuler toute l'année.

Usage CLI :
    python features.py --dir annee1 --out feats_annee1.csv
    python features.py --dir validation/test --out feats_test.csv      # utilise heures_a_predire.csv

Usage en module :
    from features import build_features
    feats, prefixes = build_features("annee1")
    feats_val, _ = build_features("annee2", prefixes=prefixes)   # mêmes colonnes dmop que le train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN = 60_000
H = 3_600_000

FTL_TYPES = ["EARTH", "SLEW", "INERTIAL", "NADIR", "WARMUP", "MAINTENANCE",
             "RADIO_SCIENCE", "ACROSS_TRACK", "D4PNPO", "D1PVMC", "SPECULAR"]
N_PREFIXES = 25


def _etat_minute(t_min, valeurs, n):
    """Série d'état (0/1) à la minute : la dernière valeur connue gagne (forward fill)."""
    s = pd.Series(valeurs, index=t_min).groupby(level=0).last()
    s = s[(s.index >= 0) & (s.index < n)]
    out = s.reindex(range(n)).ffill().fillna(0.0)
    return out.to_numpy()


def _intervalle_minute(b_min, e_min, n):
    """Couverture (0/1) à la minute d'une liste d'intervalles [b, e[."""
    delta = np.zeros(n + 1)
    b = np.clip(b_min, 0, n)
    e = np.clip(e_min, 0, n)
    np.add.at(delta, b, 1)
    np.add.at(delta, e, -1)
    return (np.cumsum(delta)[:n] > 0).astype(float)


def _frac_heure(arr_minute):
    return arr_minute.reshape(-1, 60).mean(axis=1)


def _grille(d: Path, t0=None, t1=None):
    """Bornes de la grille horaire (ut_ms début d'heure)."""
    h = d / "heures_a_predire.csv"
    if t0 is None and h.exists():
        hh = pd.read_csv(h)["ut_ms"]
        return int(hh.min()), int(hh.max())
    if t0 is None:
        ltd = pd.read_csv(d / "ltdata.csv")
        t0 = int(ltd["ut_ms"].min())
        t1 = int(ltd["ut_ms"].max())
    return int((t0 // H) * H), int((t1 // H) * H)


def build_features(dossier, prefixes=None, t0=None, t1=None):
    d = Path(dossier)
    h0, h1 = _grille(d, t0, t1)
    heures = np.arange(h0, h1 + H, H)
    nh = len(heures)
    n = nh * 60
    feats = pd.DataFrame(index=pd.Index(heures, name="ut_ms"))

    def idx_min(ts):
        return ((np.asarray(ts) - h0) // MIN).astype(int)

    # ltdata : tendances journalières, interpolées à l'heure 
    ltd = pd.read_csv(d / "ltdata.csv").sort_values("ut_ms").set_index("ut_ms")
    union = ltd.index.union(feats.index)
    ltd_h = ltd.reindex(union).interpolate(method="index").reindex(feats.index).ffill().bfill()
    for c in ltd_h.columns:
        feats[f"lt_{c}"] = ltd_h[c].to_numpy()

    # saaf : angles du Soleil, moyenne horaire 
    sa = pd.read_csv(d / "saaf.csv")
    sa["ut_ms"] = (sa["ut_ms"] // H) * H
    sa_h = sa.groupby("ut_ms")[["sa", "sx", "sy", "sz"]].mean()
    sa_h = sa_h.reindex(feats.index).interpolate(limit_direction="both")
    for c in sa_h.columns:
        feats[f"saaf_{c}"] = sa_h[c].to_numpy()

    # ftl : modes de pointage + communications
    ftl = pd.read_csv(d / "ftl.csv")
    b, e = idx_min(ftl["utb_ms"]), idx_min(ftl["ute_ms"])
    for t in FTL_TYPES:
        m = (ftl["type"] == t).to_numpy()
        feats[f"ftl_{t}"] = _frac_heure(_intervalle_minute(b[m], e[m], n)) if m.any() else 0.0
    m = ftl["flagcomms"].astype(bool).to_numpy()
    feats["ftl_comms"] = _frac_heure(_intervalle_minute(b[m], e[m], n))

    # evtf : éclipses, occultations, orbite, contacts 
    ev = pd.read_csv(d / "evtf.csv").sort_values("ut_ms")
    t_ev = idx_min(ev["ut_ms"])
    desc = ev["description"]

    def etat(start_pat, end_pat):
        is_s = desc.str.contains(start_pat, regex=True).to_numpy()
        is_e = desc.str.contains(end_pat, regex=True).to_numpy()
        sel = is_s | is_e
        return _frac_heure(_etat_minute(t_ev[sel], is_s[sel].astype(float), n))

    feats["ev_umbra"] = etat(r"^MAR_UMBRA_START", r"^MAR_UMBRA_END")
    feats["ev_penumbra"] = etat(r"^MAR_PENUMBRA_START", r"^MAR_PENUMBRA_END")
    feats["ev_occ_mars"] = etat(r"^OCC_MARS_START", r"^OCC_MARS_END")
    feats["ev_mrb_range"] = etat(r"^MRB_/_RANGE_\d+KM_START", r"^MRB_/_RANGE_\d+KM_END")

    # comptages d'événements par heure
    def compte(masque):
        c = np.bincount(np.clip(t_ev[masque] // 60, 0, nh - 1), minlength=nh)
        return c

    feats["ev_aos"] = compte(desc.str.contains("_AOS_", regex=False).to_numpy())
    feats["ev_los"] = compte(desc.str.contains("_LOS_", regex=False).to_numpy())
    feats["ev_km_ascend"] = compte(desc.str.contains("KM_ASCEND").to_numpy())
    feats["ev_km_descend"] = compte(desc.str.contains("KM_DESCEND").to_numpy())
    feats["ev_apocentre"] = compte(desc.str.contains("APOCENTRE_PASSAGE").to_numpy())
    feats["ev_pericentre"] = compte(desc.str.contains("^PERICENTRE_PASSAGE", regex=True).to_numpy())

    # phase orbitale : temps écoulé depuis le dernier péricentre / durée de l'orbite
    peri = np.sort(ev.loc[desc.str.contains("^PERICENTRE_PASSAGE", regex=True), "ut_ms"].to_numpy())
    if len(peri) > 2:
        pos = np.searchsorted(peri, heures, side="right") - 1
        pos = np.clip(pos, 0, len(peri) - 2)
        prev, nxt = peri[pos], peri[pos + 1]
        phase = np.clip((heures - prev) / np.maximum(nxt - prev, 1), 0, 1)
        feats["orb_phase"] = phase
        feats["orb_phase_sin"] = np.sin(2 * np.pi * phase)
        feats["orb_phase_cos"] = np.cos(2 * np.pi * phase)

    # dmop : nombre de commandes par heure et par famille (préfixe 4 car.)
    dm = pd.read_csv(d / "dmop.csv")
    dm["pref"] = dm["subsystem"].str[:4]
    if prefixes is None:
        prefixes = dm["pref"].value_counts().head(N_PREFIXES).index.tolist()
    dm["heure"] = (dm["ut_ms"] // H) * H
    cnt = dm.groupby(["heure", "pref"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=prefixes, fill_value=0).reindex(feats.index, fill_value=0)
    for p in prefixes:
        feats[f"dmop_{p}"] = cnt[p].to_numpy()
    feats["dmop_total"] = dm.groupby("heure").size().reindex(feats.index, fill_value=0).to_numpy()

    return feats, prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dossier avec dmop, evtf, ftl, saaf, ltdata (.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feats, _ = build_features(a.dir)
    feats.to_csv(a.out)
    print(f"{feats.shape[0]} heures x {feats.shape[1]} features -> {a.out}")
    print(f"NaN restants : {int(feats.isna().sum().sum())}")


if __name__ == "__main__":
    main()