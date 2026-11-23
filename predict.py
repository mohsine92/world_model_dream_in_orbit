import argparse
from pathlib import Path
import pandas as pd
 
ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = Path(a.test_dir)
 
init = pd.read_csv(d / "power_init.csv")
cols = [c for c in init.columns if c != "ut_ms"]
heures = pd.read_csv(d / "heures_a_predire.csv")
pred = heures.copy()
for c in cols:
    pred[c] = init[c].mean()
pred.to_csv(a.out, index=False)
print(f"{len(pred)} heures x {len(cols)} circuits -> {a.out}")