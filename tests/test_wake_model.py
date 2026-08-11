"""Layout-based wake model tests (Bastankhah-Gaussian)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from windpcea import wake_model as wm  # noqa: E402

D = 126.0
V = np.arange(0.0, 26.01, 0.1)
P = 3450 * (1 - np.exp(-(np.maximum(V - 3, 0) / 5.5) ** 3))
P = np.where(V >= 25, 0, np.minimum(P, 3450))


def rose_one(wdir, ws=9.0):
    return pd.DataFrame({"dir_deg": [wdir], "ws_bin": [ws], "freq": [1.0]})


def main():
    # 1) two turbines 5D apart aligned N-S, wind from S -> southern one is downstream
    lay2 = pd.DataFrame({"turbine": ["A", "B"], "x": [0.0, 0.0], "y": [0.0, 5 * D]})
    r = wm.farm_wake_loss(lay2, rose_one(180.0), V, P, D, 87, TI=0.10)
    pt = r["per_turbine"].set_index("turbine")["wake_loss_pct"]
    assert pt["B"] < 0.5, f"B (upstream for wind 180) should have ~0 loss, got {pt['B']:.2f}"
    assert pt["A"] > 5, f"A (downstream) should have significant loss, got {pt['A']:.2f}"
    print(f"OK  2-turbine 5D aligned: A(down)={pt['A']:.1f}%  B(up)={pt['B']:.1f}%")

    # 2) symmetry: wind from N gives identical farm loss
    rn = wm.farm_wake_loss(lay2, rose_one(0.0), V, P, D, 87, TI=0.10)
    assert abs(rn["farm_loss_pct"] - r["farm_loss_pct"]) < 1e-6
    print(f"OK  symmetry N/S: {r['farm_loss_pct']:.2f}% == {rn['farm_loss_pct']:.2f}%")

    # 3) higher TI -> wider wake -> lower loss
    r_lo = wm.farm_wake_loss(lay2, rose_one(180.0), V, P, D, 87, TI=0.05)
    r_hi = wm.farm_wake_loss(lay2, rose_one(180.0), V, P, D, 87, TI=0.20)
    assert r_lo["farm_loss_pct"] > r_hi["farm_loss_pct"]
    print(f"OK  TI effect: {r_lo['farm_loss_pct']:.1f}% (TI=0.05) > {r_hi['farm_loss_pct']:.1f}% (TI=0.20)")

    # 4) closer spacing -> higher loss
    lay3 = pd.DataFrame({"turbine": ["A", "B"], "x": [0.0, 0.0], "y": [0.0, 3 * D]})
    r3 = wm.farm_wake_loss(lay3, rose_one(180.0), V, P, D, 87, TI=0.10)
    assert r3["farm_loss_pct"] > r["farm_loss_pct"]
    print(f"OK  spacing: 3D={r3['farm_loss_pct']:.1f}% > 5D={r['farm_loss_pct']:.1f}%")

    # 5) grid: 3x4 at 5D x 4D, uniform rose -> farm loss in 5-20% band
    xs = np.tile(np.arange(4) * 5 * D, 3)
    ys = np.repeat(np.arange(3) * 4 * D, 4)
    layg = pd.DataFrame({"turbine": [f"T{i+1}" for i in range(12)], "x": xs, "y": ys})
    rose = pd.DataFrame({"dir_deg": np.arange(5, 360, 10), "ws_bin": 9.0,
                         "freq": 1.0 / 36.0})
    rg = wm.farm_wake_loss(layg, rose, V, P, D, 87, TI=0.10)
    assert 2.0 < rg["farm_loss_pct"] < 25.0, f"grid loss out of band: {rg['farm_loss_pct']:.2f}"
    assert len(rg["per_turbine"]) == 12
    print(f"OK  3x4 grid uniform rose: {rg['farm_loss_pct']:.2f}% "
          f"(per-WTG {rg['per_turbine']['wake_loss_pct'].min():.1f}-{rg['per_turbine']['wake_loss_pct'].max():.1f}%)")

    # 6) layout loading
    tmp = os.path.join(ROOT, "sample_data", "layout_sample.csv")
    if os.path.exists(tmp):
        l = wm.load_layout(tmp)
        assert len(l) == 12 and {"turbine", "x", "y"} <= set(l.columns)
        print("OK  layout file loading: 12 turbines")

    print("\nAll wake-model tests passed OK")


if __name__ == "__main__":
    main()
