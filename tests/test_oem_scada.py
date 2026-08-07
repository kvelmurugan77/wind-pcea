"""OEM SCADA compatibility tests.

Generates realistic export files in the conventions of major manufacturers and
verifies that the loader parses each one correctly (columns, date formats,
text statuses, units, delimiters, Chinese headers, wide format, preamble rows).

Run:  python tests/test_oem_scada.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "sample_data")
TMP = os.path.join(HERE, "_oem_tmp")
sys.path.insert(0, ROOT)

from windpcea import scada  # noqa: E402

STATUS_TEXT = {100: "Running", 200: "Fault", 300: "Maintenance",
               400: "Grid Loss", 500: "Curtailment", 600: "High Temp"}
STATUS_CN = {100: "运行", 200: "故障", 300: "维护", 400: "电网故障",
             500: "限电", 600: "结冰"}


def base_day():
    d = pd.read_csv(os.path.join(SAMPLE, "scada_sample.csv"))
    d = d[d["timestamp"].str.startswith("2024-01-01")].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


def write_tmp(name, df, **kwargs):
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, name)
    df.to_csv(path, index=False, **kwargs)
    return path


def check(name, result, n_turb, pmin=150, pmax=2600, min_rows=500):
    df, prof = result
    assert df["turbine"].nunique() == n_turb, f"[{name}] turbines {df['turbine'].nunique()} != {n_turb}"
    assert len(df) >= min_rows, f"[{name}] too few rows {len(df)}"
    mp = df["power_kw"].mean()
    assert pmin < mp < pmax, f"[{name}] mean power {mp:.1f} kW out of range"
    assert df["timestamp"].notna().all(), f"[{name}] unparsed timestamps"
    print(f"  OK  {name:<34} profile={prof:<10} rows={len(df):>6} meanP={mp:8.1f} kW")
    return df, prof


def main():
    base = base_day()
    base_mp = base["power_kw"].mean()

    # 1) Vestas: Turbine Name, Active Power (kW), ... text state codes
    v = pd.DataFrame({
        "Timestamp": base["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Turbine Name": base["turbine_id"],
        "Active Power (kW)": base["power_kw"],
        "Wind Speed (m/s)": base["wind_speed_mps"],
        "Nacelle Position (deg)": base["nacelle_dir_deg"],
        "Ambient Temperature (degC)": base["temp_c"],
        "Turbine State Code": base["status_code"].map(STATUS_TEXT),
    })
    check("vestas.csv", scada.load_scada(write_tmp("vestas.csv", v)), 12, pmax=base_mp * 1.1)

    # 2) Siemens Gamesa: ActivePower(kW), numeric state, compact headers
    s = pd.DataFrame({
        "Timestamp": base["timestamp"],
        "WTG Name": base["turbine_id"],
        "ActivePower(kW)": base["power_kw"],
        "WindSpeed(m/s)": base["wind_speed_mps"],
        "NacellePosition(deg)": base["nacelle_dir_deg"],
        "AmbientTemp(degC)": base["temp_c"],
        "TurbineState": base["status_code"],
    })
    check("sgre.csv", scada.load_scada(write_tmp("sgre.csv", s)), 12, pmax=base_mp * 1.1)

    # 3) Suzlon: 'Date Time', 'WTG No', 'Gen Active Power', text status
    z = pd.DataFrame({
        "Date Time": base["timestamp"].dt.strftime("%d/%m/%Y %H:%M"),
        "WTG No": base["turbine_id"],
        "Gen Active Power (kW)": base["power_kw"],
        "Wind Speed (m/s)": base["wind_speed_mps"],
        "Nacelle Direction (deg)": base["nacelle_dir_deg"],
        "Turbine Status": base["status_code"].map(STATUS_TEXT),
    })
    check("suzlon.csv", scada.load_scada(write_tmp("suzlon.csv", z)), 12, pmax=base_mp * 1.1)

    # 4) Envision: semicolon CSV, dd.mm.yyyy dates, European decimal commas
    e = pd.DataFrame({
        "Date": base["timestamp"].dt.strftime("%d.%m.%Y %H:%M:%S"),
        "Device Name": base["turbine_id"],
        "Active Power (kW)": base["power_kw"],
        "Wind Speed (m/s)": base["wind_speed_mps"],
        "Wind Direction (deg)": base["nacelle_dir_deg"],
        "Status Code": base["status_code"],
    })
    for c in ["Active Power (kW)", "Wind Speed (m/s)", "Wind Direction (deg)"]:
        e[c] = e[c].astype(str).str.replace(".", ",", regex=False)
    check("envision_semicolon.csv",
          scada.load_scada(write_tmp("envision_semicolon.csv", e, sep=";")), 12,
          pmax=base_mp * 1.1)

    # 5) Nordex: separate Date & Time columns, P-avg/V-avg/D-avg/T-avg, WEC
    n = pd.DataFrame({
        "Date": base["timestamp"].dt.strftime("%Y-%m-%d"),
        "Time": base["timestamp"].dt.strftime("%H:%M:%S"),
        "WEC": base["turbine_id"],
        "P-avg (kW)": base["power_kw"],
        "V-avg (m/s)": base["wind_speed_mps"],
        "D-avg (deg)": base["nacelle_dir_deg"],
        "T-avg (degC)": base["temp_c"],
        "Status": base["status_code"],
    })
    check("nordex.csv", scada.load_scada(write_tmp("nordex.csv", n)), 12, pmax=base_mp * 1.1)

    # 6) Goldwind: Chinese headers, Chinese text status
    g = pd.DataFrame({
        "时间": base["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "机组号": base["turbine_id"],
        "有功功率(kW)": base["power_kw"],
        "风速(m/s)": base["wind_speed_mps"],
        "机舱位置(°)": base["nacelle_dir_deg"],
        "环境温度(℃)": base["temp_c"],
        "机组状态": base["status_code"].map(STATUS_CN),
    })
    dfg, profg = check("goldwind.csv", scada.load_scada(write_tmp("goldwind.csv", g)), 12,
                       pmax=base_mp * 1.1)
    assert profg == "goldwind", f"goldwind auto-detected as {profg}"
    assert set(dfg["status"].dropna().unique()) <= {100, 200, 300, 400, 500, 600}

    # 7) Inox: power in MW
    i = pd.DataFrame({
        "Date Time": base["timestamp"],
        "Turbine ID": base["turbine_id"],
        "Active Power (MW)": base["power_kw"] / 1000.0,
        "Wind Speed (m/s)": base["wind_speed_mps"],
        "Nacelle Direction (deg)": base["nacelle_dir_deg"],
        "Turbine Status": base["status_code"],
    })
    check("inox_mw.csv", scada.load_scada(write_tmp("inox_mw.csv", i)), 12,
          pmax=base_mp * 1.1)   # MW -> kW scaling verified by mean-power range

    # 8) Wide format with units, 2 turbines (T01, T02)
    two = base[base["turbine_id"].isin(["T01", "T02"])].copy()
    ts = pd.date_range("2024-01-01", periods=144, freq="10min")
    wide = pd.DataFrame({"Timestamp": ts})
    for tid in ["T01", "T02"]:
        sub = two[two["turbine_id"] == tid].set_index("timestamp").reindex(ts)
        wide[f"{tid} Active Power (kW)"] = sub["power_kw"].values
        wide[f"{tid} Wind Speed (m/s)"] = sub["wind_speed_mps"].values
        wide[f"{tid} Nacelle Direction (deg)"] = sub["nacelle_dir_deg"].values
        wide[f"{tid} Ambient Temperature (degC)"] = sub["temp_c"].values
    check("wide_2turbines.csv", scada.load_scada(write_tmp("wide_2turbines.csv", wide)), 2,
          pmax=base_mp * 1.1, min_rows=200)

    # 9) Preamble metadata rows before the header (Envision-style export)
    v2 = v.head(3000)
    lines = ["Wind farm: Demo Farm", "Export time: 2025-01-01 00:00",
             "Created by WindPCEA test"]
    body = v2.to_csv(index=False).splitlines()
    p = os.path.join(TMP, "preamble.csv")
    with open(p, "w") as f:
        f.write("\n".join(lines + body))
    check("preamble.csv", scada.load_scada(p), 12, pmax=base_mp * 1.1)

    # 10) Envision truncated 10-char headers (real-world export style):
    #     PCTimeSt, AssetNam, Amb_Tems, Amb_Wind, Grd_Prod_, Nac_Direc,
    #     Nac_Temp, Sys_Stats_ — Grd_Prod_ is the power column
    t = pd.DataFrame({
        "PCTimeSt": base["timestamp"],
        "AssetNam": base["turbine_id"],
        "Amb_Tems": base["temp_c"],
        "Amb_Wind": base["wind_speed_mps"],
        "Grd_Prod_": base["power_kw"],
        "Nac_Direc": base["nacelle_dir_deg"],
        "Nac_Temp": base["temp_c"],
        "Sys_Stats_": base["status_code"],
    })
    dft, proft = check("envision_truncated.csv",
                       scada.load_scada(write_tmp("envision_truncated.csv", t)), 12,
                       pmax=base_mp * 1.1)
    assert proft == "envision", f"envision truncated detected as {proft}"
    assert "dir_deg" in dft.columns and "status" in dft.columns

    print("\nAll OEM compatibility tests passed OK")


if __name__ == "__main__":
    main()
