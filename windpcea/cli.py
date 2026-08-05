"""Command-line interface: python -m windpcea --config config.json --scada scada.csv --outdir results"""
import argparse
import os
import sys

from . import config as cfg_mod
from .analysis import run_analysis
from .report import build_html, console_summary, export_csvs, export_excel


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WindPCEA — post-construction energy yield assessment "
                    "(DNV-style analysis from SCADA data)")
    ap.add_argument("--config", required=True, help="JSON configuration file")
    ap.add_argument("--scada", required=True, help="SCADA data file (CSV/XLSX)")
    ap.add_argument("--outdir", default="results", help="output directory")
    args = ap.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"WindPCEA — running post-construction assessment on {args.scada} ...")
    results = run_analysis(cfg, args.scada, outdir=args.outdir)

    html = build_html(results, args.outdir)
    xlsx = export_excel(results, args.outdir)
    csvs = export_csvs(results, args.outdir)

    print(console_summary(results))
    print(f"\nOutputs written to {os.path.abspath(args.outdir)}:")
    print(f"  report : {html}")
    print(f"  excel  : {xlsx}")
    for c in csvs:
        print(f"  csv    : {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
