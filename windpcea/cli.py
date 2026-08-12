"""Command-line interface: python -m windpcea --config config.json --scada scada.csv --outdir results"""
import argparse
import os
import sys

from . import config as cfg_mod
from .analysis import run_analysis
from .blockwise import run_blockwise
from .report import build_html, console_summary, export_csvs, export_excel


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WindPCEA — post-construction energy yield assessment "
                    "(DNV-style analysis from SCADA data)")
    ap.add_argument("--config", required=True, help="JSON configuration file")
    ap.add_argument("--scada", nargs="+", required=True,
                    help="SCADA data file(s) (CSV/XLSX) - multiple allowed "
                         "(e.g. per-year exports); concatenated automatically")
    ap.add_argument("--outdir", default="results", help="output directory")
    ap.add_argument("--blockwise", action="store_true",
                    help="out-of-core mode for very large files (1 GB+); "
                         "bounded memory, slower")
    ap.add_argument("--block-days", type=int, default=None,
                    help="block size in days for --blockwise (default auto)")
    args = ap.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"WindPCEA — running post-construction assessment on {args.scada} ...")
    if len(args.scada) > 1:
        print(f"  multi-file mode: {len(args.scada)} input files")
    if args.blockwise:
        print("  mode: blockwise out-of-core (bounded memory)")
        results = run_blockwise(cfg, args.scada[0] if len(args.scada) == 1 else None,
                                outdir=args.outdir, block_days=args.block_days,
                                scada_files=args.scada if len(args.scada) > 1 else None)
    else:
        results = run_analysis(cfg, args.scada[0] if len(args.scada) == 1 else None,
                               outdir=args.outdir,
                               scada_files=args.scada if len(args.scada) > 1 else None)

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
