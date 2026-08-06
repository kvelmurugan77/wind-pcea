"""Split a large SCADA CSV into chunks small enough to upload (e.g. < 25 MB).

Usage (Windows PowerShell / any terminal):
    python scripts\\split_scada.py your_scada.csv chunks --max-mb 20

Output: chunks/part_00001.csv, part_00002.csv ... each with the header row
repeated, plus a merge note. The chunks can be re-merged later (the analysis
tool sorts by turbine & timestamp anyway, so split order does not matter).

Optional: --by-month splits at month boundaries (needs the file to have a
parseable timestamp column; slower for huge files).

Run locally on your PC — works on files of any size because it streams.
"""
import argparse
import csv
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="large SCADA CSV file")
    ap.add_argument("outdir", nargs="?", default="chunks", help="output folder")
    ap.add_argument("--max-mb", type=float, default=20.0,
                    help="target chunk size in MB (default 20)")
    ap.add_argument("--by-month", action="store_true",
                    help="split at month boundaries (slower)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"File not found: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    target_bytes = int(args.max_mb * 1024 * 1024)
    n = 0
    with open(args.input, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            sys.exit("Empty file")
        if args.by_month:
            split_by_month(args.input, header, args.outdir, target_bytes)
            return
        part = 1
        rows = [header]
        size = sum(len(h) + 1 for h in header)
        for row in reader:
            rows.append(row)
            size += sum(len(c) + 1 for c in row)
            if size >= target_bytes:
                write_part(rows, args.outdir, part)
                part += 1
                n += size
                rows = [header]
                size = sum(len(h) + 1 for h in header)
        if len(rows) > 1:
            write_part(rows, args.outdir, part)
            n += size
    print(f"Split into {part} chunk(s) in {os.path.abspath(args.outdir)}")
    for f in sorted(os.listdir(args.outdir)):
        p = os.path.join(args.outdir, f)
        if os.path.isfile(p):
            print(f"  {f:>20}  {os.path.getsize(p)/1e6:6.1f} MB")


def split_by_month(path, header, outdir, target_bytes):
    import pandas as pd
    reader = pd.read_csv(path, chunksize=2_000_000)
    out = None
    current = None
    for chunk in reader:
        ts = chunk.iloc[:, 0].astype(str)
        try:
            months = pd.to_datetime(ts, errors="coerce").dt.to_period("M")
        except Exception:
            months = None
        if months is None:
            print("Could not parse timestamps for --by-month; use plain split.")
            return
        for m, g in chunk.groupby(months):
            key = str(m).replace("-", "_")
            if key != current:
                if out is not None:
                    out.close()
                fname = os.path.join(outdir, f"month_{key}.csv")
                out = open(fname, "w", newline="", encoding="utf-8")
                out.write(",".join(header) + "\n")
                current = key
                print("writing", fname)
            g.to_csv(out, index=False, header=False)
    if out is not None:
        out.close()
    print(f"Done. Chunks in {os.path.abspath(outdir)}")


def write_part(rows, outdir, part):
    fname = os.path.join(outdir, f"part_{part:05d}.csv")
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


if __name__ == "__main__":
    main()
