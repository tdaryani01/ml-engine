#!/usr/bin/env python3
"""Parse AMDuProf summary CSV into readable console + SHARE.txt."""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys


def section_table(lines: list[str], header_pat: str) -> tuple[str, list[dict]]:
    """Return (section_title, rows) for the first matching section."""
    start = -1
    title = ""
    for i, line in enumerate(lines):
        if re.search(header_pat, line, re.I):
            start = i + 1
            title = line.strip().strip('"')
            break
    if start < 0 or start >= len(lines):
        return "", []
    block: list[str] = []
    for j in range(start, len(lines)):
        curr = lines[j].strip()
        if not curr:
            if len(block) > 1:
                break
            continue
        # next section title (quoted or ALL CAPS)
        if (curr.startswith('"') and "HOTTEST" in curr.upper()) or (
            re.match(r"^[A-Z0-9\s\-()]{8,}$", curr) and len(block) > 1
        ):
            break
        block.append(curr)
    if len(block) <= 1:
        return title, []
    try:
        return title, list(csv.DictReader(io.StringIO("\n".join(block))))
    except Exception:
        return title, []


def row_val(row: dict | None, candidates: list[str]) -> str | None:
    if not row:
        return None
    keys = list(row.keys())
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    for name in candidates:
        for k in keys:
            if k and name.lower() in k.lower() and row[k] not in (None, ""):
                return str(row[k])
    return None


def to_float(v: str | None) -> float:
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in {"nan", "n/a", "-"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else 0.0


def metric_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    skip = {
        "function",
        "function name",
        "module",
        "process",
        "thread",
        "thread id",
        "source file",
        "source line",
        "offset",
        "cacheline address",
    }
    cols = []
    for k in rows[0].keys():
        if not k:
            continue
        if k.strip().lower() in skip:
            continue
        # keep numeric-looking metric columns
        if any(to_float(r.get(k)) != 0.0 for r in rows[:5]) or k.upper().startswith(
            ("IBS_", "CPU_", "CYCLES", "IPC", "L1_", "%", "SAMPLES")
        ):
            cols.append(k)
    # stable preference order
    prefer = [
        "IBS_LD_L1_DC_MISS_LAT",
        "IBS_LOAD_STORE",
        "IBS_LOAD",
        "IBS_STORE",
        "IBS_ST_L1_DC_MISS",
        "CYCLES_NOT_IN_HALT",
        "CPU_TIME (seconds)",
        "CPU_TIME",
        "SAMPLES",
        "IPC",
        "%L1_DC_MISSES",
    ]
    ordered = [c for c in prefer if c in cols]
    ordered += [c for c in cols if c not in ordered]
    return ordered


def primary_metric(cols: list[str], title: str = "") -> str:
    m = re.search(r"Sort Event\s*-\s*([^)\"]+)", title or "", re.I)
    if m:
        name = m.group(1).strip()
        for c in cols:
            if c == name or name in c:
                return c
    return cols[0] if cols else ""


def short_symbol(fn: str | None) -> str:
    if not fn:
        return "<unknown>"
    s = fn.strip()
    if "!" in s and not s.startswith("/"):
        # module!offset or module!symbol
        leaf = s.split("!")[-1]
        if re.match(r"^0x[0-9A-Fa-f]+$", leaf):
            return leaf
        s = leaf
    s = re.sub(r"^\s*static\s+", "", s)
    m = re.match(r"^(.*)\$omp\$\d+\(\s*\)\s*$", s)
    if m:
        base = m.group(1).strip()
        m2 = re.search(r"([A-Za-z_]\w*)\s*$", base)
        return f"{m2.group(1)} [omp]" if m2 else f"{base} [omp]"
    # omp outlined clones
    if "._omp_fn" in s or "omp_fn" in s:
        m = re.search(r"([A-Za-z_]\w*)\s*\(", s)
        if m:
            return f"{m.group(1)} [omp]"
    s = re.sub(
        r"^(?:void|int\w*|float|double|bool|long|short|unsigned|signed|struct\s+\S+|class\s+\S+|const\s+\S+)\s+",
        "",
        s,
    )
    m = re.search(r"([A-Za-z_]\w*)\s*(?:<[^>]*>)?\s*\(", s)
    if m:
        return m.group(1)
    return s[:48] + "..." if len(s) > 50 else s


def short_module(path: str | None) -> str:
    if not path:
        return "-"
    return os.path.basename(path)


def filter_kernel(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        fn = row_val(row, ["FUNCTION", "Function", "Function Name"]) or ""
        mod = row_val(row, ["Module", "MODULE"]) or ""
        if re.search(r"conv_kernels|libopenblas", mod, re.I) or re.search(
            r"process_bwd_dx|process_dw_nci|process_fwd|bwd_dx_tile|build_.*pad|conv2d_.*fallback|direct_conv|im2col|sgemm|dgemm|run_contract",
            fn,
            re.I,
        ):
            out.append(row)
    return out


def fmt_num(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def print_metric_table(
    rows: list[dict],
    title: str,
    name_key_candidates: list[str],
    limit: int = 12,
    name_label: str = "Name",
    kernel_only: bool = False,
) -> str:
    if kernel_only:
        rows = filter_kernel(rows)
    if not rows:
        return ""

    cols = metric_columns(rows)
    if not cols:
        return ""
    primary = primary_metric(cols, title)
    # drop all-zero on primary
    scored = []
    for row in rows:
        val = to_float(row.get(primary))
        if val <= 0:
            continue
        name = row_val(row, name_key_candidates) or "?"
        mod = short_module(row_val(row, ["Module", "MODULE"]))
        scored.append((val, name, mod, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ""

    show_cols = [primary] + [c for c in cols if c != primary][:4]
    total = sum(v for v, *_ in scored) or 1.0

    # console
    print()
    print(title)
    print("-" * 110)
    show_mod = name_label != "Module"
    hdr = f"{'Share%':>7} {fmt_header(primary):>14}"
    for c in show_cols[1:]:
        hdr += f" {fmt_header(c):>12}"
    hdr += f"  {name_label:<42}"
    if show_mod:
        hdr += " Module"
    print(hdr)
    print("-" * 110)

    md = [
        f"### {title}",
        "",
        "| Share% | "
        + " | ".join(fmt_header(c) for c in show_cols)
        + f" | {name_label}"
        + (" | Module |" if show_mod else " |"),
        "|------:|"
        + "|".join(["------:" for _ in show_cols])
        + ("|--------|--------|" if show_mod else "|--------|"),
    ]

    for val, name, mod, row in scored[:limit]:
        share = 100.0 * val / total
        if name_label == "Module":
            sym = short_module(name)
        elif name_label == "Thread":
            sym = name[:42]
        else:
            sym = short_symbol(name)
        line = f"{share:6.1f}% {fmt_num(val):>14}"
        for c in show_cols[1:]:
            line += f" {fmt_num(to_float(row.get(c))):>12}"
        line += f"  {sym:<42}"
        if show_mod:
            line += f" {mod}"
        print(line)
        if show_mod:
            md.append(
                f"| {share:.1f} | "
                + " | ".join(fmt_num(to_float(row.get(c))) for c in show_cols)
                + f" | `{sym}` | {mod} |"
            )
        else:
            md.append(
                f"| {share:.1f} | "
                + " | ".join(fmt_num(to_float(row.get(c))) for c in show_cols)
                + f" | `{sym}` |"
            )
    print("-" * 110)
    print(f"Shown {min(limit, len(scored))} / {len(scored)} (zeros on {fmt_header(primary)} dropped)")

    # Derived cache rates when IBS raw counts exist
    rate_lines = []
    for val, name, mod, row in scored[:limit]:
        loads = to_float(row.get("IBS_LOAD"))
        stores = to_float(row.get("IBS_STORE"))
        st_miss = to_float(row.get("IBS_ST_L1_DC_MISS"))
        dram = to_float(row.get("IBS_LD_LOCAL_DRAM_HIT"))
        lat = to_float(row.get("IBS_LD_L1_DC_MISS_LAT"))
        ld_st = to_float(row.get("IBS_LOAD_STORE")) or (loads + stores)
        st_miss_pct = (100.0 * st_miss / stores) if stores > 0 else 0.0
        avg_lat = (lat / loads) if loads > 0 else 0.0
        dram_per_k = (1000.0 * dram / loads) if loads > 0 else 0.0
        if name_label == "Module":
            sym = short_module(name)
        elif name_label == "Thread":
            sym = name[:42]
        else:
            sym = short_symbol(name)
        if loads > 0 or stores > 0:
            rate_lines.append((sym, mod, st_miss_pct, avg_lat, dram_per_k, loads, stores))

    if rate_lines and name_label in ("Symbol", "Module"):
        print()
        print(title + " — derived rates")
        print("-" * 110)
        print(f"{'StL1miss%':>10} {'AvgLdLat':>10} {'Dram/1kLd':>10} {'Loads':>10} {'Stores':>10}  {name_label:<42}")
        print("-" * 110)
        md.append(f"### {title} — derived rates")
        md.append("")
        md.append(f"| StL1miss% | AvgLdLat | Dram/1kLd | Loads | Stores | {name_label} |")
        md.append("|---------:|---------:|----------:|------:|-------:|--------|")
        for sym, mod, st_miss_pct, avg_lat, dram_per_k, loads, stores in rate_lines:
            print(f"{st_miss_pct:9.2f}% {avg_lat:10.1f} {dram_per_k:10.2f} {fmt_num(loads):>10} {fmt_num(stores):>10}  {sym:<42}")
            md.append(
                f"| {st_miss_pct:.2f} | {avg_lat:.1f} | {dram_per_k:.2f} | {fmt_num(loads)} | {fmt_num(stores)} | `{sym}` |"
            )
        print("-" * 110)
        md.append("")

    md.append("")
    return "\n".join(md)


def fmt_header(col: str) -> str:
    mapping = {
        "IBS_LD_L1_DC_MISS_LAT": "L1missLat",
        "IBS_LOAD_STORE": "Ld+St",
        "IBS_LOAD": "Loads",
        "IBS_STORE": "Stores",
        "IBS_ST_L1_DC_MISS": "StL1miss",
        "IBS_LD_LOCAL_DRAM_HIT": "DramHit",
        "IBS_LD_RMT_DRAM_HIT": "RmtDram",
        "CYCLES_NOT_IN_HALT": "Cycles",
        "CPU_TIME (seconds)": "CPU(s)",
        "CPU_TIME": "CPU(s)",
        "SAMPLES": "Samples",
        "%L1_DC_MISSES": "%L1miss",
        "L1_DC_MISSES (PTI)": "L1/pti",
        "IPC": "IPC",
    }
    return mapping.get(col, col[:12])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--wall-sec", type=float, default=0.0)
    ap.add_argument("--share-out", default="")
    args = ap.parse_args()

    lines = open(args.summary_csv, encoding="utf-8", errors="replace").read().splitlines()
    analysis = args.analysis
    wall = args.wall_sec
    share_bits: list[str] = [
        f"# uProf SHARE — analysis={analysis} wall={wall:.3f}s",
        "",
        "Paste this into chat.",
        "",
    ]

    print(f"\nAnalysis: {analysis}   wall: {wall:.3f}s")
    print("=" * 110)

    # Modules
    mtitle, mods = section_table(lines, r"HOTTEST MODULES|MODULE SUMMARY")
    if mods:
        share_bits.append(
            print_metric_table(
                mods,
                mtitle or "HOTTEST MODULES",
                ["MODULE", "Module"],
                limit=10,
                name_label="Module",
            )
        )

    # Threads
    ttitle, threads = section_table(lines, r"HOTTEST THREADS|THREAD SUMMARY")
    if threads:
        share_bits.append(
            print_metric_table(
                threads,
                ttitle or "HOTTEST THREADS",
                ["THREAD", "Thread", "THREAD ID"],
                limit=12,
                name_label="Thread",
            )
        )

    # Functions
    ftitle, funcs = section_table(
        lines, r"HOTTEST FUNCTIONS|FUNCTION SUMMARY"
    )
    if funcs:
        share_bits.append(
            print_metric_table(
                funcs,
                ftitle or "HOTTEST FUNCTIONS",
                ["FUNCTION", "Function", "Function Name"],
                limit=15,
                name_label="Symbol",
            )
        )
        share_bits.append(
            print_metric_table(
                funcs,
                "CONV / BLAS KERNELS",
                ["FUNCTION", "Function", "Function Name"],
                limit=15,
                name_label="Symbol",
                kernel_only=True,
            )
        )

    share_path = args.share_out or os.path.join(
        os.path.dirname(os.path.abspath(args.summary_csv)), "SHARE.txt"
    )
    with open(share_path, "w", encoding="utf-8") as f:
        f.write("\n".join(b for b in share_bits if b) + "\n")
    print(f"\n>>> Paste file: {share_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
