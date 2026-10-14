#!/usr/bin/env python
"""Local: success-rate tables (Wilson 95% CI) from results/<phase>/. Prints markdown."""

from __future__ import annotations

import argparse

from tdex.analysis.tables import load_summaries, select_tasks, success_table, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="e.g. results/phase0/sweep")
    ap.add_argument("--select", action="store_true", help="print Phase 0 task selection (SR in 30-70%%)")
    args = ap.parse_args()
    rows = success_table(load_summaries(args.root))
    print(to_markdown(rows))
    if args.select:
        sel = select_tasks(rows)
        print(f"\nIn band ({len(sel)}): " + ", ".join(sel))
        print("Decision rule: proceed if >= 3 in band, else RoboCasa fallback.")


if __name__ == "__main__":
    main()
