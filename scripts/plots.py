#!/usr/bin/env python3
"""
CLI for generating paper plots.

Usage examples
--------------
Generate all plots as PDFs (default):
    python scripts/plots.py

Generate a single plot:
    python scripts/plots.py --plot cf_large

Use LaTeX rendering (requires a TeX installation):
    python scripts/plots.py --latex

Generate a Jupyter notebook instead of running directly:
    python scripts/plots.py --format notebook

List available plots:
    python scripts/plots.py --list
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

# Suppress integration / plotting warnings that clutter output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Make sure the scripts/ directory is importable so we can find
# plot_definitions alongside this file.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_definitions import PLOTS, PLOT_NAMES  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "output"


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _make_progress_callback(plot_name: str, desc: str):
    """Return a callback that prints a simple progress bar to stderr."""
    cols = shutil.get_terminal_size((80, 24)).columns
    bar_width = min(40, cols - 40)

    def callback(current: int, total: int) -> None:
        frac = current / total if total else 1.0
        filled = int(bar_width * frac)
        bar = "=" * filled + "-" * (bar_width - filled)
        pct = frac * 100
        line = f"\r  [{bar}] {pct:5.1f}%  {plot_name}"
        sys.stderr.write(line)
        sys.stderr.flush()
        if current >= total:
            sys.stderr.write("\n")

    return callback


# ---------------------------------------------------------------------------
# Notebook generation
# ---------------------------------------------------------------------------

def _generate_notebook(
    plots: list[str],
    output_dir: Path,
    use_latex: bool,
) -> Path:
    """Generate and execute a Jupyter notebook that produces the requested plots."""
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

    nb = new_notebook()

    # --- Setup cell ---
    latex_block = ""
    if use_latex:
        latex_block = """
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\\usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
"""
    else:
        latex_block = """
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
})
"""

    setup_src = f"""\
import sys, os, warnings
from pathlib import Path

# Ensure src/ is importable
for p in [Path.cwd(), *Path.cwd().parents]:
    cand = p / "src"
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

# Ensure scripts/ is importable (for plot_definitions)
for p in [Path.cwd(), *Path.cwd().parents]:
    cand = p / "scripts"
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
{latex_block}
from plot_definitions import PLOTS

OUTPUT_DIR = "{output_dir.as_posix()}"
"""
    nb.cells.append(new_markdown_cell(
        "# Paper plots\n\n"
        "> This notebook generates the plots as they appear in the paper.\n"
    ))
    nb.cells.append(new_code_cell(setup_src))

    # --- One cell per plot ---
    for name in plots:
        func, desc = PLOTS[name]
        nb.cells.append(new_markdown_cell(f"## {desc}"))
        nb.cells.append(new_code_cell(
            f"PLOTS[{name!r}][0](OUTPUT_DIR, use_latex={'True' if use_latex else 'False'})\n"
            f"plt.show()"
        ))

    # Write notebook
    nb_dir = output_dir.parent  # plots/
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "paper_plots.ipynb"
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)

    return nb_path


def _execute_notebook(nb_path: Path) -> None:
    """Execute a notebook in-place using nbconvert."""
    import subprocess

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
    ]
    print(f"Executing notebook: {nb_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots for 'Contextuality from the Vacuum'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available plots: {', '.join(PLOT_NAMES)}",
    )
    parser.add_argument(
        "--plot", "-p",
        action="append",
        choices=PLOT_NAMES,
        metavar="NAME",
        help="Plot(s) to generate (can be repeated). Default: all.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["pdf", "notebook"],
        default="pdf",
        help="Output format: 'pdf' runs plots directly (default), "
             "'notebook' generates and executes a Jupyter notebook.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        default=False,
        help="Use LaTeX for text rendering (requires a TeX installation).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available plots and exit.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )

    args = parser.parse_args()

    if args.list:
        print("Available plots:")
        for name in PLOT_NAMES:
            _, desc = PLOTS[name]
            print(f"  {name:20s}  {desc}")
        return

    selected = args.plot or list(PLOT_NAMES)  # default: all
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use Agg backend when not generating a notebook (no GUI needed)
    if args.format == "pdf":
        import matplotlib
        matplotlib.use("Agg")

    if args.format == "notebook":
        nb_path = _generate_notebook(selected, args.output_dir, args.latex)
        _execute_notebook(nb_path)
        print(f"Notebook saved to {nb_path}")
    else:
        print(f"Generating {len(selected)} plot(s) -> {args.output_dir}/")
        t0 = time.time()
        for i, name in enumerate(selected, 1):
            func, desc = PLOTS[name]
            print(f"\n[{i}/{len(selected)}] {desc}")
            cb = None if args.no_progress else _make_progress_callback(name, desc)
            out = func(str(args.output_dir), use_latex=args.latex, progress_callback=cb)
            print(f"  -> {out}")
        elapsed = time.time() - t0
        m, s = divmod(int(elapsed), 60)
        print(f"\nDone in {m}m {s}s.")


if __name__ == "__main__":
    main()
