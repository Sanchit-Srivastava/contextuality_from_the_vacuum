#!/usr/bin/env python3
"""
Generate ``plots/paper_plots.ipynb`` programmatically.

This is a thin wrapper: the actual plotting logic lives in
``plot_definitions.py`` so that the same code can be exercised both as a
standalone script (``plots.py``) and inside the generated notebook.

Run directly::

    python scripts/generate_notebook.py          # writes plots/paper_plots.ipynb
    python scripts/generate_notebook.py --latex   # enable LaTeX rendering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Ensure sibling module is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_definitions import PLOTS, PLOT_NAMES  # noqa: E402


def generate(*, use_latex: bool = False, output_dir: str = "output") -> Path:
    """Build the notebook and return the path it was written to."""
    nb = new_notebook()

    # ---- header ----
    nb.cells.append(new_markdown_cell(
        "# Paper plots\n\n"
        "> This notebook generates the plots as they appear in the paper.\n"
    ))

    # ---- setup cell ----
    if use_latex:
        rc_block = """\
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\\usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
"""
    else:
        rc_block = """\
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
})
"""

    setup_code = f"""\
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

{rc_block}
from plot_definitions import PLOTS

OUTPUT_DIR = "{output_dir}"
"""
    nb.cells.append(new_code_cell(setup_code))

    # ---- one cell per plot ----
    for name in PLOT_NAMES:
        _, desc = PLOTS[name]
        nb.cells.append(new_markdown_cell(f"## {desc}"))
        nb.cells.append(new_code_cell(
            f"PLOTS[{name!r}][0](OUTPUT_DIR, use_latex={'True' if use_latex else 'False'})\n"
            f"plt.show()"
        ))

    # ---- write ----
    nb_dir = Path("plots")
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "paper_plots.ipynb"
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)

    print(f"Successfully generated {nb_path}")
    return nb_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots/paper_plots.ipynb")
    parser.add_argument("--latex", action="store_true",
                        help="Enable LaTeX text rendering in the notebook.")
    args = parser.parse_args()
    generate(use_latex=args.latex)
