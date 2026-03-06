# Contextuality from the Vacuum

This repository contains the code to reproduce the plots in the paper [Contextuality from the vacuum](https://arxiv.org/abs/2508.15001).

## Quick Start

```bash
git clone https://github.com/Sanchit-Srivastava/contextuality_from_the_vacuum.git
cd contextuality_from_the_vacuum
make plots
```

That single command creates a virtual environment, installs dependencies,
and generates all four paper plots as PDFs in `plots/output/`.

No LaTeX installation is required -- matplotlib's built-in math rendering
is used by default.

## Makefile Targets

| Target         | Description                                          |
| -------------- | ---------------------------------------------------- |
| `make help`    | List all available targets                           |
| `make venv`    | Create a Python virtual environment in `.venv/`      |
| `make install` | Install dependencies (creates venv if needed)        |
| `make plots`   | Generate all plots as PDFs (no LaTeX required)       |
| `make plots-latex` | Generate all plots with LaTeX rendering          |
| `make notebook`| Generate and execute a Jupyter notebook              |
| `make clean`   | Remove generated plots and notebook                  |
| `make clean-all`| Remove plots, notebook, and virtual environment     |

## Generating Individual Plots

The CLI script `scripts/plots.py` supports generating specific plots and
shows a progress bar for each:

```bash
# List available plots
.venv/bin/python scripts/plots.py --list

# Generate only the large-detector contextual fraction plot
.venv/bin/python scripts/plots.py --plot cf_large

# Generate multiple specific plots
.venv/bin/python scripts/plots.py --plot cf_large --plot wigner_large

# Use LaTeX rendering
.venv/bin/python scripts/plots.py --latex

# Generate a Jupyter notebook instead of running directly
.venv/bin/python scripts/plots.py --format notebook
```

Available plot names:

| Name              | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `cf_large`        | Contextual fraction vs gap (R/T=1, large detectors)     |
| `cf_small`        | Contextual fraction vs gap (R/T=0.1, small detectors)   |
| `wigner_large`    | Wigner negativity vs gap (R/T=1)                        |
| `wigner_appendix` | Wigner negativity, SU(2) vs HW (R/T=0.1, appendix)     |

## Prerequisites

- Python 3.8+
- GNU Make

Jupyter and LaTeX are **not** required for the default `make plots` target.

## Repository Structure

```
.
├── notebooks/
│   └── paper_plots.ipynb         # Pre-existing notebook (for interactive use)
├── scripts/
│   ├── plots.py                  # CLI entry point for plot generation
│   ├── plot_definitions.py       # Plot functions (single source of truth)
│   └── generate_notebook.py      # Generates plots/paper_plots.ipynb
├── src/
│   ├── magic/                    # Wigner negativity / discrete Wigner function
│   ├── optimization/             # Linear programming for contextual fraction
│   ├── qft/                      # UDW qutrit detector state computation
│   └── utils/                    # Operators, contexts, measurements, state checks
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
└── requirements.txt
```

## Citation

If you use this code in your research, please cite the following paper:

> arXiv: arXiv:2508.15001
