# Contextuality from the Vacuum

This repository contains the code to reproduce the plots in the paper [Contextuality from the vacuum](https://arxiv.org/abs/2508.15001).

## Quick Start

```bash
git clone https://github.com/Sanchit-Srivastava/contextuality_from_the_vacuum.git
cd contextuality_from_the_vacuum
make plots
```

That single command creates a virtual environment, installs dependencies,
and generates all eight paper plots as PDFs in `plots/output/`.

No LaTeX installation is required -- matplotlib's built-in math rendering
is used by default.

## Makefile Targets

| Target              | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `make help`         | List all available targets                           |
| `make venv`         | Create a Python virtual environment in `.venv/`      |
| `make install`      | Install dependencies (creates venv if needed)        |
| `make list-plots`   | Print all available plot names and descriptions      |
| `make plots`        | Generate plots as PDFs (no LaTeX required)           |
| `make plots-latex`  | Generate plots with LaTeX rendering                  |
| `make notebook`     | Generate and execute a Jupyter notebook              |
| `make clean`        | Remove generated plots and notebook                  |
| `make clean-all`    | Remove plots, notebook, and virtual environment      |

### Selecting which plots to generate

Both `make plots` and `make plots-latex` accept an optional `PLOTS` variable
containing a space-separated list of plot names. Omitting it generates all
plots.

```bash
# Generate all plots (default)
make plots

# Generate a single plot
make plots PLOTS="cf_large"

# Generate a subset
make plots PLOTS="cf_large wigner_small cf_fixed_romega"

# Same with LaTeX rendering
make plots-latex PLOTS="cf_large wigner_small"
```

Run `make list-plots` to see every available name.

## Available Plots

| Name                   | Paper figure | Description                                              |
| ---------------------- | ------------ | -------------------------------------------------------- |
| `cf_large`             | Fig 1(a)     | Contextual fraction vs gap (R/T=1, large detectors)      |
| `cf_small`             | Fig 1(b)     | Contextual fraction vs gap (R/T=0.1, small detectors)    |
| `wigner_large`         | Fig 1(c)     | Wigner negativity vs gap (R/T=1)                         |
| `wigner_small`         | Fig 1(d)     | Wigner negativity vs gap (R/T=0.1, small detectors)      |
| `cf_appendix`          | Fig 2(a)     | Contextual fraction, SU(2) vs HW (R/T=0.1, appendix)    |
| `wigner_appendix`      | Fig 2(b)     | Wigner negativity, SU(2) vs HW (R/T=0.1, appendix)      |
| `cf_fixed_romega`      | Fig 3(a)     | Contextual fraction, fixed RΩ=0.01, dΩ=20 (appendix C)  |
| `wigner_fixed_romega`  | Fig 3(b)     | Wigner negativity, fixed RΩ=0.01, dΩ=20 (appendix C)    |

## Generating Plots via the CLI Directly

After running `make install`, the CLI script `scripts/plots.py` can be invoked
directly for finer control:

```bash
# List available plots
.venv/bin/python scripts/plots.py --list

# Generate a single plot
.venv/bin/python scripts/plots.py --plot cf_large

# Generate multiple specific plots
.venv/bin/python scripts/plots.py --plot cf_large --plot wigner_large

# Use LaTeX rendering
.venv/bin/python scripts/plots.py --latex

# Generate a Jupyter notebook instead of PDFs
.venv/bin/python scripts/plots.py --format notebook
```

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
