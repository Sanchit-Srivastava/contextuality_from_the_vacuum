# Contextuality from the Vacuum

This repository contains the code to reproduce the plots in the paper [Contextuality from the vacuum](https://arxiv.org/abs/2508.15001).

## Workflows

There are two ways to reproduce the plots depending on your preference.

### Jupyter notebook

`notebooks/paper_plots.ipynb` is a self-contained, reader-friendly notebook
that generates all eight paper figures inline.  Each plot has its own
markdown section explaining the physical setup and the parameter choices.
Open it in JupyterLab or VS Code and run all cells:

```bash
git clone https://github.com/Sanchit-Srivastava/contextuality_from_the_vacuum.git
cd contextuality_from_the_vacuum
pip install -r requirements.txt
jupyter lab notebooks/paper_plots.ipynb
```

A pre-executed version of the notebook (with outputs included) is committed
to the repository at each tagged release, so the plots can be browsed
directly on GitHub or downloaded from the Zenodo archive without running any
code.

### Command-line (Make)

If you prefer to generate the plots as PDFs from the command line, a single
Make target handles everything — it creates a virtual environment, installs
dependencies, and writes all eight PDFs to `plots/output/`:

```bash
git clone https://github.com/Sanchit-Srivastava/contextuality_from_the_vacuum.git
cd contextuality_from_the_vacuum
make plots
```

No LaTeX installation is required — matplotlib's built-in math rendering
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
| `cf_appendix`          | Fig 2(a)     | Contextual fraction, SU(2) vs HW (d/T=10, R/T=0.1)      |
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

## Notebooks

Two notebooks are included for different purposes.

**`notebooks/paper_plots.ipynb`** — the primary reader-facing notebook.
It contains all eight plots with explicit inline code, so every parameter
choice and computation is visible.  Each figure has a markdown section
summarising the physics.  This is the recommended starting point for anyone
who wants to understand or modify the plots.

**`plots/paper_plots.ipynb`** — an auto-generated notebook produced by
`make notebook`.  It calls into `scripts/plot_definitions.py` rather than
containing explicit code, and is re-generated from scratch on each run.
A pre-executed copy (with outputs) is committed to the repository at each
tagged release for archival and Zenodo purposes.

## Repository Structure

```
.
├── notebooks/
│   └── paper_plots.ipynb         # Reader-friendly notebook: all 8 plots with explanations
├── plots/
│   └── paper_plots.ipynb         # Auto-generated notebook (make notebook); committed at release tags
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

> Lima, C., Preciado-Rivas, M. R., & Srivastava, S. (2025). *Contextuality from the vacuum*. arXiv:2508.15001. https://arxiv.org/abs/2508.15001
