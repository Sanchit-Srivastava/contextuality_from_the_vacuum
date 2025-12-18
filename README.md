# Contextuality from the Vacuum

This repository contains the code to reproduce the plots in the paper [Contextuality from the vacuum](https://arxiv.org/abs/2508.15001)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Sanchit-Srivastava/contextuality_from_the_vacuum.git
    cd contextuality_from_the_vacuum

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\\Scripts\\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Reproducing the Plots

The main plots from the paper can be reproduced using one of two methods:

### Method 1: Using the Makefile (Automated)

The easiest way to generate the plots is to use the provided Makefile, which will automatically generate and execute the notebook.

1.  **Ensure you have a virtual environment activated:**

    ```bash
    source .venv/bin/activate  # On Windows, use `.venv\\Scripts\\activate`
    ```

2.  **Run the make command:**

    ```bash
    make plots
    ```

    This command will:
    *   Generate the `plots/paper_plots.ipynb` notebook from the Python script
    *   Execute the notebook to produce all the plots
    *   Save the generated plots in the `plots/` directory

### Method 2: Using the Existing Notebook (Manual)

Alternatively, you can manually run the pre-existing notebook in the `notebooks/` directory.

1.  Open an IDE of your choice which supports Jupyter notebooks (e.g., JupyterLab, VSCode).

2.  **Run the notebook:**

    *   Navigate to the `notebooks/` directory in the Jupyter interface.
    *   Open `paper_plots.ipynb`.
    *   Run all the cells in the notebook.

    The generated plots will be saved in the current working directory.

## Repository Structure

```
.
├── notebooks/
│   └── paper_plots.ipynb       # Pre-existing Jupyter notebook to generate the paper plots
├── scripts/
│   └── generate_notebook.py    # Script to programmatically generate the notebook
├── src/
│   ├── magic/                  # Modules related to Wigner negativity
│   ├── optimization/           # Modules for linear programming
│   ├── qft/                    # Modules for quantum field theory calculations
│   └── utils/                  # Utility functions
├── .gitignore
├── LICENSE
├── Makefile                    # Makefile for automated notebook generation and execution
├── README.md
└── requirements.txt
```

## Citation

If you use this code in your research, please cite the following paper:
 
> arXiv:  	arXiv:2508.15001

