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
    git clone https://github.com/your-username/contextuality_from_the_vacuum.git
    cd contextuality_from_the_vacuum
    ```

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

The main plots from the paper can be reproduced by running the `plots/paper_plots.ipynb` Jupyter notebook.

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Run the notebook:**

    *   Navigate to the `plots/` directory in the Jupyter interface.
    *   Open `paper_plots.ipynb`.
    *   Run all the cells in the notebook.

    The generated plots will be saved in the `plots/output/` directory.

## Repository Structure

```
.
├── plots/
│   ├── paper_plots.ipynb       # Jupyter notebook to generate the paper plots
│   └── output/                 # Directory for the generated plots
├── src/
│   ├── magic/                  # Modules related to Wigner negativity
│   ├── optimization/           # Modules for linear programming
│   ├── qft/                    # Modules for quantum field theory calculations
│   └── utils/                  # Utility functions
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Citation

If you use this code in your research, please cite the following paper:

```
[Citation information for the paper to appear here]
```
