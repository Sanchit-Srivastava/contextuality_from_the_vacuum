PYTHON   ?= python3
VENV_DIR ?= .venv
PIP      := $(VENV_DIR)/bin/pip
PY       := $(VENV_DIR)/bin/python

.PHONY: help venv install plots plots-latex notebook clean clean-all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

# ---------- environment ----------

venv: $(VENV_DIR)/bin/activate  ## Create virtual environment

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)"

install: venv  ## Install dependencies into the virtual environment
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# ---------- plots ----------

plots: install  ## Generate all plots as PDFs (no LaTeX required)
	mkdir -p plots/output
	$(PY) scripts/plots.py --output-dir plots/output

plots-latex: install  ## Generate all plots with LaTeX rendering
	mkdir -p plots/output
	$(PY) scripts/plots.py --output-dir plots/output --latex

notebook: install  ## Generate and execute a Jupyter notebook with all plots
	mkdir -p plots/output
	$(PY) scripts/generate_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute plots/paper_plots.ipynb --output-dir plots --inplace
	@echo "Notebook executed: plots/paper_plots.ipynb"

# ---------- cleanup ----------

clean:  ## Remove generated plots and notebook
	rm -rf plots/output plots/paper_plots.ipynb
	@echo "Cleaned."

clean-all: clean  ## Remove generated plots and virtual environment
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed."
