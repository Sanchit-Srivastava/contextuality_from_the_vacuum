PYTHON   ?= python3
VENV_DIR ?= .venv
PIP      := $(VENV_DIR)/bin/pip
PY       := $(VENV_DIR)/bin/python

# Optional: space-separated list of plot names to generate.
# Leave unset (or empty) to generate all plots.
# Example: make plots PLOTS="cf_large wigner_small"
PLOTS    ?=

# Build the --plot flags from $(PLOTS), or nothing if PLOTS is empty.
_PLOT_FLAGS := $(foreach p,$(PLOTS),--plot $(p))

.PHONY: help venv install plots plots-latex list-plots notebook clean clean-all

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

list-plots: install  ## List all available plot names
	$(PY) scripts/plots.py --list

plots: install  ## Generate plots as PDFs (use PLOTS="name1 name2" to select; default: all)
	mkdir -p plots/output
	$(PY) scripts/plots.py --output-dir plots/output $(_PLOT_FLAGS)

plots-latex: install  ## Generate plots with LaTeX rendering (use PLOTS="name1 name2" to select; default: all)
	mkdir -p plots/output
	$(PY) scripts/plots.py --output-dir plots/output --latex $(_PLOT_FLAGS)

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
