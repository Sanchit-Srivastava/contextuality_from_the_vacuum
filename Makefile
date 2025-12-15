.PHONY: plots

plots:
	@echo "Generating and running notebook..."
	source .venv/bin/activate && python3 scripts/generate_notebook.py
	source .venv/bin/activate && jupyter nbconvert --to notebook --execute plots/paper_plots.ipynb --output-dir plots --inplace
	@echo "Done."
