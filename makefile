.PHONY: docs

help:
	@echo "Available commands:"
	@echo " - test          : runs pytest on entire project in verbose mode"
	@echo " - clean         : removes unnecessary files from project"
	@echo " - style         : runs isort, flake8, and black on entire project"
	@echo " - pre-commit    : adds all files to staging area and runs all installed pre-commit hooks"
	@echo " - check         : runs style, test, clean and finally pre-commit"
	@echo " - docs          : generates project documentation locally, saves in /Docs folder"
	@echo " - gitkeep       : recursively deletes all .gitkeep files from the project (only do this on project completion, gitkeep files are what keeps the empty template structure)"

test:
	pytest -v

clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -exec rm -rf {} +

style:
	isort -rc -y
	flake8
	black .

pre-commit:
	git add -A
	pre-commit run
	git add -A

check: style test clean pre-commit

docs:
	pdoc --html --force --output-dir Docs src

gitkeep:
	find . -name ".gitkeep" -exec rm -rf {} +
