.PHONY: install install-dev check-src test help

help:
	@echo "make"
	@echo "    install"
	@echo "        Install the application to the active Python's site-packages"
	@echo "    install-dev"
	@echo "        Install the application to the active Python's site-packages in development mode"
	@echo "    check-src"
	@echo "        Run mypy on the source code"
	@echo "    test"
	@echo "        Run tests"
	@echo "    help"
	@echo "        Show this help message and exit"

install:
	poetry run python -m pip install -U pip
	poetry install --only main

install-dev:
	poetry run python -m pip install -U pip
	poetry install
	poetry run pre-commit install

check-src:
	poetry run mypy src/  --check-untyped-defs --explicit-package-base

test:
	poetry run pytest tests/
