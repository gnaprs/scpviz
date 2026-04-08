.PHONY: install install-dev test test-fast test-cov lint build docs docs-serve clean

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-fast:
	pytest tests/ -m "not slow and not external"

test-cov:
	pytest tests/ --cov=src/scpviz --cov-report=term-missing

lint:
	flake8 src/ tests/

build:
	pip install build
	python -m build

docs:
	mkdocs build

docs-serve:
	mkdocs serve

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
