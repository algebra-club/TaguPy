.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr dist/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr mypy_report

lint: ## check style with flake8
	poetry run flake8 --config=./tox.ini tagupy tests

test: ## run tests quickly with the default Python
	poetry run pytest --doctest-modules

test-all: ## run tests on every Python version with tox
	poetry run tox -q

coverage: ## check code coverage quickly with the default Python
	poetry run coverage run \
		--omit="*/type/*" \
		--source tagupy \
		-m pytest --doctest-modules
	poetry run coverage report -m
	poetry run coverage html
	$(BROWSER) htmlcov/index.html

typing: ## check typing is working correctly
	-poetry run mypy --config-file ./pyproject.toml ./tagupy/
	$(BROWSER) mypy_report/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/tagupy.rst
	rm -f docs/modules.rst
	poetry run sphinx-apidoc -o docs/ tagupy
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	poetry run watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	poetry publish

dist: clean## builds source and wheel package
	poetry build
	ls -l dist

install: ## install the package to the active Python's site-packages
	poetry install
