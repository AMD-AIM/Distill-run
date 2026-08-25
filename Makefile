PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PROFILE ?= rocm714
ENGINES ?= swift,trl,easydistill

.PHONY: submodules venv dev install doctor check lint fmt test smoke-multigpu wheelhouse clean

submodules:
	git submodule sync --recursive
	git submodule update --init --recursive

venv:
	python3 -m venv --system-site-packages .venv
	$(PIP) install -q --upgrade pip

dev:
	python3 scripts/setup_envs.py --profile "$(PROFILE)" --engines "$(ENGINES)"

install: dev

doctor:
	$(PY) -m distill_run doctor

check: lint test

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

test:
	$(PY) -m pytest -q

smoke-multigpu:
	DISTRIBUTED_PROCESSES=2 RUN_ID=make-fsdp2 ./run.sh trl-fsdp --max-steps 1
	DISTRIBUTED_PROCESSES=6 RUN_ID=make-zero3 ./run.sh trl-zero3 --max-steps 1

wheelhouse:
	mkdir -p wheelhouse
	$(PIP) download --dest wheelhouse --no-deps \
		-r requirements/$(PROFILE)-swift.txt \
		-r requirements/$(PROFILE)-trl.txt \
		-r requirements/$(PROFILE)-easydistill.txt \
		ms-swift==4.5.2
	$(PIP) wheel --wheel-dir wheelhouse --no-deps . third_party/trl third_party/easydistill

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
