# Simulated grasping - common workflows.
# Every target is safe to run from a fresh clone in this order.

# Windows (including Git Bash / MSYS) puts venv executables in Scripts\, not bin/.
ifeq ($(OS),Windows_NT)
  PYTHON  ?= python
  VENVBIN := Scripts
  EXE     := .exe
else
  PYTHON  ?= python3
  VENVBIN := bin
  EXE     :=
endif

VENV ?= .venv
PY   := $(VENV)/$(VENVBIN)/python$(EXE)
PIP  := $(VENV)/$(VENVBIN)/pip$(EXE)
RUFF := $(VENV)/$(VENVBIN)/ruff$(EXE)

WORKERS  ?= 4
EPISODES ?= 10000
DATA     ?= data/grasp10k
RUN      ?= runs/grasp_cnn
TRIALS   ?= 200

.PHONY: help install assets check baseline dataset train evaluate media test lint clean all

help:
	@echo "make install    create $(VENV) and install simgrasp (editable)"
	@echo "make assets     fetch the Franka Panda MJCF from MuJoCo Menagerie (~33 MB)"
	@echo "make check      verify the install end to end"
	@echo "make baseline   scripted-grasp success over $(TRIALS) trials"
	@echo "make dataset    collect $(EPISODES) episodes into $(DATA)"
	@echo "make train      train the grasp network into $(RUN)"
	@echo "make evaluate   seen vs held-out category success for every policy"
	@echo "make media      render the README GIF and prediction figure"
	@echo "make test       run the test suite"
	@echo ""
	@echo "Override any of: WORKERS=$(WORKERS) EPISODES=$(EPISODES) DATA=$(DATA) RUN=$(RUN)"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -q --upgrade pip

install: $(VENV)
	$(PIP) install -e ".[dev]"

assets: | $(VENV)
	$(PY) scripts/fetch_assets.py

check:
	$(PY) scripts/check_install.py

baseline:
	$(PY) scripts/run_baseline.py --episodes $(TRIALS) --workers $(WORKERS)

dataset:
	$(PY) scripts/collect_dataset.py --episodes $(EPISODES) --workers $(WORKERS) \
		--split seen --out $(DATA)

train:
	$(PY) scripts/train.py --data $(DATA) --out $(RUN)

evaluate:
	$(PY) scripts/evaluate.py --checkpoint $(RUN)/best.pt --workers $(WORKERS)

media:
	$(PY) scripts/make_media.py --gif --episodes 4 --successes-only
	$(PY) scripts/make_media.py --figure --checkpoint $(RUN)/best.pt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(RUFF) check src scripts tests

# The whole project, end to end. Hours, not minutes.
all: install assets check baseline dataset train evaluate media

clean:
	rm -rf $(DATA) $(RUN) results media/*.gif media/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
