PYTHON ?= python3
OUT ?= runs/smoke

.PHONY: setup test smoke score
setup:
	$(PYTHON) -m venv .venv
test:
	$(PYTHON) -m unittest discover -s tests -v
smoke:
	$(PYTHON) -m cacheboard.smoke --out "$(OUT)"
score:
	$(PYTHON) -m cacheboard.evaluate --labels "$(OUT)/private/labels.jsonl" --predictions "$(OUT)/predictions.csv"
