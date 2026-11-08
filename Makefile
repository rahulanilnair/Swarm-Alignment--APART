# One command recomputes every number, table and figure in the paper from stored results.
#   make test    kit and analysis tests
#   make paper   rule scores, verify-then-judge, merge, statistics, numbers.tex, PDF
# LLM monitor outputs are stored under results/*/predictions and monitor_cache; re-scoring them
# needs model access (see cacheboard-kit/README.md) and is not part of `make paper`.
PYTHON ?= python3
E1 := results/e1_fixtures
E2 := results/e2_live
E0 := results/e0_rahul
E3 := results/e3_replay_harness
E3A := results/e3_replay_agent
# Add an LLM monitor here only once its predictions cover every episode (see results/*/log_*.txt).
LLM_E1 := claude_haiku55_low claude_sonnet55_low claude_opus55_low deepseek41flash_nothink glm53_nothink
HEATMAP := consistency_v1,claude_haiku55_low,claude_sonnet55_low,claude_opus55_low
MONITORS := cross_actor_rule_v1,consistency_v1,claude_haiku55_low,claude_sonnet55_low,claude_opus55_low,deepseek41flash_nothink,glm53_nothink,claude_sonnet55_low+verify

.PHONY: test fixtures-check numbers pdf paper
test:
	cd cacheboard-kit && $(PYTHON) -m unittest discover -s tests
	cd analysis && uv run pytest -q

fixtures-check:  ## regenerate the scripted inputs and check they match the stored ones
	rm -rf /tmp/cacheboard-e1-check && cd cacheboard-kit && $(PYTHON) -m cacheboard.smoke --out /tmp/cacheboard-e1-check --seed 42 --seeds 30
	cmp /tmp/cacheboard-e1-check/monitor_inputs/transcript_receipts.jsonl $(E1)/monitor_inputs/transcript_receipts.jsonl
	cmp /tmp/cacheboard-e1-check/private/labels.jsonl $(E1)/private/labels.jsonl

numbers:
	cd cacheboard-kit && $(PYTHON) -m cacheboard.score --run ../$(E1) --rules --derive
	cd cacheboard-kit && $(PYTHON) -m cacheboard.score --run ../$(E1) --merge --only cross_actor_rule_v1 consistency_v1 cross_or_consistency_v1 $(LLM_E1) $(addsuffix +verify,$(LLM_E1))
	cd analysis && uv run python analyze.py --run ../$(E1) --manuscript ../manuscript --table-monitors $(HEATMAP) > /dev/null
	cd analysis && uv run python paper.py --e1 ../$(E1) --e0 ../$(E0) $(if $(wildcard $(E2)/summary.json),--e2 ../$(E2)) $(if $(wildcard $(E3)/predictions.csv),--e3 ../$(E3)) $(if $(wildcard $(E3A)/predictions.csv),--e3agent ../$(E3A)) --manuscript ../manuscript --monitors $(MONITORS)

pdf:
	cd manuscript && tectonic -X compile main.tex

paper: numbers pdf
