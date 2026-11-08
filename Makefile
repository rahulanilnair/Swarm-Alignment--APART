# One command recomputes every number, table and figure in the paper from stored results.
#   make test    kit and analysis tests
#   make paper   rule scores, verify-then-judge, merge, statistics, numbers.tex, PDF
# LLM monitor outputs are stored under results/*/predictions and monitor_cache; re-scoring them
# needs model access (see cacheboard-kit/README.md) and is not part of `make paper`.
PYTHON ?= python3
E1 := results/e1_fixtures
E2 := results/e2_all
E2DEV := results/e2_live
E2GLM := results/e2_glm
E0 := results/e0_rahul
E3 := results/e3_replay_harness
E3A := results/e3_replay_agent
# E4, the pre-registered held-out test (analysis/PREREGISTRATION.md): development runs vs fresh runs.
RUNS := results/live_runs
DEV_RUNS := $(addprefix $(RUNS)/,main-s0 main-s1 disclosed-s0 disclosed-s1)
HELD_A := $(addprefix $(RUNS)/,main-s2 main-s3 disclosed-s2 disclosed-s3 glm-s0)
HELD_B :=
E4 := results/e4_heldout_harness
E4A := results/e4_heldout_agent
E4N := results/e4_natural
# E5, adaptive colluders (amendment to the pre-registration): collusion runs and the plain runs with the same seeds
COLLUDE_RUNS := $(wildcard $(RUNS)/collude-s*)
E5 := results/e5_collude
E4B := results/e4b_heldout_harness
E4BA := results/e4b_heldout_agent
# Add an LLM monitor here only once its predictions cover every episode (see results/*/log_*.txt).
LLM_E1 := claude_haiku55_low claude_sonnet55_low claude_opus55_low deepseek41flash_nothink glm53_nothink claude_haiku55_low_nowarn
HEATMAP := consistency_v1,claude_haiku55_low,claude_sonnet55_low,claude_opus55_low
MONITORS := cross_actor_rule_v1,consistency_v1,claude_haiku55_low,claude_sonnet55_low,claude_opus55_low,deepseek41flash_nothink,glm53_nothink,claude_sonnet55_low+verify

.PHONY: test fixtures-check numbers pdf paper heldout-build
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
	for d in $(E3) $(E3A) $(E4) $(E4A) $(E4N) $(if $(HELD_B),$(E4B) $(E4BA)) $(E2) $(E2DEV) $(E2GLM) $(if $(wildcard $(E5)/monitor_inputs),$(E5)); do (cd cacheboard-kit && $(PYTHON) -m cacheboard.score --run ../$$d --rules --derive > /dev/null && $(PYTHON) -m cacheboard.score --run ../$$d --merge > /dev/null); done
	cd analysis && uv run python paper.py --e1 ../$(E1) --e0 ../$(E0) $(if $(wildcard $(E2)/summary.json),--e2 ../$(E2)) $(if $(wildcard $(E3)/predictions.csv),--e3 ../$(E3)) $(if $(wildcard $(E3A)/predictions.csv),--e3agent ../$(E3A)) \
	  $(if $(wildcard $(E4A)/predictions.csv),--e4 ../$(E4) --e4agent ../$(E4A) --e4natural ../$(E4N) $(if $(HELD_B),--e4b ../$(E4B) --e4bagent ../$(E4BA)) --e2dev ../$(E2DEV) --e2glm ../$(E2GLM) \
	  $(if $(COLLUDE_RUNS),--collude-runs $(addprefix ../,$(foreach r,$(COLLUDE_RUNS),$(if $(wildcard $(r)/results.jsonl),$(r)))) --plain-runs $(addprefix ../,$(wildcard $(RUNS)/main-s*)) --e5 ../$(E5)) --dev-runs $(addprefix ../,$(DEV_RUNS)) --heldout-runs $(addprefix ../,$(HELD_A) $(HELD_B))) \
	  --manuscript ../manuscript --monitors $(MONITORS)

heldout-build:  ## rebuild the E4 episodes from the stored live runs (no model calls)
	cd cacheboard-kit && for src in harness agent; do $(PYTHON) -m cacheboard.replay --runs $(addprefix ../,$(DEV_RUNS) $(HELD_A)) \
	  --seeds 2 --transcript $$src --calibration-runs $(addprefix ../,$(DEV_RUNS)) --out ../results/e4_heldout_$$src; done
	$(if $(HELD_B),cd cacheboard-kit && for src in harness agent; do $(PYTHON) -m cacheboard.replay --runs $(addprefix ../,$(DEV_RUNS) $(HELD_B)) \
	  --seeds 2 --transcript $$src --calibration-runs $(addprefix ../,$(DEV_RUNS)) --out ../results/e4b_heldout_$$src; done)
	cd cacheboard-kit && $(PYTHON) -m cacheboard.episodes --runs $(addprefix ../,$(HELD_A) $(HELD_B)) --out ../$(E4N)
	cd cacheboard-kit && $(PYTHON) -m cacheboard.episodes --runs $(addprefix ../,$(foreach r,$(DEV_RUNS) $(HELD_A) $(HELD_B),$(if $(findstring glm-,$(r)),,$(r)))) --out ../$(E2)
	cd cacheboard-kit && $(PYTHON) -m cacheboard.episodes --runs $(addprefix ../,$(foreach r,$(HELD_A) $(HELD_B),$(if $(findstring glm-,$(r)),$(r)))) --out ../$(E2GLM)
	cd cacheboard-kit && $(PYTHON) -m cacheboard.episodes --runs $(addprefix ../,$(COLLUDE_RUNS) $(foreach r,$(COLLUDE_RUNS),$(if $(wildcard $(subst collude-,main-,$(r))/results.jsonl),$(subst collude-,main-,$(r))))) --out ../$(E5)

pdf:
	cd manuscript && tectonic -X compile main.tex

paper: numbers pdf
