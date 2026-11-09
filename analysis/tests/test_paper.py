"""Tests for the headline trace table in paper.py. Run from analysis/: uv run pytest -q"""
from pathlib import Path

import paper

REPO = Path(paper.__file__).resolve().parents[1]


def numbers_with(keys):
    numbers = paper.Numbers()
    for k in keys:
        numbers.add(k, 1, "{:d}")
    return numbers


def test_trace_table_skips_rows_whose_numbers_were_not_built(tmp_path):
    a_only = [k for _, keys, _, _ in paper.TRACE for k in keys if "EfourB" not in k]
    out = tmp_path / "trace.tex"
    paper.trace_table(numbers_with(a_only), out)
    text = out.read_text()
    assert "held-out A: hidden caught" in text
    assert "Held-out B" not in text
    assert "--" not in text.split(r"\midrule")[1]


def test_trace_table_lists_held_out_b_once_it_exists(tmp_path):
    every = [k for _, keys, _, _ in paper.TRACE for k in keys]
    out = tmp_path / "trace.tex"
    paper.trace_table(numbers_with(every), out)
    text = out.read_text()
    assert text.count("Held-out B") == 4
    f = "results/e4b_heldout_agent/predictions/unlogged_cross_read_v1.csv"
    assert rf"\repo{{{f}}}{{{f}}}" in text
    assert r"\url{" not in text
    rows = [line for line in text.splitlines() if line.endswith(r"\\") and "&" in line][1:]
    assert len(rows) == len(paper.TRACE)


def test_add_totals_sums_held_out_a_and_b_only_when_both_exist():
    numbers = paper.Numbers()
    for k, v in {"EfourXHiddenK": 56, "EfourXHiddenN": 56, "EfourBXHiddenK": 499, "EfourBXHiddenN": 500,
                 "EfouragentXHiddenK": 56}.items():
        numbers.add(k, v, "{:d}")
    paper.add_totals(numbers, [("EfourAB", "Efour", "EfourB"), ("EfourABagent", "Efouragent", "EfourBagent")],
                     ["XHidden"])
    assert (numbers.items["EfourABXHiddenK"], numbers.items["EfourABXHiddenN"]) == ("555", "556")
    assert "EfourABagentXHiddenK" not in numbers.items


def test_repo_link_points_globs_at_their_directory():
    assert paper.repo_link("results/e2_live/") == r"\repo{results/e2_live}{results/e2_live/}"
    assert paper.repo_link("results/live_runs/collude-s*/receipts.jsonl") == \
        r"\repo{results/live_runs}{results/live_runs/collude-s*/receipts.jsonl}"
    assert paper.repo_link("results/e4_heldout_*/predictions/x.csv").startswith(r"\repo{results}{")


def test_every_trace_link_target_exists_in_the_repository():
    import re
    for _, _, files, _ in paper.TRACE:
        for f in files:
            target = re.match(r"\\repo\{([^}]*)\}", paper.repo_link(f)).group(1)
            assert (REPO / target).exists(), target
