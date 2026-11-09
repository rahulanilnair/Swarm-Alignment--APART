"""Tests for the headline trace table in paper.py. Run from analysis/: uv run pytest -q"""
import paper


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
    assert r"\url{results/e4b_heldout_agent/predictions/unlogged_cross_read_v1.csv}" in text
    rows = [line for line in text.splitlines() if line.endswith(r"\\") and "&" in line][1:]
    assert len(rows) == len(paper.TRACE)
