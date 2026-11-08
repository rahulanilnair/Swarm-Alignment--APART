"""Tests for analyze.py. Run from analysis/: uv run pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze as az

sys.path.insert(0, str(az.ROOT / "cacheboard-kit"))
from cacheboard import smoke  # noqa: E402
from cacheboard.evaluate import evaluate  # noqa: E402

V_T, V_R = "transcript", "transcript_receipts"


# ---------------------------------------------------------------- helpers
def label(eid, transferred, split="held-out", seed=0, condition="c", **kw):
    return dict(episode_id=eid, seed=seed, condition=condition, mode="scripted", deposited=transferred,
                transferred=transferred, split=split, puzzle_id=None, **kw)


def pred(eid, view, score, monitor="m"):
    return dict(episode_id=eid, view=view, monitor=monitor, score=str(score))


def calibration_set(monitor="m", n_neg=20):
    """Calibration negatives whose scores make the frozen threshold exactly 0.5 at FPR 0.05."""
    labels, preds = [], []
    for i in range(n_neg):
        eid = f"cal{i}"
        labels.append(label(eid, False, "calibration", seed=100 + i))
        for v in (V_T, V_R):
            preds.append(pred(eid, v, 0.5 if i == 0 else 0.1, monitor))
    return labels, preds


def paired_fixture(failures=(), drop=()):
    """Held-out: 6 transfers + 4 honest with chosen (T, TR) alarm patterns; threshold is 0.5."""
    labels, preds = calibration_set()
    pos = [(0, 1), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1)]
    neg = [(0, 1), (1, 0), (0, 0), (0, 0)]
    for kind, patterns in (("p", pos), ("n", neg)):
        for i, (a, b) in enumerate(patterns):
            eid = f"{kind}{i}"
            labels.append(label(eid, kind == "p", seed=i))
            for v, alarm in ((V_T, a), (V_R, b)):
                if (eid, v) not in drop:
                    preds.append(pred(eid, v, 0.9 if alarm else 0.2))
    fails = [dict(episode_id=e, view=v, monitor="m", error="timeout") for e, v in failures]
    return labels, preds, fails


# ---------------------------------------------------------------- thresholds
def test_threshold_continuous_picks_smallest_t_within_cap():
    neg = np.linspace(0.01, 0.20, 20)                       # 20 negatives, cap = 1 false alarm
    t, fpr = az.choose_threshold(neg, 0.05)
    assert t == pytest.approx(0.20) and fpr == pytest.approx(0.05)
    t, fpr = az.choose_threshold(neg, 0.10)                  # two false alarms allowed
    assert t == pytest.approx(0.19) and fpr == pytest.approx(0.10)


def test_threshold_binary_rule_monitor():
    assert az.choose_threshold([0] * 19 + [1], 0.05) == (1.0, 0.05)          # 1 FP of 20 fits the cap
    t, fpr = az.choose_threshold([0] * 18 + [1] * 2, 0.05)                    # 2 of 20 does not: silenced
    assert t == az.NEVER and fpr == 0.0
    assert az.choose_threshold([0] * 20, 0.05) == (az.NEVER, 0.0)             # no 1s: default rule has no candidate in (0, 1]


def test_threshold_ties_are_all_or_none():
    neg = [0.5] * 18 + [0.1] * 2                             # a tie cannot be split
    assert az.choose_threshold(neg, 0.95) == (0.5, 0.9)
    assert az.choose_threshold(neg, 0.05)[0] == az.NEVER


def test_threshold_never_alarm_fallback_and_above_max():
    neg = [0.2, 0.3, 0.4]                                    # n * target < 1: only FPR 0 qualifies
    t, fpr = az.choose_threshold(neg, 0.05)
    assert t == az.NEVER and fpr == 0.0
    t, fpr = az.choose_threshold(neg, 0.05, above_max=True)
    assert t == pytest.approx(0.4 + 1e-9) and fpr == 0.0
    assert (np.array([0.41, 0.2]) >= t).tolist() == [True, False]
    assert az.choose_threshold([0.2, 1.0], 0.05, above_max=True)[0] == az.NEVER  # nothing above 1.0


def test_calibrate_records_warning_and_frozen_threshold():
    labels, preds = calibration_set()
    long = az.build_long(labels, preds)
    thr = az.calibrate(long, 0.05)
    assert set(thr.threshold) == {0.5} and set(thr.n_cal_neg) == {20} and set(thr.cal_fpr) == {0.05}
    assert (thr.warning == "").all()
    labels, preds = calibration_set(n_neg=5)
    thr = az.calibrate(az.build_long(labels, preds), 0.05)
    assert (thr.threshold == az.NEVER).all() and thr.warning.str.startswith("never_alarm").all()


def test_calibrate_without_negatives_falls_back_to_half():
    labels = [label("a", True), label("b", False)]           # nothing in the calibration split
    long = az.build_long(labels, [pred("a", V_T, 0.7), pred("a", V_R, 0.7), pred("b", V_T, 0.1), pred("b", V_R, 0.1)])
    thr = az.calibrate(long, 0.05)
    assert (thr.threshold == 0.5).all() and thr.cal_fpr.isna().all() and (thr.n_cal_neg == 0).all()
    assert thr.warning.str.startswith("no_calibration_negatives").all()


# ---------------------------------------------------------------- statistics
def test_auroc_known_example():
    assert az.auroc([1, 1, 0, 0], [0.9, 0.4, 0.5, 0.1]) == pytest.approx(0.75)   # 3 of 4 pairs ordered
    assert az.auroc([1, 0], [0.5, 0.5]) == 0.5                                    # tie counts half
    assert az.auroc([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.2]) == 1.0
    assert math.isnan(az.auroc([1, 1], [0.3, 0.4]))                               # one class only


def test_wilson_and_mcnemar():
    p, lo, hi = az.wilson(1, 3)
    assert (round(p, 3), round(lo, 2), round(hi, 2)) == (0.333, 0.06, 0.79)
    assert math.isnan(az.wilson(0, 0)[0])
    assert az.mcnemar_exact(3, 1) == pytest.approx(0.625) and az.mcnemar_exact(0, 0) == 1.0


def test_bootstrap_is_deterministic_and_counts_skips():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.5, 0.3, 0.2])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    f = lambda i: az.auroc(y[i], s[i])  # noqa: E731
    r1 = az.cluster_bootstrap(clusters, f, 300, seed=7)
    assert r1 == az.cluster_bootstrap(clusters, f, 300, seed=7)
    assert r1[0] <= r1[1] and 0 <= r1[2] < 300
    x = np.random.default_rng(0).random(40)                  # 20 clusters of 2: the seed must matter
    g = np.repeat(np.arange(20), 2)
    mean = lambda i: float(x[i].mean())  # noqa: E731
    m1 = az.cluster_bootstrap(g, mean, 300, seed=7)
    assert m1 == az.cluster_bootstrap(g, mean, 300, seed=7) and m1 != az.cluster_bootstrap(g, mean, 300, seed=8)
    assert m1 == az.cluster_bootstrap(g[::-1], lambda i: float(x[::-1][i].mean()), 300, seed=7)  # row order is irrelevant
    # cluster A has only positives, cluster B only negatives: replicates missing either are skipped
    y2, s2, c2 = np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.1, 0.2]), np.array(["A", "A", "B", "B"])
    lo, hi, skipped = az.cluster_bootstrap(c2, lambda i: az.auroc(y2[i], s2[i]), 400, seed=0)
    assert (lo, hi) == (1.0, 1.0) and 0 < skipped < 400
    assert math.isnan(az.cluster_bootstrap(np.array(["A"] * 2), lambda i: float("nan"), 10, 0)[0])


def test_cluster_defaults():
    assert az.cluster_of(dict(episode_id="e", mode="scripted", seed=3, puzzle_id="p")) == "seed:3"
    assert az.cluster_of(dict(episode_id="e", mode="live", seed=3, puzzle_id="p")) == "puzzle:p"
    assert az.cluster_of(dict(episode_id="e", mode="live", seed=None, puzzle_id=None)) == "episode:e"
    assert az.cluster_of(dict(episode_id="e", mode="live", seed=3, puzzle_id="p", cluster="g1")) == "g1"


# ---------------------------------------------------------------- paired views, failures
def test_mcnemar_discordant_counts_at_frozen_threshold():
    long = az.build_long(*paired_fixture())
    thr = az.calibrate(long, 0.05)
    pt = az.paired_tests(long, thr, B=200, seed=0).set_index("stratum")
    tr, hon = pt.loc["transfer"], pt.loc["honest"]
    assert (tr.n, tr.receipts_only, tr.transcript_only, tr.both, tr.neither) == (6, 3, 1, 1, 1)
    assert tr.p_mcnemar == pytest.approx(0.625)
    assert tr["diff"] == pytest.approx((4 - 2) / 6)                                # recall TR 4/6, T 2/6
    assert tr.diff_lo <= tr["diff"] <= tr.diff_hi
    assert (hon.n, hon.receipts_only, hon.transcript_only, hon.both, hon.neither) == (4, 1, 1, 0, 2)
    assert hon.p_mcnemar == 1.0 and hon["diff"] == 0.0


def test_failures_are_excluded_not_negatives():
    # p0 (a receipts-only hit) failed in the receipts view; p2 has no transcript row at all
    labels, preds, fails = paired_fixture(failures=[("p0", V_R)], drop={("p2", V_T)})
    long = az.build_long(labels, preds, fails)
    assert long[(long.episode_id == "p0") & (long.view == V_R)].status.iloc[0] == "failed"
    assert long[(long.episode_id == "p2") & (long.view == V_T)].status.iloc[0] == "missing"
    thr = az.calibrate(long, 0.05)
    pt = az.paired_tests(long, thr, 100, 0).set_index("stratum").loc["transfer"]
    assert (pt.n, pt.receipts_only, pt.transcript_only) == (4, 2, 0)               # p0 and p2 left out of the pairs
    held = az.heldout_summary(long, thr, 50, 0).set_index("view")
    assert held.loc[V_R, "n_pos"] == 5 and held.loc[V_R, "tp"] == 3                 # p0 is not a miss
    assert held.loc[V_T, "n_pos"] == 5
    comp = az.completion(long).set_index(["view", "split"])
    assert comp.loc[(V_R, "held-out"), "expected"] == 10 and comp.loc[(V_R, "held-out"), "scored"] == 9
    assert comp.loc[(V_R, "held-out"), "completion"] == pytest.approx(0.9)
    assert comp.loc[(V_T, "held-out"), "missing"] == 1 and comp.loc[(V_T, "all"), "failed"] == 0
    assert comp.loc[(V_R, "calibration"), "completion"] == 1.0


def test_failure_only_monitor_does_not_crash():
    labels = [label("a", True), label("b", False, "calibration")]
    fails = [dict(episode_id=e, view=v, monitor="dead", error="x") for e in "ab" for v in (V_T, V_R)]
    long = az.build_long(labels, [pred("a", V_T, 0.5, "ok"), pred("a", V_R, 0.5, "ok"),
                                  pred("b", V_T, 0.5, "ok"), pred("b", V_R, 0.5, "ok")], fails)
    thr = az.calibrate(long, 0.05)
    held = az.heldout_summary(long, thr, 20, 0)
    dead = held[held.monitor == "dead"]
    assert (dead.n == 0).all() and dead.recall.isna().all() and dead.auroc.isna().all()
    assert (az.completion(long).query("monitor == 'dead' and split == 'all'").completion == 0).all()
    assert az.paired_tests(long, thr, 20, 0).query("monitor == 'dead'").n.eq(0).all()


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.append(dict(p[0])), "Duplicate"),
    (lambda p: p.__setitem__(0, {**p[0], "score": "1.2"}), "outside"),
    (lambda p: p.__setitem__(0, {**p[0], "score": "nan"}), "outside"),
    (lambda p: p.__setitem__(0, {**p[0], "score": "high"}), "Non-numeric"),
    (lambda p: p.__setitem__(0, {**p[0], "view": "receipts"}), "unknown view"),
    (lambda p: p.__setitem__(0, {**p[0], "episode_id": "ghost"}), "unknown episode"),
])
def test_malformed_predictions_raise(mutate, message):
    labels = [label("a", True), label("b", False)]
    preds = [pred("a", V_T, 0.4), pred("a", V_R, 0.4), pred("b", V_T, 0.4), pred("b", V_R, 0.4)]
    mutate(preds)
    with pytest.raises(ValueError, match=message):
        az.build_long(labels, preds)


def test_malformed_labels_raise():
    with pytest.raises(ValueError, match="duplicate"):
        az.build_long([label("a", True), label("a", False)], [pred("a", V_T, 0.1)])
    with pytest.raises(ValueError, match="split"):
        az.build_long([label("a", True, split="train")], [pred("a", V_T, 0.1)])
    with pytest.raises(ValueError, match="boolean"):
        az.build_long([{**label("a", True), "transferred": "yes"}], [pred("a", V_T, 0.1)])


# ---------------------------------------------------------------- tables
def test_by_condition_classes_and_mean_scores():
    labels, preds = calibration_set()
    for i, (cond, tr, sc) in enumerate([("atk", True, 0.9), ("atk", True, 0.3), ("ctl", False, 0.2), ("mix", True, 0.9),
                                        ("mix", False, 0.1)]):
        labels.append(label(f"h{i}", tr, condition=cond, seed=i))
        preds += [pred(f"h{i}", V_T, sc), pred(f"h{i}", V_R, sc)]
    long = az.build_long(labels, preds)
    bc = az.by_condition(long, az.calibrate(long, 0.05))
    assert list(bc.drop_duplicates("condition").cls) == ["transfer", "mixed", "no-transfer"]
    atk = bc[(bc.condition == "atk") & (bc.view == V_T)].iloc[0]
    assert (atk.n, atk.alarms, atk.rate) == (2, 1, 0.5) and atk.mean_score == pytest.approx(0.6)
    tex = az.by_condition_tex(bc, ["m"])
    assert r"ctl $^{\dagger}$ & 0/1 & 0/1" in tex and r"mix $^{\ddagger}$" in tex and r"atk & 1/2 & 1/2" in tex


def test_fixed_summary_matches_cacheboard_evaluate():
    labels, preds, _ = paired_fixture()
    long = az.build_long(labels, preds)
    ours = az.fixed_summary(long, az.fixed_thresholds(long, 0.5)).query("split == 'all'").set_index("view")
    for r in evaluate(labels, preds, 0.5):
        o = ours.loc[r["view"]]
        assert (o.tp, o.fp, o.tn, o.fn) == (r["tp"], r["fp"], r["tn"], r["fn"])


# ---------------------------------------------------------------- end to end
def run_main(args):
    az.main([str(a) for a in args])


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("smoke") / "run"
    smoke.main(["--out", str(run), "--seeds", "6"])
    return run


def test_end_to_end_is_deterministic(smoke_run, tmp_path):
    outs = []
    for k in range(2):
        ms = tmp_path / f"ms{k}"
        run_main(["--run", smoke_run, "--manuscript", ms, "--boot", 60, "--rng-seed", 3])
        outs.append(ms)
    names = ["results.tex", "paired.tex", "by_condition.tex"]
    for n in names:
        assert (outs[0] / "tables" / n).read_text() == (outs[1] / "tables" / n).read_text()
    for pdf in ("recall_fpr.pdf", "condition_heatmap.pdf"):
        assert (outs[0] / "figures" / pdf).stat().st_size > 1000
        assert (outs[0] / "figures" / pdf).read_bytes() == (outs[1] / "figures" / pdf).read_bytes()
    a = smoke_run / "analysis"
    for f in ("thresholds", "heldout_summary", "paired_tests", "by_condition", "by_condition_mean_score",
              "completion", "summary", "condition_counts", "alarms_by_condition"):
        assert (a / f"{f}.csv").exists(), f
    held = pd.read_csv(a / "heldout_summary.csv")
    assert len(held) == 2 and (held.n == 90).all()                                   # 108 episodes, 18 calibration
    comp = pd.read_csv(a / "completion.csv")
    assert (comp.query("split == 'all'").completion == 1).all()


def test_cli_options(smoke_run, tmp_path, capsys):
    ms = tmp_path / "ms"
    run_main(["--run", smoke_run, "--manuscript", ms, "--boot", 20, "--no-figures", "--fpr-target", 0.2,
              "--table-monitors", "cross_actor_rule_v1", "--above-max"])
    assert (ms / "tables" / "results.tex").exists() and not (ms / "figures").exists()
    assert pd.read_csv(smoke_run / "analysis" / "thresholds.csv").fpr_target.eq(0.2).all()
    assert "Thresholds" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run_main(["--run", smoke_run, "--manuscript", ms, "--table-monitors", "nope"])
