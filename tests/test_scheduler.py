import numpy as np

from vam.runtime.scheduler import ACCURACY, Config, CostModel, Features, Scheduler, calibrate, features_for


def _true_ms(f: Features) -> float:
    return 30 + 0.4 * f.K + 0.3 * f.Lmax + 0.25 * f.C + 0.5 * f.F + 10 * f.R


def _rows(seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for K in (10, 40, 90):
        for cfg in (Config(4, False), Config(8, True, 6), Config(16, True, 3), Config(16, False)):
            f = features_for(cfg, K, [150, 150, 120, 90])
            rows.append((f, _true_ms(f) + rng.normal(0, 2)))
    return rows


def test_cost_model_fit_recovers_slopes():
    m = CostModel.fit(_rows())
    f = features_for(Config(8, True, 6), 50, [150, 150, 150, 150])
    assert abs(m.predict(f) - _true_ms(f)) < 15
    assert m.w[1] > 0 and m.w[4] > 0


def test_features_for():
    f = features_for(Config(8, True, 6), 30, [150, 120, 90, 60, 30])
    assert (f.K, f.Lmax, f.C, f.F, f.R) == (30, 150, 70, 8, 1)
    f = features_for(Config(4, False), 30, [150, 120, 90])
    assert (f.Lmax, f.C, f.F, f.R) == (150, 0, 4, 0)  # 4 frames -> 2 windows


def test_scheduler_picks_best_that_fits_and_shrinks():
    m = CostModel.fit(_rows())
    big = Scheduler(10_000, m).pre_plan(180.0)
    assert ACCURACY[big.config] == max(ACCURACY.values())
    tight = Scheduler(80, m).pre_plan(180.0)
    assert ACCURACY[tight.config] <= ACCURACY[big.config]
    assert tight.predicted_ms <= 80 or tight.alternatives == 0
    # post-scan with many keyframes and long windows costs more -> same or cheaper config
    s = Scheduler(150, m)
    pre = s.pre_plan(180.0)
    post = s.post_plan(120, [150, 150, 150, 150])
    assert ACCURACY[post.config] <= ACCURACY[pre.config]
    assert post.stage == "post" and s.plans[-1] is post


def test_calibrate_uses_runner_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VAM_CACHE", str(tmp_path))
    calls = []

    def run(cfg, video):
        calls.append((cfg, video))
        f = features_for(cfg, 40 + 10 * video, [150, 140, 130, 120])
        return f, _true_ms(f)

    m1 = calibrate([0, 1, 2], run)
    n = len(calls)
    assert n == 4 + 2 * 3 * 4 and m1.n_fit == 24
    m2 = calibrate([0, 1, 2], run)
    assert len(calls) == n and np.allclose(m1.w, m2.w)


def test_planner_hook_changes_sampler_budget(h264_video):
    import vam
    from vam.sampling.keyframe import KeyframeSampler, OracleKeyframeScorer

    path, fps, n = h264_video
    v = vam.load_video(path)
    seen = {}

    def planner(kf, scores, chosen, wins):
        seen["K"] = len(kf)
        return 6, False, 3

    s = KeyframeSampler(16, OracleKeyframeScorer([(20.0, 30.0)], 2.0), window_s=2.0, segments=2, planner=planner)
    idx = s.select(v)
    assert seen["K"] > 0 and len(idx) == 6 and s.num_frames == 6
    v.close()


def test_cost_model_fit_ignores_one_off_stall():
    rows = _rows()
    f0, ms0 = rows[0]
    rows.append((f0, ms0 + 400))  # the same configuration measured once more, with a stall
    m = CostModel.fit(rows)
    f = features_for(Config(8, True, 6), 50, [150, 150, 150, 150])
    assert abs(m.predict(f) - _true_ms(f)) < 20
    assert (m.w >= 0).all() and m.resid_p75 >= 0


def test_budgeted_sampler_calibrates_and_plans(h264_video):
    """016: one class takes a ms budget: calibrates on a clip, plans per video, decodes through the keyframe path."""
    import vam
    from vam.sampling.keyframe import OracleKeyframeScorer

    path, fps, n = h264_video
    scorer = OracleKeyframeScorer([(20.0, 30.0)], 2.0)
    s = vam.BudgetedSampler(10_000, model=None, calibrate_on=[path], scorer=scorer, cache=False, repeats=1,
                            window_s=2.0, decode_backend="serial")
    assert s.cost.n_fit >= 4 and repr(s).startswith("BudgetedSampler(budget_ms=10000")
    v = vam.load_video(path)
    batch = s(v)
    assert s.last_plan is not None and s.last_plan.budget_ms == 10_000
    assert len(batch) == s.num_frames == s.last_plan.config.frames
    assert batch.indices.max() < n
    v.close()
