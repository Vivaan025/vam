import torch

import vam


def test_feature_cache_lru_and_budget():
    c = vam.FeatureCache(budget_mb=1 / 1024)  # 1 KB
    t = torch.zeros(64, dtype=torch.float32)  # 256 bytes each
    for k in range(4):
        c.put(k, t)
    assert len(c) == 4
    assert c.get(0) is not None  # touch 0 so it becomes most recent
    c.put(4, t)  # evicts 1, the least recently used
    assert c.get(1) is None
    assert c.get(0) is not None
    assert 0 < c.hit_rate < 1


def test_profiler_records_spans():
    p = vam.Profiler()
    with p.span("a"):
        torch.ones(10).sum()
    with p.span("b"):
        pass
    assert [s.name for s in p.spans] == ["a", "b"]
    assert "total" in p.report()
    assert set(p.as_dict()) == {"a", "b"}


def test_cuda_usable_is_bool_and_cached():
    from vam.runtime import profiler as prof

    first = prof.cuda_usable()
    assert isinstance(first, bool)
    assert prof.cuda_usable() is first
