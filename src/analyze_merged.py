"""
광주+전남 합본 쌍둥이 — 타입 분해 + 분산계수 φ 튜닝.

- 각 쌍둥이를 (사전-사전 / 본-본 / 사전-본) × (동일지역 / 교차지역) 으로 분류
- φ(분산계수): 1.0=이항(동전던지기). <1=sub-binomial(실제 투표는 변동 작음 → 쌍둥이↑)
- 근접쌍 분포(거리 d=0,1,2)로 φ 캘리브레이션
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_provinces import load_units, empirical_twins  # noqa: E402
import twin_model  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def load_merged():
    parts = []
    for path, tag in [(DATA/"crawl_provinces"/"4600_전라남도.json", "전남"),
                      (DATA/"crawl_provinces"/"2900_광주광역시.json", "광주")]:
        c, df, ia, ib = load_units(str(path))
        df["region"] = tag + "·" + df["region"]
        df["tag"] = tag
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_units(df):
    """동별 모수 + 단위 라벨(사전/본, 지역). 반환: 동테이블 D, 단위라벨(gubun,tag)."""
    D = twin_model.build_dong_table(df)
    # build_dong_table 은 region,dong 순서 유지. 단위 = [모든동 사전] + [모든동 본]
    tags = [r.split("·")[0] for r in D.region]
    gubun = ["사전"] * len(D) + ["본"] * len(D)
    tag2 = tags + tags
    return D, np.array(gubun), np.array(tag2)


def simulate_typed(D, gubun, tag, phi=1.0, R=100000, seed=7):
    N = D.N.to_numpy()
    re, rb = D.re.to_numpy(), D.rb.to_numpy()
    pAe, pBe, pAb, pBb = D.pAe.to_numpy(), D.pBe.to_numpy(), D.pAb.to_numpy(), D.pBb.to_numpy()
    rb_cond = np.clip(rb / np.clip(1 - re, 1e-9, 1), 0, 1)
    BIG = int(N.max()) * 4 + 10
    rng = np.random.default_rng(seed)

    def draw(V, pA, pB):
        if phi == 1.0:
            A = rng.binomial(V, pA)
            B = rng.binomial(np.clip(V - A, 0, None),
                             np.clip(pB/np.clip(1-pA, 1e-9, 1), 0, 1))
        else:
            A = np.clip(np.round(rng.normal(V*pA, np.sqrt(phi*V*pA*(1-pA)))), 0, V)
            B = np.clip(np.round(rng.normal(V*pB, np.sqrt(phi*V*pB*(1-pB)))), 0, V)
            A = A.astype(np.int64); B = np.minimum(B, V).astype(np.int64)
        return A, B

    total = np.zeros(R, int)
    by_type = Counter()          # 누적 기대 쌍수 by (gubunpair, regionrel)
    for r in range(R):
        Ve = rng.binomial(N, re)
        Vb = rng.binomial(np.clip(N - Ve, 0, None), rb_cond)
        Ae, Be = draw(Ve, pAe, pBe)
        Ab, Bb = draw(Vb, pAb, pBb)
        A = np.concatenate([Ae, Ab]); B = np.concatenate([Be, Bb])
        key = A.astype(np.int64) * BIG + B
        order = np.argsort(key, kind="stable")
        ks = key[order]
        npairs = 0
        i, n = 0, len(ks)
        while i < n:
            j = i
            while j + 1 < n and ks[j+1] == ks[i]:
                j += 1
            if j > i:
                idx = order[i:j+1]
                for a in range(len(idx)):
                    for b in range(a+1, len(idx)):
                        npairs += 1
                        u, v = idx[a], idx[b]
                        gp = tuple(sorted([gubun[u], gubun[v]]))
                        rel = "교차" if tag[u] != tag[v] else "동일"
                        by_type[(gp, rel)] += 1
            i = j + 1
        total[r] = npairs
    return total, {k: v/R for k, v in by_type.items()}


def main():
    df = load_merged()
    sp = df[df.gubun.isin(["관내사전투표", "선거일투표"])]
    obs = empirical_twins(sp)
    D, gubun, tag = build_units(df)
    print(f"광주+전남: 동 {len(D)} → 단위 {2*len(D)} | 실제 관측 {obs}쌍\n")

    print("=== ① 타입 분해 (φ=1, 기대 쌍수) ===")
    total, types = simulate_typed(D, gubun, tag, phi=1.0, R=100000)
    print(f"기대 총 쌍수 {total.mean():.3f}")
    for (gp, rel), v in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {'·'.join(gp):<8} {rel}: {v:.3f}쌍")
    print(f"  실제 5쌍은: 사전·사전 동일4 + 사전·사전 교차1")

    print("\n=== ② φ 튜닝 (sub-binomial) ===")
    print(f"{'φ':>5} | {'기대쌍':>7} | {'P(=5)':>7} | {'P(>=5)':>7}")
    for phi in [1.0, 0.7, 0.5, 0.3, 0.15, 0.05]:
        t, _ = simulate_typed(D, gubun, tag, phi=phi, R=100000, seed=11)
        print(f"{phi:>5} | {t.mean():>7.2f} | {(t==5).mean():>6.2%} | {(t>=5).mean():>6.2%}")


if __name__ == "__main__":
    main()
