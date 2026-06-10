"""
최종 모델: 실제 데이터 주입 → 쌍둥이 투표동 쌍 개수 확률.

모델 구성:
  · 단위   : 동×구분 (관내사전투표 / 선거일투표)
  · 투표율 : 사전 = 100%(결정론), 본투표 = 실제 평균
  · 지지율 : 구분별 평균 (사전투표자가 A를 더 지지) + Dirichlet 단위간 변동(conc)
  · 분산   : 단위 내 sub-binomial(φ) — 실제 투표는 동전던지기보다 변동 작음
  · conc/φ : bulk 근접쌍 분포가 실제와 맞도록 캘리브레이션 (conc≈120, φ≈0.3)

출력: 기대 쌍수 + 쌍 개수 확률분포 + P(실제 관측=4쌍).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_vote_sim import SimConfig, TwinVoteSimulator

CAND = ["A", "B", "c1", "c2", "c3"]   # 민형배, 이정현, 이종욱, 강은미, 김광만

# 캘리브레이션값 (bulk 근접쌍 분포 일치)
CONC = 120.0
PHI = 0.3


def support3(g: pd.DataFrame) -> np.ndarray:
    agg = g[CAND].sum().to_numpy()
    p = agg / agg.sum()
    return np.array([p[0], p[1], p[2:].sum()])


def build_model(df: pd.DataFrame):
    split = df[df.gubun.isin(["관내사전투표", "선거일투표"])].reset_index(drop=True)
    bon = split[split.gubun == "선거일투표"]
    smap = {"관내사전투표": support3(split[split.gubun == "관내사전투표"]),
            "선거일투표": support3(bon)}
    tmap = {"관내사전투표": 1.0,
            "선거일투표": float(bon.votes.sum() / bon.voters.sum())}
    N = split.voters.to_numpy(np.int64)
    tmean = np.array([tmap[g] for g in split.gubun])
    sup = np.vstack([smap[g] for g in split.gubun])
    return split, N, tmean, sup, smap, tmap


def main(observed: int = 4) -> None:
    df = pd.read_csv("jeonnam_units.csv")
    split, N, tmean, sup, smap, tmap = build_model(df)

    cfg = SimConfig(
        voters_per_station=N,
        turnout_mean_per_unit=tmean,
        support_per_unit=sup,
        support_conc=CONC,
        dispersion=PHI,
        turnout_kappa=100_000,        # 투표율 단위간 변동은 작음(거의 고정)
        n_iter=200_000, chunk=2_000, seed=1,
    )
    r = TwinVoteSimulator(cfg).run(verbose=False)

    print("=" * 56)
    print("최종 모델 — 전남도지사 쌍둥이 투표동  (A=민형배, B=이정현)")
    print("=" * 56)
    print(f"단위 {len(split)}개 (동×구분) | conc={CONC:.0f}, φ={PHI}")
    print(f"  관내사전: 투표율 100%, A={smap['관내사전투표'][0]:.3f} B={smap['관내사전투표'][1]:.3f}")
    print(f"  선거일  : 투표율 {tmap['선거일투표']:.2f}, A={smap['선거일투표'][0]:.3f} B={smap['선거일투표'][1]:.3f}")
    print()
    print(f"기대 쌍둥이 쌍수 = {r.expected_pairs:.2f} 쌍")
    dist = r.pair_distribution(6)
    print("쌍 개수 확률분포:")
    for k, v in dist.items():
        bar = "█" * round(v * 50)
        mark = "  ← 실제" if k == observed else ""
        print(f"  {k}쌍: {v:6.1%} {bar}{mark}")
    print()
    print(f">> 실제 관측 {observed}쌍이 나올 확률:")
    print(f"   P(정확히 {observed}쌍) = {r.p_exactly(observed):.1%}")
    print(f"   P({observed}쌍 이상)  = {r.p_at_least(observed):.1%}")


if __name__ == "__main__":
    main()
