"""
실제 전남 데이터 주입 시뮬레이션 — 사전투표/본투표 분리.

  · 투표율 : 관내사전투표 = 100%, 선거일(본)투표 = 실제 평균
  · 지지율 : 사전투표 / 본투표 따로 (사전투표자가 A를 더 지지)
  · 분산   : 기본값(kappa=200, conc=500)

비교: 시뮬 기대 쌍둥이 쌍수 vs 실제 관측.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_vote_sim import SimConfig, TwinVoteSimulator

CAND = ["A", "B", "c1", "c2", "c3"]   # 민형배, 이정현, 이종욱, 강은미, 김광만


def support3(g: pd.DataFrame) -> np.ndarray:
    """[A, B, 기타] 지지율."""
    agg = g[CAND].sum().to_numpy()
    p = agg / agg.sum()
    return np.array([p[0], p[1], p[2:].sum()])


def run(units: pd.DataFrame, turnout_map: dict, support_map: dict,
        label: str, obs: int, conc: float, kappa: float) -> None:
    M = len(units)
    N = units.voters.to_numpy(np.int64)
    tmean = np.array([turnout_map[g] for g in units.gubun])
    sup = np.vstack([support_map[g] for g in units.gubun])

    cfg = SimConfig(
        voters_per_station=N,
        turnout_mean_per_unit=tmean,
        support_per_unit=sup,
        turnout_kappa=kappa,
        support_conc=conc,
        n_iter=100_000, chunk=2_000, seed=42,
    )
    r = TwinVoteSimulator(cfg).run(verbose=False)
    print(f"  conc={conc:>4.0f}, kappa={kappa:>4.0f}  →  "
          f"기대 쌍둥이 = {r.expected_pairs:.2f} 쌍   (실제 {obs}쌍)")


def main() -> None:
    df = pd.read_csv("jeonnam_units.csv")
    print("=" * 56)
    print("실제 데이터 주입 — 사전/본투표 분리  (A=민형배, B=이정현)")
    print("=" * 56)

    sa = df[df.gubun == "관내사전투표"]
    bon = df[df.gubun == "선거일투표"]
    bon_turnout = bon.votes.sum() / bon.voters.sum()

    support_map = {"관내사전투표": support3(sa), "선거일투표": support3(bon)}
    turnout_map = {"관내사전투표": 1.0, "선거일투표": float(bon_turnout)}

    split = df[df.gubun.isin(["관내사전투표", "선거일투표"])]
    print(f"\n[동×구분 594개]  사전 A={support_map['관내사전투표'][0]:.3f}/B={support_map['관내사전투표'][1]:.3f},"
          f" 본투표 A={support_map['선거일투표'][0]:.3f}/B={support_map['선거일투표'][1]:.3f}")
    print("분산 키울수록(conc↓) 기대 쌍둥이 변화:")
    for conc, kappa in [(500, 200), (120, 100), (60, 50)]:
        run(split, turnout_map, support_map, "", obs=4, conc=conc, kappa=kappa)


if __name__ == "__main__":
    main()
