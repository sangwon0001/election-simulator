"""
전남도지사 쌍둥이 투표동 — 캐노니컬 모델 단일 시·도 리포트.

twin_model(동별 전체유권자→사전/본/기권 multinomial→득표) 로 P(관측) 계산.
데이터: data/jeonnam_units.csv (build_dataset.py 산출).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import twin_model

DATA = Path(__file__).resolve().parent.parent / "data"


def main(observed: int = 4) -> None:
    df = pd.read_csv(DATA / "jeonnam_units.csv")
    # build_dong_table 는 region,dong,gubun,voters,votes,A,B 필요 (이미 있음)
    D = twin_model.build_dong_table(df)
    cnt = twin_model.simulate(D, R=200_000, seed=7)
    s = twin_model.summarize(cnt, observed)

    print("=" * 56)
    print("전남도지사 쌍둥이 투표동 — 캐노니컬 모델 (A=민형배, B=이정현)")
    print("=" * 56)
    print(f"동 {len(D)}개 → 개표단위 {2*len(D)}개 (사전+본)")
    print(f"동별 사전투표율 {D.re.mean():.1%} (범위 {D.re.min():.1%}~{D.re.max():.1%})")
    print(f"동별 본투표율   {D.rb.mean():.1%}")
    print()
    print(f"기대 쌍둥이 쌍수 = {s['expected']:.2f}")
    print("쌍 개수 확률분포:")
    for k in range(7):
        v = s[f"P{k}"]
        bar = "█" * round(v * 50)
        mark = "  ← 실제" if k == observed else ""
        print(f"  {k}쌍: {v:6.1%} {bar}{mark}")
    print()
    print(f">> P(정확히 {observed}쌍) = {s[f'P{observed}']:.1%}")
    print(f">> P({observed}쌍 이상)  = {s['P_ge_obs']:.1%}")


if __name__ == "__main__":
    main()
