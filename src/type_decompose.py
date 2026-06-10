"""
쌍둥이 유형 분해 — "쌍둥이는 어느 구분(사전/본)에서 나와야 하는가"를 모델·데이터 양쪽에서.

배경(개념): 기대 쌍둥이 밀도는 일차근사로 노이즈 σ와 무관하고, 단위들의 (A,B) 평균이
정수 격자에 얼마나 빽빽이 박히는가(패킹 밀도)로 결정된다.
  밀도 ∝ 1 / (A축 퍼짐 × B축 퍼짐),  A축 ∝ 단위크기,  B축 ∝ 단위크기 × B지지율.
→ "작은 수"(단위크기)와 "지지율"(압승)은 경쟁 가설이 아니라 곱으로 작용.

출력:
  ① 구분별 단위 구조 (투표수/A/B 중앙값, B 점유율)
  ② 관측 근접쌍 셸의 유형 분해 (데이터 자신의 밀도)
  ③ 캐노니컬 모델 기대 쌍둥이의 유형 분해 (analyze_merged.simulate_typed 재사용)

핵심 결과(2026-06-10): 지선 광주+전남은 모델·근접쌍 모두 본·본이 최다 유형인데
관측 5쌍 전부 사전·사전 (조건부 ~0.2%). 대선은 유형 분포가 모델과 일치.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_merged import load_merged, build_units, simulate_typed  # noqa: E402
from analyze_president import load_president  # noqa: E402
import twin_model  # noqa: E402


def gubun_stats(sp, label):
    print(f"\n[{label}] 단위 구조 (중앙값):")
    for g, s in sp.groupby("gubun"):
        print(f"  {g:>7}: n={len(s):>5} | 투표수 {s.votes.median():>7.0f} | "
              f"A {s.A.median():>7.0f} | B {s.B.median():>6.0f} | "
              f"B/투표수 {(s.B.sum() / s.votes.sum()):.1%}")


def typed_shells(sp, dmax: int = 10):
    """근접쌍 셸(체비셰프)을 쌍 유형별로: 사전·사전 / 본·본 / 혼합."""
    A = sp.A.to_numpy(np.int64)
    B = sp.B.to_numpy(np.int64)
    pre = (sp.gubun == "관내사전투표").to_numpy()
    out = {"사전·사전": np.zeros(dmax + 1, int), "본·본": np.zeros(dmax + 1, int),
           "혼합": np.zeros(dmax + 1, int)}
    for i in range(len(A) - 1):
        cheb = np.maximum(np.abs(A[i + 1:] - A[i]), np.abs(B[i + 1:] - B[i]))
        sel = cheb <= dmax
        if sel.any():
            for d, p2 in zip(cheb[sel], pre[i + 1:][sel]):
                t = ("사전·사전" if (pre[i] and p2)
                     else ("본·본" if (not pre[i] and not p2) else "혼합"))
                out[t][d] += 1
    return out


def typed_expectation(D, R, seed=7):
    g = np.array(["사전"] * len(D) + ["본"] * len(D))
    t = np.array(["x"] * (2 * len(D)))
    total, types = simulate_typed(D, g, t, phi=1.0, R=R, seed=seed)
    agg = {}
    for (gp, rel), v in types.items():
        agg["·".join(gp)] = agg.get("·".join(gp), 0) + v
    return total, agg


def report(total, agg):
    print(f"  모델 기대 {total.mean():.2f}쌍 | 유형별 기대:")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:.3f}쌍")


def main():
    # ════ 지선 광주+전남 ════
    df = load_merged()
    sp = df[df.gubun.isin(["관내사전투표", "선거일투표"])]
    gubun_stats(sp, "지선 광주+전남")
    ts = typed_shells(sp)
    print("  근접쌍(d=1~10 합):", {k: int(v[1:].sum()) for k, v in ts.items()},
          "| d=0:", {k: int(v[0]) for k, v in ts.items()})
    D, gubun, tag = build_units(df)
    total, types = simulate_typed(D, gubun, tag, phi=1.0, R=50_000, seed=7)
    agg = {}
    for (gp, rel), v in types.items():
        agg["·".join(gp)] = agg.get("·".join(gp), 0) + v
    report(total, agg)
    print("  관측: 사전·사전 5쌍 (본·본 0, 혼합 0)")

    # ════ 대선 전국 ════
    pdf = load_president()
    sp2 = pdf[pdf.gubun.isin(["관내사전투표", "선거일투표"])]
    gubun_stats(sp2, "대선 전국")
    D2 = twin_model.build_dong_table(pdf)
    total2, agg2 = typed_expectation(D2, R=20_000)
    report(total2, agg2)
    print("  관측: 본·본 6쌍, 혼합 1쌍 (사전·사전 0)")


if __name__ == "__main__":
    main()
