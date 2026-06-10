"""
제21대 대통령선거(2025-06-03) 쌍둥이 분석 — 지선과 동일 방법론의 반복 검증.

데이터: 공공데이터포털 CSV (data/president_20250603.csv, 롱포맷)
  시도명,구시군명,읍면동명,투표구명,후보자,득표수
  · 동마다 관내사전투표 1행군 + 선거일 투표구 k행군 (각 9행: 선거인수/투표수/후보5/무효/기권)
  · 관내사전 선거인수 ≈ 투표수 (지선과 같은 동어반복 규약)
  · 동 전체 선거인수 N = 관내사전 선거인수 + Σ선거일 투표구 선거인수

단위 정의(지선과 동일): 읍면동 × {관내사전투표, 선거일투표(투표구 합산)}.
대선은 전국이 같은 후보(A=이재명, B=김문수 — 전국 1·2위)라 **전국 전체가 한 블록**:
쌍둥이를 시·도 경계 없이 전국 모든 단위쌍에서 센다.

분석(analyze_national 과 동일 3종):
  ① 관측 쌍둥이 + 상세/유형분해  ② 캐노니컬 모델 P(≥관측)
  ③ 셸 r(d) 진단 + 사이드밴드(모형무관) 외삽
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_national import shell_counts, fit_sideband, simulate_shells, DMAX  # noqa: E402
import twin_model  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "president_20250603.csv"

SKIP = {"거소·선상투표", "관외사전투표", "재외투표", "잘못 투입·구분된 투표지"}
CAND_A = "더불어민주당 이재명"
CAND_B = "국민의힘 김문수"

R_PVAL = 100_000
R_SHELL = 300


def load_president() -> pd.DataFrame:
    """CSV → 지선 분석과 같은 모양의 단위 테이블 (계/관내사전투표/선거일투표 행)."""
    raw = pd.read_csv(CSV, encoding="utf-8-sig")
    w = raw.pivot_table(index=["시도명", "구시군명", "읍면동명", "투표구명"],
                        columns="후보자", values="득표수", aggfunc="first").reset_index()
    w = w[~w.읍면동명.isin(SKIP)].copy()
    w["region"] = w.시도명 + " " + w.구시군명
    w["dong"] = w.읍면동명
    w["is_pre"] = w.투표구명 == "관내사전투표"

    rows = []
    for (reg, dong), g in w.groupby(["region", "dong"]):
        pre = g[g.is_pre]
        day = g[~g.is_pre]
        if len(pre) != 1 or len(day) == 0:
            continue
        p = pre.iloc[0]
        e = dict(voters=int(p.선거인수), votes=int(p.투표수),
                 A=int(p[CAND_A]), B=int(p[CAND_B]))
        b = dict(voters=int(day.선거인수.sum()), votes=int(day.투표수.sum()),
                 A=int(day[CAND_A].sum()), B=int(day[CAND_B].sum()))
        rows.append(dict(region=reg, dong=dong, gubun="관내사전투표", **e))
        rows.append(dict(region=reg, dong=dong, gubun="선거일투표", **b))
        rows.append(dict(region=reg, dong=dong, gubun="계",
                         voters=e["voters"] + b["voters"],
                         votes=e["votes"] + b["votes"],
                         A=e["A"] + b["A"], B=e["B"] + b["B"]))
    return pd.DataFrame(rows)


def empirical_twins_detail(split: pd.DataFrame):
    cnt, details = 0, []
    for (a, b), g in split.groupby(["A", "B"]):
        if len(g) >= 2:
            cnt += len(g) * (len(g) - 1) // 2
            details.append((int(a), int(b), g))
    return cnt, details


def main():
    df = load_president()
    split = df[df.gubun.isin(["관내사전투표", "선거일투표"])].reset_index(drop=True)
    n_dong = (df.gubun == "계").sum()
    print(f"대선 2025-06-03 | 동 {n_dong} → 단위 {len(split)} (전국 한 블록) | "
          f"A=이재명, B=김문수", flush=True)

    obs, details = empirical_twins_detail(split)
    print(f"\n=== ① 관측 쌍둥이: {obs}쌍 ===")
    from collections import Counter
    type_cnt = Counter()
    for a, b, g in details:
        who = "; ".join(f"{r.region} {r.dong}({r.gubun[:2]})" for r in g.itertuples())
        m = len(g)
        type_cnt["·".join(sorted(r.gubun[:2] for r in g.itertuples()))] += m * (m - 1) // 2
        print(f"  A={a} B={b} ({m}개 단위): {who}")
    print("  유형 분해: " + ", ".join(f"{t} {c}쌍" for t, c in type_cnt.items()))

    # ---- ② 캐노니컬 모델 ----
    D = twin_model.build_dong_table(df)
    print(f"\n=== ② 캐노니컬 모델 (동 {len(D)}, R={R_PVAL:,}) ===", flush=True)
    cnt = twin_model.simulate(D, R=R_PVAL, seed=7)
    p_ge = float((cnt >= obs).mean())
    print(f"기대 {cnt.mean():.2f}쌍 | P(≥{obs}) = {p_ge:.2%}"
          + ("  (시뮬 최대 미달 — 상한 추정)" if p_ge == 0 else ""))

    # ---- ③ 셸 + 사이드밴드 ----
    print(f"\n=== ③ 셸 (R_shell={R_SHELL}) ===", flush=True)
    sh_obs = shell_counts(split.A, split.B)
    sh_sim = simulate_shells(D, R=R_SHELL, seed=11)
    print(f"{'d':>3} | {'관측':>7} | {'시뮬':>9} | {'관측/시뮬':>8} | {'ρ_obs':>8} | {'ρ_sim':>8}")
    for d in range(DMAX + 1):
        lat = 1.0 if d == 0 else 8.0 * d
        ratio = sh_obs[d] / sh_sim[d] if sh_sim[d] > 0 else float("nan")
        print(f"{d:>3} | {sh_obs[d]:>7} | {sh_sim[d]:>9.2f} | {ratio:>8.2f} | "
              f"{sh_obs[d]/lat:>8.2f} | {sh_sim[d]/lat:>8.2f}")

    print("\n=== ④ 사이드밴드(모형무관) ===")
    for quad, lbl in [(False, "선형"), (True, "2차")]:
        a_hat, se, _ = fit_sideband(sh_obs, quadratic=quad)
        pv = float(poisson.sf(obs - 1, mu=max(a_hat, 1e-12)))
        lo = float(poisson.sf(obs - 1, mu=max(a_hat - 1.96 * se, 1e-12)))
        hi = float(poisson.sf(obs - 1, mu=a_hat + 1.96 * se))
        print(f"  {lbl}: ρ̂(0) = {a_hat:.2f} ± {se:.2f} → P(≥{obs}) = {pv:.2%} "
              f"[{lo:.2%} ~ {hi:.2%}]")


if __name__ == "__main__":
    main()
