"""
제8회 전국동시지방선거(2022-06-01) 시·도지사 쌍둥이 분석 — 판별 실험.

질문: 2026 지선의 "d=0 원자 + 관내사전 국재화"가 지방선거 일반의 성질인가,
2026 한정인가? (FINDINGS §7.2 한계 (5), 논문 §7.3 (a))

데이터: 공공데이터포털 xlsx (data/local2022_20220601.xlsx, '시·도지사' 시트)
  동별 소계/관내사전투표/선거일투표 3행. 규약은 2026 크롤과 동일
  (소계=계, 관내사전 선거인수=투표수). 후보 헤더 행(구시군별)에서 후보명 추출,
  시·도별 총득표 상위 2인을 (A,B)로. 동일 양대후보 시·도는 자동 합본(2022엔 없음 예상).

분석: analyze_national 과 동일 — 관측/유형, 캐노니컬 P, 셸 r(d), 사이드밴드.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_national import shell_counts, fit_sideband, simulate_shells, DMAX  # noqa: E402
from analyze_provinces import empirical_twins  # noqa: E402
import twin_model  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
XLSX = DATA / "local2022_20220601.xlsx"

SKIP = {"합계", "거소투표", "관외사전투표", "잘못 투입·구분된 투표지", "재외투표"}
GUBUN_MAP = {"소계": "계", "관내사전투표": "관내사전투표", "선거일투표": "선거일투표"}

R_PVAL = 50_000
R_SHELL = 300


def _int(x) -> int:
    if x is None or x == "":
        return 0
    return int(str(x).replace(",", ""))


def load_2022() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """시·도별 (단위 df, 후보명 리스트). df: region,dong,gubun,voters,votes,A,B."""
    ws = openpyxl.load_workbook(XLSX, read_only=True)["시·도지사"]
    rows = list(ws.iter_rows(values_only=True))[2:]
    names: dict[str, list[str]] = {}
    recs: dict[str, list[dict]] = {}
    for r in rows:
        sido, sgg, dong, gubun = r[0], r[1], r[2], r[3]
        if not sido:
            continue
        if sgg and not dong and r[6]:                      # 후보 헤더 행
            if sido not in names:
                names[sido] = [str(c).replace("\n", " ").strip()
                               for c in r[6:12] if c not in (None, "", 0, "0")]
            continue
        if not dong or dong in SKIP or gubun not in GUBUN_MAP:
            continue
        recs.setdefault(sido, []).append(dict(
            region=sgg, dong=dong, gubun=GUBUN_MAP[gubun],
            voters=_int(r[4]), votes=_int(r[5]),
            cand=[_int(c) for c in r[6:12]]))
    out = {}
    for sido, rs in recs.items():
        df = pd.DataFrame(rs)
        tot = np.array(df[df.gubun == "계"].cand.tolist()).sum(axis=0)
        iA, iB = np.argsort(tot)[::-1][:2]
        df["A"] = df.cand.apply(lambda c: c[iA])
        df["B"] = df.cand.apply(lambda c: c[iB])
        top2 = (names[sido][iA] if iA < len(names[sido]) else f"후보{iA}",
                names[sido][iB] if iB < len(names[sido]) else f"후보{iB}")
        out[sido] = (df.drop(columns="cand"), list(top2))
    return out


def main():
    blocks = load_2022()
    # 동일 양대후보 시·도 합본 감지
    pair2sido = Counter()
    for sido, (df, top2) in blocks.items():
        pair2sido[tuple(top2)] += 1
    merged = [p for p, n in pair2sido.items() if n > 1]
    print(f"시·도 {len(blocks)}개 | 동일 양대후보 블록: {merged if merged else '없음'}\n")

    obs_total = 0
    nat_cnt = None
    sh_obs = np.zeros(DMAX + 1, np.int64)
    sh_obs_ee = np.zeros(DMAX + 1, np.int64)
    sh_sim = np.zeros(DMAX + 1)
    twin_rows = []

    for k, (sido, (df, top2)) in enumerate(sorted(blocks.items())):
        split = df[df.gubun.isin(["관내사전투표", "선거일투표"])].reset_index(drop=True)
        obs = empirical_twins(split)
        obs_total += obs
        sh_obs += shell_counts(split.A, split.B)
        ee = split[split.gubun == "관내사전투표"]
        sh_obs_ee += shell_counts(ee.A, ee.B)
        D = twin_model.build_dong_table(df)
        cnt = twin_model.simulate(D, R=R_PVAL, seed=4000 + k)
        nat_cnt = cnt if nat_cnt is None else nat_cnt + cnt
        sh_sim += simulate_shells(D, R=R_SHELL, seed=5000 + k)
        print(f"{sido:>10} | 단위 {len(split):>4} | 관측 {obs} | 기대 {cnt.mean():.2f} "
              f"| A={top2[0].split()[-1]} B={top2[1].split()[-1]}", flush=True)
        for (a, b), g in split.groupby(["A", "B"]):
            if len(g) >= 2:
                who = "; ".join(f"{r.region} {r.dong}({r.gubun[:2]})"
                                for r in g.itertuples())
                gt = "·".join(sorted(r.gubun[:2] for r in g.itertuples()))
                twin_rows.append((sido, int(a), int(b), len(g), who, gt))

    p_ge = float((nat_cnt >= obs_total).mean()) if obs_total > 0 else 1.0
    print("\n=== 2022 지선 전국 ===")
    print(f"관측 {obs_total}쌍 | 캐노니컬 기대 {nat_cnt.mean():.2f} | "
          f"P(≥{obs_total}) = {p_ge:.2%}")
    type_cnt = Counter()
    for sido, a, b, m, who, gt in twin_rows:
        print(f"  {sido} A={a} B={b}: {who}")
        type_cnt[gt] += m * (m - 1) // 2
    print("  유형 분해:", dict(type_cnt) if type_cnt else "쌍둥이 없음")

    print(f"\n=== 셸: 관측 vs 시뮬 ===")
    print(f"{'d':>3} | {'관측':>6} | {'사전사전':>6} | {'시뮬':>8} | {'r(d)':>6} | {'ρ_obs':>7}")
    for d in range(DMAX + 1):
        lat = 1.0 if d == 0 else 8.0 * d
        ratio = sh_obs[d] / sh_sim[d] if sh_sim[d] > 0 else float("nan")
        print(f"{d:>3} | {sh_obs[d]:>6} | {sh_obs_ee[d]:>6} | {sh_sim[d]:>8.2f} | "
              f"{ratio:>6.2f} | {sh_obs[d]/lat:>7.2f}")

    print("\n=== 사이드밴드 ===")
    for quad, lbl in [(False, "선형"), (True, "2차")]:
        a_hat, se, _ = fit_sideband(sh_obs, quadratic=quad)
        pv = float(poisson.sf(obs_total - 1, mu=max(a_hat, 1e-12)))
        print(f"  {lbl}: ρ̂(0) = {a_hat:.2f} ± {se:.2f} → P(≥{obs_total}) = {pv:.2%}")


if __name__ == "__main__":
    main()
