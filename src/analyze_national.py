"""
전국 17개 시·도 합산 쌍둥이 분석 — 정식 스크립트.

① 관측: 전국 쌍둥이(동×구분) 쌍수 + 상세 목록.
   광주+전남은 양대후보(민형배/이정현)가 같아 한 블록으로 합본 → 교차쌍 포함.
   나머지 시·도는 후보가 달라 시·도 내부 쌍만 의미 있음.
② 캐노니컬 모델(twin_model): 시·도별 독립 시뮬(시드 분리) 합산 → 전국 P(≥관측)
③ 근접쌍 셸 n(d) (체비셰프 거리 d, L∞ 셸 격자점 8d): 관측 vs 시뮬
   → r(d) = 관측/시뮬 비율이 d=0 에서만 튀는지, 전 구간에서 높은지 진단.
     · r(d) 평탄(전 구간 >1) → 모델이 근접쌍 전반을 과소생성(플러그인 과평활 편향)
       ⇒ "d=0 만 특별하다"는 주장 약화
     · d≥1 에선 r≈1, d=0 만 튐 → 정확일치에 진짜 원자(atom) 존재 ⇒ 의심 강화
④ 사이드밴드(모형무관): ρ(d)=n(d)/(8d) 를 d≥1 에서 Poisson-ML 적합, d=0 외삽
   → 기대 쌍둥이 ρ̂(0) + Poisson P(≥관측). 이항·φ·turnout 가정 전부 불필요.

비순서쌍 정규화 메모: 순서쌍 차이밀도가 국소상수 c 면 셸 d 의 비순서쌍 수는
(8d/2)·c = 4d·c, 원점(d=0)은 c/2. 따라서 ρ(d)≡n(d)/(8d) (d≥1) 의 d→0 외삽이
그대로 기대 쌍둥이 수가 된다. (스크립트 안에서 시뮬로 자가검증함)
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_provinces import load_units, empirical_twins  # noqa: E402
import twin_model  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

DMAX = 15           # 셸 최대 거리
R_PVAL = 50_000     # 캐노니컬 P값용 반복수 (시·도별)
R_SHELL = 300       # 셸 곡선용 반복수 (시·도별)


# ---------------------------------------------------------------- 셸 카운트
def shell_counts(A, B, dmax: int = DMAX) -> np.ndarray:
    """비순서쌍의 체비셰프 거리별 셸 카운트. out[d] = #{i<j : max(|ΔA|,|ΔB|)=d}."""
    A = np.asarray(A, np.int64)
    B = np.asarray(B, np.int64)
    out = np.zeros(dmax + 1, np.int64)
    for i in range(len(A) - 1):
        cheb = np.maximum(np.abs(A[i + 1:] - A[i]), np.abs(B[i + 1:] - B[i]))
        small = cheb[cheb <= dmax]
        if small.size:
            out += np.bincount(small, minlength=dmax + 1)
    return out


def simulate_shells(D: pd.DataFrame, R: int = R_SHELL, dmax: int = DMAX,
                    seed: int = 11) -> np.ndarray:
    """캐노니컬 모델 R회 실현의 평균 셸 카운트 (twin_model.simulate 와 동일한 추첨)."""
    N = D.N.to_numpy()
    re, rb = D.re.to_numpy(), D.rb.to_numpy()
    pAe, pBe = D.pAe.to_numpy(), D.pBe.to_numpy()
    pAb, pBb = D.pAb.to_numpy(), D.pBb.to_numpy()
    rb_cond = np.clip(rb / np.clip(1 - re, 1e-9, 1), 0, 1)
    pBe_c = np.clip(pBe / np.clip(1 - pAe, 1e-9, 1), 0, 1)
    pBb_c = np.clip(pBb / np.clip(1 - pAb, 1e-9, 1), 0, 1)
    rng = np.random.default_rng(seed)
    acc = np.zeros(dmax + 1)
    for _ in range(R):
        Ve = rng.binomial(N, re)
        Vb = rng.binomial(N - Ve, rb_cond)
        Ae = rng.binomial(Ve, pAe); Be = rng.binomial(Ve - Ae, pBe_c)
        Ab = rng.binomial(Vb, pAb); Bb = rng.binomial(Vb - Ab, pBb_c)
        acc += shell_counts(np.concatenate([Ae, Ab]), np.concatenate([Be, Bb]), dmax)
    return acc / R


# ---------------------------------------------------------------- 사이드밴드 적합
def fit_sideband(n: np.ndarray, quadratic: bool = False,
                 dlo: int = 1, dhi: int = DMAX):
    """ρ(d)=n(d)/(8d) 에 Poisson-ML 로 (선형|2차) 적합 → ρ̂(0) 와 부트스트랩 se."""
    d = np.arange(dlo, dhi + 1, dtype=float)
    nd = n[dlo:dhi + 1].astype(float)
    lat = 8.0 * d                                   # 셸 격자점 수

    def nll(theta, counts):
        mu = lat * np.polyval(theta[::-1], d)       # theta = (a, b[, c])
        if (mu <= 0).any():
            return 1e12
        return float((mu - counts * np.log(mu)).sum())

    k = 3 if quadratic else 2
    x0 = np.zeros(k); x0[0] = max((nd / lat).mean(), 1e-6)
    res = minimize(nll, x0, args=(nd,), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    a_hat = res.x[0]

    rng = np.random.default_rng(123)
    boots = []
    for _ in range(300):                            # Poisson 재표집 부트스트랩
        nb = rng.poisson(np.maximum(nd, 0.0))
        rb_ = minimize(nll, res.x, args=(nb.astype(float),), method="Nelder-Mead",
                       options={"maxiter": 20000})
        boots.append(rb_.x[0])
    return a_hat, float(np.std(boots)), res.x


# ---------------------------------------------------------------- 메인
MERGE_CODES = {"2900", "4600"}      # 광주, 전남 — 동일 양대후보라 합본


def load_block(paths_list: list[str]) -> tuple[str, pd.DataFrame]:
    """한 블록(시·도 1개 또는 광주+전남 합본) 로드. region 에 시·도 접두어."""
    parts = []
    for p in paths_list:
        city, df, iA, iB = load_units(p)
        df = df.copy()
        df["region"] = city[:2] + "·" + df["region"]
        parts.append((city, df))
    name = "+".join(c for c, _ in parts)
    return name, pd.concat([d for _, d in parts], ignore_index=True)


def main():
    paths = sorted(glob(str(DATA / "crawl_provinces" / "*.json")))
    blocks, merged = [], []
    for path in paths:
        code = Path(path).name.split("_")[0]
        (merged if code in MERGE_CODES else blocks).append(path)
    blocks = [[p] for p in blocks] + ([merged] if merged else [])

    obs_total = 0
    nat_cnt = None                       # 전국 합산 시뮬 카운트 (R_PVAL,)
    sh_obs = np.zeros(DMAX + 1, np.int64)        # 관측 셸 (전체 쌍)
    sh_obs_ee = np.zeros(DMAX + 1, np.int64)     # 관측 셸 (사전-사전 쌍만)
    sh_sim = np.zeros(DMAX + 1)                  # 시뮬 평균 셸
    sim_exp_total = 0.0
    twin_rows = []
    prov_rows = []

    for k, block_paths in enumerate(blocks):
        city, df = load_block(block_paths)
        split = df[df.gubun.isin(["관내사전투표", "선거일투표"])].reset_index(drop=True)
        obs = empirical_twins(split)
        obs_total += obs

        # 관측 셸 (전체 / 사전-사전)
        sh_obs += shell_counts(split.A, split.B)
        ee = split[split.gubun == "관내사전투표"]
        sh_obs_ee += shell_counts(ee.A, ee.B)

        # 캐노니컬 시뮬 — P값용(시드 분리) + 셸용
        D = twin_model.build_dong_table(df)
        cnt = twin_model.simulate(D, R=R_PVAL, seed=1000 + k)
        nat_cnt = cnt if nat_cnt is None else nat_cnt + cnt
        sim_exp_total += cnt.mean()
        sh_sim += simulate_shells(D, R=R_SHELL, seed=2000 + k)

        prov_rows.append((city, len(split), obs, cnt.mean()))
        print(f"{city:>10} | 단위 {len(split):>4} | 관측 {obs} | 기대 {cnt.mean():.2f}",
              flush=True)

        # 쌍둥이 상세
        for (a, b), g in split.groupby(["A", "B"]):
            if len(g) >= 2:
                who = "; ".join(f"{r.region} {r.dong}({r.gubun[:2]})"
                                for r in g.itertuples())
                gtypes = tuple(sorted(r.gubun[:2] for r in g.itertuples()))
                twin_rows.append((city, int(a), int(b), len(g), who, gtypes))

    # ---- ② 전국 캐노니컬 P값 ----
    p_ge = float((nat_cnt >= obs_total).mean())
    print("\n=== ① 관측 / ② 캐노니컬 모델 (전국 합산) ===")
    print(f"전국 관측 쌍둥이 {obs_total}쌍 | 시뮬 기대 {nat_cnt.mean():.2f}쌍 "
          f"| P(≥{obs_total}) = {p_ge:.2%}  (R={R_PVAL:,}, 시·도별 시드 분리)")

    print("\n쌍둥이 상세:")
    from collections import Counter
    type_cnt = Counter()
    for city, a, b, m, who, gtypes in twin_rows:
        print(f"  {city} A={a} B={b} ({m}개 단위): {who}")
        type_cnt["·".join(gtypes)] += m * (m - 1) // 2
    print("  유형 분해: " + ", ".join(f"{t} {c}쌍" for t, c in type_cnt.items()))

    # ---- ③ 셸 비교 ----
    print("\n=== ③ 근접쌍 셸: 관측 vs 캐노니컬 시뮬 ===")
    print(f"{'d':>3} | {'관측':>6} | {'사전사전':>6} | {'시뮬':>8} | {'관측/시뮬':>8} | "
          f"{'ρ_obs':>8} | {'ρ_sim':>8}")
    for d in range(DMAX + 1):
        lat = 1.0 if d == 0 else 8.0 * d
        ratio = sh_obs[d] / sh_sim[d] if sh_sim[d] > 0 else float("nan")
        print(f"{d:>3} | {sh_obs[d]:>6} | {sh_obs_ee[d]:>6} | {sh_sim[d]:>8.2f} | "
              f"{ratio:>8.2f} | {sh_obs[d]/lat:>8.2f} | {sh_sim[d]/lat:>8.2f}")

    # ---- ④ 사이드밴드 외삽 ----
    print("\n=== ④ 사이드밴드(모형무관) 기대 쌍둥이 ===")
    for name, quad in [("선형", False), ("2차", True)]:
        a_hat, se, theta = fit_sideband(sh_obs, quadratic=quad)
        pv = float(poisson.sf(obs_total - 1, mu=max(a_hat, 1e-12)))
        pv_lo = float(poisson.sf(obs_total - 1, mu=max(a_hat - 1.96 * se, 1e-12)))
        pv_hi = float(poisson.sf(obs_total - 1, mu=a_hat + 1.96 * se))
        print(f"  {name}적합 d∈[1,{DMAX}]: ρ̂(0) = {a_hat:.2f} ± {se:.2f} "
              f"→ P(≥{obs_total}) = {pv:.2%}  [95% 구간 {pv_lo:.2%} ~ {pv_hi:.2%}]")

    # 자가검증: 시뮬 셸의 d=0 vs 시뮬 사이드밴드 외삽 (정규화 일관성)
    a_sim, se_sim, _ = fit_sideband(np.round(sh_sim * R_SHELL).astype(int))
    print(f"\n  [자가검증] 시뮬 d=0 = {sh_sim[0]:.2f} vs 시뮬 사이드밴드 외삽 "
          f"{a_sim / R_SHELL:.2f} ± {se_sim / R_SHELL:.2f} (일치해야 정상)")

    pd.DataFrame(prov_rows, columns=["시도", "단위수", "관측", "기대"]).to_csv(
        DATA / "national_summary.csv", index=False, encoding="utf-8-sig")
    print("\n→ national_summary.csv 저장")


if __name__ == "__main__":
    main()
