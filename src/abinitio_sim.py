"""
광주+전남 ab-initio 시뮬레이션 — 관측값을 버리고 '분포만 주고' 쌍둥이를 센다.

기존 캐노니컬 모델은 동별 관측 모수 조건부(파라메트릭 부트스트랩).
여기서는 동 개수만 고정하고, 매 실현마다 모든 동의 모수를 적합된 분포에서 새로 뽑는다:

    유권자수 N      ~ LogNormal(실측 log-적률)
    사전투표율 re    ~ Beta(실측 적률)        # 전체 선거인수 대비
    본투표율 rb|잔여 ~ Beta(실측 적률)        # (1-re) 중 본투표 비율
    사전 A지지율    ~ Beta,  사전 B|비A 지지율 ~ Beta
    본   A지지율    ~ Beta,  본   B|비A 지지율 ~ Beta

전남/광주는 구조가 달라(농촌 면 vs 도시 동) 시·도별로 따로 적합 후 합본.
모수 간 상관(작은 면일수록 A지지·사전율 높음 등)은 의도적으로 끊는다 — "대충 분포만
때려도" 쌍둥이 기대치가 비슷하게 나오는지가 질문이므로. 비교용으로 완전 풀링
(전남+광주 한 분포) 변형도 돌린다.

의미: '이런 유권자·지지율 분포를 가진 가상의 전남급 지역'의 비조건부 쌍둥이 발생률.
관측 조건화("실제값 흔들기") 비판이 원천적으로 적용되지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_merged import load_merged  # noqa: E402
import twin_model  # noqa: E402

R = 100_000
SEED = 7


def beta_moments(x: np.ndarray) -> tuple[float, float]:
    """적률법 Beta 적합 (분산 과대 시 균등에 가깝게 가드)."""
    x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4)
    m, v = x.mean(), x.var()
    v = min(v, m * (1 - m) * 0.95)
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def fit_group(D: pd.DataFrame) -> dict:
    """한 그룹(시·도)의 동 모수 분포 적합."""
    rb_cond = np.clip(D.rb / np.clip(1 - D.re, 1e-9, None), 0, 1)
    pBe_c = np.clip(D.pBe / np.clip(1 - D.pAe, 1e-9, None), 0, 1)
    pBb_c = np.clip(D.pBb / np.clip(1 - D.pAb, 1e-9, None), 0, 1)
    logN = np.log(D.N.to_numpy(float))
    return dict(
        n=len(D), logN_mu=logN.mean(), logN_sd=logN.std(),
        re=beta_moments(D.re), rb=beta_moments(rb_cond),
        pAe=beta_moments(D.pAe), pBe=beta_moments(pBe_c),
        pAb=beta_moments(D.pAb), pBb=beta_moments(pBb_c),
        Nmin=int(D.N.min()), Nmax=int(D.N.max()),
    )


def describe(f: dict, label: str):
    bm = lambda ab: ab[0] / (ab[0] + ab[1])  # noqa: E731
    print(f"  [{label}] 동 {f['n']} | N 중앙 ~{np.exp(f['logN_mu']):.0f} "
          f"(log-sd {f['logN_sd']:.2f}) | 사전율 {bm(f['re']):.1%} | "
          f"본율(잔여) {bm(f['rb']):.1%} | 사전 A {bm(f['pAe']):.1%} "
          f"B|비A {bm(f['pBe']):.1%} | 본 A {bm(f['pAb']):.1%} B|비A {bm(f['pBb']):.1%}")


def simulate_abinitio(fits: list[dict], R: int = R, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cnt = np.zeros(R, dtype=int)
    ns = [f["n"] for f in fits]
    for r in range(R):
        Ns, res, rbs = [], [], []
        pAes, pBes, pAbs, pBbs = [], [], [], []
        for f in fits:
            n = f["n"]
            Ns.append(np.clip(rng.lognormal(f["logN_mu"], f["logN_sd"], n),
                              f["Nmin"], f["Nmax"]).astype(np.int64))
            res.append(rng.beta(*f["re"], n))
            rbs.append(rng.beta(*f["rb"], n))
            pAes.append(rng.beta(*f["pAe"], n))
            pBes.append(rng.beta(*f["pBe"], n))
            pAbs.append(rng.beta(*f["pAb"], n))
            pBbs.append(rng.beta(*f["pBb"], n))
        N = np.concatenate(Ns)
        re_ = np.concatenate(res); rb_ = np.concatenate(rbs)
        pAe = np.concatenate(pAes); pBe = np.concatenate(pBes)
        pAb = np.concatenate(pAbs); pBb = np.concatenate(pBbs)

        Ve = rng.binomial(N, re_)
        Vb = rng.binomial(N - Ve, rb_)
        Ae = rng.binomial(Ve, pAe); Be = rng.binomial(Ve - Ae, pBe)
        Ab = rng.binomial(Vb, pAb); Bb = rng.binomial(Vb - Ab, pBb)
        A = np.concatenate([Ae, Ab]); B = np.concatenate([Be, Bb])
        BIG = int(A.max()) + int(B.max()) + 10
        key = np.sort(A.astype(np.int64) * BIG + B)
        _, c = np.unique(key, return_counts=True)
        cnt[r] = int((c * (c - 1) // 2).sum())
    return cnt


def report(cnt: np.ndarray, label: str, obs: int = 5):
    print(f"\n=== {label} ===")
    print(f"기대 {cnt.mean():.3f}쌍 | P(≥{obs}) = {(cnt >= obs).mean():.2%} "
          f"| P(={obs}) = {(cnt == obs).mean():.2%}")
    for k in range(8):
        p = (cnt == k).mean()
        print(f"  {k}쌍: {p:7.2%} {'█' * round(p * 50)}{'  ← 관측' if k == obs else ''}")


def main():
    df = load_merged()
    D = twin_model.build_dong_table(df)
    tags = np.array([r.split("·")[0] for r in D.region])
    print(f"광주+전남 동 {len(D)} (전남 {(tags=='전남').sum()}, 광주 {(tags=='광주').sum()})")
    print("\n적합된 분포 (주요값):")
    f_jn = fit_group(D[tags == "전남"])
    f_gj = fit_group(D[tags == "광주"])
    describe(f_jn, "전남")
    describe(f_gj, "광주")

    cnt = simulate_abinitio([f_jn, f_gj], R=R, seed=SEED)
    report(cnt, f"ab-initio (시·도별 적합, R={R:,})")

    f_all = fit_group(D)
    cnt2 = simulate_abinitio([f_all], R=R, seed=SEED + 1)
    report(cnt2, "ab-initio (완전 풀링 — 한 분포)")

    print("\n[비교] 관측 조건부 캐노니컬: 기대 1.47 | P(≥5)=1.74%  | 실제 관측 5쌍")


if __name__ == "__main__":
    main()
