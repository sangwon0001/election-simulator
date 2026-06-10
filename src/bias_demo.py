"""
플러그인(관측모수 재추첨) 시뮬레이션의 '노이즈 이중계산' 편향 — 해석적 증명.

캐노니컬 모델은 각 단위를 관측값 p̂=A/V 중심으로 이항 재추첨한다.
그러면 두 단위의 시뮬 득표차 = (관측차) + 새노이즈(분산 2σ²) 인데,
관측차 자체가 이미 (진짜평균차) + 노이즈(2σ²) 라서, 앙상블 평균으로 보면
시뮬 차이분포는 진실 대비 노이즈가 두 번 컨볼브된다(분산 4σ²).

정확일치 확률 = 차이분포의 0점 밀도이므로:
  · 진짜 평균이 같은 쌍(Δμ=0): 플러그인이 일치확률을 1/√2 ≈ 0.707 로 과소평가
  · 평균이 멀리 떨어진 쌍:      밀도가 평탄해 편향 없음 (~1.0)
(A,B) 2차원 쌍둥이에선 편향이 제곱 → 진짜동일 쌍 일치확률 약 1/2 과소평가.

여기서는 1차원(A만)으로 전 과정을 이항 pmf 내적으로 '정확히' 계산한다. MC 없음.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

V = 2000          # 단위당 투표자수
N_SHARED = 50     # 진짜 지지율이 '정확히 같은' 쌍 수 (단위 100개)
N_SINGLE = 100    # 지지율 제각각인 단위 수
T = 200           # 관측 데이터셋 반복(플러그인 기대값의 앙상블 평균용)
SEED = 42


def tie_matrix(p: np.ndarray) -> np.ndarray:
    """G[i,j] = P(A_i = A_j | Bin(V,p_i), Bin(V,p_j)) — 이항 pmf 내적(정확)."""
    a = np.arange(V + 1)
    F = binom.pmf(a[None, :], V, p[:, None])      # (M, V+1)
    return F @ F.T


def main():
    rng = np.random.default_rng(SEED)
    p_shared = rng.uniform(0.55, 0.90, N_SHARED)
    p = np.concatenate([np.repeat(p_shared, 2),            # [0,1],[2,3],... 동일쌍
                        rng.uniform(0.55, 0.90, N_SINGLE)])
    M = len(p)
    shared_idx = [(2 * k, 2 * k + 1) for k in range(N_SHARED)]
    shared_mask = np.zeros((M, M), bool)
    for i, j in shared_idx:
        shared_mask[i, j] = True
    upper = np.triu(np.ones((M, M), bool), k=1)
    other_mask = upper & ~shared_mask

    # ---- 진실: 진짜 p 로 계산한 정확일치 확률 ----
    G_true = tie_matrix(p)
    true_shared = G_true[shared_mask].sum()
    true_other = G_true[other_mask].sum()

    # ---- 플러그인: 관측 한 번 → p̂ 재추첨, 을 T회 앙상블 평균 ----
    plug_shared = np.zeros(T)
    plug_other = np.zeros(T)
    for t in range(T):
        A_obs = rng.binomial(V, p)
        G_plug = tie_matrix(A_obs / V)
        plug_shared[t] = G_plug[shared_mask].sum()
        plug_other[t] = G_plug[other_mask].sum()

    print(f"단위 {M}개 (진짜동일 {N_SHARED}쌍 + 제각각 {N_SINGLE}) | V={V} | 관측반복 T={T}")
    print()
    print(f"{'쌍 유형':<14} {'진실 E[일치]':>12} {'플러그인 E[일치]':>16} {'비율':>8}")
    r_sh = plug_shared.mean() / true_shared
    r_ot = plug_other.mean() / true_other
    print(f"{'진짜동일 쌍':<14} {true_shared:>12.4f} {plug_shared.mean():>16.4f} {r_sh:>8.3f}")
    print(f"{'제각각 쌍':<14} {true_other:>12.4f} {plug_other.mean():>16.4f} {r_ot:>8.3f}")
    print()
    print(f"이론 예측: 진짜동일 쌍 비율 ≈ 1/√2 = {1/np.sqrt(2):.3f}, 제각각 ≈ 1")
    print(f"→ (A,B) 2차원 쌍둥이에선 진짜동일 쌍 일치확률이 ≈ {r_sh**2:.2f} 배로 과소평가됨")


if __name__ == "__main__":
    main()
