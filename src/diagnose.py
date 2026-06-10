"""
모델 갭 진단: 실제 vs 시뮬의 (A,B) 근접 쌍둥이 분포 비교.

거리 d(체비셰프: max(|ΔA|,|ΔB|)) 이내인 단위 쌍의 개수를
실제 데이터와 시뮬레이션에서 각각 세어 곡선을 비교한다.
  · 전 구간에서 실제가 더 많다  → 모델 전체 분산/스프레드가 과대(단위들이 덜 뭉침)
  · d=0 에서만 실제가 튄다       → 이산화/특이 군집 효과(모델 구조 누락)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

from twin_vote_sim import SimConfig, TwinVoteSimulator

CAND = ["A", "B", "c1", "c2", "c3"]


def s3(g):
    a = g[CAND].sum().to_numpy(); p = a / a.sum()
    return np.array([p[0], p[1], p[2:].sum()])


def near_pairs(A, B, dists=(0, 1, 2, 3, 5, 10)):
    """거리 d 이내 쌍 개수 (체비셰프 거리)."""
    A = np.asarray(A); B = np.asarray(B)
    n = len(A)
    out = {d: 0 for d in dists}
    for i in range(n):
        dA = np.abs(A[i+1:] - A[i])
        dB = np.abs(B[i+1:] - B[i])
        cheb = np.maximum(dA, dB)
        for d in dists:
            out[d] += int((cheb <= d).sum())
    return out


def main():
    df = pd.read_csv(DATA/"jeonnam_units.csv")
    split = df[df.gubun.isin(["관내사전투표", "선거일투표"])].reset_index(drop=True)

    # ---- 실제 ----
    realA = split["A"].to_numpy(); realB = split["B"].to_numpy()
    real = near_pairs(realA, realB)

    # ---- 시뮬 (지지율 거의 고정: 모델 천장) ----
    smap = {"관내사전투표": s3(split[split.gubun == "관내사전투표"]),
            "선거일투표": s3(split[split.gubun == "선거일투표"])}
    tmap = {"관내사전투표": 1.0,
            "선거일투표": split[split.gubun == "선거일투표"].votes.sum()
                          / split[split.gubun == "선거일투표"].voters.sum()}
    N = split.voters.to_numpy(np.int64)
    tmean = np.array([tmap[g] for g in split.gubun])
    sup = np.vstack([smap[g] for g in split.gubun])

    # 단일 실현(realization)을 여러 번 뽑아 평균 근접쌍 곡선
    rng = np.random.default_rng(0)
    R = 200
    acc = None
    sim = TwinVoteSimulator(SimConfig(
        voters_per_station=N, turnout_mean_per_unit=tmean, support_per_unit=sup,
        turnout_kappa=100000, support_conc=100000, n_iter=2, chunk=2, seed=1))
    # _simulate_chunk 직접 호출로 (A,B) 표본 얻기
    for _ in range(R):
        # 내부 메서드 재사용: 한 번에 batch=1 실현
        batch_A, batch_B = _draw_once(sim)
        np_ = near_pairs(batch_A, batch_B)
        if acc is None:
            acc = {d: 0 for d in np_}
        for d in np_:
            acc[d] += np_[d]
    simavg = {d: acc[d] / R for d in acc}

    print("거리 d 이내 쌍 개수  (체비셰프 max(|ΔA|,|ΔB|))")
    print(f"{'d':>4} | {'실제':>8} | {'시뮬평균':>8}")
    for d in sorted(real):
        print(f"{d:>4} | {real[d]:>8} | {simavg[d]:>8.1f}")
    print("\nd=0 이 정확히 일치(쌍둥이) 개수.")


def _draw_once(sim: TwinVoteSimulator):
    """시뮬 1회 실현의 (A,B) 벡터 반환."""
    rng = sim._rng
    M = sim._N.size
    N = sim._N[None, :]
    turnout = rng.beta(sim._beta_a[None, :], sim._beta_b[None, :], size=(1, M))
    turnout = np.where(sim._turnout_deterministic[None, :], 1.0, turnout)
    V = rng.binomial(N, turnout)
    alpha = sim._alpha
    alpha_b = alpha[None, None, :] if alpha.ndim == 1 else alpha[None, :, :]
    K = alpha_b.shape[-1]
    gam = rng.gamma(np.broadcast_to(alpha_b, (1, M, K)))
    P = gam / gam.sum(axis=2, keepdims=True)
    pA = P[..., 0]; pB = P[..., 1]
    X_A = rng.binomial(V, pA)
    rem = V - X_A
    denom = 1.0 - pA
    pB_cond = np.clip(np.where(denom > 1e-12, pB / denom, 0.0), 0, 1)
    X_B = rng.binomial(rem, pB_cond)
    return X_A[0], X_B[0]


if __name__ == "__main__":
    main()
