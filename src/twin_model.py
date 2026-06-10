"""
쌍둥이 투표동 — 최종(캐노니컬) 모델.

동별 전체유권자 N 에서:
    N ──multinomial──> 사전투표자 / 본투표자 / 기권   (동별 사전·본 투표율)
                          │            │
                     사전 득표갈림   본 득표갈림        (동별 사전·본 지지율)
각 동은 (사전, 본) 두 개의 개표단위를 만들고, 전체 단위 중 (A,B) 동시 일치 쌍을 센다.

모든 모수(유권자수·사전투표율·본투표율·사전지지율·본지지율)는 동별 실측값.
랜덤성: 누가 사전/본/기권하는지(multinomial) + 표가 어떻게 갈리는지(이항).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GUBUN = {"계", "관내사전투표", "선거일투표"}


def build_dong_table(df: pd.DataFrame) -> pd.DataFrame:
    """region,dong,gubun,voters,votes,A,B 가진 df → 동별 모수 테이블.
    (A,B 컬럼은 호출 전에 시·도별 상위 2후보로 식별돼 있어야 함)"""
    rows = []
    for (reg, dong), sub in df.groupby(["region", "dong"]):
        gg = {r.gubun: r for r in sub.itertuples()}
        if not GUBUN <= set(gg):
            continue
        N = gg["계"].voters
        e, b = gg["관내사전투표"], gg["선거일투표"]
        if not N or N <= 0:
            continue
        rows.append(dict(
            region=reg, dong=dong, N=int(N),
            re=min(e.votes / N, 1.0),                 # 사전투표율(전체 대비)
            rb=min(b.votes / N, 1.0),                 # 본투표율(전체 대비)
            pAe=e.A / max(e.votes, 1), pBe=e.B / max(e.votes, 1),   # 사전 지지율
            pAb=b.A / max(b.votes, 1), pBb=b.B / max(b.votes, 1),   # 본 지지율
        ))
    return pd.DataFrame(rows)


def simulate(D: pd.DataFrame, R: int = 200_000, seed: int = 7) -> np.ndarray:
    """R회 실현. 각 실현의 쌍둥이 쌍 개수 배열 반환."""
    N = D.N.to_numpy()
    re, rb = D.re.to_numpy(), D.rb.to_numpy()
    pAe, pBe = D.pAe.to_numpy(), D.pBe.to_numpy()
    pAb, pBb = D.pAb.to_numpy(), D.pBb.to_numpy()
    rb_cond = np.clip(rb / np.clip(1 - re, 1e-9, 1), 0, 1)
    pBe_c = np.clip(pBe / np.clip(1 - pAe, 1e-9, 1), 0, 1)
    pBb_c = np.clip(pBb / np.clip(1 - pAb, 1e-9, 1), 0, 1)
    BIG = int(N.max()) * 4 + 10
    rng = np.random.default_rng(seed)
    cnt = np.zeros(R, dtype=int)
    for r in range(R):
        Ve = rng.binomial(N, re)                      # 사전투표자
        Vb = rng.binomial(N - Ve, rb_cond)            # 본투표자
        Ae = rng.binomial(Ve, pAe); Be = rng.binomial(Ve - Ae, pBe_c)
        Ab = rng.binomial(Vb, pAb); Bb = rng.binomial(Vb - Ab, pBb_c)
        A = np.concatenate([Ae, Ab]); B = np.concatenate([Be, Bb])
        key = np.sort(A.astype(np.int64) * BIG + B)
        _, c = np.unique(key, return_counts=True)
        cnt[r] = int((c * (c - 1) // 2).sum())
    return cnt


def summarize(cnt: np.ndarray, observed: int | None = None) -> dict:
    out = {"expected": float(cnt.mean())}
    for k in range(7):
        out[f"P{k}"] = float((cnt == k).mean())
    if observed is not None:
        out["observed"] = observed
        out["P_ge_obs"] = float((cnt >= observed).mean())
    return out
