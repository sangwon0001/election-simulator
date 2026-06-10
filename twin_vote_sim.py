"""
전남도지사 선거 '쌍둥이 표' 몬테카를로 시뮬레이션
=====================================================

목표: 서로 다른 '두 투표소'에서 A후보 득표수와 B후보 득표수가 동시에 똑같은
      사건(쌍둥이 투표소)이 전남 전체 투표소 중 한 쌍이라도 발생할 확률 추정.
      즉 투표소 i, j 에 대해 (A_i == A_j) AND (B_i == B_j) 인 충돌(생일 문제).

방법론: Hybrid Monte Carlo (Divide & Conquer)
  - 상수(Real)  : 투표소 수 M, 투표소별 유권자 수 N_i  → 실제 데이터 고정값
  - 변수(Gen)   : 투표율 V_i, 후보별 지지율 P_i, 최종 득표 X_i → 확률분포 난수
      · V_i : Beta 분포     (0~1 경계 보장 → 투표율 100% 초과 불가)
      · P_i : Dirichlet 분포 (지지율 벡터의 미세 변동, 합=1 보장)
      · X_i : 이항분포 분해  (다항분포와 동치, 완전 벡터화용)

최적화: for문 대신 numpy 벡터화. (iter, station) 2차원 배열을 청크 단위로 처리.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    # --- 상수 (실제 데이터로 교체) ---
    # data_loader.py 로 추출한 투표소별 선거인수 배열을 넣으세요.
    # 미지정 시 합성 데이터(전남 대략치)로 자동 생성합니다.
    voters_per_station: np.ndarray | None = None
    n_stations: int = 860            # 전남 투표소 수 대략치 (실제값으로 교체)
    mean_voters: int = 1500          # 투표소당 평균 유권자 (합성용)

    # --- 투표율 (Beta) ---
    turnout_mean: float = 0.60       # 전남 평균 투표율 (스칼라 기본값)
    turnout_kappa: float = 200.0     # 집중도(=a+b). 클수록 투표소 간 편차 작음
    # 단위별 투표율(선택): 길이 M 배열. 주면 turnout_mean 대신 단위별 적용.
    #   예) 관내사전투표 단위는 ~1.0, 선거일투표 단위는 ~0.35
    turnout_mean_per_unit: np.ndarray | None = None
    turnout_kappa_per_unit: np.ndarray | None = None

    # --- 지지율 (Dirichlet) ---
    # [A후보, B후보, 기타] 기준 지지율. 합은 1이 되도록.
    base_support: tuple[float, ...] = (0.75, 0.15, 0.10)
    support_conc: float = 500.0      # 농도. 클수록 투표소 간 지지율 변동 작음
    # 단위별 지지율(선택): (M, K) 배열. 주면 base_support 대신 단위별 적용.
    #   예) 사전투표 단위는 A지지율↑
    support_per_unit: np.ndarray | None = None
    # 단위별 Dirichlet 농도벡터 alpha(선택): (M, K). 주면 support_per_unit·conc를
    #   무시하고 이 alpha를 그대로 사용(단위별 농도까지 직접 제어).
    alpha_per_unit: np.ndarray | None = None

    # --- 단위 내 분산계수 φ (과/저분산) ---
    #   1.0 = 이항분포(동전던지기 가정). <1 = sub-binomial(실제 투표는 동전보다
    #   변동이 작음). φ를 주면 정규근사 추출(분산 = φ·V·p(1-p))을 사용.
    dispersion: float = 1.0

    # --- 몬테카를로 ---
    n_iter: int = 100_000            # 반복 횟수
    chunk: int = 2_000               # 한 번에 처리할 반복 수 (메모리 제어)
    seed: int = 42

    def __post_init__(self) -> None:
        base = np.asarray(self.base_support, dtype=float)
        if not np.isclose(base.sum(), 1.0):
            raise ValueError(f"base_support 합이 1이 아닙니다: {base.sum()}")
        if not (0.0 < self.turnout_mean < 1.0):
            raise ValueError("turnout_mean 은 (0,1) 범위여야 합니다.")
        self._base = base
        if self.support_per_unit is not None:
            s = np.asarray(self.support_per_unit, dtype=float)
            if not np.allclose(s.sum(axis=1), 1.0):
                raise ValueError("support_per_unit 각 행 합이 1이 아닙니다.")
            self._support_unit = s
        else:
            self._support_unit = None
        # 단위별 투표율은 (0,1] 허용 (사전투표 ~1.0 케이스)
        if self.turnout_mean_per_unit is not None:
            t = np.asarray(self.turnout_mean_per_unit, dtype=float)
            if np.any((t <= 0) | (t > 1)):
                raise ValueError("turnout_mean_per_unit 은 (0,1] 범위여야 합니다.")
            self._turnout_unit = t
        else:
            self._turnout_unit = None


# ---------------------------------------------------------------------------
# 시뮬레이터
# ---------------------------------------------------------------------------
@dataclass
class TwinVoteSimulator:
    cfg: SimConfig
    _rng: np.random.Generator = field(init=False)
    _N: np.ndarray = field(init=False)            # (M,) 투표소별 유권자 수
    _beta_a: float = field(init=False)
    _beta_b: float = field(init=False)
    _alpha: np.ndarray = field(init=False)        # Dirichlet 농도 벡터

    def __post_init__(self) -> None:
        c = self.cfg
        self._rng = np.random.default_rng(c.seed)

        # Pass 1: 투표소 세팅 (상수 로드 또는 합성)
        if c.voters_per_station is not None:
            self._N = np.asarray(c.voters_per_station, dtype=np.int64)
        else:
            self._N = self._synthesize_voters()

        M = self._N.size
        # Beta(a,b): mean=a/(a+b)=turnout, a+b=kappa. 단위별 배열 또는 스칼라.
        tmean = c._turnout_unit if c._turnout_unit is not None \
            else np.full(M, c.turnout_mean)
        tkappa = c.turnout_kappa_per_unit if c.turnout_kappa_per_unit is not None \
            else np.full(M, c.turnout_kappa)
        tkappa = np.asarray(tkappa, dtype=float)
        # 투표율 mean=1.0(사전투표)인 단위는 Beta가 정의 안 되므로 결정론 처리 플래그
        self._turnout_deterministic = (tmean >= 1.0 - 1e-9)      # (M,)
        safe_mean = np.clip(tmean, 1e-6, 1.0 - 1e-9)
        self._beta_a = safe_mean * tkappa                        # (M,)
        self._beta_b = (1.0 - safe_mean) * tkappa                # (M,)

        # Dirichlet 농도 alpha: 우선순위 (1)단위별 raw alpha → (2)단위별 지지율×conc
        #   → (3)공통 지지율×conc
        if c.alpha_per_unit is not None:
            self._alpha = np.asarray(c.alpha_per_unit, dtype=float)   # (M, K)
        elif c._support_unit is not None:
            self._alpha = c._support_unit * c.support_conc           # (M, K)
        else:
            self._alpha = c._base * c.support_conc                   # (K,)

    # ----- Pass 1 보조: 합성 유권자 수 -----
    def _synthesize_voters(self) -> np.ndarray:
        c = self.cfg
        rng = np.random.default_rng(c.seed + 1)
        # 로그정규로 투표소 규모 분포 모사(소형~대형 혼재), 평균≈mean_voters
        sigma = 0.5
        mu = np.log(c.mean_voters) - sigma**2 / 2
        n = rng.lognormal(mu, sigma, size=c.n_stations)
        return np.clip(n, 50, None).astype(np.int64)

    # ----- 핵심: 한 청크(batch) 시뮬레이션 -----
    def _simulate_chunk(self, batch: int) -> np.ndarray:
        """
        batch회 반복을 동시에 시뮬레이션.
        반환: (batch,) bool — 각 반복에서 쌍둥이 표가 한 곳이라도 발생했는가.
        """
        rng = self._rng
        M = self._N.size
        N = self._N[None, :]                       # (1, M) 브로드캐스트

        # Pass 2: 투표자 수 V (Beta 투표율 × 유권자 수). beta 모수는 단위별 (M,) 배열.
        turnout = rng.beta(self._beta_a[None, :], self._beta_b[None, :], size=(batch, M))
        # 투표율 ~100% (사전투표) 단위는 결정론적으로 turnout=1 고정
        turnout = np.where(self._turnout_deterministic[None, :], 1.0, turnout)
        V = rng.binomial(N, turnout)               # (batch, M) 정수 투표자 수

        # Pass 3: 지지율 P ~ Dirichlet. 단위별 alpha(M,K)는 Gamma 트릭으로 벡터화.
        #   Dirichlet(alpha) ≡ G_k~Gamma(alpha_k) 정규화. rng.gamma는 shape 배열 지원.
        alpha = self._alpha
        if alpha.ndim == 1:                        # 공통 (K,) → (1,1,K) 브로드캐스트
            alpha_b = alpha[None, None, :]
        else:                                      # 단위별 (M,K) → (1,M,K)
            alpha_b = alpha[None, :, :]
        K = alpha_b.shape[-1]
        gam = rng.gamma(np.broadcast_to(alpha_b, (batch, M, K)))   # (batch, M, K)
        P = gam / gam.sum(axis=2, keepdims=True)
        pA = P[..., 0]                             # (batch, M)
        pB = P[..., 1]

        # Pass 4: 득표수 추출
        phi = self.cfg.dispersion
        if phi == 1.0:
            # 이항분해 (정확): X_A~Bin(V,pA), X_B~Bin(V-X_A, pB/(1-pA))
            X_A = rng.binomial(V, pA)
            rem = V - X_A
            denom = 1.0 - pA
            pB_cond = np.clip(np.where(denom > 1e-12, pB / denom, 0.0), 0.0, 1.0)
            X_B = rng.binomial(rem, pB_cond)
        else:
            # sub/over-binomial: 정규근사, 분산 = φ·V·p(1-p) (A, B 독립근사)
            mA = V * pA
            mB = V * pB
            X_A = np.round(rng.normal(mA, np.sqrt(phi * V * pA * (1 - pA))))
            X_B = np.round(rng.normal(mB, np.sqrt(phi * V * pB * (1 - pB))))
            X_A = np.clip(X_A, 0, V).astype(np.int64)
            X_B = np.clip(X_B, 0, V).astype(np.int64)

        # Pass 5: 쌍둥이 표 검증 (생일 문제)
        #   서로 다른 두 투표소 i, j 에서 (A_i==A_j) AND (B_i==B_j) 인 충돌 탐지.
        #   (A, B) 쌍을 단일 정수키로 인코딩: key = A * BIG + B
        BIG = int(self._N.max()) + 1               # B는 V<=N 이므로 B < BIG 보장
        key = X_A.astype(np.int64) * BIG + X_B.astype(np.int64)  # (batch, M)
        key.sort(axis=1)                           # 행별 정렬
        # 정렬 후 인접값이 같으면 곧 (A,B) 쌍 충돌 = 쌍둥이 투표소 존재
        eq = (np.diff(key, axis=1) == 0)           # (batch, M-1)
        collision = eq.any(axis=1)                 # 한 쌍이라도 충돌?
        # 쌍(pair) 개수: 같은 키가 연속 m개면 C(m,2)쌍. 인접 동일 개수 합산은
        # '연속 run 길이-1'의 합 = (m-1) 합이라 run당 (m-1). 그러나 pair 수는
        # C(m,2). 대부분 m=2라 둘이 일치하므로 인접동일 카운트로 근사 가능하나
        # 정확히는 run 길이로 계산.
        n_pairs = np.empty(key.shape[0], dtype=np.int64)
        for i in range(key.shape[0]):              # batch는 작게(chunk) 유지되므로 OK
            if not collision[i]:
                n_pairs[i] = 0; continue
            vals, counts = np.unique(key[i], return_counts=True)
            n_pairs[i] = int((counts * (counts - 1) // 2).sum())
        return collision, n_pairs                  # (batch,), (batch,)

    # ----- 전체 몬테카를로 -----
    def run(self, verbose: bool = True) -> "SimResult":
        c = self.cfg
        total = c.n_iter
        hits = 0
        done = 0
        total_pairs = 0
        MAXBIN = 64                                 # 쌍 개수 히스토그램 상한
        pair_hist = np.zeros(MAXBIN + 1, dtype=np.int64)

        while done < total:
            batch = min(c.chunk, total - done)
            # 청크 단위 결과(발생 여부, 쌍 개수)
            chunk_any, chunk_pairs = self._simulate_chunk(batch)
            hits += int(chunk_any.sum())
            total_pairs += int(chunk_pairs.sum())
            pair_hist += np.bincount(np.clip(chunk_pairs, 0, MAXBIN),
                                     minlength=MAXBIN + 1)
            done += batch
            if verbose:
                print(f"  진행 {done:>7,}/{total:,}  누적 적중 {hits:>6,}"
                      f"  추정확률 {hits/done:.4%}", end="\r")

        if verbose:
            print()
        p = hits / total
        # Wilson 95% 신뢰구간
        z = 1.959963985
        denom = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denom
        half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
        return SimResult(
            prob=p, hits=hits, n_iter=total,
            ci_low=max(0.0, center - half), ci_high=min(1.0, center + half),
            n_stations=int(self._N.size), total_voters=int(self._N.sum()),
            expected_pairs=total_pairs / total,
            pair_hist=pair_hist,
        )


@dataclass
class SimResult:
    prob: float
    hits: int
    n_iter: int
    ci_low: float
    ci_high: float
    n_stations: int
    total_voters: int
    expected_pairs: float = 0.0       # 1회 선거당 기대 쌍둥이 쌍 개수
    pair_hist: np.ndarray | None = None   # 쌍 개수별 빈도 (index=쌍수)

    # ----- 쌍둥이 쌍 개수 확률 -----
    def p_exactly(self, k: int) -> float:
        """정확히 k쌍이 나올 확률."""
        if self.pair_hist is None or k >= self.pair_hist.size:
            return 0.0
        return float(self.pair_hist[k] / self.pair_hist.sum())

    def p_at_least(self, k: int) -> float:
        """k쌍 이상 나올 확률."""
        if self.pair_hist is None:
            return 0.0
        return float(self.pair_hist[k:].sum() / self.pair_hist.sum())

    def pair_distribution(self, upto: int = 6) -> dict[int, float]:
        tot = self.pair_hist.sum() if self.pair_hist is not None else 1
        return {k: float(self.pair_hist[k] / tot) for k in range(upto + 1)}

    def __str__(self) -> str:
        s = (
            "\n=== 쌍둥이 표 시뮬레이션 결과 ===\n"
            f"투표소 수        : {self.n_stations:,}\n"
            f"총 유권자(상수)  : {self.total_voters:,}\n"
            f"반복(iteration)  : {self.n_iter:,}\n"
            f"적중(쌍둥이 발생): {self.hits:,}\n"
            f"추정 확률        : {self.prob:.4%}\n"
            f"95% 신뢰구간     : [{self.ci_low:.4%}, {self.ci_high:.4%}]\n"
            f"기대 쌍둥이 쌍수 : {self.expected_pairs:.3f} 쌍/선거\n"
        )
        if self.pair_hist is not None:
            dist = self.pair_distribution(6)
            s += "쌍 개수 분포     : " + \
                 ", ".join(f"{k}쌍 {v:.1%}" for k, v in dist.items()) + "\n"
        return s


# ---------------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = SimConfig(
        n_stations=860,
        mean_voters=1500,
        turnout_mean=0.60,
        turnout_kappa=200.0,
        base_support=(0.75, 0.15, 0.10),
        support_conc=500.0,
        n_iter=100_000,
        chunk=2_000,
        seed=42,
    )
    sim = TwinVoteSimulator(cfg)
    print(f"투표소 {cfg.n_stations}곳 / 총 유권자 {sim._N.sum():,}명 / "
          f"{cfg.n_iter:,}회 반복 시작...")
    result = sim.run(verbose=True)
    print(result)
