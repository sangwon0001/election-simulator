"""
크롤링한 22개 시군구 JSON → 전남 개표단위 데이터셋 병합/전처리
=============================================================

출력:
  jeonnam_units.csv  : 개표단위(동×구분)별 선거인수·후보별 득표 정제 테이블
  jeonnam_voters.npy : 시뮬레이터 주입용 단위별 선거인수(N_i) 배열

추가: 실제 데이터에서 '쌍둥이 개표단위'(서로 다른 두 단위의 A표·B표 동시 일치)가
      실제로 존재하는지 경험적으로 직접 확인.

A후보 = 민형배(더불어민주당), B후보 = 이정현(국민의힘)
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# 합계/특수투표(개표단위 아님) 제외 키워드
SKIP_DONG = {"합계", "거소투표", "관외사전투표", "잘못투입·구분된투표지",
             "잘못투입·구분된 투표지", "관외사전", "거소·선상투표"}


def load_all() -> pd.DataFrame:
    recs = []
    for p in sorted(glob.glob(str(DATA/"jeonnam_manual"/"jeonnam_46*.json"))):
        d = json.load(open(p))
        if d.get("loading"):
            raise RuntimeError(f"{p} 가 로딩상태로 저장됨 — 재크롤 필요")
        region = d["townName"]
        for r in d["rows"]:
            dong, gubun = r["dong"], r["gubun"]
            if dong in SKIP_DONG:
                continue
            recs.append({"region": region, "dong": dong, "gubun": gubun,
                         "voters": r["voters"], "votes": r["votes"],
                         "A": r["A"], "B": r["B"],
                         "c1": r["c1"], "c2": r["c2"], "c3": r["c3"]})
    return pd.DataFrame(recs)


def collisions(df: pd.DataFrame, label: str) -> None:
    """(A,B) 쌍 충돌(쌍둥이 단위) 경험적 탐지."""
    pairs = list(zip(df["A"], df["B"]))
    cnt = Counter(pairs)
    dup = {k: v for k, v in cnt.items() if v >= 2}
    n = len(pairs)
    n_collision_pairs = sum(v * (v - 1) // 2 for v in dup.values())
    print(f"\n[{label}] 단위 수={n}")
    print(f"  서로 다른 두 단위가 (A,B) 동시 일치하는 '쌍둥이' 그룹: {len(dup)}개")
    print(f"  쌍둥이 쌍(pair) 총 개수: {n_collision_pairs}")
    if dup:
        # 예시 출력 (최대 8개)
        shown = sorted(dup.items(), key=lambda kv: -kv[1])[:8]
        for (a, b), v in shown:
            members = df[(df.A == a) & (df.B == b)][["region", "dong", "gubun"]]
            who = ", ".join(f"{m.region}·{m.dong}({m.gubun})" for m in members.itertuples())
            print(f"    A={a}, B={b}  ×{v}곳 → {who}")


def main() -> None:
    df = load_all()
    print(f"전체 행(동×구분 포함): {len(df)}  | 시군구 {df.region.nunique()}개")

    # ---- 단위 정의별 데이터셋 ----
    # (1) 동 단위 '계' (각 동의 총합)
    dong_total = df[df.gubun == "계"].reset_index(drop=True)
    # (2) 동×구분 중 실제 투표 단위(선거일투표 / 관내사전투표)
    unit_split = df[df.gubun.isin(["선거일투표", "관내사전투표"])].reset_index(drop=True)

    print(f"\n동 단위(계): {len(dong_total)}개")
    print(f"동×구분(선거일+관내사전): {len(unit_split)}개")

    # ---- 경험적 쌍둥이(충돌) 확인 ----
    collisions(dong_total, "동 단위(계)")
    collisions(unit_split, "동×구분(선거일/관내사전)")

    # ---- 저장 ----
    # CSV: 동×구분 전체(분석용)
    out = df.copy()
    out.to_csv(DATA/"jeonnam_units.csv", index=False, encoding="utf-8-sig")
    # 시뮬레이터용 N_i: 동 단위 '계' 선거인수 (가장 자연스러운 개표단위)
    N = dong_total["voters"].to_numpy(dtype=np.int64)
    np.save(DATA/"jeonnam_voters.npy", N)

    # 실제 전남 지지율(시뮬레이터 base_support 캘리브레이션용)
    tot = df[df.gubun == "계"][["A", "B", "c1", "c2", "c3"]].sum()
    total_valid = tot.sum()
    print(f"\n[전남 실제 지지율] (유효표 {int(total_valid):,} 기준)")
    names = ["민형배(더민주)=A", "이정현(국힘)=B", "이종욱(진보)", "강은미(정의)", "김광만(무소속)"]
    for nm, v in zip(names, tot):
        print(f"  {nm:>18}: {v/total_valid:.4f}  ({int(v):,})")

    print(f"\n저장 완료: jeonnam_units.csv ({len(out)}행), "
          f"jeonnam_voters.npy ({N.size}개 단위, 합 {N.sum():,})")


if __name__ == "__main__":
    main()
