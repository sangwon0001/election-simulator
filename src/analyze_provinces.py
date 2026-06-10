"""
시·도별 쌍둥이 투표동 분석 — crawl_provinces/*.json 전부에 대해.

각 도마다:
  · 상위 2후보(A,B) 자동 식별 (총득표 기준)
  · 경험적 쌍둥이 쌍수: 동 단위(계) / 동×구분(관내사전+선거일)
  · 시뮬 기대 쌍수 + P(관측) — 캐노니컬 모델(twin_model: 동별 multinomial)

출력: 표 + provinces_summary.csv
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import twin_model

DATA = Path(__file__).resolve().parent.parent / "data"

SKIP = {"합계", "거소투표", "관외사전투표", "잘못투입·구분된투표지",
        "잘못투입·구분된 투표지", "거소·선상투표", "관외사전"}


def load_units(path: str) -> tuple[str, pd.DataFrame, int, int]:
    d = json.load(open(path))
    city = d["city"]
    # 후보 인덱스: 합계 행들의 cand 합으로 상위2 식별
    cand_tot = None
    recs = []
    for tname, t in d["towns"].items():
        for r in t["rows"]:
            cand = r.get("cand") or []
            if r["dong"] == "합계" and r["gubun"] in ("", None):
                cand_tot = (np.array(cand) if cand_tot is None
                            else cand_tot + np.array(cand))
            if r["dong"] in SKIP:
                continue
            recs.append({"region": tname, "dong": r["dong"], "gubun": r["gubun"],
                         "voters": r["voters"], "votes": r["votes"], "cand": cand})
    df = pd.DataFrame(recs)
    iA, iB = np.argsort(cand_tot)[::-1][:2]
    df["A"] = df["cand"].apply(lambda c: c[iA] if len(c) > iA else None)
    df["B"] = df["cand"].apply(lambda c: c[iB] if len(c) > iB else None)
    df = df.dropna(subset=["A", "B"]).copy()
    df["A"] = df["A"].astype(int); df["B"] = df["B"].astype(int)
    return city, df, int(iA), int(iB)


def empirical_twins(df: pd.DataFrame) -> int:
    pairs = Counter(zip(df["A"], df["B"]))
    return sum(v * (v - 1) // 2 for v in pairs.values() if v >= 2)


def sim_prob(df: pd.DataFrame, observed: int, R: int = 100_000) -> tuple[float, float]:
    """캐노니컬 모델(twin_model): 동별 전체유권자→사전/본/기권 multinomial→득표."""
    D = twin_model.build_dong_table(df)
    if len(D) < 2:
        return float("nan"), float("nan")
    cnt = twin_model.simulate(D, R=R, seed=7)
    exp = float(cnt.mean())
    p_obs = float((cnt >= observed).mean()) if observed > 0 else 1.0
    return exp, p_obs


def main():
    rows = []
    for path in sorted(glob.glob(str(DATA/"crawl_provinces"/"*.json"))):
        city, df, iA, iB = load_units(path)
        split = df[df.gubun.isin(["관내사전투표", "선거일투표"])]
        dong = df[df.gubun == "계"]
        obs_split = empirical_twins(split)
        obs_dong = empirical_twins(dong)
        exp, p_obs = sim_prob(df, obs_split, R=100_000)
        # A 지지율(압승도)
        valid = df[df.gubun == "계"][["A", "B"]].sum()
        share_A = valid["A"] / df[df.gubun == "계"]["votes"].sum()
        rows.append({"시도": city, "동수": len(dong), "동구분수": len(split),
                     "A점유율": round(share_A, 3),
                     "실제_동": obs_dong, "실제_동구분": obs_split,
                     "시뮬_기대": round(exp, 2),
                     "P(관측이상)": round(p_obs, 3)})
        print(f"{city:>10} | 동{len(dong):>3} 동구분{len(split):>3} | "
              f"A {share_A:.0%} | 실제 동{obs_dong} 동구분{obs_split} | "
              f"시뮬기대 {exp:.2f} | P(≥관측) {p_obs:.1%}")
    out = pd.DataFrame(rows)
    out.to_csv(DATA/"provinces_summary.csv", index=False, encoding="utf-8-sig")
    print("\n→ provinces_summary.csv 저장")


if __name__ == "__main__":
    main()
