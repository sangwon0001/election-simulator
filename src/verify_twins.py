"""
쌍둥이 단위 자료 무결성 검증 (논문 §2.4).

표 1의 9쌍 18개 개표단위 전수에 대해:
  ① 내부 산술 일치성: 관내사전 + 선거일 = 계 (voters/votes/A/B), A+B ≤ 투표수
  ② 독립 수집 경로 교차 대조: 전남 9개 단위는 자동 크롤(crawl_provinces)과
     별도 세션 수동 크롤(jeonnam_units.csv) 양쪽 존재 → 수치 대조
2026-06-10 실행 결과: 전체 통과.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_provinces import load_units  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

TWINS = {  # 파일 → 쌍둥이 단위 (region, dong)
    "2800_인천광역시.json": [("연수구", "송도1동"), ("연수구", "송도2동")],
    "2900_광주광역시.json": [("광산구", "송정1동")],
    "4600_전라남도.json": [("고흥군", "금산면"), ("장성군", "북하면"), ("함평군", "엄다면"),
                       ("여수시", "삼일동"), ("신안군", "하의면"), ("화순군", "이양면"),
                       ("강진군", "병영면"), ("보성군", "노동면"), ("신안군", "팔금면")],
    "4700_경상북도.json": [("의성군", "점곡면"), ("영덕군", "창수면")],
    "4800_경상남도.json": [("하동군", "양보면"), ("거창군", "위천면")],
    "5300_전북특별자치도.json": [("익산시", "웅포면"), ("진안군", "성수면")],
}


def main() -> int:
    manual = pd.read_csv(DATA / "jeonnam_units.csv")
    fail = 0
    for fname, dongs in TWINS.items():
        city, df, iA, iB = load_units(str(DATA / "crawl_provinces" / fname))
        for reg, dong in dongs:
            sub = df[(df.region == reg) & (df.dong == dong)]
            g = {r.gubun: r for r in sub.itertuples()}
            e, b, t = g["관내사전투표"], g["선거일투표"], g["계"]
            ok_sum = all(getattr(e, c) + getattr(b, c) == getattr(t, c)
                         for c in ["voters", "votes", "A", "B"])
            ok_ab = (e.A + e.B <= e.votes) and (b.A + b.B <= b.votes)
            cross = "N/A"
            if fname.startswith("4600"):
                m = manual[(manual.region == reg) & (manual.dong == dong)
                           & (manual.gubun == "관내사전투표")]
                ok_x = (len(m) == 1 and m.iloc[0].voters == e.voters
                        and m.iloc[0].votes == e.votes
                        and m.iloc[0].A == e.A and m.iloc[0].B == e.B)
                cross = "일치" if ok_x else "불일치!"
                if not ok_x:
                    fail += 1
            if not (ok_sum and ok_ab):
                fail += 1
            print(f"{city[:2]} {reg} {dong:<6} 산술{'✓' if ok_sum else '✗'} "
                  f"합계{'✓' if ok_ab else '✗'} 교차:{cross} "
                  f"| 사전 A={e.A} B={e.B} V={e.votes}")
    print(f"\n→ {'전체 통과' if fail == 0 else f'{fail}건 실패'}")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
