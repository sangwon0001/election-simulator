"""
Pass 1 데이터 로더: 전라남도 투표소별 선거인수(N_i) 추출 & 전처리
================================================================

[데이터 소스]
중앙선거관리위원회 선거통계시스템 (info.nec.go.kr)
  → 역대선거 → 투표소별 개표결과 / 선거인수
  → 시·도: 전라남도 선택 → 엑셀(xls) 다운로드

선관위는 공식 OpenAPI(data.go.kr '중앙선거관리위원회_투표소별 개표결과')도 제공하나,
선거별 커버리지가 들쭉날쭉하므로 가장 안전한 방법은 통계시스템에서
'투표구별 선거인수' 엑셀을 직접 받아 아래 전처리를 적용하는 것입니다.

[전처리 단계]
  1. 헤더 정리: 선관위 엑셀은 병합셀/2단 헤더가 흔함 → skiprows로 잘라냄
  2. '합계'·'소계' 행 제거: 시군구/읍면동 소계가 섞여 있으면 이중계산
  3. 선거인수 컬럼만 추출 → 정수 배열화
  4. 결측/0 제거 후 numpy 배열 반환

이 모듈의 load_jeonnam_voters() 결과를 SimConfig.voters_per_station 에 넣으세요.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_jeonnam_voters(
    xls_path: str | Path,
    voter_col: str = "선거인수",
    sheet: str | int = 0,
    skiprows: int = 0,
    drop_keywords: tuple[str, ...] = ("합계", "소계", "계", "관외", "거소", "선상", "재외"),
) -> np.ndarray:
    """
    선관위 엑셀에서 투표소별 선거인수 배열을 추출.

    Parameters
    ----------
    xls_path : 다운로드한 .xls/.xlsx 경로
    voter_col : 선거인수 컬럼명 (엑셀 실제 헤더에 맞게 조정)
    skiprows : 상단 병합 헤더 줄 수 (엑셀 열어보고 조정)
    drop_keywords : 투표소가 아닌 집계행/특수투표 제거용 키워드

    Returns
    -------
    np.ndarray (int64) : 투표소별 선거인수 N_i
    """
    path = Path(xls_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. info.nec.go.kr 에서 전남 투표구별 선거인수 엑셀을 받으세요."
        )

    df = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows)
    df.columns = [str(c).strip() for c in df.columns]

    if voter_col not in df.columns:
        raise KeyError(
            f"'{voter_col}' 컬럼 없음. 실제 컬럼: {list(df.columns)}\n"
            f"voter_col 인자를 실제 헤더명으로 바꾸세요."
        )

    # 집계행/특수투표 제거: 모든 문자열 컬럼에서 키워드 포함 행 탈락
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols):
        mask_drop = pd.Series(False, index=df.index)
        for col in str_cols:
            for kw in drop_keywords:
                mask_drop |= df[col].astype(str).str.contains(kw, na=False)
        df = df[~mask_drop]

    # 선거인수 정수화 (콤마 제거)
    s = (
        df[voter_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    voters = pd.to_numeric(s, errors="coerce").dropna()
    voters = voters[voters > 0].astype(np.int64).to_numpy()

    if voters.size == 0:
        raise ValueError("유효한 선거인수 행이 0개. skiprows/voter_col 재확인.")
    return voters


def summarize(voters: np.ndarray) -> str:
    return (
        f"투표소 {voters.size:,}곳 | 총 선거인 {voters.sum():,}명 | "
        f"평균 {voters.mean():.0f} | 중앙값 {np.median(voters):.0f} | "
        f"최소 {voters.min():,} | 최대 {voters.max():,}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python data_loader.py <선관위_엑셀경로> [선거인수컬럼명]")
        print("예시  : python data_loader.py jeonnam_2022.xls 선거인수")
        sys.exit(0)

    col = sys.argv[2] if len(sys.argv) > 2 else "선거인수"
    N = load_jeonnam_voters(sys.argv[1], voter_col=col)
    print(summarize(N))
    np.save(Path(__file__).resolve().parent.parent / "data" / "jeonnam_voters.npy", N)
    print("→ jeonnam_voters.npy 저장. 시뮬레이션에서 np.load 로 불러오세요.")
