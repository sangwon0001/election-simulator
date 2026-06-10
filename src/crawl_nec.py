"""
중앙선관위 선거통계시스템 자동 크롤러 (Playwright).

제9회 지방선거(2026-06-03) 시·도지사선거 개표단위별(읍면동×구분) 결과를
시·도(cityCode)별로 자동 순회하며 수집한다.

  · 시·도 선택 → townCode(시군구) 드롭다운 자동 채워짐 → 시군구마다 검색·파싱
  · 후보 수는 시·도마다 다르므로 후보 컬럼을 일반적으로 추출
  · 결과: crawl/<cityCode>_<시도명>.json (시군구별 rows 포함)

사용:
  .venv/bin/python crawl_nec.py            # 기본: 9개 도(道)
  .venv/bin/python crawl_nec.py 4600 4700  # 특정 시도만
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# 정중한 크롤: 요청 간 딜레이(초). 환경변수 DELAY 로 조정 가능.
DELAY = float(os.environ.get("DELAY", "2.0"))
CITY_DELAY = float(os.environ.get("CITY_DELAY", "4.0"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

HOME_URL = "https://info.nec.go.kr/main/main_load.xhtml"
# '개표단위별 개표결과' 메뉴(투·개표 VC → VCCP08). 직접 진입 불가, 메뉴 경유 필요.
MENU_URL = ("https://info.nec.go.kr/main/showDocument.xhtml"
            "?electionId=0020260603&topMenuId=VC&secondMenuId=VCCP08")
SEARCH_BTN = '.f_search input[type=image][alt="검색"]'

# 전체 시·도 코드→이름
ALL_CITIES = {
    "1100": "서울특별시", "2600": "부산광역시", "2700": "대구광역시",
    "2800": "인천광역시", "2900": "광주광역시", "3000": "대전광역시",
    "3100": "울산광역시", "5100": "세종특별자치시",
    "4100": "경기도", "5200": "강원특별자치도", "4300": "충청북도",
    "4400": "충청남도", "5300": "전북특별자치도", "4600": "전라남도",
    "4700": "경상북도", "4800": "경상남도", "4900": "제주특별자치도",
}
# 비교용 기본 대상: 읍면동 구조를 갖는 9개 도(道)
DEFAULT_CITIES = {c: ALL_CITIES[c] for c in
    ["4100", "5200", "4300", "4400", "5300", "4600", "4700", "4800", "4900"]}

OUT = Path(__file__).resolve().parent.parent / "data" / "crawl_provinces"


def to_int(s: str):
    s = re.sub(r"[^0-9-]", "", s or "")
    return int(s) if s not in ("", "-") else None


def parse_table(page) -> list[dict]:
    """현재 페이지 표를 파싱. 후보 컬럼은 가변(앞4 + 후보들 + 계,무효,기권)."""
    return page.evaluate(
        """() => {
        const num = s => { const n=parseInt((s||'').replace(/[^0-9-]/g,''),10); return isNaN(n)?null:n; };
        const rows = [...document.querySelectorAll('table tbody tr')].map(tr =>
            [...tr.querySelectorAll('th,td')].map(c => c.innerText.trim()));
        if (!rows.length) return [];
        const out = [];
        for (const r of rows) {
            const nc = r.length;
            // 앞: 0동명 1구분 2선거인수 3투표수 | 끝: 계, 무효투표수, 기권자수
            const cand = r.slice(4, nc-3).map(num);   // 후보별 득표(가변)
            out.push({
                dong: r[0], gubun: r[1],
                voters: num(r[2]), votes: num(r[3]),
                cand: cand,
                valid: num(r[nc-3]), invalid: num(r[nc-2]), abstain: num(r[nc-1]),
            });
        }
        return out;
    }"""
    )


def candidate_names(page) -> list[str]:
    """헤더에서 후보명 추출 (있으면)."""
    return page.evaluate(
        """() => {
        const ths = [...document.querySelectorAll('table thead th')].map(t=>t.innerText.trim());
        return ths;
    }"""
    )


def set_select(page, sel_id: str, value: str) -> None:
    """hidden select 도 처리: 값 설정 + change 이벤트 발생 (MCP 방식)."""
    page.evaluate(
        """([id, v]) => { const s=document.getElementById(id);
           s.value=v; s.dispatchEvent(new Event('change',{bubbles:true})); }""",
        [sel_id, value],
    )


def click_search(page) -> None:
    page.evaluate(
        """() => document.querySelector('.f_search input[type=image][alt="검색"]').click()"""
    )


def get_towns(page) -> list[dict]:
    return page.evaluate(
        """() => [...document.querySelectorAll('#townCode option')]
            .filter(o=>o.value!=='-1').map(o=>({code:o.value, name:o.text}))"""
    )


def crawl_city(page, city_code: str, city_name: str) -> dict:
    # 시·도 선택 → townCode 채워질 때까지 대기
    set_select(page, "cityCode", city_code)
    page.wait_for_function(
        "() => document.querySelectorAll('#townCode option').length > 1",
        timeout=15000,
    )
    towns = get_towns(page)
    print(f"  {city_name}: 시군구 {len(towns)}개", flush=True)

    result = {}
    for t in towns:
        rows = []
        for attempt in range(4):
            set_select(page, "townCode", t["code"])
            try:
                with page.expect_navigation(wait_until="networkidle", timeout=25000):
                    click_search(page)
            except Exception:
                pass  # 네비게이션 감지 실패해도 아래서 상태 검증
            cur = page.evaluate("() => document.getElementById('townCode')?.value")
            nrows = page.evaluate("() => document.querySelectorAll('table tbody tr').length")
            if cur == t["code"] and nrows > 3:
                rows = parse_table(page)
                break
            # 빈 응답(throttle 의심) → 백오프 후 재시도
            time.sleep(DELAY * (attempt + 2))
        result[t["name"]] = {"code": t["code"], "nRows": len(rows), "rows": rows}
        flag = "" if rows else "  <<< 빈응답"
        print(f"    {t['name']:>6}: {len(rows)}행, 합계선거인 "
              f"{rows[0]['voters'] if rows else '?'}{flag}", flush=True)
        time.sleep(DELAY)            # 정중한 간격
    return {"city": city_name, "headers": candidate_names(page), "towns": result}


def main():
    args = sys.argv[1:]
    cities = {c: ALL_CITIES.get(c, c) for c in args} if args else DEFAULT_CITIES
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()
        # 홈 진입(세션 확보) → 메뉴 클릭 경유로 리포트 페이지 로드
        page.goto(HOME_URL, wait_until="domcontentloaded")
        page.goto(MENU_URL, wait_until="domcontentloaded")
        # #electionCode 는 hidden 이므로 attached 상태만 대기
        page.wait_for_selector("#electionCode", state="attached", timeout=20000)
        # 시·도지사선거 선택 보장 (electionCode=3)
        set_select(page, "electionCode", "3")
        page.wait_for_timeout(500)

        for code, name in cities.items():
            print(f"[{code}] {name} 크롤링...")
            try:
                data = crawl_city(page, code, name)
                (OUT / f"{code}_{name}.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8")
                print(f"  → 저장: crawl/{code}_{name}.json")
            except Exception as e:
                print(f"  !! 실패 {name}: {e}")
            time.sleep(CITY_DELAY)      # 시·도 간 간격
        browser.close()
    print("완료.")


if __name__ == "__main__":
    main()
