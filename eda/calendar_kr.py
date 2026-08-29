"""韓國/首爾日曆 2020–2026:공휴일 + 사회적 거리두기 政策期間。

2020 的公休日表是**人工輸入**並對照官方公告核過的。2021–2026 是用 `holidays`
套件（0.102，PUBLIC 類別）產生後**寫死進本檔**的 —— 產生時先驗證它能完全重現
2020 那張手工表（18/18 逐日吻合,含兩個臨時공휴일與落在週六的 현충일),所以
它對農曆換算與 대체공휴일 規則的處理是可信的。寫死而不是在執行期呼叫套件,
是為了讓結果不依賴套件版本;`verify_against_library()` 可以隨時重新對帳。

核心用途:資料沒有日期欄位,但我們知道每個 (대상연월, 요일) 格子聚合了哪些
日期。所以可以精確算出每個格子的曝光組成 —— 見 cell_exposure()。

**兩件事會靜默毀掉每日正規化,所以這裡都改成大聲失敗:**

1. 問一個沒有公休日資料的年份,以前會回報「零個公休日」。現在 cell_exposure()
   直接丟 ValueError。沒有涵蓋不等於沒有假日。
2. `dose` / `dose_alt` 只在 2020 有定義（거리두기 단계是為那年的因果設計編碼的,
   而該設計已放棄）。2020 年以外現在回傳 None,不再拿 None 去做除法而 TypeError。
"""
from datetime import date, timedelta

# ---------------------------------------------------------------- 공휴일 2020
# 확신도 high = 官方年度공휴일; medium = 臨時공휴일/需核對
HOLIDAYS_2020 = {
    date(2020, 1, 1):  ("신정", "high"),
    date(2020, 1, 24): ("설 연휴", "high"),
    date(2020, 1, 25): ("설날", "high"),
    date(2020, 1, 26): ("설 연휴", "high"),
    date(2020, 1, 27): ("설 대체공휴일", "high"),
    date(2020, 3, 1):  ("삼일절(일요일)", "high"),
    date(2020, 4, 15): ("제21대 국회의원선거 임시공휴일", "medium"),
    date(2020, 4, 30): ("부처님오신날", "high"),
    date(2020, 5, 5):  ("어린이날", "high"),
    date(2020, 6, 6):  ("현충일(토요일)", "high"),
    date(2020, 8, 15): ("광복절(토요일)", "high"),
    date(2020, 8, 17): ("임시공휴일", "medium"),
    date(2020, 9, 30): ("추석 연휴 첫날", "high"),
    date(2020, 10, 1): ("추석", "high"),
    date(2020, 10, 2): ("추석 연휴", "high"),
    date(2020, 10, 3): ("개천절(토요일)", "high"),
    date(2020, 10, 9): ("한글날", "high"),
    date(2020, 12, 25): ("성탄절", "high"),
}
# 법정공휴일은 아니지만 다수 사업장이 휴무 -> 별도 취급, 기본은 제외
SOFT_HOLIDAYS_2020 = {date(2020, 5, 1): ("근로자의 날", "medium")}

# ------------------------------------------------------------ 공휴일 2021–2026
# `holidays` 0.102 PUBLIC 類別產生。확신도 check = 只出現在單一年份的新增/재지정,
# 投入使用前要對照 공공데이터포털「특일 정보」核過。
HOLIDAYS_2021_2026 = {
    # ---- 2021
    date(2021,  1,  1): ("신정연휴", "high"),
    date(2021,  2, 11): ("설날 전날", "high"),
    date(2021,  2, 12): ("설날", "high"),
    date(2021,  2, 13): ("설날 다음날", "high"),
    date(2021,  3,  1): ("삼일절", "high"),
    date(2021,  5,  5): ("어린이날", "high"),
    date(2021,  5, 19): ("부처님오신날", "high"),
    date(2021,  6,  6): ("현충일", "high"),
    date(2021,  8, 15): ("광복절", "high"),
    date(2021,  8, 16): ("광복절 대체공휴일", "high"),
    date(2021,  9, 20): ("추석 전날", "high"),
    date(2021,  9, 21): ("추석", "high"),
    date(2021,  9, 22): ("추석 다음날", "high"),
    date(2021, 10,  3): ("개천절", "high"),
    date(2021, 10,  4): ("개천절 대체공휴일", "high"),
    date(2021, 10,  9): ("한글날", "high"),
    date(2021, 10, 11): ("한글날 대체공휴일", "high"),
    date(2021, 12, 25): ("기독탄신일", "high"),
    # ---- 2022
    date(2022,  1,  1): ("신정연휴", "high"),
    date(2022,  1, 31): ("설날 전날", "high"),
    date(2022,  2,  1): ("설날", "high"),
    date(2022,  2,  2): ("설날 다음날", "high"),
    date(2022,  3,  1): ("삼일절", "high"),
    date(2022,  3,  9): ("대통령 선거일", "medium"),
    date(2022,  5,  5): ("어린이날", "high"),
    date(2022,  5,  8): ("부처님오신날", "high"),
    date(2022,  6,  1): ("지방선거일", "medium"),
    date(2022,  6,  6): ("현충일", "high"),
    date(2022,  8, 15): ("광복절", "high"),
    date(2022,  9,  9): ("추석 전날", "high"),
    date(2022,  9, 10): ("추석", "high"),
    date(2022,  9, 11): ("추석 다음날", "high"),
    date(2022,  9, 12): ("추석 대체공휴일", "high"),
    date(2022, 10,  3): ("개천절", "high"),
    date(2022, 10,  9): ("한글날", "high"),
    date(2022, 10, 10): ("한글날 대체공휴일", "high"),
    date(2022, 12, 25): ("기독탄신일", "high"),
    # ---- 2023
    date(2023,  1,  1): ("신정연휴", "high"),
    date(2023,  1, 21): ("설날 전날", "high"),
    date(2023,  1, 22): ("설날", "high"),
    date(2023,  1, 23): ("설날 다음날", "high"),
    date(2023,  1, 24): ("설날 대체공휴일", "high"),
    date(2023,  3,  1): ("삼일절", "high"),
    date(2023,  5,  5): ("어린이날", "high"),
    date(2023,  5, 27): ("부처님오신날", "high"),
    date(2023,  5, 29): ("부처님오신날 대체공휴일", "high"),
    date(2023,  6,  6): ("현충일", "high"),
    date(2023,  8, 15): ("광복절", "high"),
    date(2023,  9, 28): ("추석 전날", "high"),
    date(2023,  9, 29): ("추석", "high"),
    date(2023,  9, 30): ("추석 다음날", "high"),
    date(2023, 10,  2): ("임시공휴일", "medium"),
    date(2023, 10,  3): ("개천절", "high"),
    date(2023, 10,  9): ("한글날", "high"),
    date(2023, 12, 25): ("기독탄신일", "high"),
    # ---- 2024
    date(2024,  1,  1): ("신정연휴", "high"),
    date(2024,  2,  9): ("설날 전날", "high"),
    date(2024,  2, 10): ("설날", "high"),
    date(2024,  2, 11): ("설날 다음날", "high"),
    date(2024,  2, 12): ("설날 대체공휴일", "high"),
    date(2024,  3,  1): ("삼일절", "high"),
    date(2024,  4, 10): ("국회의원 선거일", "medium"),
    date(2024,  5,  5): ("어린이날", "high"),
    date(2024,  5,  6): ("어린이날 대체공휴일", "high"),
    date(2024,  5, 15): ("부처님오신날", "high"),
    date(2024,  6,  6): ("현충일", "high"),
    date(2024,  8, 15): ("광복절", "high"),
    date(2024,  9, 16): ("추석 전날", "high"),
    date(2024,  9, 17): ("추석", "high"),
    date(2024,  9, 18): ("추석 다음날", "high"),
    date(2024, 10,  1): ("국군의 날", "medium"),
    date(2024, 10,  3): ("개천절", "high"),
    date(2024, 10,  9): ("한글날", "high"),
    date(2024, 12, 25): ("기독탄신일", "high"),
    # ---- 2025
    date(2025,  1,  1): ("신정연휴", "high"),
    date(2025,  1, 27): ("임시공휴일", "medium"),
    date(2025,  1, 28): ("설날 전날", "high"),
    date(2025,  1, 29): ("설날", "high"),
    date(2025,  1, 30): ("설날 다음날", "high"),
    date(2025,  3,  1): ("삼일절", "high"),
    date(2025,  3,  3): ("삼일절 대체공휴일", "high"),
    date(2025,  5,  5): ("부처님오신날; 어린이날", "high"),
    date(2025,  5,  6): ("부처님오신날 대체공휴일; 어린이날 대체공휴일", "high"),
    date(2025,  6,  3): ("대통령 선거일", "medium"),
    date(2025,  6,  6): ("현충일", "high"),
    date(2025,  8, 15): ("광복절", "high"),
    date(2025, 10,  3): ("개천절", "high"),
    date(2025, 10,  5): ("추석 전날", "high"),
    date(2025, 10,  6): ("추석", "high"),
    date(2025, 10,  7): ("추석 다음날", "high"),
    date(2025, 10,  8): ("추석 대체공휴일", "high"),
    date(2025, 10,  9): ("한글날", "high"),
    date(2025, 12, 25): ("기독탄신일", "high"),
    # ---- 2026
    date(2026,  1,  1): ("신정연휴", "high"),
    date(2026,  2, 16): ("설날 전날", "high"),
    date(2026,  2, 17): ("설날", "high"),
    date(2026,  2, 18): ("설날 다음날", "high"),
    date(2026,  3,  1): ("삼일절", "high"),
    date(2026,  3,  2): ("삼일절 대체공휴일", "high"),
    date(2026,  5,  1): ("노동절", "check"),  # 2026 신설/재지정 — 확인 필요
    date(2026,  5,  5): ("어린이날", "high"),
    date(2026,  5, 24): ("부처님오신날", "high"),
    date(2026,  5, 25): ("부처님오신날 대체공휴일", "high"),
    date(2026,  6,  3): ("지방선거일", "medium"),
    date(2026,  6,  6): ("현충일", "high"),
    date(2026,  7, 17): ("제헌절", "check"),  # 2026 신설/재지정 — 확인 필요
    date(2026,  8, 15): ("광복절", "high"),
    date(2026,  8, 17): ("광복절 대체공휴일", "high"),
    date(2026,  9, 24): ("추석 전날", "high"),
    date(2026,  9, 25): ("추석", "high"),
    date(2026,  9, 26): ("추석 다음날", "high"),
    date(2026, 10,  3): ("개천절", "high"),
    date(2026, 10,  5): ("개천절 대체공휴일", "high"),
    date(2026, 10,  9): ("한글날", "high"),
    date(2026, 12, 25): ("기독탄신일", "high"),
}

# 2026 起 `holidays` 把 노동절(근로자의 날) 從 bank 類移進 public 類,並新增
# 제헌절 —— 兩者都只出現在 2026,型態符合法令變更而不是資料瑕疵。**제헌절
# (2026-07-17) 落在資料窗內(資料到 2026-07),所以這一條必須查證,不能沿用。**
# 2025 年以前 근로자의 날 不是법정공휴일,照舊留在 SOFT_HOLIDAYS。
NEEDS_VERIFICATION = {date(2026, 5, 1), date(2026, 7, 17)}

HOLIDAYS = {**HOLIDAYS_2020, **HOLIDAYS_2021_2026}
HOLIDAY_YEARS = frozenset(d.year for d in HOLIDAYS)

# 법정공휴일은 아니지만 다수 사업장이 휴무 -> 별도 취급, 기본은 제외
SOFT_HOLIDAYS = dict(SOFT_HOLIDAYS_2020)
SOFT_HOLIDAYS.update({date(y, 5, 1): ("근로자의 날", "medium")
                      for y in range(2021, 2026)})  # 2026 은 위 NEEDS_VERIFICATION 참조

# ------------------------------------------- 서울 사회적 거리두기 단계 (2020)
# 2020-06-28 이전에는 3단계 체계가 없었으므로 명칭이 다르다. `index`는 강도를
# 비교 가능하게 만들기 위한 **모델링 선택**이지 공식 수치가 아니다.
# ALT_INDEX 는 견고성 확인용 대안 코딩.
# ⚠️ 2020 년에만 정의된다. 이를 쓰던 인과 설계는 폐기되었다.
POLICY = [
    (date(2020, 1, 1),  date(2020, 1, 19), "pre",              0.0, 0.0),
    (date(2020, 1, 20), date(2020, 2, 22), "first-case/경계",   0.5, 1.0),
    (date(2020, 2, 23), date(2020, 3, 21), "위기경보 심각",     1.5, 2.0),
    (date(2020, 3, 22), date(2020, 4, 19), "강화된 거리두기",   2.5, 3.0),
    (date(2020, 4, 20), date(2020, 5, 5),  "사회적 거리두기",   1.5, 2.0),
    (date(2020, 5, 6),  date(2020, 8, 15), "생활 속 거리두기",  1.0, 1.0),
    (date(2020, 8, 16), date(2020, 8, 29), "수도권 2단계",      2.0, 2.0),
    (date(2020, 8, 30), date(2020, 9, 13), "수도권 2.5단계",    2.5, 3.0),
    (date(2020, 9, 14), date(2020, 10, 11), "수도권 2단계",     2.0, 2.0),
    (date(2020, 10, 12), date(2020, 11, 18), "수도권 1단계",    1.0, 1.0),
    (date(2020, 11, 19), date(2020, 11, 23), "수도권 1.5단계",  1.5, 1.5),
    (date(2020, 11, 24), date(2020, 12, 7), "수도권 2단계b",    2.0, 2.0),
    (date(2020, 12, 8), date(2020, 12, 31), "수도권 2.5단계b",  2.5, 3.0),
]
POLICY_LABELS = [p[2] for p in POLICY]
POLICY_YEARS = frozenset({2020})

# 서울 한정 추가 조치 (단계와 별개로 겹쳐서 발효) — 기록용, 아직 모델에 미반영
SEOUL_EXTRA = [
    (date(2020, 5, 9), None, "이태원 클럽발 유흥시설 집합금지"),
    (date(2020, 9, 28), date(2020, 10, 11), "추석 특별방역기간"),
    (date(2020, 11, 7), None, "거리두기 5단계 체계로 개편 (그 이전은 3단계 체계)"),
    (date(2020, 12, 23), None, "수도권 5인 이상 사적모임 금지"),
]


def _days_of(ym, dow_n):
    """해당 월에서 dow_n(1=월 .. 7=일)에 해당하는 모든 날짜."""
    y, m = ym // 100, ym % 100
    d, out = date(y, m, 1), []
    while d.month == m:
        if d.isoweekday() == dow_n:
            out.append(d)
        d += timedelta(days=1)
    return out


def occurrences(ym, dow_n):
    return len(_days_of(ym, dow_n))


def require_year(ym):
    """공휴일 자료가 없는 해를 조용히 '휴일 0일'로 만들지 않는다."""
    y = ym // 100
    if y not in HOLIDAY_YEARS:
        raise ValueError(
            f"no holiday table for {y} (have {min(HOLIDAY_YEARS)}-{max(HOLIDAY_YEARS)}). "
            f"Extend HOLIDAYS_2021_2026 before using {ym}; an absent table is not an "
            f"empty one, and every per-day normalisation depends on it.")
    return y


def holidays_in(ym, dow_n, include_soft=False):
    require_year(ym)
    tab = dict(HOLIDAYS)
    if include_soft:
        tab.update(SOFT_HOLIDAYS)
    return [d for d in _days_of(ym, dow_n) if d in tab]


def _policy_of(d):
    """2020 년 밖에서는 (None, None, None). 호출측에서 None 을 처리해야 한다."""
    for lo, hi, label, idx, alt in POLICY:
        if lo <= d <= hi:
            return label, idx, alt
    return None, None, None


def cell_exposure(ym, dow_n, include_soft=False):
    """(대상연월, 요일) 격자 하나의 노출 구성.

    격자가 어떤 날짜들을 합친 것인지는 달력에서 정확히 알 수 있으므로,
    날짜 컬럼이 없어도 노출은 정확히 계산된다.

    `dose` / `dose_alt` 는 2020 년에만 정의되고 그 밖에서는 None 이다.
    """
    require_year(ym)
    days = _days_of(ym, dow_n)
    hol = set(holidays_in(ym, dow_n, include_soft))
    in_policy = ym // 100 in POLICY_YEARS
    share, idx, alt = {}, 0.0, 0.0
    for d in days:
        label, i, a = _policy_of(d)
        share[label] = share.get(label, 0) + 1 / len(days)
        if in_policy:
            idx += i / len(days)
            alt += a / len(days)
    return dict(ym=ym, dow_n=dow_n, n_days=len(days),
                n_holiday=len(hol), holiday_frac=len(hol) / len(days),
                dose=idx if in_policy else None,
                dose_alt=alt if in_policy else None,
                n_unverified=sum(d in NEEDS_VERIFICATION for d in days),
                shares=share, dates=[d.isoformat() for d in days])


def panel(yms, include_soft=False):
    return [cell_exposure(ym, d, include_soft) for ym in yms for d in range(1, 8)]


def verify_against_library():
    """이 파일의 표를 `holidays` 패키지와 다시 대조한다. 설치돼 있을 때만.

    返回 (n_checked, [差異]). 2020 은 손으로 검증한 기준선이므로 여기서
    어긋나면 패키지 쪽을 의심해야 한다 —— 생성 시점에는 18/18 로 일치했다.
    """
    try:
        import holidays as _h
        from holidays.constants import PUBLIC
    except ImportError:
        return None, ["holidays package not installed"]
    diffs = []
    for y in sorted(HOLIDAY_YEARS):
        lib = set(_h.SouthKorea(years=y, categories=(PUBLIC,)))
        mine = {d for d in HOLIDAYS if d.year == y}
        for d in sorted(mine - lib):
            diffs.append(f"{d} in this file, not in library ({HOLIDAYS[d][0]})")
        for d in sorted(lib - mine):
            diffs.append(f"{d} in library, not in this file")
    return len(HOLIDAYS), diffs


if __name__ == "__main__":
    import sys
    DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    yms = [int(a) for a in sys.argv[1:]] or [202001 + i // 12 * 100 + i % 12
                                             for i in range(12)]
    print(f"{'ym':>7} {'dow':>4} {'n':>2} {'hol':>3} {'dose':>5} {'alt':>5}  composition")
    for r in panel(yms):
        comp = "  ".join(f"{k} {v:.2f}" for k, v in sorted(r["shares"].items(),
                                                          key=lambda x: -x[1]))
        dose = "  —  " if r["dose"] is None else f"{r['dose']:>5.2f}"
        alt = "  —  " if r["dose_alt"] is None else f"{r['dose_alt']:>5.2f}"
        print(f"{r['ym']:>7} {DOW[r['dow_n']]:>4} {r['n_days']:>2} "
              f"{r['n_holiday']:>3} {dose} {alt}  {comp}")
    n, diffs = verify_against_library()
    print(f"\nholiday table: {n} dates, {min(HOLIDAY_YEARS)}-{max(HOLIDAY_YEARS)}")
    print("library cross-check:", "clean" if not diffs else f"{len(diffs)} diffs")
    for d in diffs:
        print("   ", d)
    print(f"needs manual verification: {sorted(str(d) for d in NEEDS_VERIFICATION)}")
