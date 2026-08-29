#!/usr/bin/env python
"""Build the daily case panel from the KDCA 전수감시 workbook.

Source: 공공데이터포털 15127820, 질병관리청_코로나19 확진자 발생현황(전수감시)_20230831
        https://www.data.go.kr/data/15127820/fileData.do  (공공누리 1유형)
Direct file: /cmm/cmm/fileDownload.do?atchFileId=FILE_000000002906082&fileDetailSn=1

What the workbook does and does not cross:
  시도별 발생 / 시도별 사망  -> 시도 x DAY, no age        (Seoul is one column)
  연령별(10세단위)          -> DAY x 9 age bands, national only
  연령별 사망               -> YEAR x age band, not daily
  시군구별                  -> cumulative rates only, not daily
So an age x region cross does not exist anywhere in the public release.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CASES

SRC = sys.argv[1] if len(sys.argv) > 1 else f"{CASES}/kdca_20230831.xlsx"
OUT = str(CASES)
os.makedirs(OUT, exist_ok=True)


def sheet(xl, name, header_row=4):
    df = xl.parse(name, header=header_row).dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    first = df.columns[0]
    df = df[pd.to_datetime(df[first], errors="coerce").notna()].copy()
    df[first] = pd.to_datetime(df[first])
    df = df.rename(columns={first: "date"})
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].replace("-", 0), errors="coerce").fillna(0).astype("int64")
    return df.reset_index(drop=True)


def main():
    xl = pd.ExcelFile(SRC)
    nat = sheet(xl, "발생별(국내발생+해외유입), 사망")
    age = sheet(xl, "연령별(10세단위)")
    sex = sheet(xl, "성별(남+여)")
    sido = sheet(xl, "시도별 발생(17개시도+검역)")
    sido_d = sheet(xl, "시도별 사망(17개시도+검역) ")

    panel = pd.DataFrame({
        "date": sido["date"],
        "seoul_cases": sido["서울"],
        "seoul_deaths": sido_d.set_index("date").reindex(sido["date"])["서울"].to_numpy(),
        "nat_cases": nat.set_index("date").reindex(sido["date"])["계(명)"].to_numpy(),
        "nat_deaths": nat.set_index("date").reindex(sido["date"])["사망"].to_numpy(),
    })
    AGE_COLS = [c for c in age.columns if c.endswith("세") or c == "80세이상"]
    a = age.set_index("date").reindex(sido["date"])
    for c in AGE_COLS:
        panel[f"nat_age_{c}"] = a[c].to_numpy()
    panel["seoul_share"] = panel["seoul_cases"] / panel["nat_cases"].replace(0, pd.NA)

    panel.to_csv(f"{OUT}/daily_panel.csv", index=False)
    panel.to_parquet(f"{OUT}/daily_panel.parquet")

    print(f"rows {len(panel)}   {panel.date.min().date()} .. {panel.date.max().date()}")
    print("\n=== 2020 monthly totals (Seoul vs national), and Seoul's share ===")
    m = panel[panel.date.dt.year == 2020].groupby(panel.date.dt.month).agg(
        days=("date", "size"), seoul=("seoul_cases", "sum"), seoul_deaths=("seoul_deaths", "sum"),
        national=("nat_cases", "sum"))
    m["seoul_share"] = (m.seoul / m.national).round(3)
    m["seoul_per_day"] = (m.seoul / m.days).round(1)
    print(m.to_string())

    print("\n=== national age composition of cases, 2020 by month (share) ===")
    a20 = panel[panel.date.dt.year == 2020].groupby(panel.date.dt.month)[
        [f"nat_age_{c}" for c in AGE_COLS]].sum()
    print((a20.div(a20.sum(axis=1), axis=0)).round(3).rename(
        columns=lambda c: c.replace("nat_age_", "")).to_string())
    print(f"\nwrote {OUT}/daily_panel.csv and .parquet")


if __name__ == "__main__":
    main()
