#!/usr/bin/env python
"""Phase 16 - 欄位盤點與完整度剖面 (descriptive, all 12 months).

../reference/EDA.md 的原則是「純描述性的圖不做」,所以前 15 個 phase 全是為了殺死題目而做的
檢定。這一支補回被跳過的那一層:每個欄位的實際域、鍵的唯一性、以及最重要的
「不存在的列」—— 這份資料只發布出現過的組合,缺列不是 0 而是「無此觀測」,
而缺列佔了邏輯網格的絕大部分。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
from common import PARQUET, PARQUET_GLOB, ROOT, YMS, YM_LABEL, AGE_LABEL, AGES, DOW_LABEL

PQ = PARQUET_GLOB
R = {}


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=10")
    con.execute("SET memory_limit='12GB'")
    con.execute("SET enable_progress_bar=false")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{PQ}')")

    # ---------------------------------------------------------------- 16.1 欄位域
    print("=== 16.1  欄位實際域 (12 個月全掃) ===")
    dom = con.execute("""
    SELECT count(*) n_rows,
           count(DISTINCT ym) n_ym, count(DISTINCT dow_n) n_dow,
           count(DISTINCT arr_hour) n_hour,
           count(DISTINCT o_dong) n_o, count(DISTINCT d_dong) n_d,
           count(DISTINCT sex) n_sex, count(DISTINCT age) n_age,
           count(DISTINCT mtype) n_mtype,
           min(arr_hour) h0, max(arr_hour) h1,
           min(pop) p0, max(pop) p1, sum(pop) psum,
           min(mean_min) m0, max(mean_min) m1,
           sum(masked::INT) n_masked,
           sum((pop IS NULL)::INT) n_null_pop,
           sum((mean_min IS NULL)::INT) n_null_min,
           sum((NOT masked AND pop IS NULL)::INT) n_unmasked_null
    FROM r""").df().iloc[0].to_dict()
    for k, v in dom.items():
        print(f"  {k:<18} {v:,}" if isinstance(v, int) else f"  {k:<18} {v}")
    R["domains"] = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in dom.items()}

    codes = con.execute("""
    SELECT count(DISTINCT c) n FROM (
      SELECT o_dong c FROM r UNION SELECT d_dong FROM r)""").fetchone()[0]
    print(f"\n  代碼空間(出發∪到達)= {codes}")
    oo = con.execute("""
    SELECT (SELECT count(*) FROM (SELECT DISTINCT o_dong c FROM r
              EXCEPT SELECT DISTINCT d_dong FROM r)) origin_only,
           (SELECT count(*) FROM (SELECT DISTINCT d_dong c FROM r
              EXCEPT SELECT DISTINCT o_dong FROM r)) dest_only""").df().iloc[0]
    print(f"  只當出發 / 只當到達 = {int(oo.origin_only)} / {int(oo.dest_only)}")
    R["code_space"] = dict(n_codes=int(codes), origin_only=int(oo.origin_only),
                           dest_only=int(oo.dest_only))

    vals = {c: [x[0] for x in con.execute(
                f"SELECT DISTINCT {c} v FROM r ORDER BY 1").fetchall()]
            for c in ("dow_n", "sex", "age", "mtype")}
    for c, v in vals.items():
        print(f"  {c:<8} = {v}")
    R["value_sets"] = vals

    # ---------------------------------------------------------------- 16.2 鍵唯一性
    # 全表一次 GROUP BY 會溢寫超過可用磁碟。ym 與 arr_hour 在單檔內是常數,所以
    # 「檔內 (요일,출발,도착,성별,나이,유형) 唯一」與「全域主鍵唯一」等價。
    print("\n=== 16.2  主鍵唯一性(逐檔,288 檔)===")
    con.execute("""CREATE TABLE dup(ym INT, o_dong INT, d_dong INT, c BIGINT,
                                identical BOOLEAN, pop DOUBLE, masked BOOLEAN)""")
    for ym in YMS:
        for hh in range(24):
            p = f"{PARQUET}/ym={ym}/h{hh:02d}.parquet"
            con.execute(f"""
            INSERT INTO dup
            SELECT {ym}, o_dong, d_dong, count(*),
                   (count(DISTINCT pop) <= 1 AND count(DISTINCT mean_min) = 1
                    AND count(DISTINCT masked) = 1),
                   min(pop), bool_or(masked)
            FROM read_parquet('{p}')
            GROUP BY dow_n, o_dong, d_dong, sex, age, mtype HAVING count(*) > 1""")
    du = con.execute("""SELECT count(*) n_groups, sum(c) n_rows, max(c) max_c,
    sum(c) - count(*) n_extra, sum(identical::INT) n_identical,
    sum(masked::INT) n_masked, sum(coalesce(pop,0) * (c-1)) dup_pop,
    sum((o_dong = d_dong)::INT) n_selfloop FROM dup""").df().iloc[0]
    print(f"  重複鍵組 {int(du.n_groups):,}  涉及列 {int(du.n_rows):,}"
          f"  每組最大重複度 {int(du.max_c)}")
    print(f"  其中「兩列完全相同(pop/시간/遮蔽皆同)」= {int(du.n_identical):,}"
          f"  ({du.n_identical/du.n_groups:.2%})")
    print(f"  -> 主鍵{'唯一' if du.n_groups == 0 else '不唯一'}"
          f";多餘列佔全體 {du.n_extra/int(dom['n_rows']):.4%}")
    print(f"  重複造成的重複計量 = {du.dup_pop:,.0f} / {dom['psum']:,.0f}"
          f" = {du.dup_pop/dom['psum']:.4%}")
    dm = con.execute("""SELECT ym, count(*) n, sum(coalesce(pop,0)*(c-1)) v
                    FROM dup GROUP BY 1 ORDER BY 1""").df()
    print("  逐月: " + "  ".join(f"{YM_LABEL[int(y)]}={int(n):,}"
                                 for y, n in zip(dm.ym, dm.n)))
    dr = con.execute("""SELECT (o_dong BETWEEN 1101000 AND 1125999) o_s,
    (d_dong BETWEEN 1101000 AND 1125999) d_s, count(*) n FROM dup
    GROUP BY 1,2 ORDER BY 3 DESC""").df()
    print("  區位: " + "  ".join(
        f"{'首爾' if o else '外部'}→{'首爾' if d else '外部'}={int(n):,}"
        for o, d, n in zip(dr.o_s, dr.d_s, dr.n)))
    R["key_unique"] = dict(
        dup_groups=int(du.n_groups), extra_rows=int(du.n_extra),
        max_multiplicity=int(du.max_c), n_identical=int(du.n_identical),
        dup_pop=float(du.dup_pop), dup_pop_share=float(du.dup_pop / dom["psum"]),
        n_selfloop=int(du.n_selfloop), by_ym=dm.to_dict("records"),
        by_region=dr.to_dict("records"))

    # ---------------------------------------------------------------- 16.3 逐月穩定
    print("\n=== 16.3  逐月契約穩定性 ===")
    mon = con.execute("""
    SELECT ym, count(*) n_rows, sum(masked::INT) n_masked,
           count(DISTINCT o_dong) n_codes,
           count(DISTINCT o_dong) FILTER (o_dong BETWEEN 1101000 AND 1125999) n_seoul_dong,
           count(DISTINCT o_dong // 1000) FILTER (o_dong BETWEEN 1101000 AND 1125999) n_gu,
           count(DISTINCT mtype) n_mtype, count(DISTINCT age) n_age,
           count(DISTINCT arr_hour) n_hour, count(DISTINCT dow_n) n_dow,
           min(pop) minpop, max(mean_min) maxmin, sum(pop) psum
    FROM r GROUP BY ym ORDER BY ym""").df()
    mon["mask_rate"] = mon.n_masked / mon.n_rows
    print(mon.assign(ym=[YM_LABEL[y] for y in mon.ym]).to_string(index=False,
          formatters={"n_rows": "{:,}".format, "n_masked": "{:,}".format,
                      "mask_rate": "{:.4f}".format, "psum": "{:.4g}".format}))
    R["monthly"] = mon.to_dict("records")

    # 코드 집합의 월간 차분
    sets = {ym: set(x[0] for x in con.execute(
                f"SELECT DISTINCT o_dong FROM r WHERE ym={ym}").fetchall()) for ym in YMS}
    base = sets[YMS[0]]
    diffs = [dict(ym=ym, added=sorted(sets[ym] - base), dropped=sorted(base - sets[ym]))
             for ym in YMS if sets[ym] != base]
    print(f"\n  相對 202001 有代碼異動的月份: {len(diffs)}")
    for d in diffs:
        print(f"    {d['ym']}  +{d['added']}  -{d['dropped']}")
    seoul = {ym: {c for c in s if 1101000 <= c <= 1125999} for ym, s in sets.items()}
    print(f"  首爾行政洞集合跨 12 月是否完全相同: "
          f"{all(seoul[y] == seoul[YMS[0]] for y in YMS)}  (n={len(seoul[YMS[0]])})")
    R["code_drift"] = dict(diffs=diffs, seoul_stable=bool(
        all(seoul[y] == seoul[YMS[0]] for y in YMS)), n_seoul_dong=len(seoul[YMS[0]]))

    # ------------------------------------------------- 16.4 結構性稀疏:不存在的列
    print("\n=== 16.4  結構性稀疏 —— 缺列不是 0,是「無此觀測」===")
    NS = len(seoul[YMS[0]])
    UNIV = NS * NS * 24 * 7 * 2 * 16 * 9
    print(f"  首爾內部邏輯網格 = {NS}洞 × {NS}洞 × 24時 × 7曜 × 2性 × 16齡 × 9型"
          f" = {UNIV:,} 格/月")

    lad = con.execute(f"""
    WITH s AS (SELECT * FROM r WHERE ym = 202001
                 AND o_dong BETWEEN 1101000 AND 1125999
                 AND d_dong BETWEEN 1101000 AND 1125999)
    SELECT (SELECT count(*) FROM (SELECT DISTINCT o_dong,d_dong FROM s)) l1,
           (SELECT count(*) FROM (SELECT DISTINCT o_dong,d_dong,arr_hour FROM s)) l2,
           (SELECT count(*) FROM (SELECT DISTINCT o_dong,d_dong,arr_hour,dow_n FROM s)) l3,
           (SELECT count(*) FROM (SELECT DISTINCT o_dong,d_dong,arr_hour,dow_n,sex
                                  FROM s)) l4,
           (SELECT count(*) FROM s) l6""").df().iloc[0]
    l1, l2, l3, l4, l6 = (int(lad[k]) for k in ("l1", "l2", "l3", "l4", "l6"))
    ladder = [
        ("OD 洞對",              l1, NS * NS),
        ("OD × 시간",            l2, l1 * 24),
        ("OD × 시간 × 요일",     l3, l2 * 7),
        ("… × 성별",             l4, l3 * 2),
        ("… × 나이 × 유형(=列)", l6, l4 * 16 * 9),
    ]
    print(f"\n  2020-01 首爾內部,逐層條件填充率:")
    print(f"  {'層級':<24}{'實有':>16}{'上層×分支':>18}{'條件填充率':>12}")
    for name, have, poss in ladder:
        print(f"  {name:<24}{have:>16,}{poss:>18,}{have/poss:>11.1%}")
    print(f"\n  絕對填充率 = {l6:,} / {UNIV:,} = {l6/UNIV:.3%}")
    print(f"  -> 邏輯網格的 {1-l6/UNIV:.2%} 沒有列。這不是缺失值,是「該格無人移動或不足門檻」。")
    R["sparsity"] = dict(n_seoul_dong=NS, universe=UNIV, rows_seoul_internal=l6,
                         fill_abs=l6 / UNIV,
                         ladder=[dict(level=n, have=h, possible=p, cond_fill=h / p)
                                 for n, h, p in ladder])

    # OD 對的覆蓋分布
    odc = con.execute("""
    SELECT n_cells, count(*) n_pairs FROM (
      SELECT o_dong, d_dong, count(*) n_cells FROM r
      WHERE ym=202001 AND o_dong BETWEEN 1101000 AND 1125999
        AND d_dong BETWEEN 1101000 AND 1125999
      GROUP BY 1,2)
    GROUP BY 1 ORDER BY 1""").df()
    q = con.execute("""
    SELECT quantile_cont(n_cells,[0.05,0.25,0.5,0.75,0.95]) q, max(n_cells) mx,
           avg(n_cells) avg_cells FROM (
      SELECT o_dong,d_dong,count(*) n_cells FROM r
      WHERE ym=202001 AND o_dong BETWEEN 1101000 AND 1125999
        AND d_dong BETWEEN 1101000 AND 1125999 GROUP BY 1,2)""").df().iloc[0]
    print(f"\n  每個 OD 洞對的實有格數(2020-01,滿格 = 24×7×2×16×9 = {24*7*2*16*9:,}):")
    print(f"    p05={q.q[0]:.0f}  p25={q.q[1]:.0f}  中位={q.q[2]:.0f}"
          f"  p75={q.q[3]:.0f}  p95={q.q[4]:.0f}  最大={int(q.mx):,}"
          f"  平均={q.avg_cells:.0f}")
    print(f"    從未出現的 OD 洞對 = {NS*NS - l1:,} / {NS*NS:,} ({1-l1/(NS*NS):.1%})")
    R["od_coverage"] = dict(full_cells=24 * 7 * 2 * 16 * 9,
                            pcts=[float(x) for x in q.q], max=int(q.mx),
                            mean=float(q.avg_cells),
                            pairs_present=l1, pairs_possible=NS * NS)

    selfl = con.execute("""
    SELECT count(*) n_rows, sum(pop) v FROM r
    WHERE o_dong = d_dong AND o_dong BETWEEN 1101000 AND 1125999""").df().iloc[0]
    print(f"\n  洞內移動(o_dong = d_dong)存在:{int(selfl.n_rows):,} 列"
          f"({selfl.n_rows/int(dom['n_rows']):.1%}),量體 {selfl.v/dom['psum']:.1%}")
    R["self_loop"] = dict(rows=int(selfl.n_rows), pop=float(selfl.v),
                          row_share=float(selfl.n_rows / int(dom["n_rows"])),
                          pop_share=float(selfl.v / dom["psum"]))

    # 시간 × 요일 격자는 빠짐없이 있는가
    gaps = con.execute("""
    SELECT count(*) FROM (
      SELECT ym, arr_hour, dow_n FROM r GROUP BY 1,2,3)""").fetchone()[0]
    print(f"\n  (월 × 시간 × 요일) 格子: 實有 {gaps} / 應有 {12*24*7} "
          f"-> {'完整' if gaps == 12*24*7 else '有缺'}")
    R["cell_grid"] = dict(have=int(gaps), expect=12 * 24 * 7)

    # ---------------------------------------------------------------- 16.5 遮蔽剖面
    print("\n=== 16.5  遮蔽剖面(12 個月,首爾內部)===")
    mk_age = con.execute("""
    SELECT age, count(*) n, avg(masked::INT) rate FROM r
    WHERE o_dong BETWEEN 1101000 AND 1125999 AND d_dong BETWEEN 1101000 AND 1125999
    GROUP BY 1 ORDER BY 1""").df()
    mk_age["age_l"] = [AGE_LABEL[a] for a in mk_age.age]
    print("  年齡別:")
    for _, x in mk_age.iterrows():
        print(f"    {x.age_l:<7}{x.rate:>8.1%}   (n={int(x.n):,})")
    mk_h = con.execute("""
    SELECT arr_hour, avg(masked::INT) rate,
           sum(3.0) FILTER (masked) / (sum(pop) FILTER (NOT masked) + sum(3.0) FILTER (masked)) ub
    FROM r WHERE o_dong BETWEEN 1101000 AND 1125999
      AND d_dong BETWEEN 1101000 AND 1125999
    GROUP BY 1 ORDER BY 1""").df()
    print("\n  到達時別(格子遮蔽率 / 被遮蔽量上界佔比):")
    print("    " + "  ".join(f"{int(h):02d}:{r:.0%}/{u:.0%}"
                             for h, r, u in zip(mk_h.arr_hour, mk_h.rate, mk_h.ub)))
    mk_t = con.execute("""
    SELECT mtype, avg(masked::INT) rate,
           sum(3.0) FILTER (masked) / (sum(pop) FILTER (NOT masked) + sum(3.0) FILTER (masked)) ub
    FROM r WHERE o_dong BETWEEN 1101000 AND 1125999
      AND d_dong BETWEEN 1101000 AND 1125999
    GROUP BY 1 ORDER BY 2 DESC""").df()
    print("\n  이동유형별:")
    for _, x in mk_t.iterrows():
        print(f"    {x.mtype}  격자 {x.rate:>6.1%}   량 상한 {x.ub:>6.1%}")
    R["masking"] = dict(by_age=mk_age.to_dict("records"), by_hour=mk_h.to_dict("records"),
                        by_mtype=mk_t.to_dict("records"))

    # ---------------------------------------------------------------- 16.6 值域量化
    print("\n=== 16.6  發布值的量化結構 ===")
    qz = con.execute("""
    SELECT age, sex, min(pop) mn, count(DISTINCT pop) nd FROM r
    WHERE NOT masked GROUP BY 1,2 ORDER BY 1,2""").df()
    piv = qz.pivot(index="age", columns="sex", values="mn")
    print("  各年齡層的最小未遮蔽發布值(≈ 該層最粗的擴大權重下界):")
    for a in AGES:
        print(f"    {AGE_LABEL[a]:<7} F={piv.loc[a,'F']:>6.2f}   M={piv.loc[a,'M']:>6.2f}")
    exact3 = con.execute("SELECT count(*) FROM r WHERE NOT masked AND pop = 3.0").fetchone()[0]
    print(f"\n  pop 恰等於 3.00 的列 = {exact3:,} ({exact3/int(dom['n_rows']):.2%})")
    print(f"  pop 相異值總數 = {con.execute('SELECT count(DISTINCT pop) FROM r').fetchone()[0]:,}")
    R["quantisation"] = dict(min_by_age_sex=qz.to_dict("records"), n_pop_exactly_3=int(exact3))

    # ---------------------------------------------------------------- 16.7 對帳
    print("\n=== 16.7  與 ETL 報告對帳 ===")
    rep = json.load(open(f"{ROOT}/eda/etl_report.json"))
    etl_rows = sum(x["rows"] for x in rep)
    etl_mask = sum(x["masked"] for x in rep)
    print(f"  ETL 報告 {len(rep)} 檔,列數 {etl_rows:,},遮蔽 {etl_mask:,}")
    print(f"  parquet 實查  列數 {int(dom['n_rows']):,},遮蔽 {int(dom['n_masked']):,}")
    print(f"  -> 列數{'一致' if etl_rows == int(dom['n_rows']) else '不一致'}"
          f" / 遮蔽{'一致' if etl_mask == int(dom['n_masked']) else '不一致'}")
    print(f"  每檔 wc -l 對帳全過: {all(x['rows'] == x['expected'] for x in rep)}"
          f" / 星期別無法解析的列: {sum(x['bad_dow'] for x in rep)}")
    R["reconcile"] = dict(n_files=len(rep), etl_rows=etl_rows, etl_masked=etl_mask,
                          pq_rows=int(dom["n_rows"]), pq_masked=int(dom["n_masked"]),
                          wc_all_ok=bool(all(x["rows"] == x["expected"] for x in rep)),
                          bad_dow=sum(x["bad_dow"] for x in rep))

    with open(f"{ROOT}/eda/results_p16.json", "w") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False, default=float)
    print("\nwrote results_p16.json")


if __name__ == "__main__":
    main()
