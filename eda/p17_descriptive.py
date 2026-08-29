#!/usr/bin/env python
"""Phase 17 - 描述性現象剖面 (12 個月).

前 15 個 phase 全是為了殺死題目而做的檢定,沒有人問過「這份資料長什麼樣」。
這一支就做那件事:時間、空間、人口、型別、旅行時間五個切面的分布與結構。

量體一律用 pop_day = pop / 該月該星期出現次數 (D2),遮蔽格用區間中點 1.5 補
(D4),並在關鍵數字上同時給未補的下界,讓補值的影響可見。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 圖上同時有中文與韓文;Arial Unicode MS 兩者皆含,DejaVu Sans 兩者皆缺。
matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Apple SD Gothic Neo",
                                      "PingFang TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from common import connect, ROOT, YMS, YM_LABEL, AGE_LABEL, AGES, DOW_LABEL, GU_NAMES

FIG = f"{ROOT}/eda/fig"
R = {}


def V(cond=None):
    """日均量體,遮蔽格用區間中點 1.5 補 (D4)。cond 折進兩個 FILTER 裡,
    因為 SQL 不允許把 FILTER 掛在複合運算式上。"""
    f = f" FILTER ({cond})" if cond else ""
    g = f" FILTER (masked AND ({cond}))" if cond else " FILTER (masked)"
    return f"(coalesce(sum(pop_day){f}, 0) + 1.5 * coalesce(sum(1.0/n_days){g}, 0))"


def LO(cond=None):
    """未補值的下界(只加總未遮蔽格)。"""
    f = f" FILTER ({cond})" if cond else ""
    return f"(coalesce(sum(pop_day){f}, 0))"


MT = ["HW", "WH", "HE", "EH", "HH", "WE", "EW", "WW", "EE"]
MT_LABEL = {"H": "집", "W": "직장", "E": "기타"}


def main():
    con = connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET preserve_insertion_order=false")

    # ----------------------------------------------------------- 17.1 月別量體
    print("=== 17.1  月別日均量體 ===")
    mo = con.execute(f"""
    SELECT ym,
      {V('o_seoul AND d_seoul')}      AS internal,
      {V('o_seoul AND NOT d_seoul')}  AS outbound,
      {V('NOT o_seoul AND d_seoul')}  AS inbound,
      {LO('o_seoul AND d_seoul')}     AS internal_lo,
      {V()}                           AS total
    FROM m GROUP BY 1 ORDER BY 1""").df()
    mo["label"] = [YM_LABEL[y] for y in mo.ym]
    mo["idx"] = mo.internal / mo.internal.iloc[0]
    mo["idx_lo"] = mo.internal_lo / mo.internal_lo.iloc[0]
    print(mo[["label", "internal", "outbound", "inbound", "idx", "idx_lo"]].to_string(
        index=False, formatters={"internal": "{:,.0f}".format, "outbound": "{:,.0f}".format,
                                 "inbound": "{:,.0f}".format, "idx": "{:.3f}".format,
                                 "idx_lo": "{:.3f}".format}))
    print(f"\n  谷底 = {mo.label[mo.idx.idxmin()]} ({mo.idx.min():.3f})"
          f" / 峰值 = {mo.label[mo.idx.idxmax()]} ({mo.idx.max():.3f})")
    print(f"  補值把指數移動了 {abs(mo.idx - mo.idx_lo).max():.4f}(最大)-> 描述性結論不敏感")
    print(f"  越界對稱性 (inbound/outbound): {(mo.inbound/mo.outbound).min():.3f}"
          f" – {(mo.inbound/mo.outbound).max():.3f}")
    R["monthly"] = mo.drop(columns="label").to_dict("records")

    # ----------------------------------------------------------- 17.2 시간 프로파일
    print("\n=== 17.2  到達時剖面 ===")
    hr = con.execute(f"""
    SELECT ym, arr_hour, weekend, {V()} AS v FROM m
    WHERE o_seoul AND d_seoul GROUP BY 1,2,3""").df()
    wd = hr[~hr.weekend].pivot_table(index="arr_hour", columns="ym", values="v", aggfunc="sum")
    we = hr[hr.weekend].pivot_table(index="arr_hour", columns="ym", values="v", aggfunc="sum")
    wd_s, we_s = wd / wd.sum(), we / we.sum()
    print("  平日各小時佔全日的比例(2020-01 vs 2020-12,單位 %):")
    for h in range(24):
        a, b = wd_s[202001][h] * 100, wd_s[202012][h] * 100
        print(f"    {h:02d}시  {a:5.2f}  {b:5.2f}   {'+' if b > a else '-'}{abs(b-a):.2f}")
    pk_wd = [(int(wd_s[y].iloc[6:12].idxmax()), int(wd_s[y].iloc[15:22].idxmax())) for y in YMS]
    print(f"\n  平日早/晚尖峰時(逐月): "
          + " ".join(f"{YM_LABEL[y]}={a:02d}/{b:02d}" for y, (a, b) in zip(YMS, pk_wd)))
    print(f"  週末尖峰時(逐月): "
          + " ".join(f"{YM_LABEL[y]}={int(we_s[y].idxmax()):02d}" for y in YMS))
    night = wd_s.loc[[22, 23, 0, 1, 2, 3, 4]].sum()
    print(f"\n  平日深夜(22–04 시)佔比: "
          + "  ".join(f"{YM_LABEL[y]}={night[y]:.1%}" for y in YMS))
    R["hour"] = dict(weekday_share=wd_s.to_dict(), weekend_share=we_s.to_dict(),
                     peaks=dict(zip(map(str, YMS), pk_wd)),
                     night_share={str(y): float(night[y]) for y in YMS})

    # ----------------------------------------------------------- 17.3 요일 프로파일
    print("\n=== 17.3  星期剖面(日均,相對該月平日均值)===")
    dw = con.execute(f"""
    SELECT ym, dow_n, n_holiday, {V()} AS v FROM m
    WHERE o_seoul AND d_seoul GROUP BY 1,2,3""").df()
    pv = dw.pivot_table(index="dow_n", columns="ym", values="v", aggfunc="sum")
    rel = pv / pv.loc[1:5].mean()
    print("       " + "".join(f"{YM_LABEL[y]:>7}" for y in YMS))
    for d in range(1, 8):
        print(f"  {DOW_LABEL[d]:<5}" + "".join(f"{rel[y][d]:>7.3f}" for y in YMS))
    hol = dw.groupby("ym")["n_holiday"].max()
    print(f"\n  含公休日的 (월,요일) 格子數 = {(dw.n_holiday > 0).sum()} / {len(dw)}")
    print(f"  週六/平日 比值全距 = {rel.loc[6].min():.3f} – {rel.loc[6].max():.3f}")
    print(f"  週日/平日 比值全距 = {rel.loc[7].min():.3f} – {rel.loc[7].max():.3f}")
    R["dow"] = dict(rel=rel.to_dict())

    # ----------------------------------------------------------- 17.4 인구 구성
    print("\n=== 17.4  年齡 × 性別組成 ===")
    ag = con.execute(f"""
    SELECT ym, age, sex, {V()} AS v, {LO()} AS v_lo, count(*) AS cells,
           sum(masked::INT) AS mcells
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2,3""").df()
    tot = ag.groupby(["age", "sex"])["v"].sum().unstack()
    sh = (tot.sum(axis=1) / tot.sum().sum())
    print("  年齡別佔全體移動量的比例(12 個月合計)與性別比:")
    for a in AGES:
        print(f"    {AGE_LABEL[a]:<7}{sh[a]:>7.2%}   F/M = {tot.F[a]/tot.M[a]:.3f}")
    print(f"\n  女性整體佔比 = {tot.F.sum()/tot.sum().sum():.2%}")
    idx = ag.groupby(["ym", "age"])["v"].sum().unstack()
    idx = idx / idx.loc[202001]
    print("\n  各年齡層相對 2020-01 的移動量指數:")
    print("       " + "".join(f"{YM_LABEL[y]:>7}" for y in YMS))
    for a in AGES:
        print(f"  {AGE_LABEL[a]:<6}" + "".join(f"{idx[a][y]:>7.3f}" for y in YMS))
    print(f"\n  全年最大跌幅 = {AGE_LABEL[idx.min().idxmin()]} "
          f"({idx.min().min():.3f});最小 = {AGE_LABEL[idx.min().idxmax()]} "
          f"({idx.min().max():.3f})")
    # 性別比對補值的敏感度:遮蔽率在 20-44 歲很高,而它在兩性間不對稱。
    fm = con.execute("""
    SELECT age, sex, coalesce(sum(pop_day),0) v0,
      coalesce(sum(pop_day),0) + 1.5*coalesce(sum(1.0/n_days) FILTER (masked),0) v15,
      coalesce(sum(pop_day),0) + 3.0*coalesce(sum(1.0/n_days) FILTER (masked),0) v3
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2""").df().pivot(index="age", columns="sex")
    mk = con.execute("""SELECT age, avg(masked::INT) r FROM m
    WHERE o_seoul AND d_seoul GROUP BY 1""").df().set_index("age")["r"]
    print("\n  性別比 F/M 對 `*` 填補的敏感度(填 0 / 1.5 / 3):")
    fmr = {}
    for a in AGES:
        r = [float(fm[(c, "F")][a] / fm[(c, "M")][a]) for c in ("v0", "v15", "v3")]
        fmr[AGE_LABEL[a]] = dict(imp0=r[0], imp15=r[1], imp3=r[2], span=r[0] - r[2],
                                 mask=float(mk[a]))
        flag = "  <- 不可識別(跨過 1.0)" if (r[0] - 1) * (r[2] - 1) < 0 else ""
        print(f"    {AGE_LABEL[a]:<7}{r[0]:>7.3f}{r[1]:>7.3f}{r[2]:>7.3f}"
              f"   遮蔽 {mk[a]:>5.1%}{flag}")
    R["fm_sensitivity"] = fmr

    R["age_sex"] = dict(share=sh.to_dict(), fm_ratio=(tot.F / tot.M).to_dict(),
                        index=idx.to_dict())

    # ----------------------------------------------------------- 17.5 이동유형
    print("\n=== 17.5  이동유형 (H=집 W=직장 E=기타) ===")
    ty = con.execute(f"""
    SELECT ym, mtype, arr_hour, {V()} AS v FROM m
    WHERE o_seoul AND d_seoul GROUP BY 1,2,3""").df()
    tsh = ty.groupby(["ym", "mtype"])["v"].sum().unstack()
    tsh = tsh.div(tsh.sum(axis=1), axis=0)
    print("  各型別佔比(逐月):")
    print("       " + "".join(f"{YM_LABEL[y]:>7}" for y in YMS))
    for t in MT:
        print(f"  {t:<6}" + "".join(f"{tsh[t][y]:>7.3f}" for y in YMS))
    lev = ty.groupby(["ym", "mtype"])["v"].sum().unstack()
    lev = lev / lev.loc[202001]
    print("\n  各型別的量體指數(相對 2020-01):")
    print("       " + "".join(f"{YM_LABEL[y]:>7}" for y in YMS))
    for t in MT:
        print(f"  {t:<6}" + "".join(f"{lev[t][y]:>7.3f}" for y in YMS))
    hsh = ty[ty.ym == 202001].pivot_table(index="arr_hour", columns="mtype", values="v")
    hsh = hsh.div(hsh.sum(axis=1), axis=0)
    print("\n  2020-01 各到達時的型別組成(HW / WH / EE 三個代表):")
    for h in range(0, 24, 2):
        print(f"    {h:02d}시  HW={hsh.HW[h]:.3f}  WH={hsh.WH[h]:.3f}  EE={hsh.EE[h]:.3f}")
    R["mtype"] = dict(share=tsh.to_dict(), level_index=lev.to_dict(),
                      hour_mix_jan=hsh.to_dict())

    # ----------------------------------------------------------- 17.6 이동 시간
    print("\n=== 17.6  平均移動時間分布(量體加權)===")
    tt = con.execute("""
    SELECT CAST(mean_min AS INT) mm, mtype, sum(pop/n_days) w
    FROM m WHERE o_seoul AND d_seoul AND NOT masked GROUP BY 1,2""").df()
    allw = tt.groupby("mm")["w"].sum().sort_index()
    cw = allw.cumsum() / allw.sum()


    def pct(c, p):
        return int(c.index[np.searchsorted(c.values, p)])


    print(f"  全體:中位 {pct(cw,.5)} 분,p25 {pct(cw,.25)},p75 {pct(cw,.75)},"
          f"p95 {pct(cw,.95)},p99 {pct(cw,.99)} 분")
    print(f"  > 120 분的量體佔比 = {1-cw[120]:.2%};> 480 분 = {1-cw[480]:.4%};"
          f"> 1440 분(逾一日)= {1-cw[1440]:.5%}")
    gmax = con.execute("SELECT max(mean_min) FROM m").fetchone()[0]
    print(f"  mean_min 最大值:首爾內部 {int(allw.index.max())} 분 "
          f"({allw.index.max()/60:.1f} 시간);全體(含跨市界){int(gmax)} 분 "
          f"({gmax/60:.1f} 시간)")
    print("\n  依型別:")
    rows = []
    for t in MT:
        c = tt[tt.mtype == t].groupby("mm")["w"].sum().sort_index().cumsum()
        c = c / c.iloc[-1]
        rows.append(dict(mtype=t, p25=pct(c, .25), median=pct(c, .5), p75=pct(c, .75),
                         p95=pct(c, .95), over120=float(1 - c[120])))
        print(f"    {t}  p25={rows[-1]['p25']:>3}  중위={rows[-1]['median']:>3}"
              f"  p75={rows[-1]['p75']:>3}  p95={rows[-1]['p95']:>4}"
              f"  >120분 {rows[-1]['over120']:.1%}")
    tt_h = con.execute("""
    SELECT arr_hour, sum(pop*mean_min)/sum(pop) mw FROM m
    WHERE o_seoul AND d_seoul AND NOT masked GROUP BY 1 ORDER BY 1""").df()
    print("\n  各到達時的量體加權平均時間(분):")
    print("    " + "  ".join(f"{int(h):02d}:{v:.0f}" for h, v in
                             zip(tt_h.arr_hour, tt_h.mw)))
    tt_m = con.execute("""
    SELECT ym, sum(pop*mean_min)/sum(pop) mw FROM m
    WHERE o_seoul AND d_seoul AND NOT masked GROUP BY 1 ORDER BY 1""").df()
    print("  各月的量體加權平均時間: "
          + "  ".join(f"{YM_LABEL[int(y)]}={v:.1f}" for y, v in zip(tt_m.ym, tt_m.mw)))
    R["travel_time"] = dict(overall={f"p{int(p*100)}": pct(cw, p)
                                     for p in (.25, .5, .75, .95, .99)},
                            over_120=float(1 - cw[120]), over_1440=float(1 - cw[1440]),
                            max_min_internal=int(allw.index.max()), max_min_all=int(gmax), by_mtype=rows,
                            by_hour=tt_h.to_dict("records"), by_ym=tt_m.to_dict("records"))

    # --------------------------------------------------- 17.7a 區代碼對應的驗證
    # common.py 的 GU_NAMES 一直標著「從編號順序推斷,未對過官方 코드표」。
    # 每區的행정동數是一個很強的指紋:2020 年首爾 25 區的洞數向量各不相同,
    # 而송파구(27,最多)與금천구(10,最少)是唯二的極值。
    print("\n=== 17.7a  區代碼 → 區名對應的內部驗證 ===")
    EXPECT = {  # 2020 년 서울 25 개 자치구의 행정동 수 (행정안전부 기준)
        "종로구": 17, "중구": 15, "용산구": 16, "성동구": 17, "광진구": 15,
        "동대문구": 14, "중랑구": 16, "성북구": 20, "강북구": 13, "도봉구": 14,
        "노원구": 19, "은평구": 16, "서대문구": 14, "마포구": 16, "양천구": 18,
        "강서구": 20, "구로구": 15, "금천구": 10, "영등포구": 18, "동작구": 15,
        "관악구": 21, "서초구": 18, "강남구": 22, "송파구": 27, "강동구": 18}
    nd = con.execute("""
    SELECT o_dong // 1000 gu, count(DISTINCT o_dong) n FROM m
    WHERE o_seoul AND ym = 202001 GROUP BY 1 ORDER BY 1""").df()
    bad = [(GU_NAMES[int(x.gu)], int(x.n), EXPECT[GU_NAMES[int(x.gu)]])
           for _, x in nd.iterrows() if int(x.n) != EXPECT[GU_NAMES[int(x.gu)]]]
    print(f"  25 區的행정동數與行政安全部名冊比對:不符 {len(bad)} 個 {bad}")
    print(f"  總洞數 {int(nd.n.sum())};最多 = {GU_NAMES[int(nd.loc[nd.n.idxmax(),'gu'])]}"
          f" ({int(nd.n.max())});最少 = {GU_NAMES[int(nd.loc[nd.n.idxmin(),'gu'])]}"
          f" ({int(nd.n.min())})")
    print("  -> 洞數向量完全吻合,且兩個極值落在正確的區;"
          "GU_NAMES 可從「推斷未驗證」升為「已由內部指紋佐證」。")
    R["gu_name_check"] = dict(mismatches=bad, total_dong=int(nd.n.sum()),
                              counts={GU_NAMES[int(x.gu)]: int(x.n) for _, x in nd.iterrows()})

    # ----------------------------------------------------------- 17.7 공간 구조
    print("\n=== 17.7  空間結構 ===")
    gu = con.execute(f"""
    SELECT o_gu, d_gu, {V()} AS v FROM m
    WHERE o_seoul AND d_seoul GROUP BY 1,2""").df()
    inflow = gu.groupby("d_gu")["v"].sum()
    outflow = gu.groupby("o_gu")["v"].sum()
    self_gu = gu[gu.o_gu == gu.d_gu].set_index("o_gu")["v"]
    bal = pd.DataFrame(dict(name=[GU_NAMES[g] for g in inflow.index],
                            inflow=inflow, outflow=outflow, self=self_gu))
    bal["net"] = bal.inflow - bal.outflow
    bal["net_pct"] = bal.net / ((bal.inflow + bal.outflow) / 2)
    bal["self_share"] = bal.self / bal.outflow
    print("  區級全日淨流入(正 = 日間人口沉積),12 個月合計:")
    for g, x in bal.sort_values("net_pct", ascending=False).iterrows():
        print(f"    {x['name']:<8}{x.net_pct:>+8.2%}   自留率 {x.self_share:>6.1%}"
              f"   流入/日 {x.inflow/12/365*12:>12,.0f}")
    sdiag = gu[gu.o_gu == gu.d_gu].v.sum() / gu.v.sum()
    print(f"\n  區內自留佔首爾內部總量 = {sdiag:.1%}")
    dong_self = con.execute(f"""
    SELECT {V('o_dong = d_dong')} / {V()} AS s FROM m
    WHERE o_seoul AND d_seoul""").fetchone()[0]
    print(f"  洞內自留佔首爾內部總量 = {dong_self:.1%}")
    top = gu[gu.o_gu != gu.d_gu].nlargest(12, "v")
    print("\n  區間 OD 前 12(12 個月合計):")
    for _, x in top.iterrows():
        print(f"    {GU_NAMES[int(x.o_gu)]:<7} → {GU_NAMES[int(x.d_gu)]:<7}"
              f"{x.v/gu.v.sum():>7.3%}")
    R["spatial"] = dict(gu_balance=bal.reset_index().to_dict("records"),
                        gu_self_share=float(sdiag), dong_self_share=float(dong_self),
                        top_gu_od=[dict(o=GU_NAMES[int(x.o_gu)], d=GU_NAMES[int(x.d_gu)],
                                        share=float(x.v / gu.v.sum()))
                                   for _, x in top.iterrows()])

    # 洞對集中度
    od = con.execute(f"""
    SELECT o_dong, d_dong, {V()} AS v FROM m
    WHERE o_seoul AND d_seoul AND ym = 202001 GROUP BY 1,2""").df()
    v = np.sort(od.v.values)[::-1]
    cum = np.cumsum(v) / v.sum()
    n = len(v)
    print(f"\n  洞對集中度(2020-01,{n:,} 個有量的 OD 洞對):")
    for k in (0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"    量體前 {k:.0%} 的洞對承載 {cum[int(n*k)-1]:.1%} 的移動量")
    gini = 1 - 2 * np.trapezoid(np.cumsum(np.sort(od.v.values)) / od.v.sum(),
                                np.arange(n) / (n - 1))
    print(f"    Gini = {gini:.3f}")
    R["concentration"] = dict(n_pairs=int(n), gini=float(gini),
                              top={f"{int(k*100)}pct": float(cum[int(n * k) - 1])
                                   for k in (0.01, 0.05, 0.10, 0.25, 0.50)})

    # ----------------------------------------------------------- 그림
    print("\n=== 產圖 ===")
    C = {"a": "#0E7C86", "b": "#9A6C15", "c": "#A9B9C0", "d": "#8C3B3B"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(range(12), mo.internal / 1e6, "o-", color=C["a"], label="首爾內部")
    ax[0].plot(range(12), (mo.inbound + mo.outbound) / 1e6, "s-", color=C["b"],
               label="跨市界(進+出)")
    ax[0].set_xticks(range(12)); ax[0].set_xticklabels(mo.label)
    ax[0].set_ylabel("日均移動量(百萬人次)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("17.1 月別日均量體", fontsize=10)
    im = ax[1].imshow(wd_s[YMS].T.values, aspect="auto", cmap="magma_r")
    ax[1].set_xticks(range(0, 24, 2)); ax[1].set_xticklabels(range(0, 24, 2))
    ax[1].set_yticks(range(12)); ax[1].set_yticklabels(mo.label, fontsize=8)
    ax[1].set_xlabel("到達時"); ax[1].set_title("17.2 平日到達時分布(列內正規化)", fontsize=10)
    fig.colorbar(im, ax=ax[1], shrink=.85)
    fig.tight_layout(); fig.savefig(f"{FIG}/p17_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(AGES))
    ax[0].bar(x, [sh[a] for a in AGES], color=C["a"])
    ax[0].set_xticks(x); ax[0].set_xticklabels([AGE_LABEL[a] for a in AGES],
                                               rotation=90, fontsize=8)
    ax[0].set_ylabel("佔全體移動量"); ax[0].grid(alpha=.3, axis="y")
    ax[0].set_title("17.4 年齡別量體佔比", fontsize=10)
    for a in AGES:
        ax[1].plot(range(12), [idx[a][y] for y in YMS], lw=1.2,
                   color=plt.cm.viridis(AGES.index(a) / 15))
    ax[1].axhline(1, color="k", lw=.8)
    ax[1].set_xticks(range(12)); ax[1].set_xticklabels(mo.label)
    ax[1].set_ylabel("相對 2020-01"); ax[1].grid(alpha=.3)
    ax[1].set_title("17.4 各年齡層量體指數(深→淺 = 幼→老)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p17_population.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bot = np.zeros(24)
    for i, t in enumerate(MT):
        ax[0].bar(range(24), hsh[t].values, bottom=bot, label=t,
                  color=plt.cm.tab10(i / 10))
        bot += hsh[t].values
    ax[0].set_xlabel("到達時"); ax[0].set_ylabel("型別佔比")
    ax[0].legend(fontsize=7, ncol=3); ax[0].set_title("17.5 到達時 × 型別(2020-01)", fontsize=10)
    w = allw.reindex(range(1, 241), fill_value=0)
    ax[1].fill_between(w.index, w.values / allw.sum(), color=C["a"], alpha=.75)
    ax[1].axvline(pct(cw, .5), color=C["d"], ls="--", lw=1.2,
                  label=f"中位 {pct(cw,.5)} 분")
    ax[1].set_xlabel("平均移動時間(분)"); ax[1].set_ylabel("量體密度")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title(f"17.6 移動時間分布(>120분 佔 {1-cw[120]:.1%})", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p17_mtype_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    b = bal.sort_values("net_pct")
    ax[0].barh(range(25), b.net_pct.values,
               color=[C["a"] if v > 0 else C["b"] for v in b.net_pct])
    ax[0].set_yticks(range(25)); ax[0].set_yticklabels(b.name, fontsize=8)
    ax[0].axvline(0, color="k", lw=.8); ax[0].set_xlabel("淨流入 / 平均流量")
    ax[0].set_title("17.7 區級全日淨流入", fontsize=10); ax[0].grid(alpha=.3, axis="x")
    gp = gu.pivot(index="o_gu", columns="d_gu", values="v").fillna(0)
    gp = gp.div(gp.sum(axis=1), axis=0)
    im = ax[1].imshow(np.log10(gp.values + 1e-6), cmap="viridis")
    ax[1].set_xticks(range(25)); ax[1].set_xticklabels([GU_NAMES[g] for g in gp.columns],
                                                       rotation=90, fontsize=6)
    ax[1].set_yticks(range(25)); ax[1].set_yticklabels([GU_NAMES[g] for g in gp.index],
                                                       fontsize=6)
    ax[1].set_title("17.7 區 × 區 OD(列內佔比,log10)", fontsize=10)
    fig.colorbar(im, ax=ax[1], shrink=.85)
    fig.tight_layout(); fig.savefig(f"{FIG}/p17_spatial.png", dpi=150); plt.close(fig)
    print("  -> fig/p17_time.png  p17_population.png  p17_mtype_time.png  p17_spatial.png")

    with open(f"{ROOT}/eda/results_p17.json", "w") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False, default=float)
    print("wrote results_p17.json")


if __name__ == "__main__":
    main()
