# Phase 48 — 圖 1：研究設計與管線，而且圖上沒有一個寫死的數字

腳本 `eda/p48_fig1.py` → `eda/fig/p48_figure1.png` / `.pdf` + `results_p48.json`。
`--pdf` 出向量檔。不重算任何東西。

**一句話結論：`paper_structure.md` §7 裡唯一完全沒有材料的那一張圖做出來了。
它是示意圖不是資料圖，而那正是它最容易漂的理由——所以圖上每個數字都在畫圖時
從 results 檔讀，沒有一個是寫死的。**

---

## 1. 為什麼一張示意圖需要一個階段

手畫一個寫著「102 億列」的方框，會在 inventory 變了之後繼續那樣寫下去。示意圖是
全篇最沒有人會去核對的東西，也因此是唯一一種可以錯上好幾個月而不被發現的圖。

做法：跨度、檔數列數、遮蔽率與它碰到哪些帶、實測填充、閘門項數、尺度階梯，
**全部在畫圖時從 `results_inventory/p16/p29/p36/p37/p38` 讀**。inventory 一動而這張圖
沒重跑，下一次執行就會自我矛盾，而 assert 會擋住。

## 2. 五層

```
來源            79 個月、1,896 個檔、10,186,891,962 列、424 洞
   ↓
產品沒告訴你的五件事   沒有日期欄位／합是月內加總／缺列不是 0／遮蔽只在 20–44／沒有個體
   ↓
外部錨          주민등록 ＋ 등록외국인 ／ 자치구官方檔 ／ Chae 接觸調查
   ↓
估計量＋閘門     共同抵達 A → 四統計量；引用閘門、決定論、獨立第二實作
   ↓
三個宣稱        儀器活著 ／ 每個尺度都近零 ／ 這件事有後果
```

## 3. 圖上讀進來的數字（`results_p48.json`）

| 欄位 | 值 | 來源 |
|---|---:|---|
| `span` | 202001–202607 | inventory |
| `months` | 79 | inventory |
| `parquet_files` | 1,896 | inventory |
| `parquet_rows` | 10,186,891,962 | inventory |
| `dong` | 424 | p29 |
| `masking_rate_weighted` | 0.261349 | p16，**格數加權平均** |
| `masked_bands_material` | 20-24 … 40-44 | p16 |
| `masked_band_rate_range` | 0.1091–0.6952 | p16 |
| `masked_bands_trace` | 45-49 … 60-64 | p16 |
| `masked_band_trace_max` | 0.00128（在 **50-54**，不是 45-49） | p16 |
| `masked_band_trace_cap_pct` | 0.13（對上一列取上界） | p16 |
| `masked_bands_none_low` | 0-9、10-14、15-19 | p16 |
| `masked_bands_none_high` | 65-69 … 80+ | p16 |
| `masked_span_material` / `_trace` / `_none` | 20-24 to 40-44／45-64／0-19 and 65+ | p16 |
| `masked_bands_exactly_zero` | 上面兩段 none 的聯集，保留供追溯 | p16 |
| `measured_fill` | 2.2673（佔 11.09%） | p30 |
| `p36_checks` | 216 | p36 |
| `claim1_months_clearing` | 17 | p54 |
| `claim1_shift_matches` / `_alternatives` | 3／78 | p58 |
| `claim1_p_shift` | 0.05063 | p58 |
| `claim1_p_term` | 4.97e-06 | p54，**保留追溯，已不上圖** |

⚠️ **唯一一個「聚合而非讀取」的數字是整體遮蔽率**：`results_p16.json` 存的是逐帶的
遮蔽率與格數，沒有存總數，所以圖上那個是它們的格數加權平均。那是已存值的展示用聚合，
不是新的估計，docstring 裡明講。

## 4. 兩個版式約束

- **直式 7.0 吋寬**，因為 AJE 的直式圖上限就是 7 吋（2026-08-23 查證）。
- **韓文一律羅馬化。** Hangul 在 matplotlib 內建的每一種字型下都是豆腐塊，
  而一張需要本機安裝 CJK 字型才畫得出來的投稿圖，在排版者的機器上就是壞的。

## 5. 2026-08-27：兩處修正

### 5.1 遮蔽是三分，不是二分，而且圖上那句話原本是用位置索引寫的

原本的分類只有兩類：`hit`（率 ≥ 1%）與 `zero`（率 == 0.0），然後圖上那句寫成

```python
f"... while {zero[-4]} and above lose exactly none"
```

`zero = ['0-9','10-14','15-19','65-69','70-74','75-79','80+']`，所以 `zero[-4]` 是
`'65-69'`，圖上就變成「65-69 以上完全沒掉」——**0-19 被整段吃掉了**。而且
`zero[-4]` 是**對帶清單的位置索引**：任何一帶的遮蔽率一動，這句就靜靜地指到別的帶，
不會報錯。CLAUDE.md 明文禁止的就是這一類寫法。

真相是三分，`p48_fig1.py` 自己的註解其實早就知道（「45-64 a trace of order 1e-4」），
但這件事從來沒進到任何一句話裡：

| 類別 | 帶 | 率 |
|---|---|---|
| material | 20-24 … 40-44 | 0.1091 … 0.6952 |
| trace | 45-49 … 60-64 | 7.87e-05 … 1.28e-03 |
| none | 0-9 … 15-19、65-69 … 80+ | 0.0 |

改法：在畫圖時從 `results_p16.json` 建三個集合，標籤**用結構推導、不用位置**——
`none_low` 是結束在 material 起點之前的那一段，`none_high` 是開始在 trace 終點之後的
那一段。四個 assert 守著：material 連續、trace 連續、三集合恰好切分 16 個帶、
none 恰好是兩段。圖上那句現在是：

> cells below 3 are masked, 26% of cells overall, and the loss is not spread evenly:
> 20-24 to 40-44 lose 11%-70% of their cells, 45-64 lose under 0.13%, and 0-19 and
> 65+ lose exactly none

⚠️ **順手抓到的一個數字錯誤**：trace 段的最大值是 **50-54 的 1.2825e-03**，不是
45-49 的 6.9486e-04——這一段**對年齡不是單調的**，所以「第一個 trace 帶」不等於
「最大的 trace 帶」。因此上界是 **0.13%，不是 0.1%**；0.1% 會是一句假話。
圖上的 `{trace_cap:g}%` 是對實測最大值取上界（`ceil(max*1e4)/1e2`）算出來的，
不是打上去的，所以率一動它自己會跟著動而且永遠是真上界。

### 5.2 宣稱 1 的方框不再引用 p 值

原本方框寫 `(p = 4.97e-06)`，來自 `p54` 的年內超幾何 null。`results_p58.json` 這一輪
把那個 null 換掉了：年內 null 假設同一年之內月標籤可交換，但學期是**連續的區塊**，
所以那個 p 是膨脹的。`p58` 改成把整個日曆環狀平移（circular shift）：

- `p_shift = 0.0506`，78 個替代對齊裡有 **3 個**也能把 17 個清過地板的月份全放進學期；
- 這個檢定的**最小可能 p 是 1/79 = 0.0127**，0.05 離自己的地板只有四倍，撐不住一個宣稱。

所以方框改成只講**還站得住的那半件事**：17 之 79 的計數照舊（那一半沒被動到），
p 值換成對齊計數，從 `results_p58.json` 的 `p_shift.n_shifts_at_or_above` 與
`p_shift.denominator` 讀。並加一條 assert `p58["anchors"]["A2"]["reproduces_p54"]`，
讓這張圖不可能和它引用的計數脫節。方框現在讀作：

> the co-arrival matrix tracks the Korean school year in all 79 months, and the year
> schools stayed shut is flat. The calendar is never supplied to it. Of the 17 months
> that reach the survey's floor, every one is a term month; 3 of 78 alternative
> calendar alignments match.

`claim1_p_term` 仍留在 `results_p48.json` 裡供追溯，只是不再畫到圖上。

## 6. 還缺的

✅ **2026-08-24：圖說已寫**，在 `paper/figure_captions_draft.md`。照 `p47` 的做法，
每一個數字都是從腳本印出來的數字清單抄的，不是從 PNG 上看著抄的。

⚠️ **圖說待改（不在本輪的檔案範圍內）**：`paper/figure_captions_draft.md` 第 23–27 行
那句「0-19 and 65+ lose exactly none」本來就是對的（錯的是圖，不是圖說），但它同樣
漏掉 trace 那一段，而且宣稱 1 的 p 值也要跟著換。替換文字見本輪的交付說明。

⚠️ **`p48_fig1.py` 不在任何 determinism 清單裡**，卻會寫 `results_p48.json`——正是
`p55`／`p56`／`p47` 當初被加進清單要防的那個失效模式。本輪已確認它連跑兩次
sha256 相同（`cc29d69d…`），加進 `eda/determinism_check.sh` 由另一輪處理。
