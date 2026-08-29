# Pre-flight — 讀檔本身的三個陷阱

*狀態:已繞過 · 對應決策:D1*

在做任何分析之前必須先繞過的東西。三個都是**靜默失敗** —— 不報錯,只給錯的答案。

## 1. DuckDB 1.5.5 的 CSV reader 會靜默丟列

同一個檔(`생활이동_행정동_2020.01_08시.csv`):

| reader | 讀到的列數 |
|---|---|
| `wc -l` − 1 | 5,634,192 |
| pandas(C engine) | 5,634,192 |
| **DuckDB 1.5.5** | **4,929,770** |

少 12.5%,而且:

- `ignore_errors=false` 不報錯;
- `PRAGMA threads=1` 結果一樣(不是平行切塊的問題);
- 先用 `iconv` 轉成 UTF-8 再讀,結果仍然一樣(不是編碼的問題);
- 換一組 `read_csv` 選項會得到**另一個**錯誤的列數(4,944,891),也就是說錯法本身不穩定。

**處置:** ETL 改成 pandas 讀 CSV、DuckDB 只碰 parquet,並逐檔比對 `wc -l`(`eda/etl.py` 裡列數不符就 abort)。

> 任何人接手都要先確認自己的 reader 沒有這個行為。這種錯誤不會出現在任何檢查裡 —— 總量只是「小一點」。

## 2. 標頭是 CRLF、資料是 LF

全檔只有 **1 個 `\r`**,就在第 1 行行尾。嚴格模式的 CSV parser 會從標頭嗅出 CRLF、然後在第 2 行崩潰(`The CSV Parser state machine reached an invalid state`)。

**處置:** `new_line` 明確釘成 `\n`,標頭用 `skip=1` 跳過。

## 3. macOS 把韓文目錄名存成 NFD

`생활이동_행정동_202001` 在檔案系統裡是分解形式(NFD),程式碼字面量是組合形式(NFC)。後果:

```python
glob.glob(ROOT + "/생활이동_행정동_2020*/*.csv")   # → []
os.path.basename(f).replace("시.csv", "")          # → 不替換
```

第一次跑的結果是「成功處理 0 個檔」,exit code 0。

**處置:** 路徑處理一律走 ASCII 正則:`glob(ROOT + "/*/*.csv")` + `re.search(r"(\d{4})\.(\d{2})_(\d{2})", basename)`。

## 通過的檢查

- 48 個檔的標頭 byte 完全相同(md5 一致)。
- 欄數固定 10,無引號字元,無跳脫。
- 轉檔後 241,097,168 列,逐檔對過 `wc -l`,零列遺失。
- 10.5 GB CSV → 1.0 GB parquet(zstd)。
