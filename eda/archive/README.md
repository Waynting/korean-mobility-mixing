# Frozen results, by report date

A report that has been sent is fixed for good, and so are the numbers in it. If
`eda/p31_report_audit.py` checked those numbers against the live `results_*.json`, the gate
would start failing the first time an upstream script is re-run — a failure that says nothing
about the document it is meant to protect.

So each sent document is gated against a frozen copy of exactly the files it was written from.

## `20260818/`

Snapshot taken 2026-08-19, before the p32/p33/p34 round began. Gates
`Email_Discussion/advisor_report_20260818.md`, which passed 100 of 100 checks against these
files at the moment they were frozen.

Contents: `results_{p9,p19,p21,p21b,p25,p26,p27,p28,p29,p30}.json`.

SHA-256:

```
318f1328745a21edf075f45f5e7817c368fd2fb4ec437dafa67be372820bc38f  results_p19.json
aa31cef49dd75457b9198ce473c2ba636fa462ef1e2dbcbc6f18b03e8f435578  results_p21.json
0e37355675e503a11c00dbc9e41f09f051eba2b2348749351a46c8d3fda24ec6  results_p21b.json
0402312a98794ed27ba374c98bbb7da5a641ce0fbd6ec32c8dca8c31038db4c0  results_p25.json
1fb276d78a332f98838cd33c0fde9c1cead722bfd231ed6a59ed4d5f80a27a10  results_p26.json
45edd3810b84f0e682881214a423c0690129d43e5f81cde253cbc17c9e68eb75  results_p27.json
cd3bb3666d05cbadea8f2e209434b231fc927176cd887590961a28850fd08f7b  results_p28.json
f8d02d9ddecc429effdfbaa76a0f14952c9d65c73172bd9b96f873b27e3c2cfe  results_p29.json
3d6b4a0adfd6fa9ff4790abb336043cfeb89b2cdf711489fb940f827ac6491bd  results_p30.json
5d7d057170a8014056b8422babaaa3dad4c21befb3c7a5f44b7c88613903fdb6  results_p9.json
```

## Rule for this round

`results_p26.json` and `results_p27.json` are **read-only**. `p32`–`p36` write their own files
and never mutate an existing one. When the next report goes out, freeze a new dated directory
and give it its own `(document, source)` pair in `p31`.

## `20260822/`

Snapshot taken 2026-08-24, before the 08-24 round (國家人口向量修正、洞級取後不放回、R₀ 曲線)
touches anything upstream. Gates **兩份**已寄出的文件：`Email_Discussion/advisor_report_20260819.md`
與 `Email_Discussion/advisor_report_20260821.md`（連同與它同時寄出的附件
`Email_Discussion/Technical notes 20260822.md`）。

在此之前這兩封信都是對著 `LIVE` 檢查的——`eda/README.md` 開頭那條規矩說「已寄出的文件對著凍結
快照檢查」，但實際上只有 08-18 那一輪照做。所以這一次凍結不是整理，是**補上一個一直沒建立的
前提**：只要 `p40`/`p42`/`p44`/`p45` 一重跑，那兩封信就會變紅，而那個紅什麼都不代表。

凍結的時機是可以驗證的：凍結當下 `git status` 顯示 `p31` 讀到的每一個 `results_*.json` 都與
HEAD 逐位元相同（唯一有改動的 `results_p20d.json` 不被 `p31` 讀取），所以磁碟上的檔案**就是**
兩封信寄出時的狀態。

### 兩層，因為 34 MB 不值得每一輪複製一次

**第一層——實體凍結（11 個檔，624 KB）。** 這一輪可能被重跑或被取代的、以及小到複製沒有代價的：

```
ef2c27eacb277a88f243eb85504f909012a8ee2e8ab573e4ef194b455ca4c7d6  results_inventory.json
dac1d1ea8e03a908a36ed314b16f58a7e4f26e6fdbe8973aba38d78149d1fb00  results_p32.json
11b98f96fc337ece514186a3b34f44a4c74802c30c31e2e982830f0b7e9bbbdc  results_p33.json
c6cd9326483df9a83f07ddd0c2a8963e59d5931fffdbe9951664fa1f9d716ae9  results_p35.json
8b8771ce445a9dff5de30d0d6ad83ad5ac97c22a44c61aca7fd42bc8e0033844  results_p36.json
78287de3c044cf6e63fcb76c8b464a424fc36d6223b70acf36f579d15d1b4ddd  results_p39.json
b37bd90096de512f2c99f52db4b3763115bc2650fe0f5655e054f937f42f5e99  results_p40.json
904d1bc3921955d077eb432a5ccce0d0f821453b54d40f6e76525c41e1659ed8  results_p41.json
887fa500b1fbf12b2201bea1960898293b600865371cd1db3139bdbeaadcbbb7  results_p42.json
522be4d36631be1c27c70a16e1ceda8d41c6f7b28ee3b2d44337204b6d6b441f  results_p44.json
d4e6e015089c757468af74f3cbaaeb02181c925ccc6c327fbb10d822cdd752f6  results_p45.json
```

**第二層——雜湊釘住（6 個檔，33.9 MB，不複製）。** `p34`/`p37`/`p38` 三組面板這一輪不重跑
（國家向量的修正走新階段，不動這三支），複製它們只是每一輪多付 34 MB。改成把 sha256 記在這裡，
由 `p31` 對**線上**檔案驗一次：檔案沒動就照常讀，動了就直接報紅並指到這一節。這比靜靜讀一個
已經改掉的檔案強，也是這一層唯一要保證的事。

```
d5cb1d4bf7fff96dbbb2b20f53bbd056194e96d49b9aa069229e6d8eb5870ad1  results_p34.json
ac43d2e02cc137bfffb103ad9e555400ff9f268f39b21fac1ab00b7556440c42  results_p34_W.json
ab3c1e16610fad0da56c0658b839ab13184110570c7c0846615d4035c907485d  results_p34_E.json
8b51858178530cfb1e0d6f627d63b00572bcd0274be509c4be1c57863df5cabb  results_p37.json
b3c27c984ffab5d6e6ae017f3fd7a4068c6140e56f2bbdef6d8473faa9e79dda  results_p38.json
77b6deaa98f831e3732d9b92109f6913c86395c4fe2557ef6c8fbe7e58fc2f07  results_p38_W.json
2c6b4f3e3b3bd4274156debbe18c08d8c0c37e52cd8f6f65716e5c833dec3ba8  results_p38_E.json
```

⚠️ 第二層的檔案**一旦要重跑，先把它複製進本目錄再跑**，否則就失去了凍結的意義。

### 這一輪的規矩

`p49`（洞級取後不放回）、`p50`（全國人口向量）、`p51`（修正後的全國臂）各自寫自己的結果檔，
一個既有的 `results_*.json` 都不動。08-24 那封回信寄出時，再開 `20260824/` 並給它自己的
`(document, source)` 對。

## `20260824/`

快照取於 2026-08-27，在 08-27 那一輪（`p58`–`p62`）動到任何上游之前。凍結的是
**已寄出**的 `Email_Discussion/advisor_report_20260824.md`，連同與它同時寄出的附件
`Email_Discussion/figure_captions_20260824.md`（五張圖的圖說，圖說帶數字，所以它是這一輪
語料的另一半，不是附註）。

### 這封信寄出去了，而且證據不是「commit 訊息說它寄了」

commit `175c28d`（2026-08-24 22:09，「Both outbound letters are sent…」）記的是**那兩封
對外的信**——開放資料的 문의하기 與給 NIMS 的詢問信；`advisor_report_20260824.md` 在那個
commit 裡只是被改寫成「記錄這兩封已寄出」。所以嚴格講，那個 commit 不是這封回信的寄出點。

**真正的證據是回信本身。** `Email_Discussion/advisor_suggestion_20260827.md`（2026-08-27）
引的數字只存在於**現在這個版本**的信裡：5,786 人（세종，`cebb5db` 才抓到）、k = 1.5 時
β = 0.8578（`d583ff0` 才有的敏感度臂）、以及圖 7(b) 那個「四個落在低 R₀ 區間、一個散在外面」
的五翻判定（`cebb5db` 重跑後才是 5）。所以宣緯老師讀到的就是 HEAD 這一版。

這件事決定了**凍結點取在哪裡**，值得寫清楚：

| commit | 時間 | 對這封信做了什麼 |
|---|---|---|
| `175c28d` | 08-24 22:09 | 兩封對外的信寄出；回信裡記上這件事 |
| `d583ff0` | 08-24 22:26 | 撤掉 p44 §44.7 的 Q9 那一臂、新增 `p57`；回信加第十一段 |
| `cebb5db` | 08-25 00:18 | 세종與 부천 兩個缺陷、`p50` 重建、`p51`/`p52`/`p53`/`p54` 重跑；回信整段改寫（23.5%→23.6%、40.2 倍、五翻） |

後面兩個 commit 都在 `175c28d` **之後**，而且都改了這封信與它引的結果檔。如果 08-24 當天就
凍結，凍下來的會是一份**沒有寄出去過**的版本——那正是 `bc08f96` 撤掉的那個錯誤，只是方向相反。
所以凍結點取在 HEAD，不是取在 `175c28d`。

### 凍結的時機是可以驗證的

凍結當下 `git status` 顯示 `p31` 在 08-24 那一段讀到的八個 `results_*.json` **每一個都與
HEAD 逐位元相同**（`git diff --quiet HEAD` 對八個檔各驗一次），所以磁碟上的檔案**就是**這封
信寄出時的狀態。凍結前後閘門都是 780/780，輸出逐字相同。

### 一層，因為八個檔加起來只有 85 KB

`20260822/` 分兩層是因為 `p34`/`p37`/`p38` 三組面板有 33.9 MB，複製不划算。這一輪沒有這個
問題，八個檔全部實體複製，不留雜湊釘住那一層。

```
5e240f1c14413f2500ca17dcfb37a251cf08951086e7e156c8a939f98a47f596  results_p44.json
9bbb95db90ccb07d4187df03ce2bb51bb24490d971e8b2e4e5a9da84d35c4f77  results_p49.json
81230a6f59f62408d3defd9415949f580a537d1de83db3f4a0d95561bfb46d95  results_p50.json
0a1081429b4f0cdcf887f733bb4b843ce048556a54b43e2cb2509f2d14bb866e  results_p51.json
34a45e4c41902e75e2774c462906b6c87881f30f38eb59ed4b3bf7e271dec1a0  results_p52.json
235233738b2641fd1e1fc5dc7b90a13eebb3d0e25911224f0c9e3ab9597a6673  results_p53.json
51603bb120f33650460f352df88ee2be34738d0e6454deb5e4a4061a56216020  results_p54.json
8e127fdbc98d15ba1292f21d3922585203b56cb125809551e233d08a2578c762  results_p57.json
```

每一個檔在信裡撐住哪一段：

| 檔 | 撐住信裡的哪一段 |
|---|---|
| `results_p49.json` | 第五段，洞級取後放回對不放回：−4.2854%、階梯（洞／區／市／場館）、場館比洞大 10.0 倍、有效 n = 1,104、慣例統一後 11.76 → 12.28 |
| `results_p50.json` | 第一段，全國人口向量：兩層錨、세종 5,786 人、全國總人口 52,673,955、首爾 25–29 是全國的 1.263 倍 |
| `results_p51.json` | 第二、四段，修正後的全國臂：1.17 倍、IPF 吃掉 23.6%／27.2%、四統計量跨度、人口組成單獨吃掉的 7.9%／5.6% |
| `results_p52.json` | 第七段與圖 7 圖說，R₀ 完整曲線：2.5 那一點 1.27%／11.2%、峰值 50.9%／62.6%、40.2 倍、argmax 全落在 1.2–1.5 |
| `results_p53.json` | 第三段，β 的全國臂：0.9347／0.9217、跨 scope 差 0.0130 |
| `results_p54.json` | 第六段，學期月重數：對新地板 17 個月、p = 4.97e−06、對舊地板 13 個月與 2.24e−04、兩個調查月都不過線 |
| `results_p57.json` | 第十一段，[CHAE] 留著的 etc. 旗標：10.53% 對 4.48%、旗標本身解釋 6.06%、2,016 與 3,210 兩類 100% 誤分 |
| `results_p44.json` | 第十一段的另一半，撤掉的 Q9 那一臂：r = 0.18539、原本的 [0.9096, …]、24.0%／40.3%、k = 1.5 時 β = 0.8578 |

⚠️ **`results_p44.json` 在 `20260822/` 與 `20260824/` 各有一份，而且兩份不一樣**
（`522be4d3…` 對 `5e240f1c…`）。08-21 那封信引的是 `d583ff0` 之前的 p44，08-24 這封信引的
是它撤掉 §44.7 之後重跑的版本。一個活的檔案沒辦法同時對兩封已寄出的信負責，所以兩封都不讀
活的檔案——這正是「每一份已寄出的文件有自己的 `(document, source)` 對」這條規矩要處理的情況。

### 這一輪的規矩

`p58`–`p62`（回應 08-27 那封回信的五個新階段）各自寫自己的結果檔，一個既有的
`results_*.json` 都不動；`20260822/` 與 `20260824/` 底下的檔案是唯讀的。`p31` 裡
`REPORT_0827`／`ANNEX_0827` 那一組已經先開好，來源是 `LIVE`——**因為那封信還沒寫**，
數字還會動，信要跟著動。等 08-27 那封回信寄出時，再開 `20260827/`、把該輪的結果檔複製進去、
sha256 寫進本檔，並把 `p31` 裡那一組的 `load(..., LIVE)` 改成 `load(..., ARCHIVE_0827)`。

規矩在兩個方向上都要成立，這一點沒有變：

| 文件 | 來源 | 理由 |
|---|---|---|
| 已寄出 | 凍結快照 | 它的數字不會再動，上游重跑不該讓它變紅 |
| **還沒寄出** | **`LIVE`** | 它的數字**還會動**，而信要跟著動；對著快照檢查會讓一封還在寫的信看起來永遠是對的 |
