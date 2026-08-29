"""Where the code lives versus where the 9 GB lives.

Code, results, figures and the small case files are in the repository. The
parquet is not: it sits on whichever drive happens to be mounted. Point
KOREAN_DATA_ROOT at that drive and every script follows —

    export KOREAN_DATA_ROOT=/Volumes/SeoulData

or, so it survives new shells, write it once into a `.env` at the repo root
(copy `.env.example`). The real environment always beats the `.env` file, so a
one-off `KOREAN_DATA_ROOT=... python eda/p17_descriptive.py` still overrides.

Unset, it falls back to the repo root — where the data sat while the 2020
pipeline was built — so nothing that used to work stops working.

The three trees can also be pointed at different places, which is what the
2026 expansion needs: the read-only source parquet on the external drive, the
derived files written back to the faster internal disk.

    KOREAN_PARQUET_DIR   the ym=YYYYMM/hHH.parquet tree (read)
    KOREAN_DERIVED_DIR   panel_core, gu_level, and later aggregates (written)
    KOREAN_RAW_DIR       the cp949 CSV directories the ETL reads

Run `python eda/paths.py` to print what resolves where and what is missing.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_dotenv(path=REPO / ".env"):
    """KEY=VALUE lines, `#` comments. Never overrides an exported variable."""
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


_load_dotenv()


def _dir(env, default):
    val = os.environ.get(env)
    return Path(os.path.expanduser(val)).resolve() if val else Path(default)


DATA_ROOT = _dir("KOREAN_DATA_ROOT", REPO)
PARQUET = _dir("KOREAN_PARQUET_DIR", DATA_ROOT / "parquet")
DERIVED = _dir("KOREAN_DERIVED_DIR", DATA_ROOT / "derived")
RAW = _dir("KOREAN_RAW_DIR", DATA_ROOT)

CASES = REPO / "cases"          # small enough to be committed
FIG = REPO / "eda" / "fig"
RESULTS = REPO / "eda"          # results_pNN.json, etl_report.json

PARQUET_GLOB = f"{PARQUET}/**/*.parquet"

# Str for the f-string SQL that fills most of eda/; Path for everything else.
ROOT = str(REPO)


def require(path, what=""):
    """Fail loudly. An unmounted drive must not read as an empty dataset."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"missing {what or p.name}: {p}\n"
            f"  KOREAN_DATA_ROOT = "
            f"{os.environ.get('KOREAN_DATA_ROOT', '(unset — falling back to the repo root)')}\n"
            f"  Is the drive mounted? Run `python eda/paths.py` to see what resolves where."
        )
    return p


if __name__ == "__main__":
    print(f"repo       {REPO}")
    for name, p in [("DATA_ROOT", DATA_ROOT), ("parquet", PARQUET),
                    ("derived", DERIVED), ("raw", RAW), ("cases", CASES)]:
        env = {"DATA_ROOT": "KOREAN_DATA_ROOT", "parquet": "KOREAN_PARQUET_DIR",
               "derived": "KOREAN_DERIVED_DIR", "raw": "KOREAN_RAW_DIR"}.get(name)
        src = f"${env}" if env and os.environ.get(env) else "default"
        print(f"{name:<10} {'ok ' if p.exists() else 'MISSING'} {p}  ({src})")
    n = len(list(PARQUET.glob("**/*.parquet"))) if PARQUET.exists() else 0
    print(f"\n{n} parquet files under {PARQUET}")
