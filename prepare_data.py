"""Collect every downloaded Bangla dialect dataset into one clean folder: data_clean/.

Raw files in data/ are never modified. Output CSVs are UTF-8 with BOM, so Excel shows Bangla correctly.
Common schema: id, source, split, dialect, dialect_text, standard_text, english_text,
               text_norm, dup_in_source, in_other_sources
"""
import glob
import os
import re
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "data_clean")
os.makedirs(OUT, exist_ok=True)

CANON0 = {"Barishal": "Barisal", "বরিশাল": "Barisal", "চিটাগং": "Chittagong", "সিলেট": "Sylhet",
         "ময়মনসিংহ": "Mymensingh", "রাজশাহী": "Rajshahi", "রংপুর": "Rangpur"}
CANON = {unicodedata.normalize("NFC", k): v for k, v in CANON0.items()}
PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean(x):
    if not isinstance(x, str):
        return None
    x = unicodedata.normalize("NFC", x).replace("​", "").replace("﻿", "")
    x = re.sub(r"\s+", " ", x).strip()
    return x or None


def norm(x):
    # drop punctuation/symbols by Unicode category; a plain \w regex would also delete Bangla vowel signs
    if not x:
        return None
    x = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in str(x))
    return re.sub(r"\s+", " ", x).strip()


def frame(source, rows):
    df = pd.DataFrame(rows, columns=["split", "dialect", "dialect_text", "standard_text", "english_text"])
    df.insert(0, "source", source)
    for c in ["dialect", "dialect_text", "standard_text", "english_text"]:
        df[c] = df[c].map(clean)
    df = df.dropna(subset=["dialect_text", "dialect"]).reset_index(drop=True)
    return df


parts = {}

# BanglaDial (no split, no parallel text)
b = pd.read_csv(os.path.join(RAW, "banglaDial", "BanglaDial.csv"))
parts["banglaDial"] = frame("BanglaDial", [(None, CANON.get(l, l), s, None, None) for s, l in zip(b["sentence"], b["label"])])

# Vashantor (has train/validation/test folders; also Banglish + English columns)
rows = []
for f in sorted(glob.glob(os.path.join(RAW, "vashantor", "Vashantor_CSV_Format", "*", "*.csv"))):
    split = os.path.basename(os.path.dirname(f)).lower()
    v = pd.read_csv(f)
    v.columns = [c.strip() for c in v.columns]
    dcol = [c for c in v.columns if c.endswith("_bangla_speech")][0]
    region = CANON.get(v["region_name"].iloc[0].strip(), v["region_name"].iloc[0].strip())
    rows += [(split, region, d, s, e) for d, s, e in zip(v[dcol], v["bangla_speech"], v["english_speech"])]
parts["vashantor"] = frame("Vashantor", rows)

# ONUBAD (sentence level: 3 dialects -> standard)
o = pd.read_excel(os.path.join(RAW, "onubad", "Sentence.xlsx"))
rows = []
for col, name in [("Chittagong Language", "Chittagong"), ("Sylhet Language", "Sylhet"), ("Barisal Language", "Barisal")]:
    rows += [(None, name, d, s, e) for d, s, e in zip(o[col], o["Standard Bangla Lanuguage"], o["English Translation"])]
parts["onubad"] = frame("ONUBAD", rows)

# Bangla Dialect Dataset ("Local Language"): standard + 6 dialects (Rakhine column left out: not a Bangla dialect)
l = pd.read_excel(os.path.join(RAW, "bdialect_local", "Local_Language_Dataset.xlsx"))
rows = []
for col in l.columns:
    key = unicodedata.normalize("NFC", str(col))
    if key in CANON:
        rows += [(None, CANON[key], d, s, None) for d, s in zip(l[col], l["শুদ্ধ বাংলা ভাষা"])]
parts["locallang"] = frame("LocalLang", rows)

alldf = pd.concat(parts.values(), ignore_index=True)
alldf["text_norm"] = alldf["dialect_text"].map(norm)
alldf["dup_in_source"] = alldf.duplicated(["source", "dialect", "text_norm"], keep="first")
src_sets = {s: set(g["text_norm"]) for s, g in alldf.groupby("source")}
alldf["in_other_sources"] = [
    ",".join(s for s, st in src_sets.items() if s != src and t in st) for src, t in zip(alldf["source"], alldf["text_norm"])]
alldf.insert(0, "id", range(1, len(alldf) + 1))

for name, key in [("banglaDial", "BanglaDial"), ("vashantor", "Vashantor"), ("onubad", "ONUBAD"), ("locallang", "LocalLang")]:
    alldf[alldf["source"] == key].to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8-sig")
alldf.to_csv(os.path.join(OUT, "all_sources.csv"), index=False, encoding="utf-8-sig")

print(alldf.groupby("source").agg(rows=("id", "size"), dialects=("dialect", "nunique"),
                                 dup_rows=("dup_in_source", "sum"),
                                 in_other=("in_other_sources", lambda s: (s != "").sum())).to_string())
print("\ntotal rows:", len(alldf))
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
