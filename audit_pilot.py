"""Pilot audit of public Bangla dialect datasets (Mendeley Data).

Loads BanglaDial, Vashantor, ONUBAD and the 'Local Language' (Bangla Dialect) dataset,
then reports: sizes, duplicates, label conflicts, cross-source overlap, and a
random-split vs cross-source classification gap.
"""
import glob
import os
import re
import sys
import unicodedata

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm(s):
    # drop punctuation/symbols by Unicode category; a plain \w regex would also delete Bangla vowel signs
    if not s:
        return None
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in str(s))
    return re.sub(r"\s+", " ", s).strip()


# canonical dialect names
CANON = {"Barishal": "Barisal", "Barisal": "Barisal", "Chittagong": "Chittagong", "Sylhet": "Sylhet",
         "Noakhali": "Noakhali", "Mymensingh": "Mymensingh", "Rajshahi": "Rajshahi", "Rangpur": "Rangpur",
         "বরিশাল": "Barisal", "চিটাগং": "Chittagong", "সিলেট": "Sylhet", "ময়মনসিংহ": "Mymensingh",
         "রাজশাহী": "Rajshahi", "রংপুর": "Rangpur"}

rows = []  # source, dialect, text, standard

# --- BanglaDial
b = pd.read_csv(os.path.join(D, "banglaDial", "BanglaDial.csv"))
for t, l in zip(b["sentence"], b["label"]):
    rows.append(("BanglaDial", CANON.get(l, l), t, None))

# --- Vashantor
for f in glob.glob(os.path.join(D, "vashantor", "Vashantor_CSV_Format", "*", "*.csv")):
    v = pd.read_csv(f)
    v.columns = [c.strip() for c in v.columns]
    dcol = [c for c in v.columns if c.endswith("_bangla_speech")][0]
    region = CANON.get(v["region_name"].iloc[0].strip(), v["region_name"].iloc[0].strip())
    for d, s in zip(v[dcol], v["bangla_speech"]):
        rows.append(("Vashantor", region, d, s))

# --- ONUBAD
o = pd.read_excel(os.path.join(D, "onubad", "Sentence.xlsx"))
std = o["Standard Bangla Lanuguage"]
for col, name in [("Chittagong Language", "Chittagong"), ("Sylhet Language", "Sylhet"), ("Barisal Language", "Barisal")]:
    for d, s in zip(o[col], std):
        rows.append(("ONUBAD", name, d, s))

# --- Local Language (Bangla Dialect Dataset)
l = pd.read_excel(os.path.join(D, "bdialect_local", "Local_Language_Dataset.xlsx"))
for col in l.columns:
    if col in CANON:
        for d, s in zip(l[col], l["শুদ্ধ বাংলা ভাষা"]):
            rows.append(("LocalLang", CANON[col], d, s))

df = pd.DataFrame(rows, columns=["source", "dialect", "text", "standard"])
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)
df["n"] = df["text"].map(norm)
df = df[df["n"].str.len() > 0].reset_index(drop=True)
df["ns"] = df["standard"].map(lambda x: norm(x) if isinstance(x, str) else None)

print("=== 1. SIZES (source x dialect) ===")
print(pd.crosstab(df["dialect"], df["source"], margins=True).to_string())

print("\n=== 2. WITHIN-SOURCE DUPLICATES / LABEL CONFLICTS ===")
for src, g in df.groupby("source"):
    dup = g.duplicated(["n", "dialect"], keep=False).sum()
    conflict_texts = g.groupby("n")["dialect"].nunique()
    conflict_texts = conflict_texts[conflict_texts > 1]
    conflict_rows = g[g["n"].isin(conflict_texts.index)].shape[0]
    print(f"{src:11s} rows={len(g):6d} exact-dup rows(same label)={dup:5d} "
          f"texts with >1 label={len(conflict_texts):5d} (rows={conflict_rows})")

print("\n=== 3. CROSS-SOURCE OVERLAP (dialect text, normalised) ===")
sets = {s: set(g["n"]) for s, g in df.groupby("source")}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, c = names[i], names[j]
        inter = len(sets[a] & sets[c])
        print(f"{a:10s} & {c:10s}: {inter:5d} shared texts "
              f"({inter / len(sets[a]):.1%} of {a}, {inter / len(sets[c]):.1%} of {c})")

print("\n=== 4. SHARED STANDARD-BANGLA SENTENCES ACROSS SOURCES ===")
ss = {s: set(g["ns"].dropna()) for s, g in df.groupby("source") if g["ns"].notna().any()}
nn = list(ss)
for i in range(len(nn)):
    for j in range(i + 1, len(nn)):
        inter = len(ss[nn[i]] & ss[nn[j]])
        print(f"{nn[i]:10s} & {nn[j]:10s}: {inter:5d} shared standard sentences "
              f"(of {len(ss[nn[i]])} / {len(ss[nn[j]])})")

print("\n=== 5. MEANING-LEAKAGE INSIDE VASHANTOR / ONUBAD / LocalLang ===")
print("(a standard sentence appearing under several dialects; a random row split would put")
print(" the same meaning in train and test)")
for src in ["Vashantor", "ONUBAD", "LocalLang"]:
    g = df[df["source"] == src]
    k = g.groupby("ns")["dialect"].nunique()
    print(f"{src:10s} unique standard sentences={len(k):5d}, appearing under >=2 dialects={(k >= 2).sum():5d}")

print("\n=== 6. CLASSIFICATION: random split vs cross-source ===")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
except ImportError:
    print("scikit-learn not installed: pip install scikit-learn")
    sys.exit(0)

SHARED = ["Barisal", "Chittagong", "Noakhali", "Sylhet"]


def model():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=300))


def fit_eval(tr, te, label, dedup=False):
    if dedup:  # drop texts present on both sides so the test is truly unseen
        both = set(tr["n"]) & set(te["n"])
        tr, te = tr[~tr["n"].isin(both)], te[~te["n"].isin(both)]
        label += f" [overlap removed: {len(both)}]"
    m = model().fit(tr["n"], tr["dialect"])
    p = m.predict(te["n"])
    print(f"{label:55s} acc={accuracy_score(te['dialect'], p):.3f} macroF1={f1_score(te['dialect'], p, average='macro'):.3f}"
          f"  (train={len(tr)}, test={len(te)})")


bd = df[(df.source == "BanglaDial") & df.dialect.isin(SHARED)].drop_duplicates("n")
tr, te = train_test_split(bd, test_size=0.2, random_state=0, stratify=bd["dialect"])
fit_eval(tr, te, "BanglaDial -> BanglaDial (random split, 4 dialects)")
for tgt in ["Vashantor", "ONUBAD"]:
    t = df[(df.source == tgt) & df.dialect.isin(SHARED)]
    fit_eval(bd, t, f"BanglaDial -> {tgt} (cross-source)", dedup=True)

va = df[(df.source == "Vashantor") & df.dialect.isin(SHARED)].drop_duplicates("n")
tr, te = train_test_split(va, test_size=0.2, random_state=0, stratify=va["dialect"])
fit_eval(tr, te, "Vashantor -> Vashantor (random row split)")
std_ids = va["ns"].astype("category").cat.codes
grp = va["ns"].drop_duplicates().sample(frac=0.8, random_state=0)
fit_eval(va[va["ns"].isin(grp)], va[~va["ns"].isin(grp)], "Vashantor -> Vashantor (meaning-grouped split)")
t = df[(df.source == "BanglaDial") & df.dialect.isin(SHARED)]
fit_eval(va, t, "Vashantor -> BanglaDial (cross-source)", dedup=True)

print("\n=== 7. SHORTCUT CHECK on BanglaDial (12 classes) ===")
b12 = df[df.source == "BanglaDial"].drop_duplicates("n")
tr, te = train_test_split(b12, test_size=0.2, random_state=0, stratify=b12["dialect"])
m = model().fit(tr["n"], tr["dialect"])
print(f"full text char n-gram            acc={accuracy_score(te['dialect'], m.predict(te['n'])):.3f}")
lens = lambda d: pd.DataFrame({"len": d["n"].str.len(), "words": d["n"].str.split().str.len()})
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(lens(tr), tr["dialect"])
print(f"length-only features (no content) acc={accuracy_score(te['dialect'], rf.predict(lens(te))):.3f}"
      f"  (majority-class baseline={te['dialect'].value_counts(normalize=True).max():.3f})")
