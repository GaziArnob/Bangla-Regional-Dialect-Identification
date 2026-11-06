# Part 12. Robustness: 3 seeds x 3 transformers with up to 15 epochs, ensembles per seed, significance tests, and a
# cross-corpus test on two dialect corpora that are not among BanglaDial's sources. Designed for a Kaggle notebook with
# 2 x T4 (runs as "Save & Run All"); every result is printed as an @@TAG@@{json} line and saved to /kaggle/working.
import json, os, subprocess, sys, time, unicodedata, re, urllib.request, pickle
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gensim", "nltk", "git+https://github.com/csebuetnlp/normalizer"], check=True)
W = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
emit = lambda tag, obj: print(f"@@{tag}@@" + json.dumps(obj, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- Parts 1-4 of the notebook: same data, cleaning, split
NB = ("https://raw.githubusercontent.com/GaziArnob/Bangla-Regional-Dialect-Identification/"
      "ab8b8c50103b54979815fa7181b6019e693afefb/colab/BanglaDialect_Midterm.ipynb")
for c in json.load(urllib.request.urlopen(NB))["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("## Part 5"):
        break
    if c["cell_type"] == "code" and not s.lstrip().startswith("!"):
        exec(s, globals())
print("Parts 1-4 done", len(train_df), len(val_df), len(test_df), flush=True)

import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
LABELS = sorted(df["label"].unique()); L2I = {l: i for i, l in enumerate(LABELS)}

# ---------------------------------------------------------------- cross-corpus test sets (not BanglaDial sources)
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})).read()

ANCH = "https://data.mendeley.com/public-files/datasets/gbkszkt8z3/files/357a3e75-3a04-4644-9cf1-a71c6b592ce0/file_downloaded"
CHAT = "https://data.mendeley.com/public-files/datasets/wtms9xbkkw/files/0cc046b6-9f98-484e-bd8f-0bfbf9c7d8c1/file_downloaded"
open(f"{W}/ancholik.csv", "wb").write(fetch(ANCH)); open(f"{W}/chatgaiyya.csv", "wb").write(fetch(CHAT))
an = pd.read_csv(f"{W}/ancholik.csv")
ch = pd.read_csv(f"{W}/chatgaiyya.csv")
print("ANCHOLIK columns:", list(an.columns), "| ChatgaiyyaAlap columns:", list(ch.columns), flush=True)
amap = {"Standard_Bangla": "Standard_Bangla", "Chittagong": "Chittagong", "Sylhet": "Sylhet", "Noakhali": "Noakhali",
        "Mymensingh": "Mymensingh", "Barishal": "Barisal", "Barisal": "Barisal"}
an_long = pd.concat([pd.DataFrame({"sentence": an[c], "label": amap[c]}) for c in an.columns if c in amap])
low = {c: str(c).lower() for c in ch.columns}
sources = [("ANCHOLIK-NER", an_long)]
try:
    std_col = next(c for c in ch.columns if "standard" in low[c] or low[c] in ("bangla", "bengali", "standard_bangla", "বাংলা", "প্রমিত"))
    ctg_col = next(c for c in ch.columns if c != std_col and any(t in low[c] for t in ("chit", "ctg", "chatg", "dialect", "regional", "চট্টগ্রাম", "চাটগাঁ")))
    sources.append(("ChatgaiyyaAlap", pd.concat([pd.DataFrame({"sentence": ch[ctg_col], "label": "Chittagong"}),
                                                 pd.DataFrame({"sentence": ch[std_col], "label": "Standard_Bangla"})])))
except StopIteration:
    print("ChatgaiyyaAlap: could not identify the columns, skipped:", list(ch.columns), flush=True)
bd_keys = set(raw["key"]) | set(df["key"])          # every BanglaDial sentence
CROSS = {}
for name, d in sources:
    d = d.dropna().copy()
    d["sentence"] = d["sentence"].astype(str).map(lambda s: unicodedata.normalize("NFC", s).strip())
    d["key"] = d["sentence"].map(norm_key)
    n0 = len(d); d = d[d["key"] != ""]
    overlap = d["key"].isin(bd_keys)
    n_over = int(overlap.sum()); d = d[~overlap].drop_duplicates("key")
    CROSS[name] = d[["sentence", "label"]].reset_index(drop=True)
    emit("CROSSDATA", {"corpus": name, "rows": n0, "overlap_with_BanglaDial": n_over, "kept": len(d),
                       "per_class": d["label"].value_counts().to_dict()})

# ---------------------------------------------------------------- the selected char SVM (Part 5 setting), calibrated
PREP_KW = dict(mask_digits=True)
txt = lambda s: [preprocess(t, **PREP_KW) for t in s]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True, max_features=200000)
Xtr = vec.fit_transform(txt(train_df["sentence"]))
svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(Xtr, train_df["label"])
cal = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3, method="sigmoid").fit(Xtr, train_df["label"])
assert list(cal.classes_) == LABELS
SVM = {"val": cal.predict_proba(vec.transform(txt(val_df["sentence"]))), "test": cal.predict_proba(vec.transform(txt(test_df["sentence"]))),
       "test_hard": np.array([L2I[l] for l in svm.predict(vec.transform(txt(test_df["sentence"])))])}
for k, d in CROSS.items():
    SVM[k] = cal.predict_proba(vec.transform(txt(d["sentence"])))
    SVM[k + "_hard"] = np.array([L2I[l] for l in svm.predict(vec.transform(txt(d["sentence"])))])
print("SVM test macro-F1", round(f1_score([L2I[l] for l in test_df["label"]], SVM["test_hard"], average="macro"), 4), flush=True)

data = {"labels": LABELS, "train": (train_df["sentence"].tolist(), [L2I[l] for l in train_df["label"]]),
        "val": (val_df["sentence"].tolist(), [L2I[l] for l in val_df["label"]]),
        "test": (test_df["sentence"].tolist(), [L2I[l] for l in test_df["label"]]),
        "cross": {k: (d["sentence"].tolist(), [L2I[l] for l in d["label"]]) for k, d in CROSS.items()}}
pickle.dump(data, open(f"{W}/data.pkl", "wb"))

# ---------------------------------------------------------------- one fine-tuning run = one subprocess on one GPU
WORKER = r'''
import json, os, pickle, random, sys, time, gc
import numpy as np, torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from normalizer import normalize
name, hub, use_norm, seed, W = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4]), sys.argv[5]
D = pickle.load(open(f"{W}/data.pkl", "rb"))
MAX_LEN, BATCH, EPOCHS, PATIENCE, LR = 64, 32, 15, 3, 5e-5
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
tok = AutoTokenizer.from_pretrained(hub)
prep = (lambda s: [normalize(x) for x in s]) if use_norm else list
ytr = np.array(D["train"][1]); cnt = np.bincount(ytr, minlength=12)
Wt = torch.tensor(len(ytr) / (12 * cnt), dtype=torch.float, device="cuda")
def loader(texts, y, shuffle, bs):
    enc = tok(prep(texts), truncation=True, max_length=MAX_LEN); y = np.array(y); idx = list(range(len(y)))
    def it():
        order = random.sample(idx, len(idx)) if shuffle else idx
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            out = tok.pad({k: [enc[k][j] for j in b] for k in enc}, return_tensors="pt")
            yield {k: v.cuda() for k, v in out.items()}, torch.tensor(y[b], device="cuda")
    return it
@torch.no_grad()
def logits(model, it):
    model.eval(); out = []
    for b, _ in it():
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**b).logits.float().cpu())
    return torch.cat(out).numpy()
tr = loader(*D["train"], True, BATCH); va = loader(*D["val"], False, 128); te = loader(*D["test"], False, 128)
cr = {k: loader(*v, False, 128) for k, v in D["cross"].items()}
model = AutoModelForSequenceClassification.from_pretrained(hub, num_labels=12).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = (len(ytr) + BATCH - 1) // BATCH * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler("cuda"); lossf = torch.nn.CrossEntropyLoss(weight=Wt)
best, hist, bad, t0 = -1, [], 0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for b, y in tr():
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**b).logits.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    lv = logits(model, va); f = f1_score(D["val"][1], lv.argmax(1), average="macro"); hist.append(round(float(f), 4))
    print(f"{name} seed {seed} epoch {ep}: val macro-F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if f > best:
        best, bad = f, 0
        np.savez(f"{W}/logits_{name.replace(' ', '_')}_s{seed}.npz", val=lv, test=logits(model, te),
                 **{k: logits(model, v) for k, v in cr.items()})
        best_ep = ep
    else:
        bad += 1
        if bad >= PATIENCE:
            break
json.dump({"model": name, "seed": seed, "best_epoch": best_ep, "val_f1": best, "hist": hist, "secs": round(time.time() - t0)},
          open(f"{W}/run_{name.replace(' ', '_')}_s{seed}.json", "w"))
print(f"DONE {name} seed {seed} best epoch {best_ep} val {best:.4f}", flush=True)
'''
open(f"{W}/worker.py", "w").write(WORKER)
MODELS = {"BanglaBERT": ("csebuetnlp/banglabert", 1), "BanglishBERT": ("csebuetnlp/banglishbert", 1), "XLM-R base": ("xlm-roberta-base", 0)}
SEEDS = [42, 1, 2]
jobs = [("XLM-R base", s) for s in SEEDS] + [(m, s) for m in ("BanglaBERT", "BanglishBERT") for s in SEEDS]   # longest first
import torch
n_gpu = max(1, torch.cuda.device_count()); running = {}
t0 = time.time()
while jobs or running:
    for g in range(n_gpu):
        if g not in running and jobs:
            m, s = jobs.pop(0)
            log = open(f"{W}/log_{m.replace(' ', '_')}_s{s}.txt", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            running[g] = (m, s, subprocess.Popen([sys.executable, f"{W}/worker.py", m, MODELS[m][0], str(MODELS[m][1]), str(s), W],
                                                 stdout=log, stderr=subprocess.STDOUT, env=env))
            print(f"[{time.time() - t0:.0f}s] start {m} seed {s} on GPU {g}", flush=True)
    for g, (m, s, p) in list(running.items()):
        if p.poll() is not None:
            print(f"[{time.time() - t0:.0f}s] finished {m} seed {s} (exit {p.returncode})", flush=True)
            if p.returncode != 0:
                print(open(f"{W}/log_{m.replace(' ', '_')}_s{s}.txt").read()[-3000:], flush=True)
            del running[g]
    time.sleep(30)

# ---------------------------------------------------------------- aggregate: seeds, ensembles, significance, cross-corpus
from scipy.stats import binomtest
sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
yv, yt = np.array(data["val"][1]), np.array(data["test"][1])
mf1 = lambda y, p: f1_score(y, p, average="macro")
R = {}
for m in MODELS:
    for s in SEEDS:
        f = f"{W}/logits_{m.replace(' ', '_')}_s{s}.npz"
        if os.path.exists(f):
            R[(m, s)] = dict(np.load(f)); R[(m, s)]["run"] = json.load(open(f"{W}/run_{m.replace(' ', '_')}_s{s}.json"))
pred = {}
summary = {"seeds": SEEDS, "models": {}, "ensembles": {}}
for m in MODELS:
    rs = [R[(m, s)] for s in SEEDS if (m, s) in R]
    f1s = [mf1(yt, r["test"].argmax(1)) for r in rs]; accs = [accuracy_score(yt, r["test"].argmax(1)) for r in rs]
    summary["models"][m] = {"test_macroF1": [round(x, 4) for x in f1s], "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else None,
                            "acc_mean": round(float(np.mean(accs)), 4), "best_epochs": [r["run"]["best_epoch"] for r in rs],
                            "hist": [r["run"]["hist"] for r in rs], "secs": [r["run"]["secs"] for r in rs]}
    for s in SEEDS:
        if (m, s) in R:
            pred[(m, s)] = R[(m, s)]["test"].argmax(1)
svm_f1 = mf1(yt, SVM["test_hard"])
summary["models"]["char SVM"] = {"mean": round(svm_f1, 4), "acc_mean": round(accuracy_score(yt, SVM["test_hard"]), 4)}
grid = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
for cname, members in {"SVM + BanglaBERT": ["BanglaBERT"], "SVM + BanglishBERT": ["BanglishBERT"], "SVM + XLM-R": ["XLM-R base"],
                       "SVM + all three": list(MODELS), "all three transformers (no SVM)": list(MODELS)}.items():
    f1s, accs, ws = [], [], []
    for s in SEEDS:
        if not all((k, s) in R for k in members):
            continue
        pv = np.mean([sm(R[(k, s)]["val"]) for k in members], 0); pt = np.mean([sm(R[(k, s)]["test"]) for k in members], 0)
        if cname.endswith("(no SVM)"):
            w = 1.0
        else:
            w = max(grid, key=lambda w: mf1(yv, (w * pv + (1 - w) * SVM["val"]).argmax(1)))
        p = (w * pt + (1 - w) * SVM["test"]).argmax(1)
        f1s.append(mf1(yt, p)); accs.append(accuracy_score(yt, p)); ws.append(w); pred[(cname, s)] = p
    summary["ensembles"][cname] = {"test_macroF1": [round(x, 4) for x in f1s], "mean": round(float(np.mean(f1s)), 4),
                                   "std": round(float(np.std(f1s, ddof=1)), 4), "acc_mean": round(float(np.mean(accs)), 4), "w": ws}
emit("SEEDS", summary)

# McNemar (exact binomial on discordant pairs) per seed + paired bootstrap of the macro-F1 difference (seed 42, 1000 draws)
def mcnemar(a, b):
    ra, rb = a == yt, b == yt; n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    return n01, n10, binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
rng = np.random.default_rng(0); B = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def boot(a, b):
    d = [mf1(yt[i], a[i]) - mf1(yt[i], b[i]) for i in B]
    return round(float(np.mean(d)), 4), [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
pairs = [("SVM + all three", "BanglaBERT"), ("SVM + all three", "SVM"), ("SVM + BanglaBERT", "BanglaBERT"), ("BanglaBERT", "SVM"),
         ("BanglishBERT", "SVM"), ("XLM-R base", "SVM"), ("BanglaBERT", "BanglishBERT"), ("BanglaBERT", "XLM-R base"), ("BanglishBERT", "XLM-R base")]
sig = []
for a, b in pairs:
    row = {"A": a, "B": b, "mcnemar": {}}
    for s in SEEDS:
        pa = SVM["test_hard"] if a == "SVM" else pred.get((a, s)); pb = SVM["test_hard"] if b == "SVM" else pred.get((b, s))
        if pa is not None and pb is not None:
            n01, n10, p = mcnemar(pa, pb); row["mcnemar"][s] = {"A_only_right": n01, "B_only_right": n10, "p": float(p)}
    pa = SVM["test_hard"] if a == "SVM" else pred.get((a, 42)); pb = SVM["test_hard"] if b == "SVM" else pred.get((b, 42))
    if pa is not None and pb is not None:
        row["bootstrap_seed42"] = boot(pa, pb)
    sig.append(row)
emit("SIG", sig)

# cross-corpus: macro-F1 over the classes present in each corpus (predictions still over all 12 classes)
cross = {}
for k, (_, y) in data["cross"].items():
    y = np.array(y); present = sorted(set(y))
    row = {"n": len(y), "classes": [LABELS[i] for i in present],
           "char SVM": {"macroF1": round(f1_score(y, SVM[k + "_hard"], labels=present, average="macro"), 4), "acc": round(accuracy_score(y, SVM[k + "_hard"]), 4)}}
    for m in MODELS:
        f1s = [f1_score(y, R[(m, s)][k].argmax(1), labels=present, average="macro") for s in SEEDS if (m, s) in R]
        row[m] = {"macroF1_mean": round(float(np.mean(f1s)), 4), "macroF1_std": round(float(np.std(f1s, ddof=1)), 4)}
    f1s = []
    for s in SEEDS:
        if all((m, s) in R for m in MODELS):
            w = summary["ensembles"]["SVM + all three"]["w"][SEEDS.index(s)]
            p = (w * np.mean([sm(R[(m, s)][k]) for m in MODELS], 0) + (1 - w) * SVM[k]).argmax(1)
            f1s.append(f1_score(y, p, labels=present, average="macro"))
            if s == 42:
                row["ensemble_per_class_F1_seed42"] = dict(zip([LABELS[i] for i in present], f1_score(y, p, labels=present, average=None).round(3).tolist()))
                row["ensemble_pred_dist_seed42"] = {LABELS[i]: int(c) for i, c in zip(*np.unique(p, return_counts=True))}
    row["SVM + all three"] = {"macroF1_mean": round(float(np.mean(f1s)), 4), "macroF1_std": round(float(np.std(f1s, ddof=1)), 4)}
    row["svm_per_class_F1"] = dict(zip([LABELS[i] for i in present], f1_score(y, SVM[k + "_hard"], labels=present, average=None).round(3).tolist()))
    cross[k] = row
emit("CROSS", cross)
json.dump({"seeds": summary, "sig": sig, "cross": cross}, open(f"{W}/part12_results.json", "w"), ensure_ascii=False, indent=1)
print("ALL DONE", round((time.time() - t0) / 60), "min", flush=True)
