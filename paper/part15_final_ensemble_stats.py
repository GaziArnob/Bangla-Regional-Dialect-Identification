# Part 15. Per-dialect F1 of the final SVM + all-three-transformers ensemble and the SVM/transformer agreement
# statistics of Section VII-C, recomputed from the saved logits of the final Part 12 run (3 seeds, up to 15 epochs),
# so that Fig. 5 and Section VII-C no longer rely on the extended-budget / initial-budget runs.
# Usage: python part15_final_ensemble_stats.py <folder with data.pkl and logits_*_s*.npz> <BanglaDialect_Midterm.ipynb> <out.json>
# The calibrated char (1,4) SVM is refitted exactly as in Part 12 (CPU only, a few minutes).
import glob, json, os, pickle, sys
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

src, nb_path, out_path = sys.argv[1:4]
g = {}
for c in json.load(open(nb_path, encoding="utf-8"))["cells"]:          # defines preprocess() and SEED, as in Part 12
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("## Part 5"):
        break
    if c["cell_type"] == "code" and not s.lstrip().startswith("!"):
        exec(s, g)
preprocess, SEED = g["preprocess"], g["SEED"]

D = pickle.load(open(os.path.join(src, "data.pkl"), "rb")); LABELS = D["labels"]
(xtr, ytr), (xva, yva), (xte, yte) = D["train"], D["val"], D["test"]
yv, yt = np.array(yva), np.array(yte)
txt = lambda s: [preprocess(t, mask_digits=True) for t in s]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True, max_features=200000)
Xtr = vec.fit_transform(txt(xtr)); Xva, Xte = vec.transform(txt(xva)), vec.transform(txt(xte))
ytr_lab = [LABELS[i] for i in ytr]
svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(Xtr, ytr_lab)
cal = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3, method="sigmoid").fit(Xtr, ytr_lab)
assert list(cal.classes_) == LABELS
L2I = {l: i for i, l in enumerate(LABELS)}
sv_val, sv_test = cal.predict_proba(Xva), cal.predict_proba(Xte)
sv_hard = np.array([L2I[l] for l in svm.predict(Xte)])
mf1 = lambda y, p: f1_score(y, p, average="macro")
print("SVM test macro-F1", round(mf1(yt, sv_hard), 4), "(Part 12: 0.7813)", flush=True)

sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
MODELS, SEEDS = ["BanglaBERT", "BanglishBERT", "XLM-R_base"], [42, 1, 2]
grid = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
ens_f1, ens_mf1, ws, agree = [], [], [], []
for s in SEEDS:
    R = {m: dict(np.load(os.path.join(src, f"logits_{m}_s{s}.npz"))) for m in MODELS}
    pv = np.mean([sm(R[m]["val"]) for m in MODELS], 0); pt = np.mean([sm(R[m]["test"]) for m in MODELS], 0)
    w = max(grid, key=lambda w: mf1(yv, (w * pv + (1 - w) * sv_val).argmax(1)))
    p = (w * pt + (1 - w) * sv_test).argmax(1)
    ws.append(w); ens_mf1.append(mf1(yt, p)); ens_f1.append(f1_score(yt, p, average=None, labels=range(12)))
    ok = np.stack([sv_hard == yt] + [R[m]["test"].argmax(1) == yt for m in MODELS])   # SVM + 3 transformers
    svm_err = ~ok[0]
    agree.append({"all_correct": ok.all(0).mean(), "all_wrong": (~ok).all(0).mean(),
                  "svm_errors_fixed_by_a_transformer": ok[1:, svm_err].any(0).mean(),
                  "svm_errors": int(svm_err.sum())})
ens_f1 = np.array(ens_f1)
res = {"source": "part15_final_ensemble_stats.py on the saved logits of the final Part 12 run (Kaggle notebook notebookc1731ffa49)",
       "labels": LABELS, "svm_macroF1": round(mf1(yt, sv_hard), 4), "w": ws,
       "ensemble_macroF1": [round(x, 4) for x in ens_mf1],
       "ensemble_f1_mean": ens_f1.mean(0).round(3).tolist(), "ensemble_f1_std": ens_f1.std(0, ddof=1).round(3).tolist(),
       "svm_f1": f1_score(yt, sv_hard, average=None, labels=range(12)).round(3).tolist(),
       "agree_per_seed": [{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in a.items()} for a in agree],
       "agree_mean": {k: round(float(np.mean([a[k] for a in agree])), 4) for k in agree[0]},
       "agree_std": {k: round(float(np.std([a[k] for a in agree], ddof=1)), 4) for k in agree[0]}}
json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("w", ws, "ensemble macro-F1", res["ensemble_macroF1"], "(Part 12: 0.8353 0.8311 0.8346)")
print("agreement", res["agree_mean"])
