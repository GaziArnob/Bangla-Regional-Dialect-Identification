"""Paper figures, drawn from colab_results.json (the Colab T4 run)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

R = json.load(open("colab_results.json", encoding="utf-8"))
X = json.load(open("colab_results_ext.json", encoding="utf-8"))   # extended transformer runs + ensembles
XTF = {t["model"]: t for t in X["TF"]}
SD = json.load(open("colab_results_seeds.json", encoding="utf-8"))["seeds"]   # 3 seeds x up to 15 epochs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # categorical slots 1-3, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
short = lambda s: s.replace("TF-IDF ", "").replace("Logistic Regression", "LR").replace("Linear SVM", "SVM") \
                   .replace("Multinomial NB", "MNB").replace("Gaussian NB", "GNB").replace("Word2Vec avg", "Word2Vec")


def hbars(ax, names, vals, colors, err=None):
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=err, color=colors, height=0.72, edgecolor="white", linewidth=1,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v + (err[i] if err is not None else 0) + 0.006, i, f"{v:.3f}", va="center", fontsize=7, color=INK)


# Fig. 2: all models on the test set; transformers and ensembles as mean +- std over 3 seeds (up to 15 epochs)
rows = [(short(r["representation"] + " + " + r["classifier"]), r["test macro-F1"], 0.0, BLUE)
        for r in R["res"] if r["representation"] not in ("fine-tuned transformer", "-")]
rows += [(k + " (fine-tuned)", v["mean"], v["std"], ORANGE) for k, v in SD["models"].items() if k != "char SVM"]
rows += [(k.replace("all three transformers (no SVM)", "3 transformers, no SVM").replace("all three", "all 3 transformers"), v["mean"], v["std"], AQUA)
         for k, v in SD["ensembles"].items()]
rows.sort(key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(3.4, 4.8))
hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows], err=[r[2] for r in rows])
ax.set_xlim(0.45, 0.87); ax.set_xlabel("test macro-F1 (transformers, ensembles: mean ± std, 3 seeds)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)],
          labels=["classical", "fine-tuned transformer", "ensemble"], loc="lower right", fontsize=7, frameon=False)
fig.savefig("fig/models.png"); plt.close(fig)

# Fig. 3: fine-tuning curves, validation macro-F1 per epoch (mean over 3 seeds, lr 5e-5)
fig, ax = plt.subplots(figsize=(3.3, 2.0))
for (name, t), c in zip([(k, v) for k, v in SD["models"].items() if k != "char SVM"], [BLUE, ORANGE, AQUA]):
    L = min(len(h) for h in t["hist"]); f = np.mean([h[:L] for h in t["hist"]], axis=0)
    ax.plot(range(1, L + 1), f, color=c, lw=2, marker="o", ms=3, markeredgecolor="white", label=name)
best = max(r["val macro-F1"] for r in R["res"] if r["representation"] != "fine-tuned transformer")
ax.axhline(best, color=INK2, ls="--", lw=1)
ax.text(6.2, best - 0.028, f"best classical (char + SVM) {best:.3f}", fontsize=7, color=INK2)
ax.set_xticks(range(1, 16, 2)); ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1")
ax.set_ylim(0.5, 0.84); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.savefig("fig/curves.png"); plt.close(fig)

# Fig. 4: cross-validation of the classical models
cv = [r for r in R["cv"] if r["classifier"] != "Majority class"]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
hbars(ax, [short(r["representation"] + " + " + r["classifier"]) for r in cv], [r["mean"] for r in cv], [BLUE] * len(cv),
      err=[r["std"] for r in cv])
ax.set_xlim(0.45, 0.85); ax.set_xlabel("macro-F1, mean ± std over 5 folds")
fig.savefig("fig/cv.png"); plt.close(fig)

# Fig. 5: per-class F1 of the SVM, BanglaBERT (mean +- std over 3 seeds) and the SVM + all-transformer ensemble
labs = R["labels"]
PC = json.load(open("colab_results_perclass.json", encoding="utf-8"))["models"]
series = [("char (1,4) + SVM", [R["svm_f1"][l] for l in labs], None, BLUE),
          ("BanglaBERT (3 seeds)", PC["BanglaBERT"]["mean"], PC["BanglaBERT"]["std"], ORANGE),
          ("SVM + all 3 transformers", X["ENS"]["SVM + all three transformers"]["f1"], None, AQUA)]
order = np.argsort(series[0][1])[::-1]
y = np.arange(len(labs)); h = 0.27
fig, ax = plt.subplots(figsize=(3.4, 3.8))
for k, (name, vals, err, c) in enumerate(series):
    ax.barh(y + (k - 1) * (h + 0.01), np.array(vals)[order], h, color=c, label=name,
            xerr=None if err is None else np.array(err)[order], error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5})
ax.set_yticks(y, [labs[i].replace("_", " ") for i in order]); ax.invert_yaxis()
ax.set_xlim(0.5, 1.0); ax.set_xlabel("test F1"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.savefig("fig/perclass.png"); plt.close(fig)

# Fig. 6: confusion matrix of the best model (% of the true class)
cm = np.array(R["cm"]); pct = cm / cm.sum(axis=1, keepdims=True) * 100
cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
tick = [l.replace("Standard_Bangla", "Standard") for l in labs]
ax.set_xticks(range(12), tick, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(12), tick, fontsize=7)
for i in range(12):
    for j in range(12):
        if pct[i, j] >= 1:
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if pct[i, j] > 55 else INK)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig("fig/confusion.png"); plt.close(fig)

# Fig. 7: accuracy by sentence length
L = R["len"]; overall = R["res"][0]["accuracy"]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ax.bar([l[0] for l in L], [l[2] for l in L], color=BLUE, width=0.62)
for i, l in enumerate(L):
    ax.text(i, l[2] + 0.02, f"{l[2]:.2f}", ha="center", fontsize=7, color=INK)
ax.axhline(overall, color=INK2, ls="--", lw=1); ax.set_ylim(0, 1.05)
ax.set_xlabel("words per sentence"); ax.set_ylabel("accuracy"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.savefig("fig/length.png"); plt.close(fig)
print("figures written")
