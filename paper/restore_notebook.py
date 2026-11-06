"""Restore the earlier (pre-faculty-revision) notebook from colab.zip as a standalone research notebook.
The code is unchanged; only course-related wording is removed and the kernel is set to python3 so Colab opens it in Python."""
import json, re, sys, zipfile

SRC_ZIP, MEMBER = "../colab.zip", "colab/BanglaDialect_Midterm.ipynb"
OUT = "BanglaDialect_Transformers.ipynb"

nb = json.loads(zipfile.ZipFile(SRC_ZIP).read(MEMBER).decode("utf-8"))

HEADER = """# Bangla Regional Dialect Identification: Classical Pipelines vs. Fine-tuned Transformers
Code for the paper *Character n-Grams or Pre-trained Transformers? A Leakage-Free Comparison for 12-Class Bangla Regional Dialect Identification*.

**Task:** given a short Bangla sentence, predict its regional dialect (12-class text classification).
**Data:** BanglaDial (Mahi & Khan, Mendeley Data, DOI 10.17632/sx6ybcps2n.2, CC BY 4.0).
**How to run:** `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`. The data is read from a public GitHub link.
"""
SCOPE_NOTE = re.compile(r"> \*\*Scope note\.\*\*.*?\n\n", re.S)
PROMPT = re.compile(r"\*\(write after you [^)]*?\)\*|\*\(write one line[^)]*?\)\*|\*\(one line:[^)]*?\)\*|\*\(read the table above\..*?\)\*|\*\(what the \"wrong for every model\".*?\)\*", re.S)
REPLACE = [
    ("(the group GitHub repository)", "(the project GitHub repository)"),
    ("every number in the report can be reproduced", "every number in the paper can be reproduced"),
    ("The assignment asks for regex cleaning of URLs, HTML, emoji and numbers. We count how many sentences each rule touches, so the report can say honestly whether the step matters for this corpus.",
     "We count how many sentences each cleaning rule touches, to show whether the step matters for this corpus."),
    ("(for the report)", ""),
    ("(n-gram setting required by the assignment)", ""),
    ("the columns are the metrics required by the assignment", "the columns are accuracy and macro-averaged precision, recall and F1"),
    ("It covers the remaining items of the assignment: the confusion matrix", "It covers the confusion matrix"),
    ("and the report should say so", "and the paper says so"),
    ("(write for the report)", ""),
    ("| 8 | Evaluation of the best model: confusion matrix, informative words, error analysis |",
     "| 8 | Evaluation of the best model: confusion matrix, informative words, error analysis |\n"
     "| 9 | Large language model: Qwen2.5-7B-Instruct, zero-shot and few-shot prompting |\n"
     "| 10 | Extended experiments: longer transformer training, ensembles, full-test LLM, QLoRA |\n"
     "| 11 | Translation probe: does the LLM understand dialectal Bangla? |\n"
     "| 12 | Robustness: 3 seeds, significance tests, cross-corpus test (standalone) |\n"
     "| 13 | More LLMs, Bangla prompt, shuffled options (standalone) |"),
    ("**Your reading of the 30 examples (required):**", "**Reading of the 30 examples:**"),
]

src = lambda c: "".join(c["source"])
for k, c in enumerate(nb["cells"]):
    s = src(c)
    if k == 0:
        s = HEADER + s[s.index("\n| Part"):]
    if c["cell_type"] == "markdown":
        s = SCOPE_NOTE.sub("", s)
        s = PROMPT.sub("*(see the paper for the interpretation)*", s)
    for a, b in REPLACE:
        s = s.replace(a, b)
    c["source"] = s.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

PART9_MD = """## Part 9. Large language model (zero-shot and few-shot prompting)
We test whether an instruction-tuned LLM can identify the dialect **without any training**, only from a prompt.
**Qwen2.5-7B-Instruct** is loaded in 4-bit (fits on a T4). The prompt lists the 12 classes as lettered options A-L and asks for the letter;
the prediction is the letter whose next-token logit is highest, so every answer is a valid class (no free-text parsing).

| Setting | Prompt |
|---|---|
| zero-shot | instruction + options + sentence |
| few-shot | the same + 1 labelled training sentence per dialect (12 examples) |

LLM inference is slow on a T4 (about 5 s per 16 zero-shot prompts), so both settings are evaluated on a stratified test subset
of 100 sentences per dialect (1,200 sentences). The character SVM is re-scored on the same subset for a like-for-like comparison.
"""
nb["cells"] += [{"cell_type": "markdown", "metadata": {}, "source": PART9_MD.splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": open("part9_llm.py", encoding="utf-8").read().splitlines(keepends=True)}]

PART10_MD = """## Part 10. Extended experiments
Run after Part 9 on a T4 GPU (about 4 hours in total). Every block prints its results as one `@@TAG@@{json}` line.

| Block | What | Time on a T4 |
|---|---|---|
| B | BanglaBERT, BanglishBERT, XLM-R with the **extended budget**: up to 8 epochs, learning rate in {2e-5, 5e-5}, early stopping (patience 2); then probability-averaging **ensembles** of the calibrated char SVM with one or all three transformers, ensemble weight tuned on validation | ~2.5 h |
| C | LLM zero-shot on the **full** test set (8,982 sentences) | ~45 min |
| D | **QLoRA** fine-tuning of Qwen2.5-7B (rank 16, 2,400 training sentences, 1 epoch, lr 5e-5, fp16 loss scaling), evaluated on the 1,200-sentence subset | ~35 min |
"""
nb["cells"] += [{"cell_type": "markdown", "metadata": {}, "source": PART10_MD.splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["!pip -q install peft"]},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": open("part10_extended.py", encoding="utf-8").read().splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["block_b()"]},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["block_c()"]},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["block_d()"]}]

PART11_MD = """## Part 11. Does the LLM understand dialectal Bangla? (translation probe)
The base Qwen2.5-7B-Instruct model (no adapters) translates 10 test sentences per class into English with greedy decoding.
Each translation is then rated by hand as correct / partial / wrong; there are no reference translations, so this needs a
Bangla speaker. The ratings used in the paper are in `translation_probe_ratings.json`. If the runtime from Part 10 is still
loaded, restart it first (Part 10 adds LoRA adapters to `llm`); the cell reloads the base model.
"""
nb["cells"] += [{"cell_type": "markdown", "metadata": {}, "source": PART11_MD.splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": open("part11_translation.py", encoding="utf-8").read().splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": ["print('\\n'.join(f\"{i:3d} {r.label[:5]} | {r.sentence} | {r.english}\" for i, r in enumerate(probe.itertuples(), 1)))"]}]

PART12_MD = """## Part 12. Robustness: 3 seeds, significance tests, cross-corpus test (standalone, Kaggle 2 x T4, ~3.3 h)
Self-contained: re-runs Parts 1-4 from GitHub, trains BanglaBERT, BanglishBERT and XLM-R with lr 5e-5 for up to 15 epochs
(patience 3) with seeds 42/1/2 on two GPUs in parallel, builds the ensembles per seed, runs McNemar and paired-bootstrap tests,
and applies all models to ANCHOLIK-NER and ChatgaiyyaAlap (Mendeley, CC BY 4.0) after removing sentences that also occur in
BanglaDial. Results: `colab_results_seeds.json`. Run it in a fresh session (Save & Run All on Kaggle).
"""
PART13_MD = """## Part 13. More LLMs, a Bangla prompt and shuffled options (standalone, Kaggle 2 x T4, ~2.4 h)
Qwen2.5-7B/14B-Instruct and BLOOMZ-7B1 (4-bit) under English and Bangla prompts, shuffled options and few-shot, on the
1,200-sentence subset. Llama-3.1-8B (needs a Hugging Face token as Kaggle secret `HF_TOKEN`) and Gemini (`GEMINI_API_KEY`)
are skipped when no key is set. Results: `colab_results_llm2.json`.
"""
for md, f in [(PART12_MD, "part12_seeds_crosscorpus.py"), (PART13_MD, "part13_llms.py")]:
    nb["cells"] += [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": open(f, encoding="utf-8").read().splitlines(keepends=True)}]

nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"].setdefault("accelerator", "GPU")
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

left = [w for w in ("CSC", "Mid-Term", "mid-term", "assignment", "instructor", "Group G", "course") if w in json.dumps(nb, ensure_ascii=False)]
print("saved", OUT, "| course words left:", left or "none")
