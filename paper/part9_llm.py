# Part 9. LLM prompting (zero-shot and few-shot) with an open instruction-tuned LLM.
# Runs after Parts 1-5 in the same session (uses df, train_df, test_df, models, reps, metrics, SEED).
import json, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "bitsandbytes>=0.43", "accelerate"], check=True)
import numpy as np, pandas as pd, torch
from collections import Counter
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM_ID, N_PER_CLASS, SHOTS_PER_CLASS = "Qwen/Qwen2.5-7B-Instruct", 100, 1
LABELS = sorted(df["label"].unique())

# LLM inference is slow on a T4 (~5 s per 16 zero-shot prompts), so both settings run on a stratified subset (100 per dialect)
sub = test_df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(N_PER_CLASS, len(g)), random_state=SEED))
pos, y_sub = test_df.index.get_indexer(sub.index), sub["label"].values

# the 12 classes become lettered options; the answer is read from the logits of the 12 letter tokens (constrained decoding)
LETTERS = "ABCDEFGHIJKL"
NAME = {l: "Standard Bangla (no regional dialect)" if l == "Standard_Bangla" else l for l in LABELS}
INSTR = ("The following Bangla sentence is written either in one of the regional dialects of Bangladesh or in Standard "
         "Bangla. Which variety is it?\n" + "\n".join(f"{LETTERS[i]}. {NAME[l]}" for i, l in enumerate(LABELS)) +
         "\nReply with the letter only.")
shots = train_df.groupby("label", group_keys=False).apply(
    lambda g: g.sample(SHOTS_PER_CLASS, random_state=SEED)).sample(frac=1, random_state=SEED)
SHOTS = "Examples:\n" + "\n".join(f"Sentence: {s}\nAnswer: {LETTERS[LABELS.index(l)]}"
                                  for s, l in zip(shots["sentence"], shots["label"])) + "\n\n"

tok = AutoTokenizer.from_pretrained(LLM_ID)
tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
assert len(set(letter_ids)) == 12


def prompt(s, few):
    msg = [{"role": "user", "content": INSTR + "\n\n" + (SHOTS if few else "") + "Sentence: " + s}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def llm_predict(sents, few, bs):
    out = []
    for i in range(0, len(sents), bs):
        enc = tok([prompt(s, few) for s in sents[i:i + bs]], return_tensors="pt", padding=True).to(llm.device)
        out += [LABELS[j] for j in llm(**enc).logits[:, -1, letter_ids].argmax(-1).tolist()]
    return np.array(out)


S, secs, P = sub["sentence"].tolist(), {}, {}
P["char (1,4) + SVM"] = models[("TF-IDF char (1,4)", "Linear SVM")].predict(reps["TF-IDF char (1,4)"][2])[pos]   # same subset
for name, few, bs in [("Qwen2.5-7B zero-shot", False, 16), ("Qwen2.5-7B few-shot", True, 8)]:
    t0 = time.time()
    P[name] = llm_predict(S, few, bs)
    secs[name] = round(time.time() - t0)
    print(f"{name}: macro-F1 {f1_score(y_sub, P[name], average='macro'):.4f}  ({secs[name]} s)", flush=True)

llm_res = pd.DataFrame({k: metrics(y_sub, p) for k, p in P.items()}).T.round(4)
print(llm_res.to_string())
print("predicted-label distribution, zero-shot:", Counter(P["Qwen2.5-7B zero-shot"]).most_common(4))
print("predicted-label distribution, few-shot :", Counter(P["Qwen2.5-7B few-shot"]).most_common(4))
