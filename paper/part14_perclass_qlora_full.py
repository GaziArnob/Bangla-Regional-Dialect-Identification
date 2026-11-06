# Part 14. (a) per-dialect test F1 averaged over the 3 seeds of Part 12, from its saved logits (attach the output of the
# Part 12 notebook as an input); (b) QLoRA fine-tuning of Qwen2.5-7B-Instruct on the FULL training set (41,916 sentences).
# A short prompt without the option list is used for (b) (~45 instead of ~160 tokens), which makes one epoch affordable
# on a T4; the answer is still read from the 12 letter logits, so A-L stand for the 12 classes in alphabetical order.
import glob, gc, json, os, pickle, random, subprocess, sys, time, urllib.request
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "bitsandbytes>=0.43", "accelerate", "peft"], check=True)
emit = lambda tag, obj: print(f"@@{tag}@@" + json.dumps(obj, ensure_ascii=False), flush=True)
import numpy as np, torch
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------- (a) per-dialect F1 over seeds
pk = glob.glob("/kaggle/input/**/data.pkl", recursive=True)
if pk:
    base = os.path.dirname(pk[0]); D = pickle.load(open(pk[0], "rb")); LABELS = D["labels"]; yt = np.array(D["test"][1])
    sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
    out = {"labels": LABELS, "models": {}}
    for m in ["BanglaBERT", "BanglishBERT", "XLM-R_base"]:
        fs = sorted(glob.glob(f"{base}/logits_{m}_s*.npz"))
        if fs:
            f1 = np.array([f1_score(yt, np.load(f)["test"].argmax(1), average=None, labels=range(12)) for f in fs])
            out["models"][m] = {"n_seeds": len(fs), "mean": f1.mean(0).round(3).tolist(), "std": f1.std(0, ddof=1).round(3).tolist()}
    emit("PERCLASS", out)
else:
    print("Part 12 output not attached: per-dialect step skipped", flush=True)

# ---------------------------------------------------------------- (b) QLoRA on the full training set
NB = ("https://raw.githubusercontent.com/GaziArnob/Bangla-Regional-Dialect-Identification/"
      "ab8b8c50103b54979815fa7181b6019e693afefb/colab/BanglaDialect_Midterm.ipynb")
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gensim", "nltk"], check=True)
for c in json.load(urllib.request.urlopen(NB))["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("## Part 5"):
        break
    if c["cell_type"] == "code" and not s.lstrip().startswith("!"):
        exec(s, globals())
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
LABELS = sorted(df["label"].unique()); L2I = {l: i for i, l in enumerate(LABELS)}; LETTERS = "ABCDEFGHIJKL"
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"); tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map={"": 0}, quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
lid = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]).to(llm.device)
short = lambda s: tok.apply_chat_template([{"role": "user", "content": "Bangla dialect of this sentence (letter A-L): " + s}],
                                          tokenize=False, add_generation_prompt=True)
llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
params = [p for p in llm.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
llm.config.use_cache = False


@torch.no_grad()
def predict(sents, bs=32):
    llm.eval(); out = []
    for i in range(0, len(sents), bs):
        enc = tok([short(s) for s in sents[i:i + bs]], return_tensors="pt", padding=True).to(llm.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out += llm(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, lid].argmax(1).tolist()
    llm.train(); return np.array(out)


tr = train_df.sample(frac=1, random_state=SEED); X, Y = tr["sentence"].tolist(), np.array([L2I[l] for l in tr["label"]])
va = val_df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(50, len(g)), random_state=SEED))   # 600 for checkpoint choice
yva = np.array([L2I[l] for l in va["label"]])
BS, ACC, LR = 8, 2, 1e-4                                        # effective batch 16
steps = len(X) // (BS * ACC)
opt = torch.optim.AdamW(params, lr=LR); sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler("cuda"); best = (-1, 0); hist = []; t0 = time.time(); k = 0; run_loss = 0.0
llm.train()
for i in range(0, steps * BS * ACC, BS):
    enc = tok([short(s) for s in X[i:i + BS]], return_tensors="pt", padding=True).to(llm.device)
    with torch.autocast("cuda", dtype=torch.float16):
        lg = llm(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, lid]
    loss = torch.nn.functional.cross_entropy(lg, torch.tensor(Y[i:i + BS], device=lg.device)) / ACC
    scaler.scale(loss).backward(); run_loss += loss.item()
    if (i // BS + 1) % ACC == 0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); k += 1
        if k % 100 == 0:
            print(f"step {k}/{steps} loss {run_loss / 100:.3f} ({time.time() - t0:.0f}s)", flush=True); run_loss = 0.0
        if k % 500 == 0 or k == steps:
            f = f1_score(yva, predict(va["sentence"].tolist()), average="macro"); hist.append([k, round(f, 4)])
            print(f"  val macro-F1 at step {k}: {f:.4f}", flush=True)
            if f > best[0]:
                best = (f, k); llm.save_pretrained("/kaggle/working/qlora_best")
train_secs = round(time.time() - t0)
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
set_peft_model_state_dict(llm, load_file("/kaggle/working/qlora_best/adapter_model.safetensors"))
yt = np.array([L2I[l] for l in test_df["label"]]); t1 = time.time(); pt = predict(test_df["sentence"].tolist())
sub = test_df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(100, len(g)), random_state=SEED))
pos = test_df.index.get_indexer(sub.index)
emit("QLORAFULL", {"n_train": len(X), "epochs": 1, "lr": LR, "eff_batch": BS * ACC, "r": 16, "prompt": "short (no option list)",
                   "best_step": best[1], "val600_hist": hist, "train_secs": train_secs, "test_secs": round(time.time() - t1),
                   "test": {"accuracy": round(accuracy_score(yt, pt), 4), "macro_F1": round(f1_score(yt, pt, average="macro"), 4)},
                   "subset1200": {"accuracy": round(accuracy_score(yt[pos], pt[pos]), 4), "macro_F1": round(f1_score(yt[pos], pt[pos], average="macro"), 4)},
                   "f1": f1_score(yt, pt, average=None, labels=range(12)).round(3).tolist(),
                   "pred_dist": {LABELS[a]: int(b) for a, b in zip(*np.unique(pt, return_counts=True))}})
print("ALL DONE", flush=True)
