# Part 13. More LLMs, a Bangla prompt and shuffled options (option-order bias), on the 1,200-sentence stratified subset.
# Local models (4-bit) are scored by the 12 letter logits as in Part 9; Gemini (API) by the first letter A-L in its reply.
# Keys come from Kaggle Secrets: HF_TOKEN (Llama licence accepted on Hugging Face) and GEMINI_API_KEY; a missing key skips that model.
import gc, json, os, random, re, subprocess, sys, time, unicodedata, urllib.request
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gensim", "nltk", "bitsandbytes>=0.43", "accelerate", "google-genai"], check=True)
emit = lambda tag, obj: print(f"@@{tag}@@" + json.dumps(obj, ensure_ascii=False), flush=True)
NB = ("https://raw.githubusercontent.com/GaziArnob/Bangla-Regional-Dialect-Identification/"
      "ab8b8c50103b54979815fa7181b6019e693afefb/colab/BanglaDialect_Midterm.ipynb")
for c in json.load(urllib.request.urlopen(NB))["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("## Part 5"):
        break
    if c["cell_type"] == "code" and not s.lstrip().startswith("!"):
        exec(s, globals())

import numpy as np, torch
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
LABELS = sorted(df["label"].unique()); LETTERS = "ABCDEFGHIJKL"
sub = test_df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(100, len(g)), random_state=SEED))
S, Y = sub["sentence"].tolist(), sub["label"].values
shots = train_df.groupby("label", group_keys=False).apply(lambda g: g.sample(1, random_state=SEED)).sample(frac=1, random_state=SEED)
def secret(k):
    """A Kaggle secret, else an environment variable, else None (a missing secret raises inside kaggle_secrets)."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(k)
    except Exception:
        return os.environ.get(k)

EN = {l: "Standard Bangla (no regional dialect)" if l == "Standard_Bangla" else l for l in LABELS}
BN = {"Barisal": "বরিশাল", "Chittagong": "চট্টগ্রাম", "Kishoreganj": "কিশোরগঞ্জ", "Mymensingh": "ময়মনসিংহ", "Narail": "নড়াইল",
      "Narsingdi": "নরসিংদী", "Noakhali": "নোয়াখালী", "Rajshahi": "রাজশাহী", "Rangpur": "রংপুর",
      "Standard_Bangla": "প্রমিত বাংলা (কোনো আঞ্চলিক ভাষা নয়)", "Sylhet": "সিলেট", "Tangail": "টাঙ্গাইল"}
HEAD = {"en": ("The following Bangla sentence is written either in one of the regional dialects of Bangladesh or in Standard "
               "Bangla. Which variety is it?", "Reply with the letter only.", "Sentence: ", "Answer: ", "Examples:"),
        "bn": ("নিচের বাংলা বাক্যটি বাংলাদেশের কোনো একটি আঞ্চলিক ভাষায় অথবা প্রমিত বাংলায় লেখা। এটি কোন ভাষারূপ?",
               "শুধু অক্ষরটি লিখুন।", "বাক্য: ", "উত্তর: ", "উদাহরণ:")}
rng = random.Random(SEED)
PERM = [rng.sample(range(12), 12) for _ in S]        # one fixed random option order per sentence (shuffled setting)
IDENT = list(range(12))


def build(i, s, lang, order, few):
    names = EN if lang == "en" else BN
    h, tail, sl, al, ex = HEAD[lang]
    opts = "\n".join(f"{LETTERS[k]}. {names[LABELS[c]]}" for k, c in enumerate(order))
    shot = ""
    if few:
        pos = {c: k for k, c in enumerate(order)}
        shot = ex + "\n" + "\n".join(f"{sl}{t}\n{al}{LETTERS[pos[LABELS.index(l)]]}" for t, l in zip(shots["sentence"], shots["label"])) + "\n\n"
    return f"{h}\n{opts}\n{tail}\n\n{shot}{sl}{s}"


SETTINGS = [("en", False, False), ("bn", False, False), ("en", True, False), ("en", False, True)]   # lang, shuffled, few-shot
sname = lambda lang, shuf, few: f"{'Bangla' if lang == 'bn' else 'English'} prompt{', shuffled options' if shuf else ''}{', few-shot' if few else ', zero-shot'}"


def score(name, setting, pred_idx, orders, secs):
    pred = np.array([LABELS[p] for p in pred_idx])
    first = float(np.mean([o.index(LABELS.index(p)) == 0 for o, p in zip(orders, pred)]))   # share answered with option A
    top = Counter(pred).most_common(2)
    emit("LLM", {"model": name, "setting": setting, "n": len(S), "secs": secs, "macro_F1": round(f1_score(Y, pred, average="macro"), 4),
                 "accuracy": round(accuracy_score(Y, pred), 4), "share_option_A": round(first, 3),
                 "top_predictions": [[k, int(v)] for k, v in top], "f1": f1_score(Y, pred, average=None, labels=LABELS).round(3).tolist()})


def run_local(name, hub, chat=True, token=None, bs=8):
    try:
        tok = AutoTokenizer.from_pretrained(hub, token=token); tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(hub, token=token, device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
    except Exception as e:
        print(f"{name}: could not load ({str(e)[:300]})", flush=True); return
    lid = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    lid2 = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS]
    if len(set(lid)) < 12:
        lid = lid2
    print(f"{name} loaded; letter ids unique: {len(set(lid)) == 12}", flush=True)
    for lang, shuf, few in SETTINGS:
        t0, out = time.time(), []
        orders = PERM if shuf else [IDENT] * len(S)
        for i in range(0, len(S), bs):
            texts = []
            for j in range(i, min(i + bs, len(S))):
                p = build(j, S[j], lang, orders[j], few)
                texts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                             if chat else p + "\n" + HEAD[lang][3])
            enc = tok(texts, return_tensors="pt", padding=True).to(llm.device)
            with torch.no_grad():
                lg = llm(**enc).logits[:, -1, lid]
            out += [orders[j][k] for j, k in zip(range(i, i + len(texts)), lg.argmax(-1).tolist())]
        score(name, sname(lang, shuf, few), out, orders, round(time.time() - t0))
    del llm; gc.collect(); torch.cuda.empty_cache()


def run_gemini():
    key = secret("GEMINI_API_KEY")
    if not key:
        print("Gemini: no GEMINI_API_KEY secret, skipped", flush=True); return
    from google import genai
    cl = genai.Client(api_key=key)
    # older models are closed to new keys and previews may reject every request, so try the newest general flash models
    # and keep the first one that answers a test request (retrying when it is overloaded); every failure is printed
    avail = {m.name.split("/")[-1] for m in cl.models.list()}
    ver = lambda a: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", a)[:1]] or [0.0]
    flash = sorted((a for a in avail if a.startswith("gemini-") and "flash" in a and not any(
        x in a for x in ("image", "tts", "audio", "live", "computer", "lite"))), key=lambda a: (ver(a), "preview" not in a), reverse=True)
    cands = list(dict.fromkeys(["gemini-3.8-flash", "gemini-flash-latest"] + flash + ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]))
    CFG = {"temperature": 0, "max_output_tokens": 4000}      # room for thinking tokens before the answer
    model = None
    for m in cands:
        for attempt in range(3):
            try:
                r = cl.models.generate_content(model=m, contents="Reply with the single letter A.", config=CFG)
                print(f"Gemini test {m}: {(r.text or '').strip()[:40]!r}", flush=True)
                model = m; break
            except Exception as e:
                print(f"Gemini test {m} failed: {str(e)[:200]}", flush=True)
                if not any(c in str(e) for c in ("503", "429", "UNAVAILABLE")):
                    break
                time.sleep(30)
        if model:
            break
    if model is None:
        print("Gemini: no usable model, skipped; flash models:", flash, flush=True); return
    print("Gemini model:", model, flush=True)
    # the free tier allows few requests per minute and per day, so 40 sentences share one request; with shuffled options
    # each sentence keeps its own option order, so that setting is sent one sentence per request on a 120-sentence subset
    G = 40
    for lang, shuf, few in SETTINGS:
        t0, bad, fails, last_err = time.time(), 0, 0, None
        idx = list(range(len(S))) if not shuf else [j for j in range(len(S)) if j % 10 == 0]
        orders = PERM if shuf else [IDENT] * len(S)
        out = {}
        groups = [[j] for j in idx] if shuf else [idx[i:i + G] for i in range(0, len(idx), G)]
        for grp in groups:
            if len(grp) == 1:
                p = build(grp[0], S[grp[0]], lang, orders[grp[0]], few)
            else:
                body = build(grp[0], "", lang, IDENT, few).rsplit(HEAD[lang][2], 1)[0]
                p = body + ("Classify each numbered sentence. Reply with one line per sentence in the form '<number>: <letter>'.\n"
                            if lang == "en" else "প্রতিটি নম্বরযুক্ত বাক্যের জন্য এক লাইনে '<নম্বর>: <অক্ষর>' আকারে উত্তর দিন।\n")
                p += "\n".join(f"{k + 1}. {S[j]}" for k, j in enumerate(grp))
            txt = ""
            if fails >= 5:                                  # quota exhausted: stop sending, count the rest as wrong
                pass
            else:
                for attempt in range(4):
                    try:
                        r = cl.models.generate_content(model=model, contents=p, config=CFG)
                        txt = (r.text or "").strip(); fails = 0; break
                    except Exception as e:
                        if last_err is None:
                            print(f"Gemini error: {str(e)[:300]}", flush=True)
                        last_err = str(e)
                        time.sleep(min(60, 10 * 2 ** attempt))
                else:
                    fails += 1
            if len(grp) == 1:
                m = re.search(r"[A-L]", txt.upper()); got = {0: m.group(0)} if m else {}
            else:
                got = {int(a) - 1: b for a, b in re.findall(r"(\d+)\s*[:.)\-]\s*\(?([A-L])\b", txt.upper())}
            for k, j in enumerate(grp):
                if k in got:
                    out[j] = orders[j][LETTERS.index(got[k])]
                else:
                    bad += 1; out[j] = (LABELS.index(Y[j]) + 1) % 12        # unparseable answers count as wrong
            time.sleep(4.2)
        print(f"Gemini {sname(lang, shuf, few)}: unparseable {bad} of {len(idx)}"
              + (f"; stopped after 5 failed requests: {last_err[:200]}" if fails >= 5 else ""), flush=True)
        keep = sorted(out)
        pred_idx = [out[j] for j in keep]
        yk, ok = Y[keep], [orders[j] for j in keep]
        pred = np.array([LABELS[p] for p in pred_idx])
        emit("LLM", {"model": f"Gemini ({model.split('/')[-1]})", "setting": sname(lang, shuf, few) + (" (120-sentence subset)" if shuf else ""),
                     "n": len(keep), "secs": round(time.time() - t0), "unparseable": bad,
                     "macro_F1": round(f1_score(yk, pred, average="macro"), 4), "accuracy": round(accuracy_score(yk, pred), 4),
                     "share_option_A": round(float(np.mean([o.index(LABELS.index(p)) == 0 for o, p in zip(ok, pred)])), 3),
                     "top_predictions": [[a, int(b)] for a, b in Counter(pred).most_common(2)],
                     "f1": f1_score(yk, pred, average=None, labels=LABELS).round(3).tolist()})


hf = secret("HF_TOKEN")
ONLY = globals().get("RUN_ONLY")                      # e.g. RUN_ONLY = ["Llama", "Gemini"] before exec, to rerun a subset
want = lambda k: ONLY is None or any(o.lower() in k.lower() for o in ONLY)
if want("Qwen2.5-7B"):
    run_local("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
if want("Llama"):
    run_local("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", token=hf) if hf else print("Llama: no HF_TOKEN secret, skipped", flush=True)
if want("BLOOMZ"):
    run_local("BLOOMZ-7B1", "bigscience/bloomz-7b1", chat=False)
if want("Qwen2.5-14B"):
    run_local("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", bs=4)
if want("Gemini"):
    run_gemini()
print("ALL LLM DONE", flush=True)
