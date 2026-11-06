# Part 10. Extended experiments (run after Parts 1-5; each block is one Colab cell, results printed as @@...@@ JSON lines).
#   B: transformers with up to 8 epochs, learning rates {2e-5, 5e-5}, early stopping; SVM + transformer ensembles
#   C: LLM zero-shot on the full test set
#   D: QLoRA fine-tuning of the LLM, evaluated on the 1,200-sentence stratified subset
import gc, json, os, random, subprocess, sys, time
import numpy as np, torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

LABELS = sorted(df["label"].unique())
lab2id = {l: i for i, l in enumerate(LABELS)}
ytr_id, yva_id, yte_id = (np.array([lab2id[l] for l in y]) for y in (y_train, y_val, y_test))
DEVICE = "cuda"
emit = lambda tag, obj: print(f"@@{tag}@@" + json.dumps(obj, ensure_ascii=False), flush=True)
sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))


# ---------------------------------------------------------------- B. transformers (longer) + ensemble
def block_b():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from normalizer import normalize
    MODELS = {"BanglaBERT": ("csebuetnlp/banglabert", True), "BanglishBERT": ("csebuetnlp/banglishbert", True),
              "XLM-R base": ("xlm-roberta-base", False)}
    MAX_LEN, BATCH, EPOCHS, PATIENCE, LRS = 64, 32, 8, 2, [2e-5, 5e-5]
    cnt = np.bincount(ytr_id, minlength=12)
    W = torch.tensor(len(ytr_id) / (12 * cnt), dtype=torch.float, device=DEVICE)

    def loader(texts, y, tok, shuffle, bs):
        enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
        idx = list(range(len(y)))
        def batches():
            order = random.sample(idx, len(idx)) if shuffle else idx
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                out = tok.pad({k: [enc[k][j] for j in b] for k in enc}, return_tensors="pt")
                yield {k: v.to(DEVICE) for k, v in out.items()}, torch.tensor(y[b], device=DEVICE)
        return batches

    @torch.no_grad()
    def logits_of(model, batches):
        model.eval()
        out = []
        for b, _ in batches():
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(**b).logits.float().cpu())
        return torch.cat(out).numpy()

    def run(name, hub, use_norm, lr):
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        tok = AutoTokenizer.from_pretrained(hub)
        prep = (lambda s: [normalize(x) for x in s]) if use_norm else list
        tr = loader(prep(train_df["sentence"]), ytr_id, tok, True, BATCH)
        va = loader(prep(val_df["sentence"]), yva_id, tok, False, 128)
        te = loader(prep(test_df["sentence"]), yte_id, tok, False, 128)
        model = AutoModelForSequenceClassification.from_pretrained(hub, num_labels=12).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps = (len(ytr_id) + BATCH - 1) // BATCH * EPOCHS
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda")
        loss_fn = torch.nn.CrossEntropyLoss(weight=W)
        best, hist, bad, t0 = (-1, 0, None, None), [], 0, time.time()
        for ep in range(1, EPOCHS + 1):
            model.train()
            for b, y in tr():
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(**b).logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
            lv = logits_of(model, va)
            f = f1_score(yva_id, lv.argmax(1), average="macro")
            hist.append(round(f, 4))
            print(f"{name} lr={lr:g} epoch {ep}: val macro-F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if f > best[0]:
                best, bad = (f, ep, lv, logits_of(model, te)), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        del model, opt; gc.collect(); torch.cuda.empty_cache()
        return {"val_f1": best[0], "epoch": best[1], "val_logits": best[2], "test_logits": best[3], "hist": hist,
                "secs": round(time.time() - t0)}

    TF = {}
    for name, (hub, use_norm) in MODELS.items():
        runs = {lr: run(name, hub, use_norm, lr) for lr in LRS}
        lr = max(runs, key=lambda k: runs[k]["val_f1"])
        r = runs[lr]
        pred = r["test_logits"].argmax(1)
        TF[name] = r
        emit("TF", {"model": name, "lr": lr, "epoch": r["epoch"], "val_f1": round(r["val_f1"], 4),
                    "hist": {str(k): v["hist"] for k, v in runs.items()}, "secs": {str(k): v["secs"] for k, v in runs.items()},
                    "test": {m: round(v, 4) for m, v in metrics(y_test, np.array(LABELS)[pred]).items()},
                    "f1": f1_score(yte_id, pred, average=None, labels=range(12)).round(3).tolist()})
    globals()["TF"] = TF

    # SVM probabilities: the selected char SVM (C=1.0), calibrated with 3-fold sigmoid scaling on the training set
    Xtr, Xva, Xte = reps["TF-IDF char (1,4)"]
    cal = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3, method="sigmoid").fit(Xtr, y_train)
    assert list(cal.classes_) == LABELS
    p_sv, p_st = cal.predict_proba(Xva), cal.predict_proba(Xte)
    ens = {"calibrated SVM alone": {m: round(v, 4) for m, v in metrics(y_test, np.array(LABELS)[p_st.argmax(1)]).items()}}
    combos = {f"SVM + {k}": [k] for k in TF} | {"SVM + all three transformers": list(TF)}
    for cname, members in combos.items():
        pv = np.mean([sm(TF[k]["val_logits"]) for k in members], axis=0)
        pt = np.mean([sm(TF[k]["test_logits"]) for k in members], axis=0)
        grid = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]                       # weight of the transformer part, tuned on validation
        wbest = max(grid, key=lambda w: f1_score(yva_id, (w * pv + (1 - w) * p_sv).argmax(1), average="macro"))
        pred = (wbest * pt + (1 - wbest) * p_st).argmax(1)
        ens[cname] = {"w_transformer": wbest,
                      "val_F1": round(f1_score(yva_id, (wbest * pv + (1 - wbest) * p_sv).argmax(1), average="macro"), 4),
                      **{m: round(v, 4) for m, v in metrics(y_test, np.array(LABELS)[pred]).items()},
                      "f1": f1_score(yte_id, pred, average=None, labels=range(12)).round(3).tolist()}
        print(cname, ens[cname]["w_transformer"], ens[cname]["macro_F1"], flush=True)
    emit("ENS", ens)


# ---------------------------------------------------------------- C. LLM zero-shot on the full test set
def block_c():
    t0 = time.time()
    zs = []
    S = test_df["sentence"].tolist()
    for i in range(0, len(S), 16):
        zs += list(llm_predict(S[i:i + 16], False, 16))
        if (i // 16) % 50 == 0:
            print("zero-shot full", i + 16, "/", len(S), round(time.time() - t0), "s", flush=True)
    zs = np.array(zs)
    globals()["zs_full"] = zs
    from collections import Counter
    emit("ZSFULL", {"n": len(S), "secs": round(time.time() - t0),
                    "test": {m: round(v, 4) for m, v in metrics(y_test, zs).items()},
                    "f1": f1_score(y_test, zs, average=None, labels=LABELS).round(3).tolist(),
                    "dist": dict(Counter(zs))})


# ---------------------------------------------------------------- D. QLoRA fine-tuning of the LLM
# Reported run: lr 5e-5 with fp16 loss scaling and fp32 LoRA weights. A first attempt with lr 2e-4 and no loss scaling
# diverged and collapsed to one class (macro-F1 0.057); see colab_results_ext.json ("QLORA1_failed").
def block_d(n_per_class=200, epochs=1, lr=5e-5, bs=8):
    from collections import Counter
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_linear_schedule_with_warmup
    global llm
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in llm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()                                  # LoRA weights in fp32
    ft = train_df.groupby("label", group_keys=False).apply(lambda g: g.sample(n_per_class, random_state=SEED)).sample(frac=1, random_state=SEED)
    X, Y = ft["sentence"].tolist(), torch.tensor([lab2id[l] for l in ft["label"]], device=DEVICE)
    lid = torch.tensor(letter_ids, device=DEVICE)
    opt = torch.optim.AdamW(params, lr=lr)
    steps = epochs * ((len(X) + bs - 1) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    llm.config.use_cache = False
    llm.train(); t0, k, hits, seen, run_loss = time.time(), 0, 0, 0, 0.0
    # the training target is the answer letter: cross-entropy over the 12 letter logits at the answer position,
    # i.e. exactly the quantity that the zero-shot readout uses for prediction
    for ep in range(epochs):
        for i in range(0, len(X), bs):
            enc = tok([prompt(s, False) for s in X[i:i + bs]], return_tensors="pt", padding=True).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = llm(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, lid]
            loss = torch.nn.functional.cross_entropy(lg, Y[i:i + bs])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sch.step(); k += 1
            hits += (lg.argmax(1) == Y[i:i + bs]).sum().item(); seen += len(lg); run_loss += loss.item()
            if k % 50 == 0:
                print(f"QLoRA step {k}/{steps} mean loss {run_loss / 50:.3f} train acc {hits / seen:.3f} ({time.time() - t0:.0f}s)", flush=True)
                hits, seen, run_loss = 0, 0, 0.0
    train_secs = round(time.time() - t0)
    llm.eval(); t0 = time.time()
    with torch.no_grad():
        ftp = llm_predict(sub["sentence"].tolist(), False, 16)
    emit("QLORA", {"n_train": len(X), "epochs": epochs, "lr": lr, "r": 16, "train_secs": train_secs, "eval_secs": round(time.time() - t0),
                   "subset": {m: round(v, 4) for m, v in metrics(y_sub, ftp).items()},
                   "f1": f1_score(y_sub, ftp, average=None, labels=LABELS).round(3).tolist(), "dist": dict(Counter(ftp))})
