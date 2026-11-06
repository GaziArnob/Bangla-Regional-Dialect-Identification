# Part 11. Does the LLM understand dialectal Bangla? Translation probe.
# Qwen2.5-7B-Instruct (base model, no adapters) translates 10 test sentences per class into English. If translations of
# dialect sentences are about as adequate as those of Standard Bangla sentences, the model understands dialect text and
# fails only at naming the dialect. Adequacy is rated by hand (correct / partial / wrong); no reference translations exist.
# Runs after Parts 1-4 (needs df, test_df, SEED).
import json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM_ID, PER_CLASS = "Qwen/Qwen2.5-7B-Instruct", 10
probe = test_df.groupby("label", group_keys=False).apply(lambda g: g.sample(PER_CLASS, random_state=SEED))
tok = AutoTokenizer.from_pretrained(LLM_ID)
tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
ASK = ("Translate the following Bangla sentence into English. It may be written in a regional dialect of Bangladesh "
       "rather than in Standard Bangla. Reply with the English translation only.\n\nSentence: ")


@torch.no_grad()
def translate(sents, bs=16, max_new=48):
    out = []
    for i in range(0, len(sents), bs):
        msgs = [tok.apply_chat_template([{"role": "user", "content": ASK + s}], tokenize=False, add_generation_prompt=True)
                for s in sents[i:i + bs]]
        enc = tok(msgs, return_tensors="pt", padding=True).to(llm.device)
        gen = llm.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        out += [t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


t0 = time.time()
probe["english"] = translate(probe["sentence"].tolist())
print(f"translated {len(probe)} sentences in {time.time() - t0:.0f} s", flush=True)
print("@@TRANS@@" + json.dumps(probe[["label", "sentence", "english"]].to_dict("records"), ensure_ascii=False), flush=True)
