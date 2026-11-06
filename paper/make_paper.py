"""Research paper (IEEE-style two-column Word document): classical pipelines vs. fine-tuned transformers
(BanglaBERT, BanglishBERT, XLM-R) for Bangla dialect identification. Code: BanglaDialect_Transformers.ipynb;
numbers: colab_results.json (Colab T4 run); figures: make_figures.py."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = "BanglaDialect_Transformers_Paper.docx"
GITHUB = "https://github.com/GaziArnob/Bangla-Regional-Dialect-Identification"
TODO = RGBColor(0xC0, 0x00, 0x00)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.625)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(10)
rpr = st.element.get_or_add_rPr()
rf = rpr.find(qn("w:rFonts"))
if rf is None:
    rf = OxmlElement("w:rFonts"); rpr.append(rf)
for a in ("w:ascii", "w:hAnsi", "w:eastAsia"):
    rf.set(qn(a), "Times New Roman")
rf.set(qn("w:cs"), "Nirmala UI")                         # Bangla text
pf = st.paragraph_format
pf.space_after, pf.space_before, pf.line_spacing = Pt(0), Pt(0), 1.0


def columns(section, n):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")):
        sp.remove(c)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(n)); c.set(qn("w:space"), "360")
    sp.append(c)


def P(text="", size=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, after=0, before=0, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.first_line_indent = Inches(0.17) if indent else None
    p.paragraph_format.space_after, p.paragraph_format.space_before = Pt(after), Pt(before)
    if text:
        add(p, text, size, bold, italic, color)
    return p


def add(p, text, size=10, bold=False, italic=False, color=None):
    """Mini markup: **bold**, *italic*, [[TODO text]] in red."""
    import re
    for tok in re.split(r"(\*\*.+?\*\*|\*.+?\*|\[\[.+?\]\])", text):
        if not tok:
            continue
        b, i, c = bold, italic, color
        if tok.startswith("**"):
            tok, b = tok[2:-2], True
        elif tok.startswith("[["):
            tok, c = "[" + tok[2:-2] + "]", TODO
        elif tok.startswith("*"):
            tok, i = tok[1:-1], True
        r = p.add_run(tok)
        r.bold, r.italic, r.font.size = b, i, Pt(size)
        if c is not None:
            r.font.color.rgb = c
    return p


def H1(num, text):
    p = P(f"{num}. {text.upper()}", 10, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=8, after=4)
    p.paragraph_format.keep_with_next = True
    for r in p.runs:
        r.font.small_caps = True
    return p


def H2(letter, text):
    p = P(f"{letter}. {text}", 10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=4, after=2)
    p.paragraph_format.keep_with_next = True
    return p


def bullet(text):
    p = P("", indent=False)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.2), Inches(-0.12)
    add(p, "• " + text)
    return p


def figure(path, caption, width=3.3):
    p = P("", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=4)
    p.add_run().add_picture(path, width=Inches(width))
    p.paragraph_format.keep_with_next = True
    c = P("", 8, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=6)
    add(c, caption, 8)


def borders(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement("w:" + side)
        val = edges.get(side)
        e.set(qn("w:val"), "single" if val else "nil")
        if val:
            e.set(qn("w:sz"), str(val)); e.set(qn("w:color"), "000000")
        b.append(e)
    tcPr.append(b)


def table(caption, rows, widths, size=7.5, bold_rows=()):
    cp = P("", 8, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=6, after=2)
    add(cp, caption, 8)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            cell = t.cell(i, j)
            cell.width = Inches(widths[j])
            cell.text = ""
            q = cell.paragraphs[0]
            q.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add(q, str(val), size, bold=(i == 0 or i in bold_rows))
            top = 8 if i == 0 else None
            bottom = 6 if i == 0 else (8 if i == len(rows) - 1 else None)
            borders(cell, top=top, bottom=bottom)
            q.paragraph_format.keep_with_next = i < len(rows) - 1
    P("", after=4, indent=False)
    return t


from results import R                                    # numbers extracted from the executed notebook

# ============================================================ title block (one column)
P("Character n-Grams, Fine-tuned Transformers, Ensembles or LLM Prompting? A Leakage-Free Comparison for 12-Class "
  "Bangla Regional Dialect Identification", 20, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=12)

body = doc.add_section(WD_SECTION.CONTINUOUS)
columns(body, 2)

# ============================================================ abstract
ab = P("", 9, indent=False, after=4)
add(ab, "*Abstract*—", 9, bold=True)
add(ab, R["abstract"], 9, bold=True)
it = P("", 9, indent=False, after=6)
add(it, "*Index Terms*—", 9, bold=True)
add(it, "Bangla dialects, dialect identification, BanglaBERT, large language models, character n-grams, TF-IDF, linear SVM.", 9, bold=True)

# ============================================================ I. Introduction
H1("I", "Introduction")
P("Bangla is spoken by more than two hundred million people, and its regional dialects differ markedly from Standard Bangla in "
  "vocabulary, pronouns, verb endings and spelling. Speakers increasingly write in their own dialect on social media, in "
  "messages and in transcripts of speech, while most Bangla NLP systems are trained on standard text and degrade on dialectal "
  "input [5]. Identifying the dialect of a sentence is a first step towards routing such input to suitable models, towards "
  "dialect-aware chatbots and speech-to-text post-processing, and towards measuring how well standard tools cope with "
  "regional language.")
P("We treat the problem as 12-class sentence classification: given one sentence, predict one of eleven regional dialects or "
  "Standard Bangla. The task is hard because the evidence is small (about six words per sentence), several dialects are "
  "geographic neighbours that share vocabulary, and the strongest cues are function words and inflections. Pre-trained "
  "transformers such as BanglaBERT [15] dominate Bangla benchmarks, but they are pre-trained mostly on standard text, "
  "and in the shared tasks on similar languages and dialects linear models on character n-grams have remained hard to beat [11]. "
  "Large language models (LLMs) are now often used as zero-shot classifiers, yet they favour the standard variety of "
  "Bangla [7], [9]. The only published benchmark on BanglaDial concluded that a linear SVM beats fine-tuned BanglaBERT [22].")
P("Our central finding is that **the answer to “classical or neural?” for Bangla dialect identification is decided less "
  "by the model than by three evaluation choices** that are rarely controlled: how long the neural models are trained, "
  "whether the test data is really unseen, and whether the model was ever exposed to dialect text. Each choice can reverse "
  "or inflate a conclusion. A four-epoch default makes BanglaBERT look worse than the SVM; an adequate budget makes it "
  "better. A corpus that looks independent can share most of its sentences with the training corpus. A general LLM that "
  "understands Standard Bangla reads dialect sentences as Standard Bangla and performs at chance. Under a protocol that "
  "controls all three, character n-grams and transformers turn out to be complementary rather than competing, and "
  "their ensemble is the strongest system.")
P("Our contributions are:")
bullet("a leakage-free protocol on BanglaDial [10] that removes duplicates and conflicting labels **before** splitting, so that no sentence appears in both training and test data;")
bullet("a preprocessing ablation showing that standard Bangla preprocessing removes dialect cues;")
bullet("a comparison of Bag of Words, word and character TF-IDF and averaged Word2Vec with Naive Bayes, Logistic Regression and linear SVM, on a single split and with five-fold cross-validation;")
bullet("fine-tuning of BanglaBERT, BanglishBERT [15] and XLM-R [17] under three training budgets and three seeds, with McNemar and bootstrap significance tests, which shows that the budget reverses the published conclusion that the SVM is better [22];")
bullet("probability-averaging ensembles of the SVM and the transformers, which reach the best macro-F1 (0.834 ± 0.002) and are significantly better than every single model;")
bullet("a cross-corpus test on two corpora that were not merged into BanglaDial, which reveals that one of them shares 90% of its sentences with BanglaDial and that the character SVM transfers better than every transformer on the independent ChatgaiyyaAlap corpus and is on par on ANCHOLIK-NER;")
bullet("three open LLMs (Qwen2.5-7B, Qwen2.5-14B, BLOOMZ-7B1) under English and Bangla prompts, shuffled options and few-shot examples, plus QLoRA fine-tuning and a translation probe, which show that the LLMs read dialectal Bangla as Standard Bangla and perform at chance level, while QLoRA on the full training set reaches 0.721 but still trails the SVM.")

# ============================================================ II. Related work
H1("II", "Related Work")
P("**Dialect corpora.** Resource building for Bangla dialects has mostly aligned dialect text with Standard Bangla. Vashantor [4] "
  "established dialect-to-standard translation for five regional dialects. ONUBAD [1] covers the Chittagong, Sylhet and Barisal "
  "dialects at word, clause and sentence level. ChatgaiyyaAlap [2] provides 4,012 Chittagonian–Standard sentence pairs from "
  "social media. BanglaDial [10], the corpus we use, merges four public corpora (Vashantor, ONUBAD, Bhashamul/REGIPA and the "
  "Bangla Dialect Dataset) into 60,729 labelled sentences for twelve classes and is explicitly imbalanced. Because it "
  "absorbs most existing dialect corpora, a cross-corpus test on BanglaDial is only meaningful on data that was not merged "
  "into it, and even corpora released separately can share text with it (Section VI-G).")
P("**Dialect and language identification.** Mahi et al. [3] built MELD for three ethnic languages written in Bengali script and "
  "reached 89–93% accuracy for language identification with conventional classifiers. In the VarDial evaluation campaigns on "
  "discriminating similar languages and dialects, linear classifiers on character n-grams were consistently strong [11]. "
  "The Vashantor authors also identify the region of origin of five dialects with a fine-tuned BanglaBERT [4]. On "
  "BanglaDial, the dataset paper reports no classification experiments [10]; the only benchmark we found, by Sadiq et al. "
  "[22], reports 82.03% accuracy for a LinearSVC on combined word and character TF-IDF, ahead of a fine-tuned BanglaBERT. "
  "It reports accuracy only, and its split and duplicate handling are not described, so its numbers are not directly "
  "comparable with ours (Section VI-B).")
P("**Pre-trained models and dialects.** BanglaBERT [15] is an ELECTRA [16] discriminator pre-trained on a large Bangla web "
  "corpus; BanglishBERT is the same architecture pre-trained on Bangla and English. Multilingual encoders such as XLM-R [17] "
  "cover Bangla as one of about one hundred languages. ANCHOLIK-NER [5] shows that models trained on Standard Bangla degrade "
  "under dialectal variation, and BIDWESH [6] makes the same point for hate-speech detection. Large language models prompted "
  "for dialect translation prefer the standard form [7], [8], and treat Bengali sociolects unevenly [9].")
P("**Gap.** Existing results on Bangla dialect identification come from single runs, report accuracy rather than "
  "macro-F1 on an imbalanced task, leave the training budget of the neural baselines unexamined, and do not test on "
  "independent data. We compare classical pipelines, fine-tuned transformers, their ensembles and LLMs under one "
  "leakage-free protocol with several seeds, significance tests and a cross-corpus test.")

# ============================================================ III. Dataset
H1("III", "Dataset")
P("**Source and licence.** BanglaDial (Mahi and Khan, Mendeley Data, v2, 2025, DOI 10.17632/sx6ybcps2n.2) is released under "
  "CC BY 4.0 [10], [26]. Each row holds one sentence and its class; ten rows that were empty after Unicode normalisation were dropped, "
  "leaving 60,719 rows.")
P("**Cleaning before splitting.** Sentences were compared after NFC normalisation, ignoring punctuation and repeated spaces. "
  "182 distinct sentences (486 rows) appear under more than one dialect label and were removed, and 353 extra copies of exact "
  "duplicates were removed. Without this step a random split places identical sentences in training and test data and inflates "
  "every score. 59,880 sentences remain (Table I).")
table("TABLE I\nData preparation and split", [
    ["Step", "Sentences"],
    ["Rows in the file", "60,719"],
    ["Removed: conflicting labels (182 sentences)", "−486"],
    ["Removed: repeated copies", "−353"],
    ["Used in the experiments", "59,880"],
    ["Train / validation / test (70/15/15)", "41,916 / 8,982 / 8,982"],
], [2.25, 1.15])
P("**Class distribution and length.** The data is imbalanced (Fig. 1): Chittagong has 8,609 sentences and Rajshahi only 885, "
  "a ratio of 9.7 : 1, so macro-F1 is the primary metric. Sentences are short, 6.0 words on average (median 5), and 24% of the "
  "test sentences have three words or fewer. The text is already clean: no URLs or HTML, digits in 0.39% of sentences, and "
  "punctuation removed by the dataset creators.")
figure("fig/class_dist.png", "Fig. 1. Class distribution after cleaning (59,880 sentences).")

# ============================================================ IV. Methods
H1("IV", "Methods")
H2("A", "Preprocessing for the classical models")
P("One reusable function implements every step, each switchable for the ablation: (1) Unicode NFC normalisation; (2) "
  "regular-expression removal of URLs, HTML and digits (or digits kept as a NUM token); (3) a Bangla tokeniser that keeps maximal "
  "runs of Bengali-block characters and, unlike a \\w-based pattern, never splits a word at a vowel sign; (4) optional removal "
  "of the NLTK Bengali stop-words; and (5) optional rule-based suffix stripping. Five variants are compared on the validation "
  "set (Section VI-A) and the best is used for all classical models.")
H2("B", "Classical representations and classifiers")
P("**Bag of Words** counts words seen at least twice. **TF-IDF word** features use unigrams or unigrams plus bigrams with "
  "sublinear term frequency; **TF-IDF character** features use 1–4-grams inside word boundaries. **Word2Vec** [12]: "
  "100-dimensional skip-gram vectors trained on the training sentences and averaged per sentence. Classifiers are a "
  "majority-class baseline, Multinomial Naive Bayes (MNB) on sparse features, Logistic Regression (LR) and linear SVM on all "
  "representations, and Gaussian NB on Word2Vec. LR and SVM use balanced class weights. One hyper-parameter per model was "
  "tuned on validation: MNB α ∈ {0.05, 0.2, 1}, LR C ∈ {1, 10}, SVM C ∈ {0.1, 1}, Gaussian NB var_smoothing ∈ "
  "{10⁻⁹, 10⁻⁶} (scikit-learn [13], gensim [14]).")
H2("C", "Fine-tuned transformers")
P("We fine-tune three pre-trained encoders end to end with a linear classification head over the 12 classes: "
  "**BanglaBERT** (csebuetnlp/banglabert) and **BanglishBERT** (csebuetnlp/banglishbert) [15], ELECTRA-base discriminators "
  "with about 110 M parameters pre-trained on Bangla and on Bangla plus English, and **XLM-R base** (xlm-roberta-base) [17], "
  "a multilingual RoBERTa model covering about 100 languages. BanglaBERT and BanglishBERT receive the sentence after the "
  "csebuetnlp text normaliser that was used during their pre-training; XLM-R receives the NFC-normalised sentence. No "
  "stop-word removal or stemming is applied, since the subword tokenisers handle morphology themselves. The three models "
  "test whether Bangla-specific pre-training, additional English pre-training, or broad multilingual pre-training suits "
  "dialectal text best.")
P("**Training recipe** (identical for all three): AdamW [18] with weight decay 0.01, linear decay with 10% warm-up, "
  "batch size 32, maximum length 64 subword tokens (enough for almost every sentence), gradient clipping at 1.0 and fp16 "
  "mixed precision. The loss is class-weighted cross-entropy with weights n/(K·n_k), the same balancing as for LR and SVM. "
  "We compare three training budgets. The **initial** budget is a common default: learning rate 3·10⁻⁵ and at most four "
  "epochs. The **extended** budget searches the learning rate in {2·10⁻⁵, 5·10⁻⁵} and trains for up to eight epochs with "
  "early stopping (patience two). Because 5·10⁻⁵ was selected for all three models, the **final** budget fixes it and "
  "trains for up to 15 epochs (patience three) with three seeds (42, 1, 2); unless stated otherwise, transformer results "
  "are the mean ± standard deviation over these seeds. In every budget the model is scored on the validation set after "
  "each epoch, the best epoch is kept, and the test set is used once with those weights (Hugging Face Transformers [19]).")
H2("D", "Large language model prompting")
P("To test whether general-purpose large language models can identify the dialect without task-specific training, we "
  "prompt three open models: **Qwen2.5-7B-Instruct** and **Qwen2.5-14B-Instruct** [20], and **BLOOMZ-7B1** [23], whose "
  "multilingual pre-training and instruction tuning include Bengali. They are loaded in 4-bit NF4 quantisation [21] so "
  "that they fit on T4 GPUs. The prompt states that the sentence is written in a regional "
  "dialect of Bangladesh or in Standard Bangla, lists the 12 classes as lettered options A–L, and asks for the letter "
  "only. Instead of parsing generated text, we read the logits of the 12 letter tokens at the first answer position and "
  "take the highest; every prediction is therefore a valid class. In the **zero-shot** setting the prompt contains only "
  "the instruction and the sentence. In the **few-shot** setting it also contains 12 labelled training sentences, one per "
  "dialect, drawn with the fixed seed and shown in random order. Two further zero-shot settings test the prompt itself: "
  "the same prompt written in **Bangla**, and **shuffled options**, where every sentence gets its own random order of "
  "the 12 options, so that a bias towards one option position can be separated from a bias towards one class. "
  "Finally, to separate the model's capacity from the lack "
  "of task data, we fine-tune it with **QLoRA** [21]: low-rank adapters (rank 16, α = 32, dropout 0.05) on all attention "
  "and feed-forward projections of the 4-bit model. A first run trains for one epoch on 2,400 training sentences (200 "
  "per dialect) with the zero-shot prompt, learning rate 5·10⁻⁵ (10% warm-up, linear decay), batch size 8 and fp16 loss "
  "scaling. A second run uses all 41,916 training sentences for one epoch with a short prompt without the option list, "
  "learning rate 10⁻⁴ and effective batch size 16; the adapter with the best macro-F1 on 600 validation sentences, "
  "checked every 500 steps, is kept. In both runs the loss is the cross-entropy over "
  "the 12 letter logits at the answer position, i.e. exactly the quantity used for prediction. To test whether the LLM "
  "understands dialect sentences at all, the base model also translates 120 test sentences (10 per class) into English "
  "with greedy decoding (Section VI-F).")
H2("E", "Ensembles")
P("Because the classical and the transformer models make different errors (Section VII-C), we combine them by averaging "
  "class probabilities. The selected character SVM is turned into a probabilistic classifier with three-fold sigmoid "
  "calibration on the training set; transformer probabilities are the softmax of the logits of the selected epoch. The "
  "ensemble prediction is the argmax of w·p_transformer + (1−w)·p_SVM, where p_transformer is one transformer or the mean "
  "of all three, and the weight w ∈ {0, 0.1, …, 1} is chosen on the validation set.")

# ============================================================ V. Experimental setup
H1("V", "Experimental Setup")
P("All models share one stratified 70/15/15 split with seed 42. Vectorisers, vocabularies and Word2Vec are fitted on the "
  "training part only; hyper-parameters and the transformer epoch are chosen on validation; each selected model is evaluated "
  "once on the 8,982 test sentences. The classical models are also evaluated with stratified five-fold cross-validation over "
  "all 59,880 sentences, refitting every vectoriser and embedding model inside each fold and reusing the validation-selected "
  "hyper-parameters; fine-tuning five times per model was outside our compute budget, so the transformers are evaluated on "
  "the single split with three seeds instead. Differences are tested on the 8,982 test sentences with the exact McNemar "
  "test on the discordant predictions of two models [24], run separately for each seed, and with a paired bootstrap "
  "(1,000 resamples of the test set) for the difference in macro-F1, which gives a 95% confidence interval [25]. For the "
  "cross-corpus test, the models trained on BanglaDial are applied unchanged to ANCHOLIK-NER [5] (Standard Bangla and "
  "five dialects, CC BY 4.0) and ChatgaiyyaAlap [2] (Chittagong and Standard Bangla, CC BY 4.0), neither of which is "
  "among BanglaDial's sources; every sentence whose normalised form also occurs in BanglaDial is removed first, and "
  "macro-F1 is computed over the classes present in each corpus. LLM inference is slow on a T4 (about 5 s per 16 zero-shot prompts), so the LLM settings are "
  "compared on a stratified subset of the test set with 100 sentences per dialect (1,200 sentences), on which the best "
  "supervised model is re-scored for a like-for-like comparison; zero-shot prompting is additionally evaluated on the "
  "full test set. "
  "Metrics are accuracy and macro-averaged F1, the primary metric (macro-precision and recall are reported in the "
  "notebook). "
  + R["hardware"])

# ============================================================ VI. Results
H1("VI", "Results")
H2("A", "Preprocessing ablation")
P("Table II compares five pipeline variants for the classical models, with TF-IDF unigram+bigram features and fixed "
  "hyper-parameters, on the validation set. Every step beyond tokenisation hurts or does nothing: stop-word removal costs "
  "2.3 points of macro-F1, stemming 0.5 and both together 2.9. Keeping digits as a NUM token is equal to deleting them "
  "(+0.0004) and was selected as the pipeline for all classical models. The NLTK stop-word list was built for Standard "
  "Bangla, so removing it deletes the pronouns, particles and auxiliaries in which dialects differ, and the suffix "
  "stripper merges the dialect-specific verb endings that it was not written for.")
table("TABLE II\nPreprocessing ablation (validation macro-F1)", [
    ["Variant", "MNB", "SVM", "Mean", "Δ vs A"],
    ["A Tokeniser only", "0.744", "0.749", "0.747", "–"],
    ["B + stop-word removal", "0.724", "0.724", "0.724", "−0.023"],
    ["C + stemming", "0.738", "0.746", "0.742", "−0.005"],
    ["D + stop-words + stemming", "0.716", "0.719", "0.718", "−0.029"],
    ["E Digits kept as NUM", "0.745", "0.749", "0.747", "+0.000"],
], [1.45, 0.45, 0.45, 0.45, 0.55])

H2("B", "Classical models, transformers and ensembles")
P("Table III and Fig. 2 give the test results; transformers and ensembles are averaged over three seeds under the final "
  "budget. Among the classical models the linear SVM on character 1–4-gram TF-IDF is best (macro-F1 0.781, accuracy "
  "0.793), followed by Logistic Regression on the same features (0.777). Character features beat the best word features "
  "by about 4 points with SVM and LR (1 point with MNB), and averaged Word2Vec vectors are the weakest representation "
  "(≤0.591), because averaging six or fewer word vectors loses the spelling detail that separates dialects. All "
  "fine-tuned transformers beat the SVM: BanglaBERT reaches 0.806 ± 0.002, BanglishBERT 0.801 ± 0.004 and XLM-R "
  "0.791 ± 0.003. The ensembles are better still: the SVM combined with one transformer reaches 0.810–0.820, and the "
  "SVM combined with all three transformers reaches **0.834 ± 0.002** macro-F1 and 0.837 accuracy, 5.2 points above the "
  "SVM and 2.7 points above BanglaBERT. Averaging the three transformers without the SVM gives 0.828 ± 0.003, so the "
  "character model adds information that the transformers do not have. The seed-to-seed standard deviations "
  "(0.001–0.004) are an order of magnitude smaller than these gaps.")
table("TABLE III\nTest results (8,982 sentences), sorted by macro-F1; transformers and ensembles: mean ± std over 3 seeds", [
    ["Model", "Acc", "Macro-F1"],
    ["SVM + all three transformers", "0.837", "0.834 ± 0.002"],
    ["All three transformers (no SVM)", "0.828", "0.828 ± 0.003"],
    ["SVM + BanglaBERT", "0.825", "0.820 ± 0.001"],
    ["SVM + BanglishBERT", "0.821", "0.817 ± 0.001"],
    ["SVM + XLM-R", "0.814", "0.810 ± 0.002"],
    ["BanglaBERT (fine-tuned)", "0.809", "0.806 ± 0.002"],
    ["BanglishBERT (fine-tuned)", "0.803", "0.801 ± 0.004"],
    ["XLM-R base (fine-tuned)", "0.792", "0.791 ± 0.003"],
    ["char (1,4) + SVM", "0.793", "0.781"],
    ["char (1,4) + LR", "0.785", "0.777"],
    ["char (1,4) + MNB", "0.775", "0.756"],
    ["BoW + MNB", "0.769", "0.755"],
    ["word (1,1) + MNB", "0.765", "0.749"],
    ["word (1,1) + SVM", "0.757", "0.742"],
    ["word (1,2) + SVM", "0.750", "0.734"],
    ["BoW + SVM", "0.746", "0.732"],
    ["Word2Vec + LR", "0.622", "0.591"],
    ["Word2Vec + Gaussian NB", "0.552", "0.512"],
    ["Majority class", "0.144", "0.021"],
], [1.9, 0.5, 0.95], bold_rows=(1,))
figure("fig/models.png", "Fig. 2. Test macro-F1 of all models: classical (blue), fine-tuned transformers (orange) and ensembles (green); error bars: std over 3 seeds.")
P("**Significance.** Table IV tests the main differences. The ensemble is better than BanglaBERT and than the SVM in "
  "every seed (McNemar p < 10⁻¹⁸), with bootstrap intervals of +0.025 to +0.039 and +0.047 to +0.062 macro-F1. "
  "BanglaBERT beats the SVM in every seed (p ≤ 4·10⁻⁴; +0.013 to +0.031). BanglishBERT beats the SVM in two of three "
  "seeds. XLM-R and the SVM are not distinguishable by McNemar's test in any seed (p ≥ 0.63), although the bootstrap "
  "interval for macro-F1 (+0.003 to +0.022) excludes zero: XLM-R makes as many correct predictions as the SVM but "
  "distributes them more evenly over the classes. BanglaBERT and BanglishBERT are not reliably different (p < 0.05 in "
  "one of three seeds; bootstrap −0.011 to +0.006), so the additional English pre-training neither helps nor hurts.")
table("TABLE IV\nSignificance of the main differences (test set): McNemar p per seed and paired-bootstrap 95% CI of Δ macro-F1 (seed 42)", [
    ["A vs. B", "p (seeds 42/1/2)", "Δ F1 [95% CI]"],
    ["Ensemble vs. BanglaBERT", "<10⁻¹⁸ (all)", "+0.032 [0.025, 0.039]"],
    ["Ensemble vs. SVM", "<10⁻³⁷ (all)", "+0.054 [0.047, 0.062]"],
    ["SVM+BanglaBERT vs. BanglaBERT", "<10⁻⁹ (all)", "+0.017 [0.012, 0.022]"],
    ["BanglaBERT vs. SVM", "4e-4 / 4e-5 / 8e-6", "+0.022 [0.013, 0.031]"],
    ["BanglishBERT vs. SVM", "0.005 / 0.002 / 0.11", "+0.025 [0.015, 0.034]"],
    ["XLM-R vs. SVM", "0.81 / 0.63 / 0.93", "+0.012 [0.003, 0.022]"],
    ["BanglaBERT vs. BanglishBERT", "0.45 / 0.22 / 0.002", "−0.002 [−0.011, 0.006]"],
    ["BanglaBERT vs. XLM-R", "8e-4 / 1e-6 / 7e-6", "+0.010 [0.001, 0.019]"],
], [1.55, 1.0, 0.95])
P("**The training budget decides the comparison.** Table V contrasts the three budgets. Under the initial budget (four "
  "epochs, learning rate 3·10⁻⁵) all three transformers stay below the SVM (0.736–0.756) and lead to the same conclusion "
  "as the earlier benchmark [22]; with longer training every model gains 4–7 points and overtakes the SVM. The final "
  "budget adds little over eight epochs (+0.003 to +0.007), and the best validation epoch was 13–15 (Fig. 3), so the "
  "models are close to, but not entirely at, convergence. A 15-epoch run costs about 32 minutes for BanglaBERT and "
  "BanglishBERT and 60 minutes for XLM-R on a T4 GPU, against seconds on a CPU for the SVM.")
table("TABLE V\nEffect of the fine-tuning budget (test macro-F1; final budget: mean ± std over 3 seeds)", [
    ["Model", "Initial (4 ep.)", "Extended (8 ep.)", "Final (15 ep.)"],
    ["BanglaBERT", "0.736", "0.802", "0.806 ± 0.002"],
    ["BanglishBERT", "0.756", "0.798", "0.801 ± 0.004"],
    ["XLM-R base", "0.737", "0.784", "0.791 ± 0.003"],
    ["char (1,4) + SVM", "0.781", "0.781", "0.781"],
], [1.15, 0.7, 0.75, 0.85])
figure("fig/curves.png", "Fig. 3. Validation macro-F1 per epoch under the final budget (mean over 3 seeds, lr 5·10⁻⁵); dashed: best classical model.")
P("**Comparison with previous work.** The only earlier benchmark on BanglaDial [22] reports 82.03% accuracy for a "
  "LinearSVC on combined word and character TF-IDF and finds it better than a fine-tuned BanglaBERT. Our character SVM "
  "reaches 79.3% accuracy and our ensemble 83.7% on a split from which duplicates and conflicting labels were removed "
  "first. The numbers are not directly comparable, because [22] reports neither macro-F1 nor how duplicates were "
  "handled; the released corpus still contains 353 duplicate copies and 486 rows with conflicting labels, so a split "
  "without this cleaning places repeated sentences in both training and test data. Our "
  "results agree with [22] under a short training budget and disagree with it once the transformers are trained longer.")

H2("C", "Cross-validation")
P("Five-fold cross-validation of the classical models (Fig. 4) confirms the single-split ranking with small standard "
  "deviations (0.002–0.007): character 1–4-grams with an SVM are best at 0.808 ± 0.004, followed by Logistic Regression "
  "on the same features (0.801 ± 0.003). The gap to the best word model (0.767, word unigram MNB) is about ten standard "
  "deviations. Cross-validated scores are about two points higher than the single-split scores, "
  "partly because each fold trains on 80% instead of 70% of the data. The transformers were evaluated on the single "
  "split with three seeds, so they should be compared with the single-split numbers in Table III.")
figure("fig/cv.png", "Fig. 4. Five-fold cross-validation of the classical models, macro-F1 (mean ± std).")

H2("D", "Per-dialect comparison")
P("Fig. 5 compares the SVM, BanglaBERT (mean over three seeds) and the best ensemble per dialect. BanglaBERT is much "
  "better than the SVM on Standard Bangla (F1 0.890 ± 0.006 vs 0.790), Mymensingh (0.813 vs 0.747), Rajshahi (0.743 vs "
  "0.695), Noakhali (0.834 vs 0.796) and Sylhet (0.870 vs 0.845), about equal on Chittagong, Kishoreganj, Barisal and "
  "Tangail, and worse only on Narail (0.814 vs 0.835). Narsingdi varies most across seeds (0.705 ± 0.017). Under the "
  "initial budget the transformers had been clearly worse on the central dialects (BanglishBERT: Narsingdi 0.567, "
  "Tangail 0.592), so this weakness was a symptom of under-training. Per-dialect ensemble scores are available for the 8-epoch "
  "extended-budget run (seed 42), whose ensemble macro-F1 of 0.836 is close to that of the final run "
  "(0.834 ± 0.002); there the ensemble is the best or within 0.003 of the best model on every dialect; its largest gains over the SVM are on Rajshahi (+0.123), "
  "Mymensingh (+0.078), Noakhali (+0.067) and the two hardest central dialects, Narsingdi and Tangail (+0.047 each). "
  "Narsingdi and Tangail nevertheless remain the hardest classes for every model.")
figure("fig/perclass.png", "Fig. 5. Per-dialect test F1 of the SVM, BanglaBERT (mean ± std over 3 seeds) and the SVM + all-transformer ensemble (8-epoch extended-budget run, seed 42; macro-F1 0.836).")

H2("E", "Large language model prompting")
P("Table VI compares three open LLMs under four prompt settings with the character SVM on the stratified subset of "
  "1,200 test sentences (100 per dialect), on which random guessing gives a macro-F1 of about 0.083. The SVM reaches "
  "0.792 on this subset, close to its full-test score. Every LLM in every setting is at or below chance level (0.017–0.071). "
  "Neither a model twice as large (Qwen2.5-14B), nor a model with Bengali in its multilingual pre-training and "
  "instruction tuning (BLOOMZ-7B1 [23]), nor a prompt written in Bangla, nor in-context examples changes this. For a "
  "sentence checked with free text generation, the model wrote the same letter that the logit readout selected, so the "
  "low scores are not an artefact of the readout.")
table("TABLE VI\nLLM macro-F1 by prompt setting (1,200-sentence stratified test subset; English/Bangla prompt and shuffled options are zero-shot)", [
    ["Model", "English", "Bangla", "Shuffled", "Few-shot"],
    ["Qwen2.5-7B-Instruct", "0.024", "0.070", "0.034", "0.066"],
    ["Qwen2.5-14B-Instruct", "0.024", "0.017", "0.025", "0.065"],
    ["BLOOMZ-7B1", "0.036", "0.052", "0.071", "0.026"],
    ["Qwen2.5-7B + QLoRA (2,400 sent.)", "0.205", "–", "–", "–"],
    ["Qwen2.5-7B + QLoRA (41,916 sent.)", "0.715", "–", "–", "–"],
    ["char (1,4) + SVM (supervised)", "0.792", "", "", ""],
    ["Random guessing (expected)", "0.083", "", "", ""],
], [1.3, 0.52, 0.52, 0.6, 0.6], bold_rows=(6,))
P("The predictions show why. The two Qwen models read dialectal Bangla as Standard Bangla: with the English zero-shot "
  "prompt both assign 1,084 of the 1,200 sentences (90%) to Standard Bangla, and with the Bangla prompt Qwen2.5-14B does "
  "so for 1,152 (96%). This is a bias about content, not about the position of the option: with a different random "
  "option order for every sentence, Qwen2.5-7B still labels 1,007 sentences (84%) and Qwen2.5-14B 987 (82%) as Standard "
  "Bangla, and only 8% and 5% of the answers are the first option. BLOOMZ has a different default, Rangpur (818 of 1,200 "
  "zero-shot), and loses even that preference when the options are shuffled, which points to an option-order bias. "
  "In-context examples move the bias rather than remove it: with one example per dialect Qwen2.5-7B assigns 51% of the "
  "sentences to Barisal, the first option. The preference for the standard form matches what was reported for dialect "
  "translation [7], [8] and sociolect evaluation [9]. The result holds on the full test set: Qwen2.5-7B zero-shot on all "
  "8,982 sentences gives a macro-F1 of 0.025, with 8,058 sentences (89.7%) labelled Standard Bangla. The LLMs are also "
  "the most expensive models: 135–1,939 s on a T4 GPU per setting for 1,200 sentences, against well under one second on "
  "a CPU for the SVM.")
P("**Fine-tuning the LLM.** With QLoRA on 2,400 training sentences the LLM does learn the task: macro-F1 rises to 0.205 "
  "(accuracy 0.243), three times the few-shot and eight times the zero-shot score, and its predictions are no longer "
  "dominated by one class. It is best on Standard Bangla (F1 0.702) and Chittagong (0.481) but still nearly fails on "
  "Noakhali, Sylhet and Rangpur (≤0.061). Its training loss was still falling (from 3.73 to 2.13), so this run is a "
  "lower bound. A first attempt with learning rate 2·10⁻⁴ and no loss scaling diverged and collapsed to a single class "
  "(macro-F1 0.057); the reported run uses 5·10⁻⁵ with fp16 loss scaling. "
  "With all 41,916 training sentences the same model reaches a test macro-F1 of **0.721** (accuracy 0.728) on the full "
  "test set and 0.715 on the 1,200-sentence subset. Validation macro-F1 rose steadily over the epoch (0.429 after 500 "
  "steps, 0.604 after 1,000, 0.744 after 2,000) and peaked at 0.767 after 2,500 of 2,619 steps. The per-dialect pattern "
  "matches that of the encoders: Chittagong (0.897), Standard Bangla (0.842) and Barisal (0.813) are easiest, Narsingdi "
  "(0.548) and Tangail (0.531) hardest. The LLM's failure under prompting is therefore not a lack of capacity but a lack "
  "of exposure to dialect text, which supervised data removes. Even so, the fine-tuned 7B model stays 0.06 below the "
  "SVM (0.781) and 0.085 below BanglaBERT (0.806), and is far more expensive: its single epoch took 16,915 s (4.7 h) on "
  "a T4, against about 130 s for one BanglaBERT epoch and seconds for the SVM. More epochs might close part of the gap, "
  "but at this cost a small encoder is the better choice.")

H2("F", "Does the LLM understand dialectal Bangla?")
P("The LLM's failure could mean either that it understands dialect sentences but cannot name the dialect, or that it "
  "does not understand them at all. To separate the two, the base model translated 120 test sentences (10 per class) "
  "into English. The translations were rated by one of the authors, a native Bangla speaker, as correct (meaning fully "
  "conveyed), partial (gist or key words right) or wrong (wrong, untranslated or hallucinated); the rater started from "
  "a first-pass rating produced by an LLM-based assistant (Claude, Anthropic) and changed 20 of the 120 labels. A score counts a correct translation as 1 and a "
  "partial one as 0.5. Table VII shows a clear gap. Standard Bangla sentences are translated well (score 0.95, no wrong "
  "translation), but dialect sentences are not (0.29 on average; 64 of 110 wrong). Narsingdi, Rangpur and Sylhet fare "
  "best (0.45) and Chittagong and Noakhali worst (0.15).")
table("TABLE VII\nTranslation adequacy of Qwen2.5-7B-Instruct, 10 test sentences per class", [
    ["Class", "Correct", "Partial", "Wrong", "Score"],
    ["Standard Bangla", "9", "1", "0", "0.95"],
    ["Narsingdi", "3", "3", "4", "0.45"],
    ["Rangpur", "3", "3", "4", "0.45"],
    ["Sylhet", "3", "3", "4", "0.45"],
    ["Barisal", "2", "2", "6", "0.30"],
    ["Narail", "2", "2", "6", "0.30"],
    ["Tangail", "2", "2", "6", "0.30"],
    ["Kishoreganj", "0", "5", "5", "0.25"],
    ["Mymensingh", "1", "2", "7", "0.20"],
    ["Rajshahi", "1", "2", "7", "0.20"],
    ["Chittagong", "1", "1", "8", "0.15"],
    ["Noakhali", "0", "3", "7", "0.15"],
    ["All dialects (110)", "18", "28", "64", "0.29"],
], [1.25, 0.55, 0.55, 0.5, 0.5], bold_rows=(1, 13))
P("The errors are characteristic. Dialect function words are read as names or places: আফনেরে (you) is "
  "translated as *Afghanistan*, কিতা (what) as *Kita*, হরি (do) as *Hari* and ক্যা "
  "(why) as *Keynes*. Dialect pronouns and negations are misread, e.g. আরে (me) as *it* and "
  "দেই ন (did not see) as *do not give*; some sentences are transliterated instead of translated, and a few "
  "outputs are off-task. The LLM therefore does not merely fail to name the dialect: it often fails to understand "
  "dialectal Bangla at all, and its Standard Bangla label is consistent with treating unfamiliar dialect words as names "
  "or noise.")

H2("G", "Cross-corpus test")
P("Table VIII applies the BanglaDial models, unchanged, to the two external corpora. The first finding concerns the data "
  "itself: **18,773 of the 20,886 ANCHOLIK-NER sentences (90%) already occur in BanglaDial**, presumably because both "
  "draw on the same earlier dialect resources, although ANCHOLIK-NER is not listed among BanglaDial's sources. Only "
  "1,960 sentences remain, dominated by Noakhali (880) and Mymensingh (828). ChatgaiyyaAlap is essentially independent: "
  "only 34 of its 8,022 sentences occur in BanglaDial. Without this check, a cross-corpus evaluation on ANCHOLIK-NER "
  "would largely re-test the training data.")
table("TABLE VIII\nCross-corpus macro-F1 over the classes present (models trained on BanglaDial; transformers: mean ± std, 3 seeds)", [
    ["Model", "ANCHOLIK-NER (1,960; 6 cl.)", "ChatgaiyyaAlap (7,982; 2 cl.)"],
    ["SVM + all three transformers", "0.540 ± 0.004", "0.895 ± 0.007"],
    ["char (1,4) + SVM", "0.535", "0.883"],
    ["BanglishBERT", "0.529 ± 0.014", "0.847 ± 0.014"],
    ["XLM-R base", "0.523 ± 0.005", "0.848 ± 0.003"],
    ["BanglaBERT", "0.519 ± 0.005", "0.850 ± 0.014"],
], [1.45, 0.95, 0.95], bold_rows=(1,))
P("On the unseen data the ranking changes, although the transformers were better in-domain: the character SVM is "
  "better than every transformer on ChatgaiyyaAlap (0.883 vs. 0.847–0.850, a gap of several standard deviations) and "
  "on par with them on ANCHOLIK-NER (0.535 vs. 0.519–0.529, within about one standard deviation of BanglishBERT), and "
  "the ensemble is again the best model. Character n-grams thus capture spelling regularities "
  "that carry over to new text, while the transformers rely more on cues specific to BanglaDial. On ChatgaiyyaAlap the "
  "ensemble recognises Chittagonian well (F1 0.956) but labels part of the Standard Bangla side as other dialects (F1 "
  "0.831), because the models still choose among all 12 classes. On ANCHOLIK-NER Mymensingh, Noakhali, Sylhet and "
  "Standard Bangla transfer moderately (0.62–0.84), while the few Barisal and Chittagong sentences left after "
  "de-duplication are mostly misclassified (about 0.20). Absolute scores drop well below the in-domain level, so "
  "BanglaDial results overestimate performance on dialect text from other sources.")

# ============================================================ VII. Error analysis
H1("VII", "Error Analysis")
H2("A", "Confusions of the character SVM")
P("The confusion matrix of the character SVM (Fig. 6) shows that errors concentrate in a block of central dialects. The "
  "most frequent confusions are Kishoreganj→Narsingdi (99 test sentences), Narsingdi→Kishoreganj (89), and "
  "Narail→Tangail and Tangail→Kishoreganj (56 each). Chittagong (F1 0.920), Barisal (0.861) and Sylhet (0.845), whose "
  "dialects are lexically distinct, are the easiest classes; Tangail (0.666) and Narsingdi (0.684) the hardest. "
  "Rajshahi, the smallest class (133 test sentences), reaches 0.695.")
figure("fig/confusion.png", "Fig. 6. Confusion matrix of char (1,4) + SVM on the test set (% of the true class).")
H2("B", "Sentence length, unseen words and informative words")
P("Accuracy rises from 0.58 for one- or two-word sentences to 0.91 for sentences of nine or more words (Fig. 7). "
  "36.4% of the test sentences contain a word never seen in training; they are classified slightly better (0.804 vs "
  "0.786), because they are also longer, which shows that character features cope well with unseen spellings.")
figure("fig/length.png", "Fig. 7. Accuracy of char (1,4) + SVM by sentence length (dashed: overall).")
P("The most informative words of Naive Bayes (log-probability ratio) are mostly genuine dialect markers: pronouns such "
  "as আঁই, আঁর (Chittagong), হামার, মোক (Rangpur) and তুমিতাইন (Sylhet), "
  "verb forms such as করতাসে, লাগতাসে (Mymensingh) and ল্যাগা (Rajshahi), and "
  "the standard forms এনেছে, এসেছিল for Standard Bangla. For Kishoreganj, Narsingdi and Tangail, "
  "however, names and place words such as বিডিও, রাজু, নরসিংদী and "
  "টাঙ্গাইল rank high. These topic artefacts of the corpus coincide with the weakest classes.")
H2("C", "Error types and model agreement")
P("Each of the 1,863 test errors of the SVM (20.7%) was assigned one type in a fixed order (Table IX). Very short "
  "sentences cause 40% of the errors. Examples from a sample of 30 errors: মন্দিরা (one word, Rangpur) contains no "
  "dialect cue at all; তা ভুল টুল করে (Kishoreganj, predicted Standard Bangla) is written in standard wording; and "
  "তুই এনতে সর সামনে আ (Kishoreganj, predicted Narsingdi) is a confusion between neighbouring dialects.")
table("TABLE IX\nError types of char (1,4) + SVM (test set)", [
    ["Type", "Errors", "%"],
    ["A Very short (1–3 words)", "750", "40.3"],
    ["B Involves Standard Bangla", "225", "12.1"],
    ["C Frequent neighbour confusion", "311", "16.7"],
    ["D Other", "577", "31.0"],
], [2.0, 0.6, 0.6])
P("This complementarity was visible before the ensemble was built. With the four-epoch transformers of the initial "
  "budget, 57.1% of the test sentences were correct for the SVM and all three transformers and only 9.2% were wrong for "
  "all of them, while at least one transformer corrected 55.9% of the SVM's errors. These agreement figures were not "
  "recomputed for the longer-trained final models, so they only motivate the ensemble; its benefit is measured directly "
  "in Table III. The ensemble exploits exactly this: character n-grams capture spelling and endings, the transformers "
  "capture context and standard-language patterns, and averaging their probabilities fixes many errors of both. The "
  "sentences that every model gets wrong are likely to be ambiguous or mislabelled and set a practical ceiling.")

# ============================================================ VIII. Limitations
H1("VIII", "Limitations")
P("The transformers used one learning rate in the final budget and three seeds; BanglaBERT's best epoch was still the last "
  "one, so its score and those of the ensembles may be slightly underestimated. The per-dialect ensemble scores (Fig. 5) and "
  "the agreement figures of Section VII-C come from earlier single-seed runs with shorter training, not from the final "
  "three-seed run. Cross-validation was run for the classical "
  "models only and reused hyper-parameters selected on the validation split, which is slightly optimistic. The comparison "
  "with the earlier benchmark [22] is indirect, because its split, its handling of duplicates and its macro-F1 are not "
  "reported. The in-domain results come from one corpus, whose labels for several districts are not documented and which "
  "contains topic words that a model can exploit without learning the dialect; the cross-corpus test covers only six of "
  "the twelve classes, and after de-duplication the ANCHOLIK-NER set is small and unbalanced. Punctuation had been "
  "removed by the dataset creators, so its role could not be studied, and the many very short sentences set a practical "
  "ceiling for any model. The LLM results cover three open models of up to 14B parameters in 4-bit precision on a "
  "1,200-sentence subset; larger open models such as Llama-3.1 and commercial models such as GPT or Gemini were not tested, and QLoRA was "
  "trained for one epoch only, with a single seed. The translation probe covers only 120 sentences, rated by a single "
  "native speaker without reference translations; because the rater started from an LLM first pass, the ratings may be "
  "anchored to it (the rater kept 100 of the 120 labels). A second independent rater with an agreement statistic such "
  "as Cohen's κ would make the probe more reliable.")

# ============================================================ IX. Conclusion
H1("IX", "Conclusion")
P("For 12-class Bangla regional dialect identification, the answer to “classical or neural?” depends on three "
  "evaluation choices more than on the model. With a common four-epoch default, fine-tuned transformers fall below a "
  "character n-gram SVM, as an earlier benchmark found; trained for up to 15 epochs and averaged over three seeds, "
  "BanglaBERT (0.806 ± 0.002), BanglishBERT (0.801 ± 0.004) and XLM-R (0.791 ± 0.003) all beat the SVM (0.781), and "
  "the differences are significant for BanglaBERT in every seed. On unseen corpora the order reverses again: the SVM "
  "transfers better than every transformer on the independent ChatgaiyyaAlap corpus and is on par on ANCHOLIK-NER, and one supposedly external corpus turns out to share 90% of its sentences "
  "with BanglaDial. Character n-grams and transformers are complementary, and their probability-averaging ensemble is the "
  "strongest system both in-domain (0.834 ± 0.002, significantly better than any single model) and across corpora. "
  "Standard Bangla preprocessing hurts, because stop-word removal and stemming delete dialect cues. Open LLMs of up to "
  "14B parameters perform at chance level under English or Bangla prompts, shuffled options or few-shot examples, "
  "because they read dialectal Bangla as Standard Bangla; a translation probe shows that they do not understand most "
  "dialect sentences. QLoRA fine-tuning on the full training set lifts Qwen2.5-7B to 0.721, so the gap is one of "
  "exposure, not capacity, but even then it stays below the SVM, after one epoch that took longer than 15 BanglaBERT epochs. Dialect identification therefore "
  "still needs supervised, dialect-specific training, careful tuning of the neural baselines, and de-duplicated "
  "external test data. Future work will add character-aware neural models, commercial LLMs and more independent "
  "dialect corpora.")

H1("", "Use of AI Assistants").runs[0].text = "USE OF AI ASSISTANTS"
P("An LLM-based assistant (Claude, Anthropic) was used to help write the experiment code, to draft and edit the text, "
  "and to produce the first-pass translation ratings of Section VI-F, which one of the authors then reviewed and "
  "corrected. All numbers were produced by the experiment code; the authors take full responsibility for the content.")

# ============================================================ References
H1("", "References").runs[0].text = "REFERENCES"
refs = [
    "N. Sultana, R. Yasmin, B. Mallik, and M. S. Uddin, “ONUBAD: A comprehensive dataset for automated conversion of Bangla regional dialects into standard Bengali dialect,” *Data in Brief*, vol. 58, Art. no. 111276, 2025, doi: 10.1016/j.dib.2025.111276.",
    "S. Chowdhury, D. R. A. Remal, S. T. Pasha, A. Islam, and S. R. H. Noori, “ChatgaiyyaAlap: A dataset for conversion from Chittagonian dialect to standard Bangla,” *Data in Brief*, vol. 59, Art. no. 111413, 2025, doi: 10.1016/j.dib.2025.111413.",
    "M. H. Mahi, A. R. Khan, M. H. Anik, S. R. H. Noori, A. Mahmud, and M. U. Mojumdar, “MELD: A multilingual ethnic dataset of Chakma, Garo, and Marma in Bengali script with English and standard Bengali translation,” *Data in Brief*, vol. 61, Art. no. 111745, 2025, doi: 10.1016/j.dib.2025.111745.",
    "F. T. J. Faria, M. B. Moin, A. A. Wase, M. Ahmmed, M. R. Sani, and T. Muhammad, “Vashantor: A large-scale multilingual benchmark dataset for automated translation of Bangla regional dialects to Bangla language,” arXiv:2311.11142, 2023.",
    "B. Paul, F. F. Preotee, S. Sarker, S. R. Refat, S. Islam, T. Muhammad, M. A. Hoque, and S. Manzoor, “ANCHOLIK-NER: A benchmark dataset for Bangla regional named entity recognition,” *PLoS ONE*, vol. 21, no. 2, Art. no. e0342786, 2026, doi: 10.1371/journal.pone.0342786.",
    "A. H. Fayaz, M. S. Uddin, R. U. Bhuiyan, Z. Sultana, M. S. Islam, B. Paul, T. Muhammad, and S. Manzoor, “BIDWESH: A Bangla regional based hate speech detection dataset,” arXiv:2507.16183, 2025.",
    "K. M. J. Sami, D. Sumit, A. Hossain, and F. Sadeque, “A comparative analysis of retrieval-augmented generation techniques for Bengali standard-to-dialect machine translation using LLMs,” in *Proc. 2nd Workshop Bangla Lang. Process. (BLP-2025)*, 2025, pp. 266–279.",
    "T. T. Prama, “LLMs for low-resource dialect translation using context-aware prompting: A case study on Sylheti,” in *Proc. 2nd Workshop Bangla Lang. Process. (BLP-2025)*, 2025, pp. 292–308.",
    "A. T. Wasi, R. Islam, M. R. Islam, F. Sadeque, T. H. Rafi, and D.-K. Chae, “Dialectal bias in Bengali: An evaluation of multilingual large language models across cultural variations,” in *Companion Proc. ACM Web Conf. 2025*, 2025, pp. 1380–1384, doi: 10.1145/3701716.3715468.",
    "M. H. Mahi and A. R. Khan, “BanglaDial: A merged and imbalanced text dataset for Bengali regional dialect analysis,” Mendeley Data, v2, 2025, doi: 10.17632/sx6ybcps2n.2.",
    "M. Zampieri, S. Malmasi, N. Ljubešić, P. Nakov, A. Ali, J. Tiedemann, Y. Scherrer, and N. Aepli, “Findings of the VarDial evaluation campaign 2017,” in *Proc. 4th Workshop NLP for Similar Languages, Varieties and Dialects (VarDial)*, 2017, pp. 1–15.",
    "T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” arXiv:1301.3781, 2013.",
    "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
    "R. Řehůřek and P. Sojka, “Software framework for topic modelling with large corpora,” in *Proc. LREC 2010 Workshop New Challenges for NLP Frameworks*, 2010, pp. 45–50.",
    "A. Bhattacharjee, T. Hasan, W. U. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, and R. Shahriyar, “BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla,” in *Findings Assoc. Comput. Linguistics: NAACL 2022*, 2022, pp. 1318–1327.",
    "K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, “ELECTRA: Pre-training text encoders as discriminators rather than generators,” in *Proc. ICLR*, 2020.",
    "A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics*, 2020, pp. 8440–8451.",
    "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. ICLR*, 2019.",
    "T. Wolf *et al.*, “Transformers: State-of-the-art natural language processing,” in *Proc. EMNLP 2020: System Demonstrations*, 2020, pp. 38–45.",
    "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
    "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in *Proc. NeurIPS*, 2023.",
    "N. Sadiq, R. Islam, R. Ahmed, M. A. Farzana, M. R. Hasan, and A. Zami, “Bangla dialect classification across 11 regions using machine learning and transformer-based models,” in *Proc. 5th Int. Conf. Electrical, Computer & Telecommunication Engineering (ICECTE)*, 2026, pp. 1–6, doi: 10.1109/ICECTE69292.2026.11429342.",
    "N. Muennighoff *et al.*, “Crosslingual generalization through multitask finetuning,” in *Proc. 61st Annu. Meeting Assoc. Comput. Linguistics*, 2023, pp. 15991–16111.",
    "T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, 1998.",
    "T. Berg-Kirkpatrick, D. Burkett, and D. Klein, “An empirical investigation of statistical significance in NLP,” in *Proc. EMNLP-CoNLL*, 2012, pp. 995–1005.",
    "M. H. Mahi, A. R. Khan, and M. U. Mojumdar, “BanglaDial: A merged and imbalanced text dataset for Bengali regional dialect analysis,” *Data in Brief*, 2025, doi: 10.1016/j.dib.2025.112200.",
]
for i, r in enumerate(refs, 1):
    p = P("", 8, indent=False, after=1)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.25), Inches(-0.25)
    add(p, f"[{i}]\t{r}", 8)

doc.save(OUT)
print("saved", OUT)
