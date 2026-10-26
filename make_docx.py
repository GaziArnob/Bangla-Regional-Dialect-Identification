"""Word (.docx) version of the mid-term proposal. Keep text in sync with make_proposal.py."""
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

OUT = "CSC4233_SecB_GXX_Proposal.docx"  # rename GXX -> your group number
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(14)
sec.top_margin, sec.bottom_margin = Mm(10), Mm(9)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Arial", Pt(8)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st.paragraph_format.space_after = Pt(1.5)
st.paragraph_format.space_before = Pt(0)
st.paragraph_format.line_spacing = 1.0


def rich(p, text, size=None, color=None):
    """Tiny markup: <b>..</b>, <i>..</i>, &amp;."""
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    for tok in re.split(r"(</?[bi]>)", text):
        if tok in ("<b>", "<i>", "</b>", "</i>"):
            rich.state[tok.strip("</>")] = not tok.startswith("</")
            continue
        if not tok:
            continue
        r = p.add_run(tok)
        r.bold, r.italic = rich.state["b"], rich.state["i"]
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    rich.state = {"b": False, "i": False}


rich.state = {"b": False, "i": False}


def para(text, size=None, after=None, container=None, align=None):
    p = (container or doc).add_paragraph()
    rich(p, text, size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    return p


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(4), Pt(1.5)
    r = p.add_run(text)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), NAVY
    p.paragraph_format.keep_with_next = True


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def set_cell(cell, text, size=7.5, bold=False, center=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    rich(p, ("<b>%s</b>" % text) if bold else text, size)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(rows, widths_mm, header=True, size=7.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j)
            cell.width = Mm(widths_mm[j])
            set_cell(cell, c, size, bold=(header and i == 0))
            if header and i == 0:
                shade(cell, "E8EEF6")
    return t


METH_ROWS = [('Component', 'Methods and details'), ('EDA', 'Class distribution, length statistics, top words per class before/after preprocessing, at least 3 figures.'), ('Preprocessing', 'One reusable function: regex cleaning (URL, HTML, emoji, digits; we report how rare these are in this corpus), Unicode NFC normalisation, Bangla tokeniser and stop-word list, optional Bangla stemming (limitations discussed).'), ('Ablation', 'At least 2 variants (stop-words kept vs removed, stemmed vs not, digits kept vs masked); effect on macro-F1.'), ('Representations', 'Bag of Words; TF-IDF (word unigram vs unigram+bigram, plus character n-grams); averaged Word2Vec (trained on the corpus) or pretrained FastText Bangla vectors.'), ('Classical classifiers', 'Majority-class baseline; Multinomial Naive Bayes (BoW, TF-IDF); Logistic Regression and Linear SVM on all three representations (Logistic Regression / Gaussian NB on embeddings); one hyper-parameter tuned per model on validation (alpha, C, max_features).'), ('Pre-trained transformers', 'BanglaBERT, BanglishBERT (Bangla-English bilingual) and XLM-R-base, fine-tuned end-to-end on the same pipeline and splits (Colab GPU, max length 64 since sentences are short; learning rate and epochs tuned on validation). They enter the same comparison table, confusion matrix and error analysis.'), ('Optional', 'Class-conditional character n-gram language models with smoothing, classified by perplexity (only if the instructor confirms it is in scope).'), ('Protocol', 'Stratified 70/15/15 split, fixed seed, vectorisers fitted on training data only, test set used once.'), ('Evaluation', 'Accuracy, macro-precision, macro-recall, macro-F1 in one comparison table; confusion matrix of the best model.'), ('Error analysis', 'Top-10 informative words per class (Naive Bayes log-probability ratio); at least 20 misclassified examples grouped into error types.'), ('Deliverable', 'One Colab notebook (Run all) and an IEEE report (max 10 pages).')]

# ---------------------------------------------------------------- content
p = doc.add_paragraph()
r = p.add_run("Bangla Regional Dialect Identification: Classical Pipelines versus Transformer Models")
r.bold, r.font.size = True, Pt(14)
p.paragraph_format.space_after = Pt(1)
para("CSC 4233 NLP, Section B  |  Mid-Term Project Proposal (M2)  |  Group G__  |  Members: [Name, ID], [Name, ID], [Name, ID]", 7.5)

heading("1. Problem and use case")
para("Given a short Bangla sentence, predict which regional dialect it is written in (multi-class text classification, 12 classes). "
     "Use case: routing and handling regional-language input in Bangla chatbots, call-centre transcripts and speech-to-text "
     "pipelines, and measuring how well standard-Bangla tools cope with regional speech. Beyond accuracy, we examine which "
     "preprocessing choices help or hurt, because dialect cues often sit in the very words (verb endings, function words) that "
     "standard preprocessing removes.")

heading("2. Dataset")
para("<b>Main dataset: BanglaDial</b> (Mahi & Khan, Mendeley Data, v2, 2025, DOI 10.17632/sx6ybcps2n.2), <b>CC BY 4.0</b> "
     "(academic use allowed). 60,729 sentences, 12 classes, mean 6 words per sentence. <b>Imbalanced</b>: Kishoreganj 8,691, "
     "Chittagong 8,660, Narail 7,743, Rangpur 5,878, Narsingdi 5,735, Tangail 5,410, Standard Bangla 4,403, Barisal 4,046, "
     "Sylhet 3,710, Mymensingh 3,096, Noakhali 2,462, Rajshahi 885 (about 10:1), so macro-F1 is the primary metric. "
     "The notebook will read the CSV from a public GitHub link (to be added).")
table([
    ["Dataset (all Mendeley Data, all CC BY 4.0)", "Size / content", "Role", "Link"],
    ["BanglaDial", "60,729 sent., 12 labels", "Main", "data.mendeley.com/datasets/sx6ybcps2n"],
    ["Vashantor", "12,500 pairs, 5 dialects, train/val/test", "Cross-source check", "data.mendeley.com/datasets/bj5jgk878b"],
    ["ONUBAD", "980 sent. x 3 dialects, with standard Bangla", "Cross-source check", "data.mendeley.com/datasets/6ft99kf89b"],
    ["Bangla Dialect Dataset", "~4,000 pairs, 6 dialects", "Cross-source check", "data.mendeley.com/datasets/sm63ryv5dt"],
    ["BD-Dialect", "Standard + 5 dialects, aligned", "Related / future", "data.mendeley.com/datasets/k769s4vk5z"],
    ["Parallel Corpus (Dialect Bias Eval.)", "15,376 pairs, 4 dialects", "Related / future", "data.mendeley.com/datasets/42gvc83xkg"],
], [44, 52, 28, 54])
para("Our own audit shows BanglaDial already contains about 99% of the sentences of Vashantor, ONUBAD and the Bangla Dialect "
     "Dataset, plus 427 repeated rows and 173 sentences carrying more than one label. We therefore deduplicate and drop "
     "conflicting labels <i>before</i> splitting, and use the other sources only after removing shared sentences. "
     "The Bangla Regional Dialects Speech Dataset is not used (restricted access).").paragraph_format.space_before = Pt(2)

heading("3. Methodology")
steps = ["<b>1 Data</b>\nBanglaDial", "<b>2 Clean + split</b>\ndedup, 70/15/15\nstratified, seed", "<b>3 Preprocess</b>\nreusable fn +\nablation",
         "<b>4 Represent</b>\nBoW, TF-IDF,\nWord2Vec", "<b>5 Models</b>\nMajority, NB, LR, SVM\nBanglaBERT, BanglishBERT, XLM-R",
         "<b>6 Evaluate</b>\nmacro-F1, confusion\nmatrix", "<b>7 Errors</b>\ntop words,\n20+ cases"]
cells = []
for i, s in enumerate(steps):
    cells.append(s)
    if i < len(steps) - 1:
        cells.append("»")
widths = []
for i in range(len(cells)):
    widths.append(5 if i % 2 else (36 if i == 8 else 22))
ft = doc.add_table(rows=1, cols=len(cells))
ft.alignment = WD_TABLE_ALIGNMENT.CENTER
ft.autofit = False
for j, c in enumerate(cells):
    cell = ft.cell(0, j)
    cell.width = Mm(widths[j])
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    lines = c.split("\n")
    for k, ln in enumerate(lines):
        rich(p, ln, 6.5)
        if k < len(lines) - 1:
            p.add_run().add_break()
    if j % 2 == 0:
        shade(cell, "E8EEF6")
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for side in ("top", "left", "bottom", "right"):
            e = OxmlElement("w:" + side)
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "1F3A5F")
            b.append(e)
        tcPr.append(b)
doc.add_paragraph().paragraph_format.space_after = Pt(0)

table(METH_ROWS, [34, 148])

heading("4. Risks")
para("(i) Very short sentences (about 6 words) limit n-gram and embedding signal. (ii) Label provenance of several BanglaDial "
     "district classes is undocumented, so label noise may cap accuracy; we report this honestly. (iii) Strong class imbalance "
     "(Rajshahi: 885); handled with stratified splits, macro-F1 and class weights. (iv) Overlap between public dialect datasets "
     "can inflate results; handled by deduplication. (v) Limited time before 5 Nov: transformer fine-tuning needs GPU time, so the "
     "cross-source check is optional. (vi) The course guideline limits the mid-term to Weeks 1-5 techniques; we request the "
     "instructor's approval to include the transformer models.")

heading("5. Member roles")
table([
    ["#", "Member (Name, ID)", "Planned role", "Notebook / report sections"],
    ["1", "[Name, ID]", "Data loading, deduplication, EDA, preprocessing function and ablation",
     "Dataset, EDA (3+ figures), stratified split; regex cleaning, Bangla tokeniser / stop-words / stemming, ablation"],
    ["2", "[Name, ID]", "Representations, classifiers, tuning, error analysis",
     "BoW, TF-IDF, Word2Vec; Naive Bayes, LR, SVM; hyper-parameters; informative words, misclassified examples"],
    ["3", "[Name, ID]", "Transformer fine-tuning, evaluation, report",
     "BanglaBERT, BanglishBERT, XLM-R; comparison table, confusion matrix; report writing and IEEE formatting"],
], [7, 38, 55, 82])
para("All members write and can explain every notebook section; each member's tasks and percentage contribution are signed off "
     "in the report appendix.", 7.2).paragraph_format.space_before = Pt(2)

doc.core_properties.title = "Bangla Regional Dialect Identification (Proposal)"
doc.save(OUT)
print("saved", OUT)
