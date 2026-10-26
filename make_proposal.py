from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = "CSC4233_SecB_GXX_Proposal.pdf"  # rename GXX -> your group number
base = ParagraphStyle("b", fontName="Helvetica", fontSize=7.6, leading=9.2, spaceAfter=1.5)
h = ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=8.6, leading=10.5, spaceBefore=3.5, spaceAfter=1.5,
                   textColor=colors.HexColor("#1f3a5f"))
title = ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceAfter=1)
small = ParagraphStyle("s", parent=base, fontSize=6.9, leading=8.2, spaceAfter=0)
P = lambda t, s=base: Paragraph(t, s)


def boxed(fl, pad=3):
    t = Table([[fl]], colWidths=[182 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1f3a5f")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f9fc")),
                           ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                           ("TOPPADDING", (0, 0), (-1, -1), pad - 1), ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 1)]))
    return t


def flow():
    steps = ["<b>1 Data</b><br/>BanglaDial", "<b>2 Clean + split</b><br/>dedup, 70/15/15<br/>stratified, seed",
             "<b>3 Preprocess</b><br/>reusable fn +<br/>ablation", "<b>4 Represent</b><br/>BoW, TF-IDF,<br/>Word2Vec",
             "<b>5 Models</b><br/>Majority, NB, LR, SVM<br/>BanglaBERT, BanglishBERT, XLM-R",
             "<b>6 Evaluate</b><br/>macro-F1, confusion<br/>matrix", "<b>7 Errors</b><br/>top words,<br/>20+ cases"]
    cell = ParagraphStyle("c", parent=small, alignment=1, fontSize=6.4, leading=7.8)
    row, widths = [], []
    for i, st in enumerate(steps):
        row.append(Paragraph(st, cell)); widths.append(21.5 * mm if i != 4 else 36 * mm)
        if i < len(steps) - 1:
            row.append(Paragraph("<b>&raquo;</b>", cell)); widths.append(2.5 * mm)
    t = Table([row], colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
          ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1)]
    for i in range(0, len(row), 2):
        st += [("BOX", (i, 0), (i, 0), 0.6, colors.HexColor("#1f3a5f")), ("BACKGROUND", (i, 0), (i, 0), colors.HexColor("#e8eef6"))]
    t.setStyle(TableStyle(st))
    return t


def roles_table():
    hdr = ["#", "Member (Name, ID)", "Planned role", "Notebook / report sections"]
    data = [
        ["1", "[Name, ID]", "Data loading, deduplication, EDA, preprocessing function and ablation", "Dataset, EDA (3+ figures), stratified split; regex cleaning, Bangla tokeniser / stop-words / stemming, ablation"],
        ["2", "[Name, ID]", "Representations, classifiers, tuning, error analysis", "BoW, TF-IDF, Word2Vec; Naive Bayes, LR, SVM; hyper-parameters; informative words, misclassified examples"],
        ["3", "[Name, ID]", "Transformer fine-tuning, evaluation, report", "BanglaBERT, BanglishBERT, XLM-R; comparison table, confusion matrix; report writing and IEEE formatting"],
    ]
    rows = [[Paragraph("<b>%s</b>" % c, small) for c in hdr]] + [[Paragraph(c, small) for c in r] for r in data]
    t = Table(rows, colWidths=[7 * mm, 38 * mm, 55 * mm, 82 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]))
    return t


METH_ROWS = [('Component', 'Methods and details'), ('EDA', 'Class distribution, length statistics, top words per class before/after preprocessing, at least 3 figures.'), ('Preprocessing', 'One reusable function: regex cleaning (URL, HTML, emoji, digits; we report how rare these are in this corpus), Unicode NFC normalisation, Bangla tokeniser and stop-word list, optional Bangla stemming (limitations discussed).'), ('Ablation', 'At least 2 variants (stop-words kept vs removed, stemmed vs not, digits kept vs masked); effect on macro-F1.'), ('Representations', 'Bag of Words; TF-IDF (word unigram vs unigram+bigram, plus character n-grams); averaged Word2Vec (trained on the corpus) or pretrained FastText Bangla vectors.'), ('Classical classifiers', 'Majority-class baseline; Multinomial Naive Bayes (BoW, TF-IDF); Logistic Regression and Linear SVM on all three representations (Logistic Regression / Gaussian NB on embeddings); one hyper-parameter tuned per model on validation (alpha, C, max_features).'), ('Pre-trained transformers', 'BanglaBERT, BanglishBERT (Bangla-English bilingual) and XLM-R-base, fine-tuned end-to-end on the same pipeline and splits (Colab GPU, max length 64 since sentences are short; learning rate and epochs tuned on validation). They enter the same comparison table, confusion matrix and error analysis.'), ('Optional', 'Class-conditional character n-gram language models with smoothing, classified by perplexity (only if the instructor confirms it is in scope).'), ('Protocol', 'Stratified 70/15/15 split, fixed seed, vectorisers fitted on training data only, test set used once.'), ('Evaluation', 'Accuracy, macro-precision, macro-recall, macro-F1 in one comparison table; confusion matrix of the best model.'), ('Error analysis', 'Top-10 informative words per class (Naive Bayes log-probability ratio); at least 20 misclassified examples grouped into error types.'), ('Deliverable', 'One Colab notebook (Run all) and an IEEE report (max 10 pages).')]


def meth_table():
    rows = [[Paragraph("<b>%s</b>" % c if i == 0 else (("<b>%s</b>" % c) if j == 0 else c), small) for j, c in enumerate(r)]
            for i, r in enumerate(METH_ROWS)]
    tb = Table(rows, colWidths=[34 * mm, 148 * mm])
    tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]))
    return tb


story = [
    P("Bangla Regional Dialect Identification: Classical Pipelines versus Transformer Models", title),
    P("CSC 4233 NLP, Section B &nbsp;|&nbsp; Mid-Term Project Proposal (M2) &nbsp;|&nbsp; Group G__ &nbsp;|&nbsp; "
      "Members: [Name, ID], [Name, ID], [Name, ID]", small),

    P("1. Problem and use case", h),
    P("Given a short Bangla sentence, predict which regional dialect it is written in (multi-class text classification, 12 classes). "
      "Use case: routing and handling regional-language input in Bangla chatbots, call-centre transcripts and speech-to-text "
      "pipelines, and measuring how well standard-Bangla tools cope with regional speech. Beyond accuracy, we examine which "
      "preprocessing choices help or hurt, because dialect cues often sit in the very words (verb endings, function words) that "
      "standard preprocessing removes."),

    P("2. Dataset", h),
    P("<b>Main dataset: BanglaDial</b> (Mahi &amp; Khan, Mendeley Data, v2, 2025, DOI 10.17632/sx6ybcps2n.2), "
      "<b>CC BY 4.0</b> (academic use allowed). 60,729 sentences, 12 classes, mean 6 words per sentence. "
      "<b>Imbalanced</b>: Kishoreganj 8,691, Chittagong 8,660, Narail 7,743, Rangpur 5,878, Narsingdi 5,735, Tangail 5,410, "
      "Standard Bangla 4,403, Barisal 4,046, Sylhet 3,710, Mymensingh 3,096, Noakhali 2,462, Rajshahi 885 (about 10:1), "
      "so macro-F1 is the primary metric. The notebook will read the CSV from a public GitHub link (to be added)."),
    Spacer(1, 2),
]

rows = [
    ["Dataset (all Mendeley Data, all CC BY 4.0)", "Size / content", "Role", "Link"],
    ["BanglaDial", "60,729 sent., 12 labels", "Main", "data.mendeley.com/datasets/sx6ybcps2n"],
    ["Vashantor", "12,500 pairs, 5 dialects, train/val/test", "Cross-source check", "data.mendeley.com/datasets/bj5jgk878b"],
    ["ONUBAD", "980 sent. x 3 dialects, with standard Bangla", "Cross-source check", "data.mendeley.com/datasets/6ft99kf89b"],
    ["Bangla Dialect Dataset", "~4,000 pairs, 6 dialects", "Cross-source check", "data.mendeley.com/datasets/sm63ryv5dt"],
    ["BD-Dialect", "Standard + 5 dialects, aligned", "Related / future", "data.mendeley.com/datasets/k769s4vk5z"],
    ["Parallel Corpus (Dialect Bias Eval.)", "15,376 pairs, 4 dialects", "Related / future", "data.mendeley.com/datasets/42gvc83xkg"],
]
rows = [[P(c, small) for c in r] for r in rows]
t = Table(rows, colWidths=[44 * mm, 52 * mm, 28 * mm, 54 * mm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story += [t, Spacer(1, 2),
          P("Our own audit shows BanglaDial already contains about 99% of the sentences of Vashantor, ONUBAD and the Bangla Dialect "
            "Dataset, plus 427 repeated rows and 173 sentences carrying more than one label. We therefore deduplicate and drop "
            "conflicting labels <i>before</i> splitting, and use the other sources only after removing shared sentences. "
            "The Bangla Regional Dialects Speech Dataset is not used (restricted access).", base),

    P("3. Methodology", h),
    flow(),
    Spacer(1, 2),
    meth_table(),

    P("4. Risks", h),
    P("(i) Very short sentences (about 6 words) limit n-gram and embedding signal. (ii) Label provenance of several BanglaDial "
      "district classes is undocumented, so label noise may cap accuracy; we report this honestly. (iii) Strong class imbalance "
      "(Rajshahi: 885); handled with stratified splits, macro-F1 and class weights. (iv) Overlap between public dialect datasets "
      "can inflate results; handled by deduplication. (v) Limited time before 5 Nov: transformer fine-tuning needs GPU time, so the cross-source check is optional. (vi) The course guideline limits the mid-term to Weeks 1-5 techniques; we request the instructor's approval to include the transformer models."),

    P("5. Member roles", h),
    roles_table(),
    P("All members write and can explain every notebook section; each member's tasks and percentage contribution are signed off in the report appendix.", small),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=10 * mm, bottomMargin=9 * mm,
                        title="Bangla Regional Dialect Identification (Proposal)")
doc.build(story)

from pypdf import PdfReader
print("pages:", len(PdfReader(OUT).pages))
