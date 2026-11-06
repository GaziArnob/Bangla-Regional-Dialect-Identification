"""Text-level numbers for the paper; values come from colab_results*.json (Colab/Kaggle T4 runs)."""
R = {
    "abstract": (
        "Bangla has many regional dialects, yet most Bangla language tools assume the standard variety. We study 12-class "
        "dialect identification on BanglaDial, removing duplicates and conflicting labels before splitting (59,880 "
        "sentences), and show that the comparison between classical and neural models is decided by three evaluation "
        "choices. First, the training budget: with a common four-epoch default, fine-tuned BanglaBERT, BanglishBERT and "
        "XLM-R fall below a linear SVM on character 1–4-gram TF-IDF (macro-F1 0.781), as an earlier benchmark found; "
        "trained for up to 15 epochs over three seeds they beat it (0.806 ± 0.002, 0.801 ± 0.004, 0.791 ± 0.003), "
        "significantly so for BanglaBERT in every seed (McNemar p ≤ 4·10⁻⁴). Second, test independence: one external "
        "dialect corpus shares 90% of its sentences with BanglaDial, and on unseen data the SVM transfers better than every "
        "transformer on the independent ChatgaiyyaAlap corpus and is on par on ANCHOLIK-NER. Third, exposure to dialect text: three open LLMs (Qwen2.5-7B, Qwen2.5-14B, BLOOMZ-7B1) "
        "perform at chance level under English or Bangla prompts, shuffled options or few-shot examples, because they read "
        "dialectal Bangla as Standard Bangla; a translation probe confirms that they do not understand most dialect "
        "sentences. QLoRA fine-tuning on the full training set lifts Qwen2.5-7B to 0.721, still below the SVM at far "
        "higher cost. Character n-grams and transformers are complementary: their probability-averaging ensemble reaches "
        "0.834 ± 0.002 macro-F1, significantly better than any single model, and is also the best model across corpora. "
        "Stop-word removal and stemming lower macro-F1, because dialect cues sit in function words and inflections."),
    "hardware": "All experiments ran on Google Colab and Kaggle with NVIDIA T4 GPUs (scikit-learn, gensim, PyTorch and Hugging Face Transformers).",
}
