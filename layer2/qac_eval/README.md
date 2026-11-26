# Layer 2a against gold data: the Quranic Arabic Corpus

Scores morphology disambiguators against QAC's morphology (whole Qur'an), and
builds gold training labels for 2a from it.

1. `parse_ws.py WIKITEXT verses.json`: the Arabic Wikisource Qur'an
   (القرآن الكريم (بالرسم الإملائي)/النص المشكول, raw wikitext) → verses.
2. `align.py QAC_MORPH.txt verses.json aligned.jsonl`: align QAC words
   (Uthmani, QAC numbering) with the standard text verse by verse; QAC tags
   mapped to CAMeL features as gold.
3. `run_system.py SYSTEM aligned.jsonl preds.jsonl`: `msa`, `msa_sup`, or a
   folder with one of our models.
4. `score.py aligned.jsonl NAME=preds.jsonl … [--heldout]`.
5. `qac_labels.py aligned.jsonl labels.json OUT_DIR`: gold 2a labels
   (dictionary readings that agree with QAC, narrowed by the vowels), split by
   sūra: test = sūras ending in 3, val = ending in 7.

QAC (corpus.quran.com, Kais Dukes) may be used for training but is not
redistributed: keep datasets containing it private.
