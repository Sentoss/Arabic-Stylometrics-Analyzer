# Arabic Stylometrics

A program for stylistic and stylometric analysis of Arabic texts, with a focus on
**Classical Arabic** (prose, poetry, and the Qur'an). It aims to go beyond
counting surface words: features are computed over morphology (lemmas, roots,
patterns, part of speech), and the analysis extends to the rhetorical devices of
the Arabic tradition (بلاغة) and to metre and rhyme in poetry.

> **Status (November 2026):** Layer 1 (spelling restoration) and layer 3
> (stylometry: attribution, verification, clustering, PCA, on a cleaned
> 26-author benchmark) work and are evaluated. See
> [`reports/layer3/`](reports/layer3/README.md) for results. Layer 2 (Classical
> morphology) is next; layers 4–6 are planned. Scope is Classical Arabic and
> MSA; dialects are deliberately out of scope.

## Why this project

- General-purpose Arabic NLP tools (CAMeL Tools, Farasa, MADAMIRA) are trained on
  modern news text and lose accuracy on Classical texts.
- Existing stylometry tools (e.g. R `stylo`, used by the KITAB project on
  OpenITI) work on surface word forms and know nothing of Arabic morphology,
  so وكتابه and بكتابهم count as unrelated words, and an edition's printing habits
  (فى for في) look like an author's style.
- Research on Classical Arabic authorship suggests authors differ more in
  syntax than in vocabulary, which argues for morphosyntactic features.
- No tool brings stylometry, Classical morphology, بلاغة and prosody together.

## Architecture

| Layer | What it does | Status |
|---|---|---|
| 1. Spelling restoration | Undoes each edition's printing conventions (dropped hamza, ى for ي, ه for ة) while keeping real distinctions; strips apparatus; flags Qur'an quotations | ✅ working |
| 2. Classical morphology | A Classical disambiguator on top of the CAMeL analyzer, trained on data derived from diacritized text (Tashkeela); later, Classical lemmas from Qabas | ⏳ in progress (Classical supplement + training labels built) |
| 3. Stylometry | Delta, clustering and classifiers over lemmas, patterns, POS sequences and function words; Impostors verification | ✅ working (first results) |
| 4. Surface rhetoric | Rule-based سجع, جناس, طباق, repetition, parallelism | planned |
| 5. Poetry | Metre (عروض) and rhyme (قافية); metre also constrains vowels for layer 2 | planned (on hold for al-Farāhīdī's MCP) |
| 6. Semantics | How words are used: word senses (عين = eye / spring / notable), each author's usage of a word learned from its contexts across OpenITI, semantic fields drawn from Classical onomasiological works (Ibn Sīda's *al-Mukhaṣṣaṣ*, al-Thaʿālibī's *Fiqh al-lugha*), collocations; the basis for telling literal from figurative use in layer 7 | planned |
| 7. Deep rhetoric | استعارة, كناية, تقديم وتأخير, حذف; literal vs figurative use built on layer 6; LLM-assisted with human review | planned |

### Data sources
- **Qur'an:** Arabic Wikisource, الرسم الإملائي edition (Ḥafṣ). Standard spelling, so no Uthmānī conversion is needed.
- **Classical prose and poetry:** al-Maktaba al-Shāmila, OpenITI.
- **Training/frequency data:** Tashkeela (diacritized Classical text).
- **Gold evaluation:** QuranMorph (Qabas lemmas, SAMA tagset), Noor-Sharaye (205k words, 17 Classical books).

## Layer 1: spelling restorer

Many Classical editions on Shamela/OpenITI were printed with conventions that
are not the author's: Cairo prints write في as فى; older prints drop the hamza
(الى, اصحاب) and write ة as ه (القضاه). Erasing these distinctions everywhere
would merge different words (على/علي, إن/أن), so the restorer *restores*
instead of normalizing.

### How it works
1. **Cleanup** (`restorer/markup.py`): removes footnote markers, page markers,
   metre labels and item numbers; tokenizes with character offsets; flags Qur'an
   quotations (﴿…﴾ certain, `(…)` after تعالى/قوله probable).
2. **Source profile** (`restorer/profile.py`): measures five conventions on
   diagnostic words (فى/في, الى/إلى, الصلاه/الصلاة, حتي/حتى, قايل/قائل). A
   convention is switched on only if ≥15% of the diagnostics use it, so
   well-edited texts pass through untouched.
3. **Candidates** (`restorer/restore.py`): CAMeL Tools proposes readings; only
   those that differ from the written word by substitutions the profile licenses
   are kept.
4. **Choice:** CAMeL's BERT disambiguator (MSA model) ranks candidates by
   context; editor-supplied diacritics act as a constraint; near-ties are settled
   by Classical word frequency (`restorer/prior.py`, built from Tashkeela) only
   when the counts are decisive. An unknown proper noun (واسط) is kept if the
   model prefers a name reading and the written form is attested in Classical
   text while the dictionary alternative is not.
5. **Output:** three aligned tiers per token: raw, restored (letters fixed,
   original diacritics kept), and analysis (lemma, POS, root, pattern), with a
   confidence level (certain / high / low).

### Results
Synthetic test: 2,559 held-out Tashkeela words with all three conventions
introduced everywhere.

| System | Accuracy on words needing a decision |
|---|---|
| Change nothing | 58.6% |
| Context model only | 91.4% |
| + Classical frequency prior | **95.4%** |

Of the corrupted words, 89.9% were restored; 99.2% of already-correct words
were left alone. Confidence labels are reliable: certain 99.5%, high 93.7%,
low 73% (low = review by hand).

Real Shamela passages (in `samples/`):

| Passage | Conventions detected | Changes | Errors |
|---|---|---|---|
| Ṭabarī, *Ṣila* | dropped hamza, ى for ي, ه for ة | 112 | 1 clear (وانهتك→وأنهتك), 2 debatable (الإصفهاني) |
| Nuwayrī, *Nihāyat al-arab* | ى for ي | 28 | 1 (past غنّى read as imperative) |
| Sibṭ Ibn al-Jawzī, *Mirʾāt al-zamān*; Ibn Ḥijja, *Thamarāt al-awrāq* | none | 0 | 0 |

### Known limits
Most remaining errors come from using a modern-Arabic model on Classical text,
which is what layer 2 addresses:
- أن vs إن after a dropped hamza (needs syntax, e.g. إن after قال).
- Classical names and nisbas missing from the dictionary (الطالقاني, الإسنوي) are left as written and flagged.
- Verb forms VII/VIII with hamzat al-waṣl can be mistaken for form IV; imperatives in verse are often missed.
- The frequency prior is built from only 2.1M words and has some noise of its own.

## Benchmark corpus (for layer 3)

Built from [OpenITI](https://github.com/OpenITI) by `python -m corpus.build DATA_DIR`
(selection in `corpus/benchmark.py`). Texts are downloaded at build time, not
stored in this repo; `corpus/books.tsv` and `corpus/manifest.tsv` record exactly
which versions and chunks were used.

| | |
|---|---|
| Authors | 26 (d. 255–429 AH), 19 with two or more secure books |
| Books | 78 used: 67 secure, 4 disputed, 7 single-work authors (impostor pool); 3 excluded as uncorrected OCR |
| Prose | 9.5M words in 4,638 chunks of ~2,000 words |
| Poetry | 1.4M words, separated from prose and excluded from chunks |
| Genres (chunks) | history 2,140 · adab 1,951 · criticism 332 · ethics 96 · law 60 · zuhd 59 |

**Disputed test cases:** al-Jāḥiẓ, *al-Maḥāsin wa-l-aḍdād* and *al-Tāj*;
Ibn Qutayba, *al-Imāma wa-l-siyāsa*; Abū l-Faraj al-Iṣfahānī, *Adab al-ghurabāʾ*.
Natural impostors include al-Bayhaqī's *al-Maḥāsin wa-l-masāwī* and Ibn ʿAbd
Rabbih's *al-ʿIqd al-farīd*.

**Cleaning:** OpenITI headers, page markers, milestones, footnote markers and
editorial tags are removed. Verses are split out of prose in all three notations
found (`%~%`, `( a % b )`, `a ... b`). Very long paragraphs are split at sentence
punctuation.

**Editorial insertions** (`corpus/editorial.py`, counts per book in
`corpus/editorial.tsv`). Brackets and quotation marks come from modern editors,
and each edition uses its own system, so the cleaner removes what editors *added*
and keeps what they only *marked*:
- **Removed** (~13,500 items): metre labels such as `[من الخفيف]`; Qur'an
  references such as `[النساء: 163]`; bare numbers; manuscript sigla such as
  `«خ»`; editors' glosses of the form `(headword: explanation)`; item numbering.
  Glosses introduced by أي, يعني or قال are kept, as they are usually the
  author's own voice.
- **Kept, re-marked:** Qur'an in `{ }` or `' '` becomes `﴿ ﴾` (a Qur'an hint);
  text in `« » " " ( )` becomes `« »` (a quotation hint: proverb, ḥadīth, speech,
  name or title, to be decided by reference matching); words restored by editors
  in `[ ]` keep their text without the brackets. About 109,000 hint spans.
- **Excluded books:** three versions are uncorrected OCR with the editor's
  introduction, apparatus and indexes mixed into the text, and OpenITI has no
  clean alternative: al-Ṣūlī's *Khilāfat al-Wāthiq*, Ḥamza al-Iṣfahānī's
  *al-Durra al-fākhira*, Miskawayh's *al-Ḥikma al-khālida*. After exclusion,
  paragraphs with editorial markers (في نسخة, page references, etc.) are a few
  hundred words, some of them authorial (al-Masʿūdī's own "ووجدت في نسخة").

After cleaning: 78 books, 4,638 chunks, 9.5M prose words; 19 authors still
have two or more secure books.

**Quotations** (`python -m corpus.quotations DATA_DIR REFS_DIR`; counts per book in
`corpus/quotations.tsv`). Much of a Classical book is other people's words, so
borrowed text is found and then classified by how the author uses it:

| Found by | Kind |
|---|---|
| word 4-grams shared with the Qur'an (OpenITI *Muṣḥaf*, basmala excluded) | Q |
| word 5-grams shared with nine ḥadīth collections (Bukhārī, Muslim, the four Sunan, Muwaṭṭaʾ, Musnad Aḥmad, Dārimī); isnād words and shared formulas (dates, piety, letter openings) are not indexed, phrases found in more than 10% of benchmark books are treated as common phrasing, and a match needs 6+ words | H |
| 5,345 proverb headwords from al-Maydānī, Abū ʿUbayd and al-Zamakhsharī (explanations not indexed) | M |
| transmission chains: a transmission verb (حدثنا, أخبرني, سمعت…) linked by عن and name links (بن, أبي…) up to the reporter's قال / أنّ; a lone "سمعت أعرابيا" is not a chain | isnād |

Each Q/H/M span is then either **reproduction** (12+ words, or its whole
sentence apart from framing such as قال تعالى, وفي الحديث, وفي المثل) or
**embedded** (woven into the author's sentence: اقتباس, a proverb in passing).
Reproduced blocks and isnāds are replaced by placeholders (`<Q-BLOCK>`,
`<H-BLOCK>`, `<M-BLOCK>`, `<ISNAD>`); embedded quotations by `<Q>`, `<H>`, `<M>`.
Their words stop counting as the author's, while the act of quoting stays
countable as a feature. Chunks are then rebuilt from the author's own words
(`corpus/manifest_authorial.tsv`: 4,378 chunks).

Result: 94.3% of prose words remain the authors' own. Removed or masked: isnāds
391k words, Qur'an 90k (67k reproduced, 23k embedded), ḥadīth 46k, proverbs 12k.
Extremes: Ibn Abī l-Dunyā's zuhd works 52–69% own words; *Taʾwīl mushkil
al-Qurʾān* 80%; *al-Aghānī* 89% (122k words of isnāds); al-Jāḥiẓ 97–99%;
Miskawayh 99.6%. A random sample of detected spans read in context showed
correct matches throughout; one embedded proverb was classified as a block.

Known limits: reported speech and quotations from sources not in the
reference sets (letters, sermons, other prose works) stay in the author's text;
shared narrative material (e.g. sīra episodes retold by historians) is
caught only where it also appears in the ḥadīth collections; verse remaining
in prose (unmarked editions) is not yet handled.

**Known limits:**
- **Unmarked verse.** 17 books (flag `verse_marked = no` in `books.tsv`) have
  less than 1% detected poetry. In adab and history works this usually means the
  edition's verses run on as ordinary lines, so some poetry remains in their
  chunks. Detecting verses by metre (layer 5) would fix this.
- **Isnāds, Qur'an quotations and reported speech are still inside prose
  chunks.** Removing them is the next step.
- **Book sizes are very unequal.** Ṭabarī's *Tārīkh*, *al-Aghānī* and *Ansāb
  al-ashrāf* contribute 400–700 chunks each, so analyses should cap chunks per
  book.

## Layer 3: stylometry

Full results, figures and caveats: [`reports/layer3/README.md`](reports/layer3/README.md).

**Pipeline.** The authorial chunks go through layer 1 (`stylometry/analyze_chunks.py`).
Features are counted per chunk (`stylometry/features.py`): restored word forms,
lemmas, function words and clitics, POS 1–3-grams, morphological patterns,
character 3–4-grams and quoting rates. Methods (`stylometry/methods.py`):
Burrows's and Cosine Delta, logistic regression and linear SVM, General
Impostors verification, Zeta, bootstrap consensus clustering and PCA.
Evaluation holds out whole books (`stylometry/evaluate.py`), and verification
is calibrated on books of known authorship (`stylometry/verify.py`).

**First results** (922 samples, 12 per book; 19 candidate authors; 67 held-out books):
- **Attribution:** the best feature sets attribute 54% of held-out books
  correctly (chance 5%). That is 62–65% for books whose genre the author also
  wrote in, and almost none of the 12 books that are an author's only work in
  their genre: genre is the main obstacle. Distinctive stylists are always
  recognized (al-Jāḥiẓ 7/7, al-Tawḥīdī 5/5, Ibn Abī l-Dunyā 4/4); compilers
  mostly are not.
- **Verification:** AUC 0.83. At the calibrated threshold, 84% of true authors
  pass on same-genre books and 15% of false candidates pass.
- **Disputed works:** none is stylistically consistent with its traditional
  author. *al-Maḥāsin wa-l-aḍdād* is closest to al-Bayhaqī (*al-Maḥāsin
  wa-l-masāwī*), with which it also shares 10.9% of its text, and the
  resemblance survives removing all shared passages. *al-Imāma wa-l-siyāsa*
  sits among the historians, far from Ibn Qutayba. *al-Tāj* is closest to a
  secretarial *adab* author. *Adab al-ghurabāʾ* is too short to judge.

**Analysing your own text** (plain text, Shamela copy, or an OpenITI file):

```bash
python analyze.py mytext.txt --data DATA_DIR --refs REFS_DIR [--candidate 0255Jahiz]
```

The command removes editorial insertions, quotations and isnāds, runs layer 1,
ranks the benchmark authors by stylistic distance, and, with `--candidate`,
gives the calibrated Impostors verdict with the features and Zeta words that
separate the text from that author. Output goes to `out/<name>/report.md`.
Candidate IDs are the OpenITI author IDs in `corpus/books.tsv`. Verdicts on
fewer than three ~2,000-word chunks are weak.


### Feature profiles

Which features characterise an author, a genre or a book, compared with the rest of the corpus:

```
python -m stylometry.profile DATA_DIR --by author        # or genre, book; --only 0255Jahiz
```

Writes `reports/profiles/profiles_<by>.md` (the most over- and under-used function words and clitics, lemmas, POS sequences and patterns, written as فَعَلَ / فاعِل, with how steady each habit is across the group's books) and a full `.tsv`. `analyze.py` now also writes `features.tsv` for any text and lists its most distinctive features in `report.md`.

## Installation

```bash
pip install camel-tools
camel_data -i morphology-db-msa-r13
camel_data -i disambig-bert-unfactored-msa
```

## Usage

```bash
# restore a text; writes OUT/<name>.restored.txt, .tokens.tsv, .profile.json
python restore_cli.py samples/tabari_sila.txt out/

# override a detected convention
python restore_cli.py book.txt out/ --force hamza_dropped_on_alif=on

# synthetic evaluation on clean (standard-spelling) text
python evaluate.py clean.txt --words 3000 [--no-prior]

# run all samples and print a report
python run_samples.py
```

## Repository layout

```
restorer/            layer 1 package (markup, profile, restore, prior, text)
restore_cli.py       command-line interface
evaluate.py          synthetic corruption test
run_samples.py       runs the Shamela samples
resources/           Classical form frequencies (from Tashkeela)
layer2/              layer 2: build_labels.py (training labels from Tashkeela vowels), probe_labels.py
stylometry/          layer 3: chunk analysis, features, methods, evaluation, verification, figures
analyze.py           analyse an outside text against the benchmark
reports/layer3/      results report, figures, evaluation and verification data
corpus/              benchmark selection, OpenITI download/cleaning, editorial cleanup,
                     quotation and isnad handling, chunking, manifests
samples/             test passages from al-Maktaba al-Shāmila
```

## Roadmap
1. ~~Layer 1: spelling restoration~~
2. ~~Layer 3: stylometry~~ (re-run with 40 chunks per book; feature profiles by author/genre/book; next: syntactic features such as clause types, masking of content words, and a larger corpus with authors spanning several genres)
3. Layer 2 (in progress): Classical disambiguator. Done: a Classical supplement to CAMeL's analyzer (passives and imperatives by rule for all verb forms; the interrogative أَ; 464 Classical vowelings, Form II verbs such as حَدَّثَ and name forms harvested from Tashkeela; lightened كَأَنْ; the lemma of رَأَى corrected from راوَنْد) and training labels for 2.1 M Tashkeela words: 79.9% settled, 16.9% ties kept as option sets, 1.1% unmatched (was 4.0%), 2.1% unknown. Next: training-format conversion, then fine-tuning CAMeLBERT-CA. See [`layer2/README.md`](layer2/README.md).
4. Layers 4–5: rule-based rhetoric; metre and rhyme (layer 5 on hold until al-Farāhīdī's MCP server is available).
5. Layer 6: semantics. Start with distributional usage profiles (how an author uses a word, compared with others) and semantic fields from *al-Mukhaṣṣaṣ*; then word senses. Semantic features follow topic closely, so they serve stylistic description more than attribution, and are compared within a genre.
6. Layer 7 (deep rhetoric) and lexicon extension from Qabas.
7. Graphical interface: load a text, see its restored spelling and morphology,
   its feature profile against the corpus, the closest authors and the
   verification against a chosen candidate, without using the command line.
8. Standalone offline package (target 250–500 MB, no GPU needed): the trained
   disambiguator and the parser compressed by 8-bit quantization and run with
   a light runtime (ONNX Runtime) instead of PyTorch; author profiles shipped
   instead of the corpus; accuracy re-measured after compression.

## License

Copyright (C) 2026 the Arabic Stylometrics contributors.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, **either version 2 of the License, or (at your option) any later
version**. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See [`LICENSE`](LICENSE) (the GPL v2 text) for details.

"Or later" keeps the code compatible with Tashkeela (GPL v2) today and allows a
move to GPL v3 if a future data source requires it.

## Data and licensing notes

| source | used for | license | consequence |
|---|---|---|---|
| [Tashkeela](https://sourceforge.net/projects/tashkeela/) (Zerrouki & Balla 2017), via the [AliOsm split](https://github.com/AliOsm/arabic-text-diacritization) (code MIT) | frequency prior (`resources/form_freq_tashkeela.tsv`), layer 2 labels, Classical variants (`layer2/resources/classical_variants.tsv`), the future layer 2 model | GPL v2 (listed as "v2 only" by some redistributors) | derived files and models are distributed under GPL v2; they cannot be combined with GPL-v3-only material unless the author permits "v2 or later" |
| [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) | analyzer, MSA disambiguator | code MIT; MSA morphology database GPL v2 | as above |
| [CamelParser](https://github.com/CAMeL-Lab/camel_parser) and its CATiB model | dependency parses (syntax experiments) | see its repository; not redistributed here | parses are produced locally |
| [OpenITI](https://openiti.org) | benchmark texts, downloaded at build time | OpenITI's own terms | texts are not redistributed here |
| al-Maktaba al-Shāmila | short sample passages for testing | excerpts | — |
| [al-Farāhīdī](https://www.faraheedy.com) (planned, layer 5) | metre content | CC BY-NC-SA | kept separate from GPL material, or used through its service only |
| Quranic Arabic Corpus, MASAQ (planned, i'rāb parser) | traditional-grammar treebanks | QAC: GPL (version to be checked); MASAQ: to be checked | check compatibility with GPL v2 before use |

