# Arabic Stylometrics

A program for stylistic and stylometric analysis of Arabic texts, with a focus on
**Classical Arabic** (prose, poetry, and the Qur'an). It aims to go beyond
counting surface words: features are computed over morphology (lemmas, roots,
patterns, part of speech), and the analysis extends to the rhetorical devices of
the Arabic tradition (بلاغة) and to metre and rhyme in poetry.

> **Status (November 2026):** Layer 1 (spelling restoration) works and is
> evaluated. Layers 2–6 are planned. Scope is Classical Arabic and MSA;
> dialects are deliberately out of scope.

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
| 2. Classical morphology | A Classical disambiguator on top of the CAMeL analyzer, trained on data derived from diacritized text (Tashkeela); later, Classical lemmas from Qabas | ⏳ next |
| 3. Stylometry | Delta, clustering and classifiers over lemmas, patterns, POS sequences and function words | planned |
| 4. Surface rhetoric | Rule-based سجع, جناس, طباق, repetition, parallelism | planned |
| 5. Poetry | Metre (عروض) and rhyme (قافية); metre also constrains vowels for layer 2 | planned |
| 6. Deep rhetoric | استعارة, كناية, تقديم وتأخير, حذف; LLM-assisted with human review | planned |

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
samples/             test passages from al-Maktaba al-Shāmila
```

## Roadmap
1. ~~Layer 1: spelling restoration~~
2. Layer 2 proof of concept: Classical disambiguator trained on Tashkeela-derived data, measured against layer 1's 95.4%.
3. Layer 3: stylometry on lemma/pattern/POS features.
4. Layers 4–5: rule-based rhetoric, metre and rhyme.
5. Layer 6 and lexicon extension from Qabas.

## Data and licensing notes
- `resources/form_freq_tashkeela.tsv` was built from
  [AliOsm/arabic-text-diacritization](https://github.com/AliOsm/arabic-text-diacritization)
  (MIT), a split of the Tashkeela corpus. Check the original Tashkeela license
  before redistributing.
- CAMeL Tools data packages carry their own licenses (the MSA morphology
  database is GPL v2).
- Sample passages are short excerpts from al-Maktaba al-Shāmila, used for testing.
