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
layer2/              layer 2 experiments (probe_labels.py: labels from Tashkeela vowels)
corpus/              benchmark selection, OpenITI download/cleaning, chunking, manifests
samples/             test passages from al-Maktaba al-Shāmila
```

## Roadmap
1. ~~Layer 1: spelling restoration~~
2. Layer 3: stylometry on lemma/pattern/POS features (next), using the MSA model as a stand-in.
3. Layer 2: Classical disambiguator. CAMeLBERT-CA fine-tuned on labels derived from Tashkeela's vowels; a first probe gives a unique label for ~74% of words after reconciling vowel-writing conventions between Tashkeela and CAMeL (sun-letter shadda, tanwīn placement, dagger alif); 19% stay ambiguous because vowels cannot decide, 5% are dictionary gaps (passives, Classical forms) (see `layer2/probe_labels.py`).
4. Layers 4–5: rule-based rhetoric, metre and rhyme.
5. Layer 6 and lexicon extension from Qabas.

## Data and licensing notes
- `resources/form_freq_tashkeela.tsv` was built from
  [AliOsm/arabic-text-diacritization](https://github.com/AliOsm/arabic-text-diacritization),
  a split of the [Tashkeela corpus](https://sourceforge.net/projects/tashkeela/)
  (Zerrouki & Balla 2017). Tashkeela is licensed **GPL v2**, so this derived
  frequency list should be treated as GPL v2 as well.
- CAMeL Tools data packages carry their own licenses (the MSA morphology
  database is GPL v2).
- Sample passages are short excerpts from al-Maktaba al-Shāmila, used for testing.
- Benchmark texts come from OpenITI and are downloaded at build time rather than
  redistributed here; OpenITI's own license terms apply to them.
