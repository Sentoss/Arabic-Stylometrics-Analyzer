# Layer 3 with more text: 40 chunks per book instead of 12

Same 78 books, same methods, same leave-one-book-out tests as
[`reports/layer3`](../layer3/README.md); only the sample grew, from 922 to
1,969 analysed chunks (≈ 4 M words). Books with ≤ 40 chunks, and the disputed
works, are used whole. Commands:

```
python -m stylometry.analyze_chunks DATA --per-book 40 --worker 0/2   # and 1/2
python -m stylometry.evaluate DATA --sample sample40.tsv --out reports/layer3_s40
python -m stylometry.verify   DATA --sample sample40.tsv --out reports/layer3_s40
```

## Attribution (19 candidate authors, 67 test books)

Book-level accuracy (share of test books whose author is named correctly);
chunk-level accuracy in brackets.

| features | logistic regression, 12/book | logistic regression, 40/book |
|---|---|---|
| surface words (300) | 54% (43%) | 55% (43%) |
| lemma + function + POS + pattern | 51% (44%) | **60%** (44%) |
| POS 1-3-grams | 40% (35%) | 48% (34%) |
| function words + clitics | 46% (35%) | 45% (33%) |

By genre (combined features, logistic regression):

| | 12/book | 40/book |
|---|---|---|
| author has another book in the same genre (56 books) | 59% | **70%** |
| author known only from other genres (11 books) | 1/11 | 1/11 |

**Reading:** more text does not make single chunks easier to attribute (chunk
accuracy is flat), but each book now gets more votes, so book verdicts are
steadier: the best result rises from 54% to 60%, and to 70% within a genre.
The genre wall is untouched: more text of the same kind cannot teach the
model what an author sounds like in a genre it has never seen.

## Verification (General Impostors)

AUC 0.837 (was 0.832); same-genre trials 0.857.

| threshold | true author passes, same genre | cross-genre | false candidates passing |
|---|---|---|---|
| 0.155 (threshold of the first report) | 45/55 (was 46/55) | 1/12 (was 1/12) | 15% (was 15%) |
| 0.015 (new automatic optimum) | 51/55 | 6/12 | 30% |

The automatic threshold moved only because the balanced-accuracy curve is
flat between these values; the score distributions are essentially the same.
We keep 0.155 for verdicts (fewer false passes).

## Disputed works: unchanged

| work | score | rank of traditional author | nearest author | after removing shared text |
|---|---|---|---|---|
| al-Maḥāsin wa-l-aḍdād (al-Jāḥiẓ) | 0.00 | 19 | al-Bayhaqī | 0.00, rank 21 |
| al-Tāj (al-Jāḥiẓ) | 0.07 | 6 | Ibn Wahb al-Kātib | 0.10, rank 5 |
| al-Imāma wa-l-siyāsa (Ibn Qutayba) | 0.00 | 15 | Abū Ḥanīfa al-Dīnawarī | 0.00, rank 15 |
| Adab al-ghurabāʾ (Abū l-Faraj) | 0.01 | 12 | al-Tanūkhī | 0.00, rank 19 |

All four stay below the 0.155 threshold. The one change: al-Imāma's nearest
author is now Abū Ḥanīfa al-Dīnawarī (a historian) instead of al-Bayhaqī,
which fits its historical-narrative genre better.

## Conclusion

The 12-chunk results were not an artefact of a small sample. More text helps
attribution within a genre, does not help across genres, and does not change
any verdict. The full corpus (4,378 chunks) is not expected to change the
picture; the next gains must come from better analysis (layer 2) and from
genre-robust features.

## Does better morphology matter? An ablation (before building layer 2)

`python -m stylometry.ablation DATA` re-runs the attribution (logistic
regression, 40 chunks/book) on versions of the corpus that differ only in
their morphological analysis. 65% of word tokens are ambiguous (the analyzer
offers more than one lemma + POS, so the MSA model had to choose); 4% are
unknown to it. Chunk accuracy (book accuracy in brackets):

| features | base | unknown words dropped | +10% wrong readings | +25% | +50% |
|---|---|---|---|---|---|
| lemmas | 39.2% (52%) | 39.3% (52%) | 39.0% (52%) | 38.4% (49%) | 39.8% (54%) |
| function words + clitics | 33.1% (45%) | 33.1% (45%) | 32.9% (46%) | 32.0% (42%) | 33.2% (45%) |
| POS 1-3-grams | 34.0% (48%) | 34.0% (48%) | 32.8% (48%) | 32.0% (49%) | 27.2% (45%) |
| patterns | 30.0% (45%) | 30.0% (45%) | 29.6% (46%) | 27.8% (42%) | 29.3% (45%) |
| combined | 43.5% (60%) | 43.6% (60%) | 41.8% (58%) | 41.3% (55%) | 40.1% (54%) |

"+50%" gives a wrong reading to half of the ambiguous words, about a third of
all words, and the combined set loses only 3.4 points; lemmas and function
words lose nothing. Only POS sequences are clearly sensitive (−7 points).
Keeping only unambiguous words (not shown in the table: combined 36.9%) hurts
because it throws away 65% of the text, including most function words, so it
says nothing about analysis quality.

**Reading:** the attribution is very robust to analysis errors. Fixing the
current errors with layer 2 should gain at most a point or two, a few more for
POS features. Caveat: random errors are not the same as systematic ones. The
one thing this test cannot measure is categories the MSA dictionary lacks
entirely (e.g. passives, read as actives everywhere), which may carry real
authorial signal. For layer 3, the bottleneck is genre, not morphology.

## Re-run with the Classical supplement in layer 1

All 1,969 chunks were re-analysed with the layer 2 supplement plugged into
layer 1 (`analyze_chunks --out analyzed_l2`; results in
[`../layer3_l2`](../layer3_l2/evaluation.json)). The disambiguator is still
CAMeL's MSA model; only the readings it may choose from changed.

**What changed in the analyses: 0.6% of words.**

| change | share of words | examples |
|---|---|---|
| lemma of رأى corrected (was راوَنْد) | 0.30% | رأيت، يرى، رآه، ترى |
| passive chosen instead of active | 0.14% | ذُكِرَ، نُسِبَ، يُسْتَعْمَلُ، جُعِلَ، قُبِضَ |
| other lemma or POS | 0.08% | رُوِيَ (was the noun رَوِيّ), سُمِّيَ (was سَمِيّ), حُكِيَ, يُرْوَى, بُنِيَ |
| reading for a word unknown before | 0.08% | أُتِيَ، اُقْتُلُوا، فَافْعَلُوا، وَأَطِيعُوا، صِلْ |

In unvowelled text only passives, harvested variants, كَأَنْ and the رأى lemma
are used freely; imperatives only for words with no other reading; the
interrogative أَ not at all, because the MSA model cannot be trusted with it
(it read the name أهرمن as أَ + هَرَّمَ). That waits for the Classical model.

**Attribution: unchanged, as the ablation predicted** (logistic regression,
chunks / books):

| features | MSA dictionary | + Classical supplement |
|---|---|---|
| lemmas | 39.2% / 52% | 39.5% / 52% |
| POS 1-3-grams | 34.0% / 48% | 34.2% / 48% |
| patterns | 30.0% / 45% | 30.2% / 48% |
| combined | 43.5% / 60% | 44.0% / 58% |

Chunk accuracy rises by 0.2–0.5 points; book accuracy moves by one book either
way. The supplement makes the analyses more correct, which matters for
syntax, rhetoric and semantics; it does not move attribution.
