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
