# Layer 2: Classical morphological disambiguator

Goal: pick the right reading (lemma, part of speech, features) of each word in
**unvowelled Classical text**, as CAMeL Tools' disambiguator does for Modern
Standard Arabic. The model will be CAMeLBERT-CA fine-tuned on Classical
text whose correct readings are known.

## Step 1: training labels from Tashkeela (done)

Nobody has hand-labelled millions of Classical words. But Tashkeela's texts are
*vowelled*, and vowels rule out readings. `build_labels.py` asks CAMeL's
analyzer for every possible reading of the bare word, then keeps only the
readings whose vowels agree with the printed ones:

| word in Tashkeela | readings of the bare form | kept |
|---|---|---|
| عَلِمَ | عَلِمَ "he knew", عُلِمَ, عَلَّمَ "he taught", عِلْم "knowledge", … | عَلِمَ |
| وَيُعِيدُهُمَا | many | one (verb أعاد, imperfect, 3ms, + هما) |
| أَقْوَى | أَقْوَى "stronger", أَقْوَى "I grow strong", … | both (tie) |

```
python layer2/build_labels.py DATA/tashkeela_aliosm/dataset/train.txt DATA/layer2/labels_train.tsv
```

Writing conventions are reconciled first (`probe_labels.norm`): sukun, dagger
alif, tanwīn al-fatḥ placement, sun-letter shadda, vowel on the connecting alif
(وَاَللَّهُ), final ى for ي (أَبِى), and the case/mood vowel before a pronoun
suffix (يَلْزَمُهُ).

### Results (AliOsm split of Tashkeela, train = 2.1 M words)

| status | share | meaning |
|---|---|---|
| unique | 63.8% | one reading survives |
| case_tie | 14.4% | lemma and POS settled; a feature the text leaves unmarked stays open |
| tie | 15.6% | readings with different lemma/POS survive; all are stored in `options` |
| no_match | 4.0% | vowels contradict every reading in the dictionary |
| oov | 2.2% | dictionary has no reading |

Ties are kept as sets (`options`), so training can count any of them as right
(partial-label training) and let sentence context choose, instead of trusting
the dictionary's most probable pick, which is often wrong (أَقْوَى → the verb).

### What the no_match words show: CAMeL's MSA dictionary lacks Classical forms

- **Passives** are absent for most verbs: ذَكَرَ but no ذُكِرَ, رَوَى but no
  رُوِيَ, يَشْتَرِط but no يُشْتَرَط. Clear passives are ~16% of no_match
  (يُشْتَرَطُ يُسْتَحَبُّ يُحْتَمَلُ ذُكِرَ رُوِيَ كُرِهَ قُبِلَ).
- **Isnād verbs**: حَدَّثَنَا, حَدَّثَنِي (only the noun حَدَث is listed).
- **Classical vowelings**: ثُلُث (only ثُلْث), جُمُعَة (only جُمْعَة), رَضَاع,
  عَجَزَ, أَرَأَيْتَ, lightened كَأَنْ.
- **Names** with odd entries: يُوسِف for يُوسُف; عَمْرو without case; سَلَمَة,
  قَتَادَة, يُونُس.

## Next steps

1. **Classical supplement** to the dictionary: generate passive readings from
   each active verb's pattern (فُعِلَ / يُفْعَلُ and the derived forms), add the
   isnād verbs, Classical vowelings and names, then rebuild the labels.
2. **Fine-tune** CAMeLBERT-CA on the labels (unique + case_tie as targets, ties
   as option sets); evaluate on the test split against the vowels.
3. Plug it into layer 1 in place of the MSA disambiguator and re-run layer 3.
