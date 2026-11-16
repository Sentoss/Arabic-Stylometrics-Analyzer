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

| status | MSA dictionary only | with the Classical supplement | meaning |
|---|---|---|---|
| unique | 63.8% | 65.4% | one reading survives |
| case_tie | 14.4% | 14.5% | lemma and POS settled; a feature the text leaves unmarked stays open |
| tie | 15.6% | 16.9% | readings with different lemma/POS survive; all are stored in `options` |
| no_match | 4.0% | **1.1%** | vowels contradict every reading in the dictionary |
| oov | 2.2% | 2.1% | dictionary has no reading |

Validation and test splits give the same figures (no_match 1.1%), although the
variants were harvested from the training split only.

Ties are kept as sets (`options`), so training can count any of them as right
(partial-label training) and let sentence context choose, instead of trusting
the dictionary's most probable pick, which is often wrong (أَقْوَى → the verb).

## Step 2: the Classical supplement (done)

The no_match words showed what CAMeL's MSA dictionary lacks for Classical
prose. `supplement.py` adds the missing readings at analysis time
(`ClassicalAnalyzer`, a drop-in for CAMeL's analyzer):

**1. Passives by rule** (`make_passive`). The MSA database has passives for only
some verbs (عُلِمَ, يُذْكَرُ) and none for most (ذُكِرَ, قِيلَ, رُوِيَ, يُشْتَرَطُ).
Every active perfect and imperfect reading now gets its passive:

| | active | passive |
|---|---|---|
| Form I | ذَكَرَ / يَذْكُرُ | ذُكِرَ / يُذْكَرُ |
| Forms II–X | عَلَّمَ، قاتَلَ، أَكْرَمَ، تَعَلَّمَ، اِسْتَخْرَجَ | عُلِّمَ، قُوتِلَ، أُكْرِمَ، تُعُلِّمَ، اُسْتُخْرِجَ |
| hollow | قالَ / يَقُولُ، أَقامَ | قِيلَ / يُقالُ، أُقِيمَ |
| defective | رَمَى / يَرْمِي، دَعا / يَدْعُو، نادَى | رُمِيَ / يُرْمَى، دُعِيَ / يُدْعَى، نُودِيَ |
| doubled | مَدَّ | مُدَّ |
| prefix only | يَجْعَلُ، يَتَصَوَّرُ | يُجْعَلُ، يُتَصَوَّرُ |

About 1% of all Tashkeela words now get a passive label.

**2. Classical variants harvested from Tashkeela** (`harvest_variants.py` →
`resources/classical_variants.tsv`, 464 entries, each attested ≥ 5 times). For a
no_match word, the text's vowels are put on the *stem* of each dictionary
reading; if that makes the reading fit, the stem has a Classical vowelling the
dictionary lacks. Precision first:

- nouns may change vowels but never gain a shadda; verbs may change only the
  middle radical (Form II gemination, or the stem vowel);
- if several lemmas or POS explain the word equally well, it is skipped
  (e.g. عَجَزَ, يَقْدِرُ; 1,073 forms);
- epenthetic vowels keep the lemma (ثُلْث → ثُلُث, سُدُس, رُبُع, خُمُس, إِبِل,
  عُنُق, فَخِذ, خُلُق); other changes make a new lemma: Form II verbs missing
  from the dictionary (حَدَث → **حَدَّث**, 4,426 tokens), passive participles
  (مُكاتَب, مُدَبَّر, مُعْتَق, مُسْتَأْمَن), Classical name forms (يُوسُف,
  يُونُس, عِكْرِمَة, مُعاذ, بُكَيْر), and words like قَبُول, رِدَّة;
- broken plurals are skipped (the name الخَطّاب must not become خُطّاب "suitors").

**3. Imperatives** (`make_imperative`). The MSA database has almost none (only a
few such as قُلْ, خُذْ, أُنْظُرْ, the last with a hamza for the connecting alif).
They are derived from the 2nd-person jussive: تَكْتُبْ → اُكْتُبْ, تَعْلَمْ →
اِعْلَمْ, تُكْرِمْ → أَكْرِمْ, تُقِيمْ → أَقِمْ, تُعَلِّمْ → عَلِّمْ, تُقاتِلْ → قاتِلْ,
تَسْتَغْفِرْ → اِسْتَغْفِرْ, تَتَّقِي → اِتَّقِ, تَرْمِي → اِرْمِ, تَدْعُو → اُدْعُ,
تَكْتُبُوا → اُكْتُبُوا; also after وَ/فَ (وَاعْلَمْ, فَانْظُرْ, فَاقْتُلْهُ).

**4. The interrogative hamza.** The database allows it only on a few words
(أَكانَ, أَلَيْسَ). The supplement adds أَ + any verb, pronoun, demonstrative or
particle: أَرَأَيْتَ, أَفَرَأَيْتُمْ, أَيَكُونُ, أَتَجْعَلُهُ, أَهُوَ, أَهٰذا. Not on bare
nouns: there a word-initial hamza is usually an elative (أَعَمُّ, أَوْرَعُ).

Imperatives and question readings are *secondary*: they count only when no
ordinary reading fits the vowels, so أَرَادَ stays the Form IV verb (not أَ + رادَ)
and ابْنِ stays "son of" (not "build!"). Remaining errors are rare collisions
such as أَعَمُّ (elative) read as أَ + عَمَّ.

**5. Fixes.** The lightened كَأَنْ (كَأَنْ لَمْ تَغْنَ بِالأَمْسِ; lemma كَأَنَّ), and
the lemma of the verb رَأَى, which the database gives as راوَنْد ("rhubarb") in
every form (رَأَى, يَرَى, رَأَيْتُ…).

**6. More writing conventions reconciled** (`probe_labels.norm`): the sun-letter
shadda after لِلْـ (لِلصَّلاة), the kasra on the article before a connecting
alif (الِابْن), and the mood vowel on a final ي (يُصَلِّيَ, يُعْطِيَهُ).

**Still unmatched (1.1%):** names the dictionary lacks or misvowels (عَمْرٍو,
قَتادَة, أُمَيَّة, مَسْلَمَة); ambiguous variants (عَجَزَ, يَقْدِرُ); passives whose
hamza seat changes (يُؤْخَذُ from يَأْخُذُ); elatives missing from the dictionary.

```
python layer2/build_labels.py ... --variants none      # labels without variants
python layer2/harvest_variants.py LABELS_without_variants.tsv
python layer2/build_labels.py DATA/tashkeela_aliosm/dataset/train.txt DATA/layer2/labels_train.tsv
```

## Next steps

1. ~~Classical supplement to the dictionary, and rebuilt labels~~ (done, above).
2. **Fine-tune** CAMeLBERT-CA on the labels (unique + case_tie as targets, ties
   as option sets); evaluate on the test split against the vowels.
3. Plug it into layer 1 in place of the MSA disambiguator and re-run layer 3.
