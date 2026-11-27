# Lane's Lexicon in the layer-2 dictionary

The layer-2 dictionary is CAMeL's MSA morphology database plus rule-generated
forms (see `layer2/README.md`). This folder adds **words** from Edward William
Lane's *Arabic-English Lexicon* (1863–93), using the Perseus Digital Library's
TEI XML as restructured and corrected by Winfried H. Leclaire
(github.com/quranic-research/lexicon_xml).

> Text provided by Perseus Digital Library, with funding from The U.S.
> Department of Education and The Max Planck Society.

## Steps

1. `parse_lane.py XML_DIR lane_rows.jsonl`: roots, vowelled headwords, verb
   forms with the Form I aorist vowel, and from the entry text the plurals
   ("pl."), infinitive nouns ("inf. n.") and feminines ("fem."). Arabic is in
   the edition's own transliteration (`internal_map.json`, from its
   `Encoding.dtd`).
2. `coverage.py lane_rows.jsonl lane_cov.jsonl`: which Lane words the
   dictionary already has.
3. `tashkeela_forms.py TASHKEELA/texts.txt forms_full.tsv`: counts of every
   fully vowelled form in the full Tashkeela (97 Classical books, 66 M words;
   the modern `msa` folder is left out).
4. `build_lane_stems.py lane_cov.jsonl forms_full.tsv ../resources/lane_stems.tsv`:
   nouns, adjectives, plurals, maṣdars and feminines as CAMeL stem entries.
   Each gets the inflection class most common among existing stems with the
   same pattern (وزن), feminine/plural status and part of speech. A word is
   added only if
   - its bare form is new to the dictionary, alone and with any proclitic
     (و، ف، ب، ل، ك، ال …) or common suffix; or
   - it competes with existing readings but Tashkeela attests its vowelling
     at least 3 times and at least 2% as often as the competing readings.
   Never added: words that with a proclitic have the same vowels as an
   existing word (كَ + ثِير = كَثِير), which vowels could not rule out.
   The Tashkeela frequency sets each new reading's prior.
5. `lexicon_db.py`: builds the merged database (cached in
   `~/.cache/arabic_stylometrics/`, not in the repository) for
   `Analyzer(lane_db())`; `build_labels.py --lane` and
   `qac_eval/run_system.py MODEL+lane` use it.

Verbs are not added yet (CAMeL's verb classes need the aorist vowel and the
weak-root type per stem); Lane's own work ends at the letter qāf, so later
letters are thin.
