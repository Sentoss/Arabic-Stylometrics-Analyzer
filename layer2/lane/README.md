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

6. `name_twins.py CORPUS lane_names.tsv`: a proper-noun reading with the same
   vowels for the 433 Lane words our corpus uses as names (after بن، أبو، أم …):
   Lane defines مُصْعَب as a stallion camel, but our texts mean the man.
7. `build_lane_verbs.py lane_cov.jsonl forms_full.tsv ARRAMOOZ/verb_triliteral.csv lane_verbs.tsv`:
   1,403 verbs (Forms I–X of triliteral roots). Each copies the full stem set
   (perfect, imperfect, imperative, passives, weak-root shapes) of an existing
   verb with the same lemma pattern, with its own radicals. The Form I imperfect
   vowel comes from Lane ("aor."), else from Arramooz's conjugation class
   (باب). Derived forms of sound roots take the form's standard vowelling
   (Form VIII with an assimilating first radical, اِدَّكَرَ، اِصْطَبَرَ, only
   from a vowelled headword).
8. `arramooz_rows.py ARRAMOOZ/data/nouns arr_rows.jsonl`: nouns from Arramooz
   (Taha Zerrouki, github.com/linuxscout/arramooz, GPL) for roots beginning
   with ك ل م ن ه و ي, where Lane is thin (his own work ends at qāf); then
   steps 2 and 4 as for Lane → `arramooz_stems.tsv` (1,244 entries).

`lexicon_db.py` merges lane_stems, lane_verbs, arramooz_stems and lane_names.

Tried and rejected: the ~8,400 rare Lane words whose bare spelling is also a
common word, with a very low prior: most of the 538 extra prose changes were
wrong (زَهْد for زُهْد، الرَيْق، بَصْرِي), and Tashkeela ties rose by 0.6 points.
