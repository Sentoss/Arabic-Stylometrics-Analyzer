# Can rules supply the heads MASAQ lacks? A test on three sūras

MASAQ (v5, CC BY 4.0) gives every word its i'rāb role but not its head.
`tools/irab_rules.py` adds heads by rule (using MASAQ's roles, case,
definiteness and clause boundaries) and compares them with الجدول في إعراب
القرآن (محمود صافي), wherever الجدول names the head explicitly. الجدول is used
here only as a yardstick; its text is not redistributed.

Sūras: al-Baqara, Yūsuf, al-Naba' (8,063 words). Gold facts could be extracted
for part of the text only (where الجدول's wording names the head and the entry
could be aligned with MASAQ's words).

| what | correct |
|---|---|
| hidden subject (ضمير مستتر) or not, per verb | 1036/1100 (94%) |
| نعت: which noun it describes | 97/117 (83%) |
| جار ومجرور attached to a word (verb, اسم فاعل…) | 373/430 (87%) |
| جار ومجرور متعلق بمحذوف خبر | 66/91 (73%) |
| جار ومجرور متعلق بمحذوف صفة | about 18/55 (33%) |
| جار ومجرور متعلق بمحذوف حال | 0/81 (0%) |
| معطوف على X / بدل من X | 9/21, 5/8 (small samples) |

All جار ومجرور together: about 69%.

**Verdict: the rules do not hold.** Mechanical facts (hidden subjects, most
attachments to an expressed verb) come out well, but the distinctions that
depend on meaning do not: حال versus attachment to the verb, صفة versus حال,
which noun a معطوف joins. These are exactly the analyses a naḥw-faithful parser
must get right. Training data with real heads is needed (الجدول or an
equivalent, or hand-checked annotation).
