"""Parse the Wikisource imla'i Qur'an wikitext into {(sura,verse): text}."""
import re, json, sys
def parse(path):
    t = open(path, encoding='utf-8').read()
    parts = re.split(r'^==\s*(سورة[^=]+?)\s*==\s*$', t, flags=re.M)
    suras = []
    for i in range(1, len(parts), 2):
        suras.append((parts[i].strip(), parts[i+1]))
    out = {}
    for si, (name, body) in enumerate(suras, 1):
        body = re.sub(r'\{\{[^{}]*\}\}', ' ', body)
        body = re.sub(r'<[^>]+>', ' ', body)
        for m in re.finditer(r'(.*?)[\(﴿]\s*([0-9٠-٩]+)\s*[\)﴾]', body, flags=re.S):
            n = int(m.group(2).translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')))
            out[(si, n)] = ' '.join(m.group(1).split())
    return suras, out
if __name__ == '__main__':
    suras, v = parse(sys.argv[1])
    print(len(suras), len(v))
    from collections import Counter
    c = Counter(s for s, _ in v)
    print([ (s,c[s]) for s in list(c)[:5]], c[2], c[114])
    for k in [(1,1),(2,1),(2,2),(9,1),(20,94),(114,6)]: print(k, v.get(k))
    json.dump({f'{a}:{b}': t for (a, b), t in v.items()}, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)
