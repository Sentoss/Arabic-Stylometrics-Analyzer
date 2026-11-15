"""Run the restorer on every sample and print what it changed."""
from pathlib import Path

from restorer import Restorer
from restore_cli import write_outputs

R = Restorer()
for path in sorted(Path("samples").glob("*.txt")):
    text, prof, res = R.run(path.read_text(encoding="utf-8"))
    write_outputs(path.stem, Path("out"), text, prof, res)
    words = [r for r in res if r.status != "not_word"]
    ch = [r for r in words if r.changed]
    print(f"\n=== {path.stem}: {len(words)} words, {len(ch)} changed")
    print("conventions:", {k: v for k, v in prof.conventions.items() if v}, prof.notes)
    print("evidence:", prof.evidence)
    for r in ch:
        print(f"  {r.raw:>14} -> {r.restored:<14} {r.confidence:<7} {','.join(r.conventions)}"
              + (f"  alt={r.alternatives}" if r.alternatives else ""))
    amb = [r for r in words if r.status in ("ambiguous", "no_compatible")]
    print(f"  -- {len(amb)} flagged for review:",
          [(r.raw, r.restored, r.alternatives[:2]) for r in amb][:25])
