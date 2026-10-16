"""Re-apply the selection key (with the simplicity tie-break) to saved summaries' finals; prints changed choices."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.method import selection_key  # noqa: E402

for p in map(Path, sys.argv[1:]):
    s = json.loads(p.read_text(encoding="utf-8"))
    old = s["best"]["from"]
    s["finals"].sort(key=selection_key, reverse=True)
    s["best"] = s["finals"][0]
    p.write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print(p, "kept", old, "->", s["best"]["from"], "(changed)" if old != s["best"]["from"] else "")
