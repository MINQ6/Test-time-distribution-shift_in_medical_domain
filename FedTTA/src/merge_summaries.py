"""client별로 쪼갠 추론 결과(<group>/client_*/summary.json)를 <group>/summary.json 으로 병합.

사용: python src/merge_summaries.py <group_dir>
"""
import glob
import json
import os
import sys


def main() -> None:
    group = sys.argv[1].rstrip("/")
    subs = sorted(glob.glob(os.path.join(group, "*", "summary.json")))
    if not subs:
        raise SystemExit(f"no */summary.json under {group}")
    base = None
    rows = []
    clients = []
    for p in subs:
        s = json.load(open(p))
        base = base or s
        rows.extend(s["rows"])
        clients.extend(s.get("clients", []))
    out = dict(base)
    out["clients"] = sorted(set(clients))
    out["rows"] = rows
    outp = os.path.join(group, "summary.json")
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=2)
    print(f"[merge] {len(subs)} files -> {outp} ({len(rows)} rows, clients={out['clients']})")


if __name__ == "__main__":
    main()
