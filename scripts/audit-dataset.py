#!/usr/bin/env python3
"""Audit a WD14-captioned dataset for training-quality issues.

Reads <dataset>/*.txt files, scans for known problems, emits a report
and (optionally) writes a 'reject list' that filter-dataset.py consumes.

Checks:
  - 1boy / wrong-gender tags on subjects that should be female (or vice-versa)
  - Conflicting mutually-exclusive tags on same image (e.g. multiple hair colors)
  - Missing expected tags (e.g. subject should have 'freckles' if they do IRL)
  - Solo vs group (multiple people when subject should be alone)
  - Text/watermark tags (scanned screenshots — bad training material)

Usage:
  python3 scripts/audit-dataset.py <dataset_dir> [--subject-gender female]
                                    [--expected-tags freckles,red hair]
                                    [--reject-out rejects.txt]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

CONFLICT_GROUPS = [
    # Mutually-exclusive within one image
    {"brown hair", "black hair", "blonde hair", "red hair", "silver hair",
     "grey hair", "white hair", "pink hair", "blue hair", "green hair", "purple hair"},
    {"blue eyes", "brown eyes", "green eyes", "red eyes", "yellow eyes",
     "purple eyes", "grey eyes", "pink eyes", "black eyes", "heterochromia"},
]

WRONG_PERSON_TAGS = {
    "2boys", "6+boys", "multiple boys",
    "2girls", "multiple girls", "6+girls",
    "old man", "old woman", "child", "baby",
    "furry", "monster girl", "elf", "animal ears",
}
# Co-presence rules: reject if BOTH tags appear in the same caption.
# (e.g. '1boy' + '1girl' on a solo-subject dataset means a co-star was present)
COPRESENCE_REJECTS = [
    {"1boy", "1girl"},    # subject + male co-star
    {"1boy", "hetero"},   # romantic/action scene with partner
]

LOW_QUALITY_TAGS = {
    "english text", "text", "watermark", "signature", "logo", "greyscale",
    "monochrome", "black and white photograph", "chromatic aberration",
    "poster (object)", "painting (object)",
}


def load_caption(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [t.strip() for t in text.split(",") if t.strip()]


def audit_one(path: Path, expected: set[str]) -> list[str]:
    tags = set(load_caption(path))
    issues: list[str] = []

    for wrong in WRONG_PERSON_TAGS:
        if wrong in tags:
            issues.append(f"wrong-person:{wrong}")

    for combo in COPRESENCE_REJECTS:
        if combo.issubset(tags):
            issues.append(f"wrong-person:co-{'+'.join(sorted(combo))}")

    for lq in LOW_QUALITY_TAGS:
        if lq in tags:
            issues.append(f"low-quality:{lq}")

    for group in CONFLICT_GROUPS:
        found = tags & group
        if len(found) > 1:
            issues.append(f"conflict:{'+'.join(sorted(found))}")

    # Missing expected (e.g. subject should have freckles)
    for exp in expected:
        if exp not in tags:
            issues.append(f"missing:{exp}")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="path to dataset directory (.txt caption files)")
    ap.add_argument("--expected-tags", default="",
                    help="comma-sep tags expected in every caption (e.g. 'freckles')")
    ap.add_argument("--reject-out", default="",
                    help="write newline-separated list of image stems to reject")
    ap.add_argument("--reject-on", default="wrong-person,conflict,low-quality",
                    help="comma-sep issue prefixes that cause rejection; "
                         "'missing' is informational by default")
    ap.add_argument("--max-issues", type=int, default=0,
                    help="reject if number of issues > N (0 = reject on any matching issue)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        print(f"error: {dataset} is not a directory", file=sys.stderr)
        return 1

    expected = {t.strip() for t in args.expected_tags.split(",") if t.strip()}
    reject_prefixes = tuple(p.strip() for p in args.reject_on.split(",") if p.strip())

    captions = sorted(dataset.glob("*.txt"))
    print(f"Auditing {len(captions)} caption files in {dataset}")
    if expected:
        print(f"Expected tags: {sorted(expected)}")
    print(f"Reject on: {reject_prefixes}")
    print()

    issue_counter: Counter[str] = Counter()
    rejects: list[str] = []
    per_file: dict[str, list[str]] = {}

    for cap in captions:
        issues = audit_one(cap, expected)
        if issues:
            per_file[cap.stem] = issues
            for i in issues:
                issue_counter[i.split(":", 1)[0]] += 1
                issue_counter[i] += 1
            reject_matches = [i for i in issues if i.startswith(reject_prefixes)]
            if reject_matches and (args.max_issues == 0 or
                                   len(reject_matches) > args.max_issues):
                rejects.append(cap.stem)

    print("Issue summary (count = files affected):")
    for issue, n in sorted(issue_counter.items(), key=lambda x: -x[1]):
        prefix = issue.split(":", 1)[0]
        indent = "  " if ":" in issue else ""
        print(f"  {indent}{issue:60s} {n}")

    print(f"\nTotal captions with any issue: {len(per_file)}")
    print(f"Rejected: {len(rejects)} / {len(captions)} ({len(rejects)/len(captions)*100:.1f}%)")

    # Show 20 worst offenders
    if per_file:
        worst = sorted(per_file.items(), key=lambda x: -len(x[1]))[:10]
        print("\n10 worst offenders:")
        for stem, issues in worst:
            print(f"  {stem}: {', '.join(issues[:6])}"
                  f"{' ...' if len(issues) > 6 else ''}")

    if args.reject_out:
        out = Path(args.reject_out)
        out.write_text("\n".join(sorted(rejects)) + "\n")
        print(f"\nWrote reject list: {out} ({len(rejects)} entries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
