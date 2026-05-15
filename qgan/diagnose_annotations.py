#!/usr/bin/env python3
# qgan/diagnose_annotations.py
#
# Run this FIRST to see exactly what annotation strings are in your EDF files.
# Then the fix in data_loader_journal.py can be tailored to the real strings.
#
# Run: python -m qgan.diagnose_annotations
#
# Output: prints the first 30 unique annotation strings from each subject,
#         then writes full list to annotation_audit.json

import os
import json
import pyedflib

SUBJECT_PATHS = [
    "data/EPCTL01.edf",
    "data/EPCTL02.edf",
    "data/EPCTL03.edf",
    "data/EPCTL04.edf",
    "data/EPCTL05.edf",
    "data/EPCTL06.edf",
    "data/EPCTL07.edf",
    "data/EPCTL09.edf",
    "data/EPCTL10.edf",
    "data/EPCTL11.edf",
]

def diagnose(paths=None):
    if paths is None:
        paths = SUBJECT_PATHS

    all_audit = {}
    all_unique_across_files = set()

    print("\n" + "="*70)
    print("  ANNOTATION AUDIT — ANPHY-Sleep EDF files")
    print("="*70)

    for path in paths:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"\n  {name}: FILE NOT FOUND")
            continue

        with pyedflib.EdfReader(path) as f:
            try:
                ann = f.readAnnotations()
                onsets = ann[0]
                descs  = ann[2]
            except Exception as e:
                print(f"\n  {name}: CANNOT READ ANNOTATIONS — {e}")
                continue

        # Collect all raw annotation strings
        raw_strings = []
        for onset, desc in zip(onsets, descs):
            s = str(desc).strip()
            raw_strings.append((float(onset), s))

        unique_descs = sorted(set(s for _, s in raw_strings))
        all_unique_across_files.update(unique_descs)
        all_audit[name] = {
            "n_annotations": len(raw_strings),
            "n_unique":      len(unique_descs),
            "unique_strings": unique_descs,
            "first_20_with_onset": [
                {"onset": o, "desc": d} for o, d in raw_strings[:20]
            ],
        }

        print(f"\n  {name}")
        print(f"  Total annotations : {len(raw_strings)}")
        print(f"  Unique strings    : {len(unique_descs)}")
        print(f"  All unique annotation strings:")
        for s in unique_descs:
            print(f"    repr: {repr(s)}")

    print("\n" + "="*70)
    print("  ALL UNIQUE ANNOTATION STRINGS ACROSS ALL FILES:")
    print("="*70)
    for s in sorted(all_unique_across_files):
        print(f"  repr: {repr(s)}")

    # Save full audit
    out = "annotation_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"all_unique": sorted(all_unique_across_files),
             "per_file":   all_audit},
            f, indent=2, ensure_ascii=False
        )
    print(f"\n  Full audit saved to: {out}")
    print("  Share the repr() strings above and the fix will be exact.\n")

if __name__ == "__main__":
    diagnose()