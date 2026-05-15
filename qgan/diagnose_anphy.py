#!/usr/bin/env python3
# qgan/diagnose_anphy.py
#
# ANPHY-Sleep has NO sleep stage annotations in the EDF signal file.
# This script finds WHERE the stage labels actually are.
#
# Run: python -m qgan.diagnose_anphy
#
# It checks ALL common ANPHY-Sleep label storage locations:
#   1. Companion *-nsrr.xml or *.xml files (NSRR format)
#   2. Companion *.csv files (stage per row)
#   3. Companion *.txt files
#   4. Companion *_annotations.edf (separate EDF annotations file)
#   5. Additional EDF signal channels (some PSG systems encode stage as a channel)
#   6. The EDF header itself (some systems use the "local recording identification")

import os
import glob
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

def diagnose_all():
    print("\n" + "="*70)
    print("  ANPHY-SLEEP LABEL LOCATION DIAGNOSTIC")
    print("="*70)

    found_anything = False

    for edf_path in SUBJECT_PATHS:
        name = os.path.basename(edf_path)
        if not os.path.exists(edf_path):
            continue

        base_dir  = os.path.dirname(os.path.abspath(edf_path))
        stem      = os.path.splitext(name)[0]   # e.g. "EPCTL01"

        print(f"\n{'─'*70}")
        print(f"  Subject: {name}")
        print(f"  Base dir: {base_dir}")
        print(f"  Stem: {stem}")

        # ── 1. Scan the data directory for companion files ──────────────────
        print(f"\n  Files in same directory:")
        try:
            all_files = os.listdir(base_dir)
            for f in sorted(all_files):
                print(f"    {f}")
        except Exception as e:
            print(f"    ERROR listing dir: {e}")

        # ── 2. Look for companion files with same stem ──────────────────────
        extensions = [
            ".xml", "-nsrr.xml", "_nsrr.xml",
            ".csv", "_stages.csv", "_labels.csv", "_annotations.csv",
            ".txt", "_stages.txt", "_labels.txt",
            "-profusion.xml", ".eannot",
            "_hypnogram.txt", "_hypnogram.csv",
            ".tsv", "_events.tsv",
        ]
        print(f"\n  Companion files matching '{stem}':")
        found_companion = False
        for ext in extensions:
            candidate = os.path.join(base_dir, stem + ext)
            if os.path.exists(candidate):
                size = os.path.getsize(candidate)
                print(f"    FOUND: {stem}{ext}  ({size} bytes)")
                found_companion = True
                found_anything  = True
                # Try to read first 10 lines
                try:
                    with open(candidate, "r", encoding="utf-8",
                              errors="replace") as f:
                        lines = [next(f) for _ in range(10)]
                    print(f"    First 10 lines:")
                    for line in lines:
                        print(f"      {repr(line.rstrip())}")
                except Exception as e:
                    print(f"    Cannot read as text: {e}")

        if not found_companion:
            print(f"    None found with known extensions")

        # ── 3. Wildcard search for any file containing the stem ─────────────
        print(f"\n  Any file in data dir containing '{stem.lower()}':")
        try:
            for f in sorted(os.listdir(base_dir)):
                if stem.lower() in f.lower() and f != name:
                    fpath = os.path.join(base_dir, f)
                    size  = os.path.getsize(fpath)
                    print(f"    {f}  ({size} bytes)")
                    found_anything = True
        except Exception as e:
            print(f"    ERROR: {e}")

        # ── 4. Check EDF signal channels ────────────────────────────────────
        print(f"\n  EDF signal channels:")
        try:
            with pyedflib.EdfReader(edf_path) as f:
                n_signals = f.signals_in_file
                labels    = f.getSignalLabels()
                freqs     = [f.getSampleFrequency(i) for i in range(n_signals)]
                n_samples = [f.getNSamples()[i] for i in range(n_signals)]
                duration  = f.getFileDuration()
                print(f"    File duration: {duration}s")
                print(f"    Channels ({n_signals} total):")
                for i, (lbl, freq, ns) in enumerate(zip(labels, freqs, n_samples)):
                    print(f"      [{i:2d}] '{lbl}'  {freq}Hz  {ns} samples")
                    # Flag any channel that looks like a stage channel
                    lbl_up = lbl.strip().upper()
                    if any(kw in lbl_up for kw in [
                        "STAGE", "HYPNO", "SCORE", "SLEEP", "ANNOT",
                        "LABEL", "CLASS", "PSG", "EPOCH"
                    ]):
                        print(f"           *** POSSIBLE STAGE CHANNEL ***")
                        # Read first 10 values
                        try:
                            sig = f.readSignal(i)
                            print(f"           First 10 values: {sig[:10].tolist()}")
                            print(f"           Unique values: {sorted(set(sig[:500].astype(int).tolist()))}")
                        except Exception as e:
                            print(f"           Cannot read: {e}")
        except Exception as e:
            print(f"    ERROR reading EDF: {e}")

        # ── 5. Check EDF header fields ───────────────────────────────────────
        print(f"\n  EDF header fields:")
        try:
            with pyedflib.EdfReader(edf_path) as f:
                print(f"    patient_name:     {repr(f.getPatientName())}")
                print(f"    patient_code:     {repr(f.getPatientCode())}")
                print(f"    recording_additional: {repr(f.getRecordingAdditional())}")
                print(f"    admincode:        {repr(f.getAdmincode())}")
                print(f"    equipment:        {repr(f.getEquipment())}")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Only process first 3 subjects to keep output manageable
        if stem == "EPCTL03":
            break

    print(f"\n{'='*70}")
    if not found_anything:
        print("""
  NO COMPANION FILES FOUND.

  The ANPHY-Sleep dataset distributes sleep stage annotations SEPARATELY
  from the EDF recording files. Common locations to check:

  1. Did you download ONLY the EDF files, or the complete dataset?
     The full download from OpenNeuro/PhysioNet typically includes:
       sub-*/ses-*/eeg/*_eeg.edf          (signal file)
       sub-*/ses-*/eeg/*_events.tsv       (stage labels — BIDS format)
       sub-*/ses-*/eeg/*_annotations.csv  (alternative)

  2. Check the dataset download page:
     https://openneuro.org/datasets/ds003768   (ANPHY-Sleep on OpenNeuro)
     or wherever you downloaded from.

  3. The ANPHY-Sleep paper (Wei et al. 2024) states annotations are in
     separate files scored by a board-certified PSG technologist.

  4. Check if there is a separate 'derivatives' or 'labels' folder in
     the dataset root (not inside 'data/').

  NEXT STEPS:
    - Print the full directory tree of your dataset folder:
        find . -type f | sort
      (Linux/Mac) or:
        Get-ChildItem -Recurse | Select FullName
      (Windows PowerShell)
    - Look for any .tsv, .csv, .xml, or .txt files
    - Share the output and the correct label format will be implemented
""")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    diagnose_all()