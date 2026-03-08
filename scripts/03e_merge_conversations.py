#!/usr/bin/env python3
"""
Merge multiple conversation JSONL files into a single unified file.

Adds unique IDs and source file timestamps to each conversation's meta,
analogous to 01e_merge_personas.py and 02e_merge_experiences.py.

Usage:
    python scripts/03e_merge_conversations.py
    python scripts/03e_merge_conversations.py --min-datetime "2026-02-27T10:00:00"
    python scripts/03e_merge_conversations.py --output data/conversations/generated/merged.jsonl
"""

import hashlib
import json
import re
from datetime import datetime
from glob import glob
from typing import List, Optional


def parse_timestamp_from_filename(file_path: str) -> Optional[datetime]:
    """Extract datetime from filename like conversations_YYYYMMDD_HHMMSS.jsonl"""
    match = re.search(r"conversations_(\d{8})_(\d{6})\.jsonl$", file_path)
    if match:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    return None


def merge_conversation_files(
    input_files: List[str],
    output_file: str,
    min_datetime: Optional[datetime] = None,
):
    """
    Merge multiple JSONL files containing conversation data.
    Adds unique ID and source file timestamp to each conversation's meta.
    Optionally skips files created before a given datetime.

    Args:
        input_files:  List of paths to input JSONL files
        output_file:  Path to output merged JSONL file
        min_datetime: If set, skip files with a parsed timestamp strictly before this value
    """
    merged_conversations = []
    skipped_files = 0
    error_lines = 0

    for file_path in sorted(input_files):  # sorted -> chronological processing order
        file_ts = parse_timestamp_from_filename(file_path)

        # --- datetime filter ---
        if min_datetime is not None:
            if file_ts is None:
                print(f"Warning: Cannot parse timestamp from '{file_path}', skipping.")
                skipped_files += 1
                continue
            if file_ts < min_datetime:
                print(
                    f"Skipping {file_path}  ({file_ts.isoformat()} < {min_datetime.isoformat()})"
                )
                skipped_files += 1
                continue

        print(f"Processing {file_path}...")
        file_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    conversation = json.loads(line)

                    if "meta" not in conversation:
                        conversation["meta"] = {}

                    # Store the source file's timestamp in meta
                    if file_ts is not None:
                        conversation["meta"]["source_timestamp"] = file_ts.isoformat()

                    # Remove record_index — will be reassigned after merge
                    conversation["meta"].pop("record_index", None)

                    # Unique ID — computed last so it reflects the full meta content
                    conversation_text = json.dumps(
                        conversation, sort_keys=True, ensure_ascii=False
                    )
                    conversation["meta"]["id"] = hashlib.md5(
                        conversation_text.encode()
                    ).hexdigest()

                    merged_conversations.append(conversation)
                    file_count += 1

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON in {file_path}, line {line_num}: {e}"
                    )
                    error_lines += 1

        print(f"  → {file_count} conversations loaded")

    # Write merged output
    with open(output_file, "w", encoding="utf-8") as f:
        for conversation in merged_conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Merge complete!")
    print(f"  Total conversations: {len(merged_conversations)}")
    print(f"  Output file:         {output_file}")
    if skipped_files:
        print(f"  Skipped files:       {skipped_files}")
    if error_lines:
        print(f"  Skipped error lines: {error_lines}")

    # Print style/cadence summary
    from collections import Counter

    styles = Counter()
    cadences = Counter()
    exhausted = 0
    for conv in merged_conversations:
        meta = conv.get("meta", {})
        styles[meta.get("conversation_style", "unknown")] += 1
        cadences[meta.get("message_cadence", "unknown")] += 1
        if meta.get("exhausted"):
            exhausted += 1

    print(f"\n  Style distribution:   {dict(styles)}")
    print(f"  Cadence distribution: {dict(cadences)}")
    print(
        f"  Exhausted:            {exhausted} / {len(merged_conversations)} "
        f"({exhausted / len(merged_conversations) * 100:.1f}%)"
        if merged_conversations
        else ""
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge multiple conversation JSONL files"
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="data/conversations/generated/conversations_*.jsonl",
        help="Glob pattern for input files (default: data/conversations/generated/conversations_*.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/conversations/generated/data.jsonl",
        help="Output merged JSONL file path",
    )
    parser.add_argument(
        "--min-datetime",
        type=str,
        default=None,
        help="Only include files with timestamps >= this ISO datetime (e.g. 2026-02-27T10:00:00)",
    )
    args = parser.parse_args()

    min_dt = None
    if args.min_datetime:
        min_dt = datetime.fromisoformat(args.min_datetime)

    input_files = glob(args.input_pattern)
    if not input_files:
        print(f"No files matched pattern: {args.input_pattern}")
        exit(1)

    print(f"Found {len(input_files)} file(s) matching '{args.input_pattern}'")
    merge_conversation_files(input_files, args.output, min_datetime=min_dt)
