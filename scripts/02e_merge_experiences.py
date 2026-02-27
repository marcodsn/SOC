import hashlib
import json
import re
from datetime import datetime
from glob import glob
from typing import List, Optional


def parse_timestamp_from_filename(file_path: str) -> Optional[datetime]:
    """Extract datetime from filename like experiences_YYYYMMDD_HHMMSS.jsonl"""
    match = re.search(r"experiences_(\d{8})_(\d{6})\.jsonl$", file_path)
    if match:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    return None


def merge_experience_files(
    input_files: List[str],
    output_file: str,
    min_datetime: Optional[datetime] = None,
):
    """
    Merge multiple JSONL files containing experience data.
    Adds unique ID and source file timestamp to each experience's meta.
    Optionally skips files created before a given datetime.

    Args:
        input_files:  List of paths to input JSONL files
        output_file:  Path to output merged JSONL file
        min_datetime: If set, skip files with a parsed timestamp strictly before this value
    """
    merged_experiences = []

    for file_path in sorted(input_files):  # sorted -> chronological processing order
        file_ts = parse_timestamp_from_filename(file_path)

        # --- datetime filter ---
        if min_datetime is not None:
            if file_ts is None:
                print(f"Warning: Cannot parse timestamp from '{file_path}', skipping.")
                continue
            if file_ts < min_datetime:
                print(
                    f"Skipping {file_path}  ({file_ts.isoformat()} < {min_datetime.isoformat()})"
                )
                continue

        print(f"Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    experience = json.loads(line)

                    if "meta" not in experience:
                        experience["meta"] = {}

                    # Store the source file's timestamp in meta
                    if file_ts is not None:
                        experience["meta"]["source_timestamp"] = file_ts.isoformat()

                    # Remove iteration number if present
                    experience["meta"].pop("iteration", None)

                    # Unique ID — computed last so it reflects the full meta content
                    experience_text = json.dumps(
                        experience, sort_keys=True, ensure_ascii=False
                    )
                    experience["meta"]["id"] = hashlib.md5(
                        experience_text.encode()
                    ).hexdigest()

                    merged_experiences.append(experience)

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON in {file_path}, line {line_num}: {e}"
                    )

    with open(output_file, "w", encoding="utf-8") as f:
        for experience in merged_experiences:
            f.write(json.dumps(experience, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(merged_experiences)} experiences into {output_file}")


if __name__ == "__main__":
    # Only merge files from this datetime onwards (set to None to merge all)
    min_dt = datetime(2026, 2, 24, 10, 0, 0)

    input_files = glob("data/experiences/generated/experiences_*.jsonl")
    output_file = "data/experiences/generated/data.jsonl"

    merge_experience_files(input_files, output_file, min_datetime=min_dt)
