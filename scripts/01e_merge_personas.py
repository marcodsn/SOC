import hashlib
import json
import re
from datetime import datetime
from glob import glob
from typing import List, Optional


def parse_timestamp_from_filename(file_path: str) -> Optional[datetime]:
    """Extract datetime from filename like persona_YYYYMMDD_HHMMSS.jsonl"""
    match = re.search(r"personas_(\d{8})_(\d{6})\.jsonl$", file_path)
    if match:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    return None


def merge_persona_files(
    input_files: List[str],
    output_file: str,
    min_datetime: Optional[datetime] = None,
):
    """
    Merge multiple JSONL files containing persona data.
    Adds unique ID and source file timestamp to each persona's meta.
    Optionally skips files created before a given datetime.

    Args:
        input_files:  List of paths to input JSONL files
        output_file:  Path to output merged JSONL file
        min_datetime: If set, skip files with a parsed timestamp strictly before this value
    """
    merged_personas = []

    for file_path in sorted(input_files):  # sorted → chronological processing order
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
                    persona = json.loads(line)

                    if "meta" not in persona:
                        persona["meta"] = {}

                    # Store the source file's timestamp in meta
                    if file_ts is not None:
                        persona["meta"]["source_timestamp"] = file_ts.isoformat()

                    # Remove iteration number if present
                    persona["meta"].pop("iteration", None)

                    # Unique ID — computed last so it reflects the full meta content
                    persona_text = json.dumps(
                        persona, sort_keys=True, ensure_ascii=False
                    )
                    persona["meta"]["id"] = hashlib.md5(
                        persona_text.encode()
                    ).hexdigest()

                    merged_personas.append(persona)

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON in {file_path}, line {line_num}: {e}"
                    )

    with open(output_file, "w", encoding="utf-8") as f:
        for persona in merged_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(merged_personas)} personas into {output_file}")


if __name__ == "__main__":
    # Only merge files from this datetime onwards (set to None to merge all)
    min_dt = datetime(2026, 2, 22, 17, 50, 0)

    input_files = glob("data/personas/generated/personas_*.jsonl")
    # output_file = (
    #     f"data/personas/generated/merged_{min_dt.strftime('%Y%m%d_%H%M%S')}.jsonl"
    # )
    output_file = "data/personas/generated/data.jsonl"

    merge_persona_files(input_files, output_file, min_datetime=min_dt)
