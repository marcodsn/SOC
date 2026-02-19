import hashlib
import json
import sys
from glob import glob
from pathlib import Path
from typing import List


def merge_experience_files(input_files: List[str], output_file: str):
    """
    Merge multiple JSONL files containing experience data.
    Adds unique ID to each experience and removes iteration number.

    Args:
        input_files: List of paths to input JSONL files
        output_file: Path to output merged JSONL file
    """
    merged_experiences = []

    for file_path in input_files:
        print(f"Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    experience = json.loads(line)

                    # Add unique ID
                    if "meta" not in experience:
                        experience["meta"] = {}

                    # Create a deterministic ID based on content
                    experience_text = json.dumps(experience, ensure_ascii=False)
                    experience_id = hashlib.md5(experience_text.encode()).hexdigest()
                    experience["meta"]["id"] = experience_id

                    # Remove iteration number if present
                    if "iteration" in experience["meta"]:
                        del experience["meta"]["iteration"]

                    merged_experiences.append(experience)

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON in {file_path}, line {line_num}: {e}"
                    )

    # Write merged data
    with open(output_file, "w", encoding="utf-8") as f:
        for experience in merged_experiences:
            f.write(json.dumps(experience, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(merged_experiences)} experiences into {output_file}")


if __name__ == "__main__":
    # Determine the base directory (SOC/)
    # Resolves to .../SOC/
    base_dir = Path(__file__).resolve().parent.parent

    # Define data directory
    data_dir = base_dir / "data" / "experiences" / "generated"

    # Input files pattern: experiences_*.jsonl
    # This matches experiences_20260212_164333.jsonl, etc.
    input_pattern = str(data_dir / "experiences_*.jsonl")
    input_files = glob(input_pattern)

    # Output file: merged_experiences.jsonl
    output_file = data_dir / "merged_experiences.jsonl"

    # Ensure we don't include the output file in the input if it already exists
    # (though the pattern 'experiences_' vs 'merged_' shouldn't overlap, it's good practice)
    if str(output_file) in input_files:
        input_files.remove(str(output_file))

    # Sort input files to ensure deterministic order
    input_files.sort()

    if not input_files:
        print(f"No input files found matching pattern: {input_pattern}")
        sys.exit(1)

    merge_experience_files(input_files, str(output_file))
