import json
import uuid
from glob import glob
from typing import List


def merge_persona_files(input_files: List[str], output_file: str):
    """
    Merge multiple JSONL files containing persona data.
    Adds unique ID to each persona and removes iteration number.

    Args:
        input_files: List of paths to input JSONL files
        output_file: Path to output merged JSONL file
    """
    merged_personas = []

    for file_path in input_files:
        print(f"Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    persona = json.loads(line)

                    # Add unique ID
                    if "meta" not in persona:
                        persona["meta"] = {}
                    persona["meta"]["id"] = str(uuid.uuid4())

                    # Remove iteration number if present
                    if "iteration" in persona["meta"]:
                        del persona["meta"]["iteration"]

                    merged_personas.append(persona)

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON in {file_path}, line {line_num}: {e}"
                    )

    # Write merged data
    with open(output_file, "w", encoding="utf-8") as f:
        for persona in merged_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(merged_personas)} personas into {output_file}")


if __name__ == "__main__":
    personas_v = "b"
    input_files = glob(f"data/personas/generated/{personas_v}_personas_*.jsonl")

    output_file = f"data/personas/generated/{personas_v}_merged_personas.jsonl"

    merge_persona_files(input_files, output_file)
