#!/usr/bin/env python3
"""
Generate conversations from paired personas and experiences.

Usage:
    python scripts/03_gen_conversations.py --num 50
    python scripts/03_gen_conversations.py --num 20 --style freeform
    python scripts/03_gen_conversations.py \\
        --model "moonshotai/Kimi-K2.5:fireworks-ai" \\
        --summarizer-model "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.conversation_generator import ConversationGenerator
from soc_2602.llm.client import LLMClient


def load_experiences(path: str, filters: dict) -> list:
    """Load and optionally filter experience records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = record.get("meta", {})

            if (
                filters.get("style")
                and meta.get("conversation_style") != filters["style"]
            ):
                continue
            if (
                filters.get("cadence")
                and meta.get("message_cadence") != filters["cadence"]
            ):
                continue

            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate conversations from experiences"
    )

    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of experiences to process (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/Kimi-K2.5:fireworks-ai",
        help="Model ID for turn generation",
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default=None,
        help="Model ID for summarization (defaults to --model if not set)",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="data/personas/generated/b_merged_personas.jsonl",
        help="Path to merged personas JSONL file",
    )
    parser.add_argument(
        "--experiences",
        type=str,
        default="data/experiences/generated/experiences_20260221_140037.jsonl",
        help="Path to merged experiences JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/conversations/generated",
        help="Output directory for generated conversations",
    )
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default="conf/prompts",
        help="Directory containing Jinja prompt templates",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of conversations to run concurrently (default: 4)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=["structured", "semi-structured", "freeform"],
        help="Filter experiences by conversation style",
    )
    parser.add_argument(
        "--cadence",
        type=str,
        default=None,
        choices=["realtime", "delayed", "async"],
        help="Filter experiences by message cadence",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within filtered experience list (for resuming runs)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Clients
    # -------------------------------------------------------------------------
    print(f"Initializing turn model: {args.model}")
    llm_client = LLMClient(model_id=args.model)

    summarizer_client = None
    if args.summarizer_model:
        print(f"Initializing summarizer model: {args.summarizer_model}")
        summarizer_client = LLMClient(model_id=args.summarizer_model)

    # -------------------------------------------------------------------------
    # Load and filter experiences
    # -------------------------------------------------------------------------
    filters = {"style": args.style, "cadence": args.cadence}
    print(f"Loading experiences from: {args.experiences}")
    experiences = load_experiences(args.experiences, filters)

    if not experiences:
        print("No experiences found after applying filters. Exiting.")
        sys.exit(1)

    # Apply start index and num cap
    experiences = experiences[args.start_index :]
    if args.num is not None:
        experiences = experiences[: args.num]

    print(f"Selected {len(experiences)} experience(s) to process")
    if args.style or args.cadence:
        print(
            f"  Filters applied — style={args.style or 'any'}, cadence={args.cadence or 'any'}"
        )

    # -------------------------------------------------------------------------
    # Output path
    # -------------------------------------------------------------------------
    output_file = (
        Path(args.output_dir)
        / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # -------------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------------
    generator = ConversationGenerator(
        llm_client=llm_client,
        summarizer_client=summarizer_client,
        persona_file=args.personas,
        experience_file=args.experiences,
        output_file=str(output_file),
        prompt_dir=args.prompt_dir,
    )

    generator.generate_batch(
        experience_records=experiences,
        batch_size=args.batch_size,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
