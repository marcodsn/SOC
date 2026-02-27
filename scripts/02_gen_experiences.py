#!/usr/bin/env python3
"""
Generate experiences for personas using iterative sampling.

Usage:
    python scripts/02_gen_experiences.py --num 50
    python scripts/02_gen_experiences.py --num 50 --model "moonshotai/Kimi-K2.5:fireworks-ai"
    python scripts/02_gen_experiences.py --num 50 --provider nim
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.experience_generator import ExperienceGenerator
from soc_2602.llm.client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="Generate conversation experiences")

    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of experiences to generate (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model ID to use (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["modal", "nim", "huggingface", "mistral", "local"],
        help="LLM provider to use (default: local)",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="data/personas/generated/data.jsonl",
        help="Path to merged personas JSONL file",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="data/experiences/seed",
        help="Directory containing seed experiences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiences/generated",
        help="Output directory for generated experiences",
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
        help="Number of experiences to run concurrently (default: 4)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Client
    # -------------------------------------------------------------------------
    print(f"Initializing model: {args.model} (provider: {args.provider})")
    llm_client = LLMClient(model_id=args.model, provider=args.provider)

    # -------------------------------------------------------------------------
    # Output path
    # -------------------------------------------------------------------------
    output_file = (
        Path(args.output_dir)
        / f"experiences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # -------------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------------
    generator = ExperienceGenerator(
        llm_client=llm_client,
        persona_file=args.personas,
        output_file=str(output_file),
        seed_dir=args.seed_dir,
        prompt_dir=args.prompt_dir,
    )

    generator.generate_batch(num_experiences=args.num, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
