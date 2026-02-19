#!/usr/bin/env python3
"""
Generate diverse personas using iterative sampling.

Usage:
    python scripts/generate_personas.py --num 50 --model "moonshotai/Kimi-K2.5:fireworks-ai"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.persona_generator import PersonaGenerator
from soc_2602.llm.client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="Generate diverse personas")
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of personas to generate (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/Kimi-K2.5:fireworks-ai",
        help="Model ID to use (default: moonshotai/Kimi-K2.5:fireworks-ai)",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="data/personas/seed",
        help="Directory containing seed personas",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/personas/generated",
        help="Output directory for generated personas",
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
        help="Batch size for asynchronous generation (default: 4)",
    )

    args = parser.parse_args()

    # Initialize LLM client
    print(f"Initializing LLM client with model: {args.model}")
    llm_client = LLMClient(model_id=args.model)

    # Initialize persona generator
    generator = PersonaGenerator(
        llm_client=llm_client,
        seed_dir=args.seed_dir,
        prompt_dir=args.prompt_dir,
    )

    # Generate personas
    generator.generate_batch(num_personas=args.num, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
