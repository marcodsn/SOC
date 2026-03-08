#!/usr/bin/env python3
"""
Generate experiences for personas using iterative sampling.

Usage:
    python scripts/02_gen_experiences.py --num 50
    python scripts/02_gen_experiences.py --num 50 --region-profile west
    python scripts/02_gen_experiences.py --num 50 --model "moonshotai/Kimi-K2.5:fireworks-ai" --provider huggingface
    python scripts/02_gen_experiences.py --num 50 --preset kimi_k2
    python scripts/02_gen_experiences.py --num 100 --region-profile east_asia --lingua-franca English
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.experience_generator import ExperienceGenerator
from soc_2602.llm.client import LLMClient
from soc_2602.utils.config import (
    get_language_settings,
    get_section,
    load_config,
)


def main():
    parser = argparse.ArgumentParser(description="Generate conversation experiences")

    # ── Generation parameters ────────────────────────────────────────────
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of experiences to generate (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of experiences to run concurrently (default: from config or 4)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Starting iteration number (for resuming runs)",
    )

    # ── Model selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use (overrides config and preset)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["modal", "nim", "huggingface", "mistral", "local"],
        help="LLM provider to use (overrides config and preset)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Named model preset from models.yaml (e.g. kimi_k2, glm5)",
    )

    # ── Region and language ──────────────────────────────────────────────
    parser.add_argument(
        "--region-profile",
        type=str,
        default=None,
        help="Region profile name — used only for logging; persona pool determines actual regions",
    )
    parser.add_argument(
        "--lingua-franca",
        type=str,
        default=None,
        help="Shared language for conversations (default: from config, usually English)",
    )

    # ── Experience-specific parameters ───────────────────────────────────
    parser.add_argument(
        "--same-region-prob",
        type=float,
        default=None,
        help="Probability of same-region persona pairing (default: from config or 0.5)",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        help="Path to merged personas JSONL file (default: from config)",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default=None,
        help="Directory containing seed experiences (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated experiences (default: from config)",
    )
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=None,
        help="Directory containing Jinja prompt templates (default: from config)",
    )

    args = parser.parse_args()

    # ── Load configuration ───────────────────────────────────────────────
    cfg = load_config(config_path=args.config)
    exp_cfg = get_section(cfg, "experience")
    lang_cfg = get_language_settings(cfg)
    paths_cfg = cfg.get("paths", {})

    # Apply CLI overrides to language settings
    if args.lingua_franca is not None:
        lang_cfg["lingua_franca"] = args.lingua_franca

    # Resolve paths from config with CLI overrides
    personas_path = args.personas or paths_cfg.get(
        "merged_personas", "data/personas/generated/data.jsonl"
    )
    seed_dir = args.seed_dir or exp_cfg.get("seed_dir", "data/experiences/seed")
    output_dir = args.output_dir or exp_cfg.get(
        "output_dir", "data/experiences/generated"
    )
    prompt_dir = args.prompt_dir or exp_cfg.get("prompt_dir", "conf/prompts")
    batch_size = args.batch_size or exp_cfg.get("batch_size", 4)

    # Same-region pairing probability
    same_region_prob = (
        args.same_region_prob
        if args.same_region_prob is not None
        else exp_cfg.get("same_region_probability", 0.5)
    )

    # ── Region profile (informational) ───────────────────────────────────
    profile_name = args.region_profile or cfg.get("default_region_profile", "global")
    print(f"Region profile: {profile_name} (persona pool determines actual regions)")
    print(f"Same-region pairing probability: {same_region_prob}")
    print(f"Lingua franca: {lang_cfg.get('lingua_franca', 'English')}")

    # ── Build LLM client ─────────────────────────────────────────────────
    llm_client = LLMClient.from_config(
        cfg,
        preset_name=args.preset,
        model_id=args.model,
        provider=args.provider,
    )
    print(f"Model: {llm_client.model_id} (provider: {llm_client.provider})")

    # ── Output path ──────────────────────────────────────────────────────
    output_file = (
        Path(output_dir)
        / f"experiences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # ── Generator ────────────────────────────────────────────────────────
    generator = ExperienceGenerator(
        llm_client=llm_client,
        persona_file=personas_path,
        output_file=str(output_file),
        seed_dir=seed_dir,
        prompt_dir=prompt_dir,
        language_settings=lang_cfg,
        same_region_probability=same_region_prob,
    )

    generator.generate_batch(
        num_experiences=args.num,
        start_iteration=args.start_iteration,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
