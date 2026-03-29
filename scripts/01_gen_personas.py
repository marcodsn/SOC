#!/usr/bin/env python3
"""
Generate diverse personas using iterative sampling.

Usage:
    python scripts/01_gen_personas.py --num 50
    python scripts/01_gen_personas.py --num 50 --region-profile west
    python scripts/01_gen_personas.py --num 50 --model "moonshotai/Kimi-K2.5:fireworks-ai" --provider huggingface
    python scripts/01_gen_personas.py --num 50 --preset kimi_k2
    python scripts/01_gen_personas.py --num 100 --region-profile east_asia --lingua-franca English
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.persona_generator import PersonaGenerator
from soc_2602.llm.client import LLMClient
from soc_2602.utils.config import (
    get_language_settings,
    get_region_profile,
    get_section,
    load_config,
)
from soc_2602.utils.sampling import StatsEngine


def main():
    parser = argparse.ArgumentParser(description="Generate diverse personas")

    # ── Generation parameters ────────────────────────────────────────────
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of personas to generate (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of personas to run concurrently (default: from config or 4)",
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
        help="Named model preset from config.yaml (e.g. kimi_k2, glm5)",
    )

    # ── Region and language ──────────────────────────────────────────────
    parser.add_argument(
        "--region-profile",
        type=str,
        default=None,
        help="Region profile name (e.g. global, west, east_asia, latam). Default: from config",
    )
    parser.add_argument(
        "--lingua-franca",
        type=str,
        default=None,
        help="Shared language for example messages (default: from config, usually English)",
    )

    # ── Age distribution ─────────────────────────────────────────────────
    parser.add_argument(
        "--age-mean",
        type=float,
        default=None,
        help="Mean age for sampling (default: from config or 25)",
    )
    parser.add_argument(
        "--age-std",
        type=float,
        default=None,
        help="Age standard deviation (default: from config or 5)",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default=None,
        help="Directory containing seed personas (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated personas (default: from config)",
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
    persona_cfg = get_section(cfg, "persona")
    lang_cfg = get_language_settings(cfg)

    # Apply CLI overrides to language settings
    if args.lingua_franca is not None:
        lang_cfg["lingua_franca"] = args.lingua_franca

    # Resolve paths from config with CLI overrides
    seed_dir = args.seed_dir or persona_cfg.get("seed_dir", "data/personas/seed")
    output_dir = args.output_dir or persona_cfg.get(
        "output_dir", "data/personas/generated"
    )
    prompt_dir = args.prompt_dir or persona_cfg.get("prompt_dir", "conf/prompts")
    batch_size = args.batch_size or persona_cfg.get("batch_size", 4)

    # Age distribution
    age_mean = (
        args.age_mean if args.age_mean is not None else persona_cfg.get("age_mean", 25)
    )
    age_std = (
        args.age_std if args.age_std is not None else persona_cfg.get("age_std", 5)
    )

    # ── Region profile ───────────────────────────────────────────────────
    profile_name = args.region_profile or cfg.get("default_region_profile", "global")
    allowed_regions = get_region_profile(cfg, profile_name)

    print(f"Region profile: {profile_name}")
    if allowed_regions:
        print(
            f"  Restricted to {len(allowed_regions)} region(s): {', '.join(allowed_regions)}"
        )
    else:
        print("  Using all regions (global)")

    # ── Build StatsEngine with region filtering ──────────────────────────
    stats_dir = Path(cfg.get("paths", {}).get("stats_dir", "data/stats"))
    stats_engine = StatsEngine(stats_dir=stats_dir, allowed_regions=allowed_regions)

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
        Path(output_dir) / f"personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # ── Generator ────────────────────────────────────────────────────────
    generator = PersonaGenerator(
        llm_client=llm_client,
        seed_dir=seed_dir,
        output_file=str(output_file),
        prompt_dir=prompt_dir,
        stats_engine=stats_engine,
        language_settings=lang_cfg,
        age_mean=age_mean,
        age_std=age_std,
    )

    generator.generate_batch(
        num_personas=args.num,
        start_iteration=args.start_iteration,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
