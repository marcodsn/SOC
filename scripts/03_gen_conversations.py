#!/usr/bin/env python3
"""
Generate conversations from paired personas and experiences.

Usage:
    python scripts/03_gen_conversations.py --num 50
    python scripts/03_gen_conversations.py --num 20 --style freeform
    python scripts/03_gen_conversations.py --num 50 --preset kimi_k2
    python scripts/03_gen_conversations.py --num 50 --provider nim
    python scripts/03_gen_conversations.py \
        --model "moonshotai/Kimi-K2.5:fireworks-ai" \
        --summarizer-model "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"
    python scripts/03_gen_conversations.py --num 100 --lingua-franca English
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_2602.generation.conversation_generator import ConversationGenerator
from soc_2602.llm.client import LLMClient
from soc_2602.utils.config import (
    get_language_settings,
    get_section,
    load_config,
)


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

    # ── Generation parameters ────────────────────────────────────────────
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of experiences to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of conversations to run concurrently (default: from config or 4)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within filtered experience list (for resuming runs)",
    )

    # ── Model selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for turn generation (overrides config and preset)",
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default=None,
        help="Model ID for summarization (defaults to --model if not set)",
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
        help="Named model preset from models.yaml for turn generation (e.g. kimi_k2, glm5)",
    )
    parser.add_argument(
        "--summarizer-preset",
        type=str,
        default=None,
        help="Named model preset for summarization (e.g. local_small)",
    )

    # ── Region and language ──────────────────────────────────────────────
    parser.add_argument(
        "--region-profile",
        type=str,
        default=None,
        help="Region profile name — informational logging only; persona pool determines regions",
    )
    parser.add_argument(
        "--lingua-franca",
        type=str,
        default=None,
        help="Shared language for conversations (default: from config, usually English)",
    )

    # ── Conversation parameters ──────────────────────────────────────────
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Hard ceiling on turns per conversation (default: from config or 50)",
    )
    parser.add_argument(
        "--instant-event-prob",
        type=float,
        default=None,
        help="Probability of instant event injection per turn (default: from config or 0.05)",
    )

    # ── Filtering ────────────────────────────────────────────────────────
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
        "--experiences",
        type=str,
        default=None,
        help="Path to merged experiences JSONL file (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated conversations (default: from config)",
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
    conv_cfg = get_section(cfg, "conversation")
    lang_cfg = get_language_settings(cfg)
    paths_cfg = cfg.get("paths", {})

    # Apply CLI overrides to language settings
    if args.lingua_franca is not None:
        lang_cfg["lingua_franca"] = args.lingua_franca

    # Resolve paths from config with CLI overrides
    personas_path = args.personas or paths_cfg.get(
        "merged_personas", "data/personas/generated/data.jsonl"
    )
    experiences_path = args.experiences or paths_cfg.get(
        "merged_experiences", "data/experiences/generated/data.jsonl"
    )
    output_dir = args.output_dir or conv_cfg.get(
        "output_dir", "data/conversations/generated"
    )
    prompt_dir = args.prompt_dir or conv_cfg.get("prompt_dir", "conf/prompts")
    batch_size = args.batch_size or conv_cfg.get("batch_size", 4)

    # Conversation parameters from config with CLI overrides
    max_turns = (
        args.max_turns if args.max_turns is not None else conv_cfg.get("max_turns", 50)
    )
    instant_event_prob = (
        args.instant_event_prob
        if args.instant_event_prob is not None
        else conv_cfg.get("instant_event_probability", 0.05)
    )
    max_active_turns = conv_cfg.get("max_active_turns", 20)
    retire_batch = conv_cfg.get("retire_batch", 8)
    max_consecutive_failures = conv_cfg.get("max_consecutive_failures", 3)

    # ── Region profile (informational) ───────────────────────────────────
    profile_name = args.region_profile or cfg.get("default_region_profile", "global")
    print(f"Region profile: {profile_name} (persona pool determines actual regions)")
    print(f"Lingua franca: {lang_cfg.get('lingua_franca', 'English')}")
    print(f"Max turns: {max_turns} | Instant event prob: {instant_event_prob}")

    # ── Build LLM clients ────────────────────────────────────────────────
    llm_client = LLMClient.from_config(
        cfg,
        preset_name=args.preset,
        model_id=args.model,
        provider=args.provider,
    )
    print(f"Turn model: {llm_client.model_id} (provider: {llm_client.provider})")

    summarizer_client = None
    if args.summarizer_model or args.summarizer_preset:
        summarizer_client = LLMClient.from_config(
            cfg,
            preset_name=args.summarizer_preset,
            model_id=args.summarizer_model,
            provider=args.provider,
        )
        print(
            f"Summarizer model: {summarizer_client.model_id} "
            f"(provider: {summarizer_client.provider})"
        )

    # ── Load and filter experiences ──────────────────────────────────────
    filters = {"style": args.style, "cadence": args.cadence}
    print(f"Loading experiences from: {experiences_path}")
    experiences = load_experiences(experiences_path, filters)

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

    # ── Output path ──────────────────────────────────────────────────────
    output_file = (
        Path(output_dir)
        / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # ── Generator ────────────────────────────────────────────────────────
    generator = ConversationGenerator(
        llm_client=llm_client,
        summarizer_client=summarizer_client,
        persona_file=personas_path,
        experience_file=experiences_path,
        output_file=str(output_file),
        prompt_dir=prompt_dir,
        language_settings=lang_cfg,
        max_turns=max_turns,
        instant_event_prob=instant_event_prob,
        max_active_turns=max_active_turns,
        retire_batch=retire_batch,
        max_consecutive_failures=max_consecutive_failures,
    )

    generator.generate_batch(
        experience_records=experiences,
        batch_size=batch_size,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
