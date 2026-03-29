#!/usr/bin/env python3
"""
SOC — Synthetic Online Conversations

Entry point for the SOC pipeline. Provides quick-access subcommands
for the most common operations without needing to remember individual
script paths.

Usage:
    python main.py personas   --num 50
    python main.py experiences --num 50
    python main.py conversations --num 20
    python main.py merge      --stage personas
    python main.py stats      --stage personas
    python main.py validate
    python main.py info
"""

import argparse
import subprocess
import sys
from pathlib import Path

# All scripts live here
SCRIPTS_DIR = Path(__file__).parent / "scripts"

# Map subcommands to their scripts
SCRIPT_MAP = {
    "personas": SCRIPTS_DIR / "01_gen_personas.py",
    "experiences": SCRIPTS_DIR / "02_gen_experiences.py",
    "conversations": SCRIPTS_DIR / "03_gen_conversations.py",
    "validate": SCRIPTS_DIR / "04_validate_data.py",
    "evaluate": SCRIPTS_DIR / "01e_evaluate_personas.py",
}

MERGE_SCRIPTS = {
    "personas": SCRIPTS_DIR / "01e_merge_personas.py",
    "experiences": SCRIPTS_DIR / "02e_merge_experiences.py",
    "conversations": SCRIPTS_DIR / "03e_merge_conversations.py",
}

STATS_SCRIPTS = {
    "personas": SCRIPTS_DIR / "01e_stats_personas.py",
    "conversations": SCRIPTS_DIR / "03e_stats_conversations.py",
}


def print_info():
    """Print pipeline overview and available commands."""
    print(
        r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   SOC — Synthetic Online Conversations                               ║
║                                                                      ║
║   A pipeline for generating realistic, persona-grounded              ║
║   synthetic dialogue datasets using large language models.           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline stages:

  1. personas       Generate diverse character personas
  2. experiences    Pair personas into conversational experiences
  3. conversations  Run turn-by-turn conversation generation

Utility commands:

  merge             Merge generated files (--stage personas|experiences|conversations)
  stats             Show dataset statistics (--stage personas|conversations)
  validate          Run data validation and quality tests
  evaluate          Evaluate persona quality with LLM-as-a-judge
  info              Show this help message

Examples:

  python main.py personas --num 50 --region-profile west
  python main.py experiences --num 50 --preset kimi_k2
  python main.py conversations --num 20 --style freeform
  python main.py merge --stage personas
  python main.py stats --stage conversations
  python main.py validate --verbose
  python main.py evaluate --preset kimi_k2

Configuration:

  conf/config.yaml    Pipeline parameters, region profiles, model presets, provider endpoints

For detailed help on any subcommand, run:

  python main.py <subcommand> --help
"""
    )


def run_script(script_path: Path, extra_args: list):
    """Run a Python script with the given extra arguments."""
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + extra_args
    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        description="SOC — Synthetic Online Conversations pipeline",
        usage="python main.py <command> [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  personas        Generate character personas\n"
            "  experiences     Generate conversational experiences\n"
            "  conversations   Generate multi-turn conversations\n"
            "  merge           Merge generated files (use --stage)\n"
            "  stats           Show dataset statistics (use --stage)\n"
            "  validate        Run data validation tests\n"
            "  evaluate        Evaluate persona quality with LLM-as-a-judge\n"
            "  info            Show pipeline overview\n"
        ),
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="info",
        choices=[
            "personas",
            "experiences",
            "conversations",
            "merge",
            "stats",
            "validate",
            "evaluate",
            "info",
        ],
        help="Pipeline command to run",
    )

    # Parse only the first argument to determine the subcommand;
    # everything else is forwarded to the underlying script.
    args, extra = parser.parse_known_args()

    if args.command == "info" and not extra:
        print_info()
        return

    # ── Generation commands ──────────────────────────────────────────────
    if args.command in SCRIPT_MAP:
        run_script(SCRIPT_MAP[args.command], extra)
        return

    # ── Merge command ────────────────────────────────────────────────────
    if args.command == "merge":
        merge_parser = argparse.ArgumentParser(prog="main.py merge")
        merge_parser.add_argument(
            "--stage",
            required=True,
            choices=["personas", "experiences", "conversations"],
            help="Which stage to merge",
        )
        merge_args, merge_extra = merge_parser.parse_known_args(extra)

        script = MERGE_SCRIPTS.get(merge_args.stage)
        if script is None or not script.exists():
            print(f"Error: No merge script for stage '{merge_args.stage}'")
            sys.exit(1)

        run_script(script, merge_extra)
        return

    # ── Stats command ────────────────────────────────────────────────────
    if args.command == "stats":
        stats_parser = argparse.ArgumentParser(prog="main.py stats")
        stats_parser.add_argument(
            "--stage",
            required=True,
            choices=["personas", "conversations"],
            help="Which stage to analyze",
        )
        stats_args, stats_extra = stats_parser.parse_known_args(extra)

        script = STATS_SCRIPTS.get(stats_args.stage)
        if script is None or not script.exists():
            print(f"Error: No stats script for stage '{stats_args.stage}'")
            sys.exit(1)

        run_script(script, stats_extra)
        return

    # Fallback
    print_info()


if __name__ == "__main__":
    main()
