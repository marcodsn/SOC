#!/usr/bin/env python3
"""
Comprehensive experience dataset statistics.

Analyzes merged or raw experience JSONL files and prints detailed statistics
covering conversation style and cadence distributions, persona pairing
patterns, instant event quality, token usage, and potential issues.

Usage:
    python scripts/02e_stats_experiences.py
    python scripts/02e_stats_experiences.py --input data/experiences/generated/data.jsonl
    python scripts/02e_stats_experiences.py --input data/experiences/generated/data.jsonl --verbose
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Data loading ─────────────────────────────────────────────────────────────


def load_experiences(path: str) -> list:
    """Load experience records from a JSONL file."""
    experiences = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                experiences.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping invalid JSON at line {line_num}: {e}")
    return experiences


# ── Text parsing ──────────────────────────────────────────────────────────────


def extract_experience_body(text: str) -> str:
    """Strip <experience> tags and return the inner content."""
    if "<experience>" in text and "</experience>" in text:
        start = text.find("<experience>") + len("<experience>")
        end = text.find("</experience>")
        return text[start:end].strip()
    return text.strip()


def count_instant_events(text: str) -> int:
    """Count the number of bullet-point instant events listed."""
    body = extract_experience_body(text)
    # Instant events section starts with "Possible instant events:"
    if "instant events" not in body.lower():
        return 0
    # Count lines starting with "- " after the instant events header
    in_events = False
    count = 0
    for line in body.splitlines():
        stripped = line.strip()
        if re.search(r"instant events", stripped, re.IGNORECASE):
            in_events = True
            continue
        if in_events:
            if stripped.startswith("- "):
                count += 1
            elif stripped and not stripped.startswith("-"):
                # A non-bullet non-empty line ends the section
                break
    return count


def count_topic_blocks(text: str) -> int:
    """Count the number of structured topic lines (lines with turn counts)."""
    body = extract_experience_body(text)
    # Structured topics look like: "- Topic name (N turns)"
    return len(re.findall(r"-\s+.+\(\d+\s+turns?\)", body))


def has_initial_state(text: str) -> bool:
    """Check whether an Initial state line is present and non-trivial."""
    body = extract_experience_body(text)
    m = re.search(r"Initial state:\s*(.+)", body)
    if not m:
        return False
    val = m.group(1).strip()
    return bool(val) and val.lower() not in {"none", "—", "-", "null", ""}


def extract_initial_state(text: str) -> str | None:
    body = extract_experience_body(text)
    m = re.search(r"Initial state:\s*(.+)", body)
    return m.group(1).strip() if m else None


def estimate_word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


# ── Printing helpers ──────────────────────────────────────────────────────────


def print_section(title: str, width: int = 70):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_distribution(counter: Counter, top_n: int = 20, total: int = 0):
    """Pretty-print a frequency distribution."""
    if not counter:
        print("  (no data)")
        return
    if total == 0:
        total = sum(counter.values())
    for key, count in counter.most_common(top_n):
        pct = (count / total) * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {str(key):>35s}  {count:>5d}  ({pct:5.1f}%)  {bar}")
    if len(counter) > top_n:
        remaining = sum(c for _, c in counter.most_common()[top_n:])
        print(
            f"  {'... and ' + str(len(counter) - top_n) + ' more':>35s}  {remaining:>5d}"
        )


def print_numeric_stats(values: list, label: str = "", show_histogram: bool = True):
    """Print min/max/mean/median/std for a list of numbers."""
    if not values:
        print("  (no data)")
        return
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = statistics.mean(values_sorted)
    median = statistics.median(values_sorted)
    stdev = statistics.stdev(values_sorted) if n > 1 else 0.0
    print(f"  Count:  {n}")
    print(f"  Min:    {values_sorted[0]:,}")
    print(f"  Max:    {values_sorted[-1]:,}")
    print(f"  Mean:   {mean:,.2f}")
    print(f"  Median: {median:,.1f}")
    print(f"  StdDev: {stdev:,.2f}")

    if show_histogram and n >= 5:
        num_bins = min(15, max(5, n // 10))
        lo, hi = values_sorted[0], values_sorted[-1]
        if lo == hi:
            print(f"  (all values are {lo:,})")
            return
        bin_width = (hi - lo) / num_bins
        bins = [0] * num_bins
        for v in values_sorted:
            idx = min(int((v - lo) / bin_width), num_bins - 1)
            bins[idx] += 1
        max_count = max(bins)
        print("\n  Distribution:")
        for i, count in enumerate(bins):
            lo_edge = lo + i * bin_width
            hi_edge = lo + (i + 1) * bin_width
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  {lo_edge:>8.1f}–{hi_edge:<8.1f}  {count:>4d}  {bar}")


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze_experiences(
    experiences: list, verbose: bool = False, top_n: int = 20
) -> None:
    total = len(experiences)
    if total == 0:
        print("\n  No experiences found in file.")
        return

    print(f"\n  Total experiences: {total}")

    # Check for IDs
    with_ids = sum(1 for e in experiences if e.get("meta", {}).get("id"))
    print(f"  With unique IDs:   {with_ids} / {total}")

    # ── Collect metadata ──────────────────────────────────────────────────
    styles = Counter()
    cadences = Counter()
    models = Counter()
    source_timestamps = Counter()
    persona_names_all = Counter()
    persona_pair_counts = Counter()

    instant_event_counts = []  # from meta list
    instant_event_text_counts = []  # from text body
    topic_block_counts = []
    has_initial_state_count = 0
    freeform_with_energy = 0
    freeform_without_energy = 0

    char_counts = []
    word_counts = []

    input_tokens = []
    output_tokens = []
    total_tokens_list = []

    structured_with_initial = 0
    structured_without_initial = 0

    for record in experiences:
        meta = record.get("meta", {})
        text = record.get("experience_text", "")

        # Style & cadence
        style = meta.get("conversation_style", "unknown")
        cadence = meta.get("message_cadence", "unknown")
        styles[style] += 1
        cadences[cadence] += 1

        # Model
        models[meta.get("model", "unknown")] += 1

        # Source timestamp (by date)
        ts = meta.get("source_timestamp")
        if ts:
            source_timestamps[ts[:10]] += 1

        # Persona names
        for name in meta.get("persona_names", []):
            persona_names_all[name] += 1

        # Persona pairs (sorted so A↔B == B↔A)
        names = sorted(meta.get("persona_names", []))
        if len(names) == 2:
            persona_pair_counts[f"{names[0]} ↔ {names[1]}"] += 1

        # Instant events from meta
        events_meta = meta.get("instant_events", [])
        instant_event_counts.append(len(events_meta))

        # Instant events and topics from text
        instant_event_text_counts.append(count_instant_events(text))
        topic_block_counts.append(count_topic_blocks(text))

        # Initial state
        if has_initial_state(text):
            has_initial_state_count += 1

        # Freeform: check for conversational energy paragraph
        if style == "freeform":
            body = extract_experience_body(text)
            if re.search(r"[Cc]onversational energy", body):
                freeform_with_energy += 1
            else:
                freeform_without_energy += 1

        # Structured/semi-structured: check initial state present
        if style in ("structured", "semi-structured"):
            if has_initial_state(text):
                structured_with_initial += 1
            else:
                structured_without_initial += 1

        # Text length
        body = extract_experience_body(text)
        char_counts.append(char_count(body))
        word_counts.append(estimate_word_count(body))

        # Token usage
        inp = meta.get("input_tokens")
        out = meta.get("output_tokens")
        if inp is not None and inp > 0:
            input_tokens.append(inp)
        if out is not None and out > 0:
            output_tokens.append(out)
        if inp is not None and out is not None and inp > 0 and out > 0:
            total_tokens_list.append(inp + out)

    # ── Conversation Style ────────────────────────────────────────────────
    print_section("CONVERSATION STYLE DISTRIBUTION")
    STYLE_TARGETS = {
        "structured": 0.25,
        "semi-structured": 0.40,
        "freeform": 0.35,
    }
    style_total = sum(styles.values())
    print(
        f"  {'Style':>20s}  {'Count':>6s}  {'Actual%':>8s}  {'Target%':>8s}  {'Δ':>6s}"
    )
    print(f"  {'─' * 55}")
    for style_name, target in sorted(STYLE_TARGETS.items(), key=lambda x: -x[1]):
        count = styles.get(style_name, 0)
        actual_pct = (count / style_total * 100) if style_total else 0
        target_pct = target * 100
        delta_str = f"{actual_pct - target_pct:+.1f}"
        print(
            f"  {style_name:>20s}  {count:>6d}  {actual_pct:>7.1f}%  {target_pct:>7.1f}%  {delta_str:>6s}"
        )
    # Any unexpected style values
    for style_name, count in styles.items():
        if style_name not in STYLE_TARGETS:
            pct = count / style_total * 100 if style_total else 0
            print(f"  {style_name:>20s}  {count:>6d}  {pct:>7.1f}%  (unexpected)")

    # ── Message Cadence ───────────────────────────────────────────────────
    print_section("MESSAGE CADENCE DISTRIBUTION")
    CADENCE_TARGETS = {
        "realtime": 0.20,
        "delayed": 0.55,
        "async": 0.25,
    }
    cadence_total = sum(cadences.values())
    print(
        f"  {'Cadence':>15s}  {'Count':>6s}  {'Actual%':>8s}  {'Target%':>8s}  {'Δ':>6s}"
    )
    print(f"  {'─' * 50}")
    for cadence_name, target in sorted(CADENCE_TARGETS.items(), key=lambda x: -x[1]):
        count = cadences.get(cadence_name, 0)
        actual_pct = (count / cadence_total * 100) if cadence_total else 0
        target_pct = target * 100
        delta_str = f"{actual_pct - target_pct:+.1f}"
        print(
            f"  {cadence_name:>15s}  {count:>6d}  {actual_pct:>7.1f}%  {target_pct:>7.1f}%  {delta_str:>6s}"
        )
    for cadence_name, count in cadences.items():
        if cadence_name not in CADENCE_TARGETS:
            pct = count / cadence_total * 100 if cadence_total else 0
            print(f"  {cadence_name:>15s}  {count:>6d}  {pct:>7.1f}%  (unexpected)")

    # ── Models ────────────────────────────────────────────────────────────
    print_section("MODEL DISTRIBUTION")
    print_distribution(models, top_n=10, total=total)

    # ── Source Timestamps ─────────────────────────────────────────────────
    if source_timestamps:
        print_section("EXPERIENCES BY SOURCE DATE")
        print_distribution(source_timestamps, top_n=30, total=total)

    # ── Persona Participation ─────────────────────────────────────────────
    print_section(f"PERSONA PARTICIPATION (top {top_n})")
    print_distribution(persona_names_all, top_n=top_n, total=total * 2)
    unique_personas = len(persona_names_all)
    print(f"\n  Unique persona names:          {unique_personas}")
    print(f"  Total persona name slots:      {total * 2}")

    # How many personas appear in multiple experiences
    multi_experience = sum(1 for c in persona_names_all.values() if c > 1)
    print(f"  Personas in >1 experience:     {multi_experience}")

    # Unique pairs
    unique_pairs = len(persona_pair_counts)
    repeated_pairs = sum(1 for c in persona_pair_counts.values() if c > 1)
    print(f"  Unique persona pairs:          {unique_pairs}")
    if repeated_pairs > 0:
        print(f"  ⚠ Repeated pairs:             {repeated_pairs}")
        if verbose:
            print_section(f"  REPEATED PERSONA PAIRS (top {top_n})")
            for pair, count in persona_pair_counts.most_common(top_n):
                if count > 1:
                    print(f"    {pair}: {count}×")

    # ── Instant Events ────────────────────────────────────────────────────
    print_section("INSTANT EVENTS PER EXPERIENCE (from meta)")
    print_numeric_stats(instant_event_counts, show_histogram=False)

    # Cross-check: meta count vs text-parsed count
    if any(a != b for a, b in zip(instant_event_counts, instant_event_text_counts)):
        mismatches = sum(
            1 for a, b in zip(instant_event_counts, instant_event_text_counts) if a != b
        )
        print(f"\n  ⚠ Meta/text instant-event count mismatch in {mismatches} records")

    zero_events = sum(1 for c in instant_event_counts if c == 0)
    if zero_events > 0:
        print(f"\n  Experiences with 0 instant events: {zero_events}")

    # ── Structured Topic Blocks ───────────────────────────────────────────
    structured_count = styles.get("structured", 0) + styles.get("semi-structured", 0)
    if structured_count > 0:
        print_section("TOPIC BLOCKS (structured / semi-structured only)")
        struct_topic_counts = []
        for record in experiences:
            meta = record.get("meta", {})
            if meta.get("conversation_style") in ("structured", "semi-structured"):
                text = record.get("experience_text", "")
                struct_topic_counts.append(count_topic_blocks(text))
        print_numeric_stats(struct_topic_counts, show_histogram=False)

        has_is = structured_with_initial
        missing_is = structured_without_initial
        print(
            f"\n  With initial state:    {has_is} / {structured_count} "
            f"({has_is / structured_count * 100:.1f}%)"
        )
        if missing_is > 0:
            print(f"  ⚠ Missing initial state: {missing_is} / {structured_count}")

    # ── Freeform Energy ───────────────────────────────────────────────────
    freeform_count = styles.get("freeform", 0)
    if freeform_count > 0:
        print_section("FREEFORM CONVERSATIONAL ENERGY")
        print(
            f"  With energy paragraph:    {freeform_with_energy} / {freeform_count} "
            f"({freeform_with_energy / freeform_count * 100:.1f}%)"
        )
        if freeform_without_energy > 0:
            print(
                f"  ⚠ Missing energy paragraph: {freeform_without_energy} / {freeform_count}"
            )

    # ── Text Length ───────────────────────────────────────────────────────
    print_section("EXPERIENCE TEXT LENGTH (body, excluding XML tags)")
    print("\n  Character count:")
    print_numeric_stats(char_counts)
    print("\n  Word count:")
    print_numeric_stats(word_counts)

    # ── Token Usage ───────────────────────────────────────────────────────
    if input_tokens or output_tokens:
        print_section("TOKEN USAGE PER EXPERIENCE")

        if input_tokens:
            print(f"\n  Input tokens  (n={len(input_tokens)}):")
            print_numeric_stats(input_tokens, show_histogram=False)

        if output_tokens:
            print(f"\n  Output tokens  (n={len(output_tokens)}):")
            print_numeric_stats(output_tokens, show_histogram=False)

        if total_tokens_list:
            print(f"\n  Total tokens per experience  (n={len(total_tokens_list)}):")
            print_numeric_stats(total_tokens_list, show_histogram=False)

        sum_in = sum(input_tokens)
        sum_out = sum(output_tokens)
        sum_tot = sum_in + sum_out
        print(f"\n  Aggregate token usage:")
        print(f"    Input  tokens total:  {sum_in:>12,d}")
        print(f"    Output tokens total:  {sum_out:>12,d}")
        print(f"    Grand total:          {sum_tot:>12,d}")
        if input_tokens:
            print(f"\n  Mean per experience:")
            print(f"    Input:   {sum_in / len(input_tokens):>10,.1f}")
            print(f"    Output:  {sum_out / len(output_tokens):>10,.1f}")
            if total_tokens_list:
                print(f"    Total:   {sum_tot / len(total_tokens_list):>10,.1f}")
    else:
        print_section("TOKEN USAGE")
        print("  (no token data found — pre-token-tracking records)")

    # ── Potential Issues ──────────────────────────────────────────────────
    print_section("POTENTIAL ISSUES")
    issues = []

    # Short experiences
    short_count = sum(1 for c in char_counts if c < 400)
    if short_count > 0:
        issues.append(
            f"  ⚠ {short_count} experiences under 400 chars (may be truncated or malformed)"
        )

    # Very long experiences
    long_count = sum(1 for c in char_counts if c > 3000)
    if long_count > 0:
        issues.append(f"  ⚠ {long_count} experiences over 3000 chars (may be verbose)")

    # No instant events
    if zero_events > 0:
        issues.append(f"  ⚠ {zero_events} experiences have no instant events")

    # Structured without initial state
    if structured_without_initial > 0:
        issues.append(
            f"  ⚠ {structured_without_initial} structured/semi-structured experiences missing an initial state"
        )

    # Freeform without conversational energy
    if freeform_without_energy > 0:
        issues.append(
            f"  ⚠ {freeform_without_energy} freeform experiences missing a conversational energy paragraph"
        )

    # Style imbalance
    if style_total >= 20 and len(styles) > 1:
        max_style = styles.most_common(1)[0]
        min_style = styles.most_common()[-1]
        if max_style[1] > 4 * min_style[1]:
            issues.append(
                f"  ⚠ Style imbalance: '{max_style[0]}' ({max_style[1]}) vs "
                f"'{min_style[0]}' ({min_style[1]})"
            )

    # Cadence imbalance
    if cadence_total >= 20 and len(cadences) > 1:
        max_cad = cadences.most_common(1)[0]
        min_cad = cadences.most_common()[-1]
        if max_cad[1] > 4 * min_cad[1]:
            issues.append(
                f"  ⚠ Cadence imbalance: '{max_cad[0]}' ({max_cad[1]}) vs "
                f"'{min_cad[0]}' ({min_cad[1]})"
            )

    # Personas missing token data
    no_token_data = total - len(input_tokens)
    if 0 < no_token_data < total:
        issues.append(
            f"  ⚠ {no_token_data} experiences have no token usage data (pre-tracking records)"
        )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No obvious issues detected")

    # ── Verbose: sample initial states ───────────────────────────────────
    if verbose:
        initial_states = Counter()
        for record in experiences:
            text = record.get("experience_text", "")
            state = extract_initial_state(text)
            if state:
                initial_states[state] += 1
        if initial_states:
            print_section(f"INITIAL STATES (top {top_n})")
            print_distribution(initial_states, top_n=top_n)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive experience dataset statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/experiences/generated/data.jsonl",
        help="Path to experiences JSONL file (default: data/experiences/generated/data.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail (repeated pairs, initial states, etc.)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top entries to show in distributions (default: 20)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  EXPERIENCE DATASET STATISTICS")
    print(f"  Source: {input_path}")
    print(f"{'═' * 70}")

    experiences = load_experiences(str(input_path))
    analyze_experiences(experiences, verbose=args.verbose, top_n=args.top_n)

    print(f"\n{'═' * 70}")
    print("  Analysis complete.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
