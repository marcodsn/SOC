#!/usr/bin/env python3
"""
Comprehensive persona dataset statistics.

Analyzes merged or raw persona JSONL files and prints detailed statistics
covering demographic distributions, token usage, text quality metrics, and
potential issues.

Usage:
    python scripts/01e_stats_personas.py
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl --verbose
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_personas(path: str) -> list:
    """Load persona records from a JSONL file."""
    personas = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                personas.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping invalid JSON at line {line_num}: {e}")
    return personas


def extract_sections(persona_text: str) -> dict:
    """Extract named sections from a persona's character card text."""
    sections = {}
    pattern = r"\*\*([^*]+)\*\*\s*\n(.*?)(?=\n\*\*[^*]+\*\*|\Z)"
    for match in re.finditer(pattern, persona_text, re.DOTALL):
        name = match.group(1).strip().rstrip(":")
        content = match.group(2).strip()
        if content:
            sections[name] = content
    return sections


def count_example_messages(persona_text: str) -> int:
    """Count the number of <START> example message blocks."""
    return persona_text.count("<START>")


def estimate_word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def print_section(title: str, width: int = 70):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_distribution(counter: Counter, label: str, top_n: int = 20, total: int = 0):
    """Pretty-print a frequency distribution."""
    if not counter:
        print("  (no data)")
        return
    if total == 0:
        total = sum(counter.values())
    for i, (key, count) in enumerate(counter.most_common(top_n)):
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


def analyze_text_quality(personas: list) -> dict:
    """Analyze text quality metrics across all personas."""
    metrics = {
        "char_counts": [],
        "word_counts": [],
        "section_counts": [],
        "example_msg_counts": [],
        "has_character_tags": 0,
        "missing_sections": Counter(),
        "duplicate_names": [],
    }

    expected_sections = {
        "Basic Information",
        "Physical & Lifestyle",
        "Personality Overview",
        "Core Traits",
        "Emotional Profile",
        "Relationships",
        "Values, Motivations & Fears",
        "Behavioral Patterns",
        "Communication Style",
        "Example Messages",
        "Summary",
    }

    name_counts = Counter()

    for record in personas:
        text = record.get("persona_text", "")
        meta = record.get("meta", {})

        metrics["char_counts"].append(char_count(text))
        metrics["word_counts"].append(estimate_word_count(text))

        if "<character>" in text and "</character>" in text:
            metrics["has_character_tags"] += 1

        sections = extract_sections(text)
        found_sections = set()
        for s_name in sections:
            for expected in expected_sections:
                if (
                    expected.lower() in s_name.lower()
                    or s_name.lower() in expected.lower()
                ):
                    found_sections.add(expected)
                    break
        metrics["section_counts"].append(len(found_sections))

        missing = expected_sections - found_sections
        for m in missing:
            metrics["missing_sections"][m] += 1

        metrics["example_msg_counts"].append(count_example_messages(text))

        name = meta.get("name", "Unknown")
        name_counts[name] += 1

    metrics["duplicate_names"] = [
        (name, count) for name, count in name_counts.items() if count > 1
    ]

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive persona dataset statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/personas/generated/data.jsonl",
        help="Path to personas JSONL file (default: data/personas/generated/data.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail (e.g. all duplicate names, low-rep regions)",
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
    print(f"  PERSONA DATASET STATISTICS")
    print(f"  Source: {input_path}")
    print(f"{'═' * 70}")

    personas = load_personas(str(input_path))
    total = len(personas)

    if total == 0:
        print("\n  No personas found in file.")
        sys.exit(0)

    print(f"\n  Total personas:  {total}")

    # Check for IDs
    with_ids = sum(1 for p in personas if p.get("meta", {}).get("id"))
    print(f"  With unique IDs: {with_ids} / {total}")

    # ── Demographics ─────────────────────────────────────────────────────
    ages = []
    regions = Counter()
    subregions = Counter()
    names = Counter()
    models = Counter()
    archetypes = Counter()
    regulation_styles = Counter()
    source_timestamps = Counter()

    for record in personas:
        meta = record.get("meta", {})
        age = meta.get("age")
        if age is not None:
            ages.append(age)
        regions[meta.get("region", "Unknown")] += 1
        subregions[meta.get("subregion", "Unknown")] += 1
        names[meta.get("name", "Unknown")] += 1
        models[meta.get("model", "Unknown")] += 1
        archetypes[meta.get("archetype", "Unknown")] += 1
        regulation_styles[meta.get("regulation_style", "Unknown")] += 1
        ts = meta.get("source_timestamp")
        if ts:
            # Group by date (first 10 chars of ISO string)
            source_timestamps[ts[:10]] += 1

    # ── Age ──────────────────────────────────────────────────────────────
    print_section("AGE DISTRIBUTION")
    print_numeric_stats(ages, "Age")

    # ── Region ───────────────────────────────────────────────────────────
    print_section(f"REGION DISTRIBUTION (top {args.top_n})")
    print_distribution(regions, "Region", top_n=args.top_n, total=total)

    print_section(f"SUBREGION DISTRIBUTION (top {args.top_n})")
    print_distribution(subregions, "Subregion", top_n=args.top_n, total=total)

    # ── Names ────────────────────────────────────────────────────────────
    print_section(f"NAME DISTRIBUTION (top {args.top_n})")
    print_distribution(names, "Name", top_n=args.top_n, total=total)
    unique_names = len(names)
    print(
        f"\n  Unique names: {unique_names} / {total} ({unique_names / total * 100:.1f}%)"
    )

    # ── Models ───────────────────────────────────────────────────────────
    print_section("MODEL DISTRIBUTION")
    print_distribution(models, "Model", top_n=10, total=total)

    # ── Archetypes ───────────────────────────────────────────────────────
    print_section("ARCHETYPE DISTRIBUTION")
    # Show expected base rates alongside actual
    ARCHETYPE_BASE_RATES = {
        "grounded_pragmatist": 0.22,
        "warm_connector": 0.18,
        "anxious_achiever": 0.14,
        "quiet_introvert": 0.12,
        "earnest_idealist": 0.10,
        "defensive_cynic": 0.08,
        "drifter": 0.07,
        "ambitious_performer": 0.05,
        "wounded_caretaker": 0.04,
    }
    arch_total = sum(archetypes.values())
    print(
        f"  {'Archetype':>25s}  {'Count':>6s}  {'Actual%':>8s}  {'Target%':>8s}  {'Δ':>6s}"
    )
    print(f"  {'─' * 60}")
    for archetype, base_rate in sorted(
        ARCHETYPE_BASE_RATES.items(), key=lambda x: -x[1]
    ):
        count = archetypes.get(archetype, 0)
        actual_pct = (count / arch_total * 100) if arch_total else 0
        target_pct = base_rate * 100
        delta = actual_pct - target_pct
        delta_str = f"{delta:+.1f}"
        print(
            f"  {archetype:>25s}  {count:>6d}  {actual_pct:>7.1f}%  {target_pct:>7.1f}%  {delta_str:>6s}"
        )
    # Unknown archetypes (pre-archetype runs)
    unknown_arch = archetypes.get("Unknown", 0)
    if unknown_arch:
        print(f"  {'Unknown (pre-archetype)':>25s}  {unknown_arch:>6d}")

    # ── Regulation Styles ────────────────────────────────────────────────
    print_section("REGULATION STYLE DISTRIBUTION")
    REGULATION_BASE_RATES = {
        "stable": 0.35,
        "expressive": 0.30,
        "suppressed": 0.20,
        "volatile": 0.15,
    }
    reg_total = sum(regulation_styles.values())
    print(
        f"  {'Style':>15s}  {'Count':>6s}  {'Actual%':>8s}  {'Target%':>8s}  {'Δ':>6s}"
    )
    print(f"  {'─' * 50}")
    for style, base_rate in sorted(REGULATION_BASE_RATES.items(), key=lambda x: -x[1]):
        count = regulation_styles.get(style, 0)
        actual_pct = (count / reg_total * 100) if reg_total else 0
        target_pct = base_rate * 100
        delta = actual_pct - target_pct
        delta_str = f"{delta:+.1f}"
        print(
            f"  {style:>15s}  {count:>6d}  {actual_pct:>7.1f}%  {target_pct:>7.1f}%  {delta_str:>6s}"
        )
    unknown_reg = regulation_styles.get("Unknown", 0)
    if unknown_reg:
        print(f"  {'Unknown':>15s}  {unknown_reg:>6d}")

    # ── Source Timestamps ────────────────────────────────────────────────
    if source_timestamps:
        print_section("PERSONAS BY SOURCE DATE")
        print_distribution(source_timestamps, "Date", top_n=30, total=total)

    # ── Token Usage ──────────────────────────────────────────────────────
    input_tokens = []
    output_tokens = []
    total_tokens = []

    for record in personas:
        meta = record.get("meta", {})
        inp = meta.get("input_tokens")
        out = meta.get("output_tokens")
        if inp is not None and inp > 0:
            input_tokens.append(inp)
        if out is not None and out > 0:
            output_tokens.append(out)
        if inp is not None and out is not None and inp > 0 and out > 0:
            total_tokens.append(inp + out)

    if input_tokens or output_tokens:
        print_section("TOKEN USAGE PER PERSONA")

        if input_tokens:
            print(f"\n  Input tokens  (n={len(input_tokens)}):")
            print_numeric_stats(input_tokens, show_histogram=False)

        if output_tokens:
            print(f"\n  Output tokens  (n={len(output_tokens)}):")
            print_numeric_stats(output_tokens, show_histogram=False)

        if total_tokens:
            print(f"\n  Total tokens per persona  (n={len(total_tokens)}):")
            print_numeric_stats(total_tokens, show_histogram=False)

        # Aggregate
        sum_in = sum(input_tokens)
        sum_out = sum(output_tokens)
        sum_tot = sum_in + sum_out
        print(f"\n  Aggregate token usage:")
        print(f"    Input  tokens total:  {sum_in:>12,d}")
        print(f"    Output tokens total:  {sum_out:>12,d}")
        print(f"    Grand total:          {sum_tot:>12,d}")
        if len(input_tokens) > 0:
            print(f"\n  Mean per persona:")
            print(f"    Input:   {sum_in / len(input_tokens):>10,.1f}")
            print(f"    Output:  {sum_out / len(output_tokens):>10,.1f}")
            if total_tokens:
                print(f"    Total:   {sum_tot / len(total_tokens):>10,.1f}")
    else:
        print_section("TOKEN USAGE")
        print("  (no token data found — pre-token-tracking records)")

    # ── Text Quality ─────────────────────────────────────────────────────
    print_section("TEXT QUALITY METRICS")
    quality = analyze_text_quality(personas)

    print("\n  Character count:")
    print_numeric_stats(quality["char_counts"], "Characters")

    print("\n  Word count:")
    print_numeric_stats(quality["word_counts"], "Words")

    print("\n  Section count (out of 11 expected):")
    print_numeric_stats(quality["section_counts"], "Sections", show_histogram=False)

    print("\n  Example message blocks (<START> count):")
    print_numeric_stats(quality["example_msg_counts"], "Examples", show_histogram=False)

    tag_pct = quality["has_character_tags"] / total * 100
    print(
        f"\n  Proper <character> tags: {quality['has_character_tags']} / {total} ({tag_pct:.1f}%)"
    )

    if quality["missing_sections"]:
        print("\n  Frequently missing sections:")
        for section, count in quality["missing_sections"].most_common():
            pct = count / total * 100
            print(f"    {section:>35s}:  {count:>4d} missing ({pct:.1f}%)")

    # ── Potential Issues ─────────────────────────────────────────────────
    print_section("POTENTIAL ISSUES")
    issues = []

    short_count = sum(1 for c in quality["char_counts"] if c < 1500)
    if short_count > 0:
        issues.append(
            f"  ⚠ {short_count} personas under 1500 characters (may be truncated)"
        )

    long_count = sum(1 for c in quality["char_counts"] if c > 5000)
    if long_count > 0:
        issues.append(
            f"  ⚠ {long_count} personas over 5000 characters (may be verbose)"
        )

    missing_tags = total - quality["has_character_tags"]
    if missing_tags > 0:
        issues.append(f"  ⚠ {missing_tags} personas missing <character> tags")

    no_examples = sum(1 for c in quality["example_msg_counts"] if c == 0)
    if no_examples > 0:
        issues.append(f"  ⚠ {no_examples} personas have no example message blocks")

    dupes = quality["duplicate_names"]
    if dupes:
        dupe_count = sum(c for _, c in dupes)
        issues.append(
            f"  ⚠ {len(dupes)} duplicate name(s) affecting {dupe_count} personas"
        )
        if args.verbose:
            for name, count in sorted(dupes, key=lambda x: -x[1])[:20]:
                issues.append(f"      {name}: {count}×")

    # Archetype distribution skew
    if arch_total >= 30:
        for archetype, base_rate in ARCHETYPE_BASE_RATES.items():
            count = archetypes.get(archetype, 0)
            actual_rate = count / arch_total
            if abs(actual_rate - base_rate) > 0.10:
                issues.append(
                    f"  ⚠ Archetype '{archetype}' is {actual_rate * 100:.1f}% "
                    f"(target {base_rate * 100:.1f}%) — deviation > 10pp"
                )

    # Regulation style skew
    if reg_total >= 30:
        for style, base_rate in REGULATION_BASE_RATES.items():
            count = regulation_styles.get(style, 0)
            actual_rate = count / reg_total
            if abs(actual_rate - base_rate) > 0.10:
                issues.append(
                    f"  ⚠ Regulation style '{style}' is {actual_rate * 100:.1f}% "
                    f"(target {base_rate * 100:.1f}%) — deviation > 10pp"
                )

    # Underrepresented regions
    if total >= 50:
        threshold = total * 0.005
        zero_regions = [
            r for r, c in regions.items() if c <= threshold and r != "Unknown"
        ]
        if zero_regions and len(zero_regions) < len(regions):
            issues.append(
                f"  ⚠ {len(zero_regions)} region(s) with very low representation (≤{threshold:.0f} personas)"
            )
            if args.verbose:
                for r in sorted(zero_regions)[:10]:
                    issues.append(f"      {r}: {regions[r]}")

    # Personas with no token data
    no_token_data = total - len(input_tokens)
    if no_token_data > 0 and no_token_data < total:
        issues.append(
            f"  ⚠ {no_token_data} personas have no token usage data (pre-tracking records)"
        )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No obvious issues detected")

    # ── Region Coverage ──────────────────────────────────────────────────
    print_section("REGION COVERAGE")
    regions_yaml_path = Path("data/stats/demographics/regions.yaml")
    if regions_yaml_path.exists():
        try:
            import yaml

            with open(regions_yaml_path, "r") as f:
                regions_config = yaml.safe_load(f)
            expected_regions = set(regions_config.get("regions", {}).keys())
            found_regions = set(regions.keys()) - {"Unknown"}
            covered = found_regions & expected_regions
            missing_regions = expected_regions - found_regions
            extra = found_regions - expected_regions

            print(f"  Expected regions:  {len(expected_regions)}")
            print(f"  Found regions:     {len(found_regions)}")
            print(
                f"  Coverage:          {len(covered)} / {len(expected_regions)} "
                f"({len(covered) / len(expected_regions) * 100:.1f}%)"
            )

            if missing_regions:
                print(f"\n  Missing regions ({len(missing_regions)}):")
                for r in sorted(missing_regions):
                    print(f"    - {r}")

            if extra:
                print(f"\n  Extra/unknown regions ({len(extra)}):")
                for r in sorted(extra):
                    print(f"    - {r}: {regions[r]}")
        except ImportError:
            print("  (install PyYAML to compare against regions.yaml)")
        except Exception as e:
            print(f"  (could not load regions.yaml: {e})")
    else:
        print("  (regions.yaml not found, skipping coverage check)")

    print(f"\n{'═' * 70}")
    print("  Analysis complete.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
