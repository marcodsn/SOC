#!/usr/bin/env python3
"""
Comprehensive persona dataset statistics.

Analyzes merged or raw persona JSONL files and prints detailed statistics
covering demographic distributions, text quality metrics, and potential
issues.

Usage:
    python scripts/01e_stats_personas.py
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl --verbose
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
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
    # Match **Section Name** followed by content until next **Section** or end
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
    """Rough word count for a text string."""
    return len(text.split())


def char_count(text: str) -> int:
    """Character count for a text string."""
    return len(text)


def print_distribution(counter: Counter, label: str, top_n: int = 20, total: int = 0):
    """Pretty-print a frequency distribution."""
    if not counter:
        print(f"  (no data)")
        return
    if total == 0:
        total = sum(counter.values())
    for i, (key, count) in enumerate(counter.most_common(top_n)):
        pct = (count / total) * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {str(key):>30s}  {count:>5d}  ({pct:5.1f}%)  {bar}")
    if len(counter) > top_n:
        remaining = sum(c for _, c in counter.most_common()[top_n:])
        print(
            f"  {'... and ' + str(len(counter) - top_n) + ' more':>30s}  {remaining:>5d}"
        )


def print_numeric_stats(values: list, label: str):
    """Print min/max/mean/median/std for a list of numbers."""
    if not values:
        print(f"  (no data)")
        return
    import statistics

    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = statistics.mean(values_sorted)
    median = statistics.median(values_sorted)
    stdev = statistics.stdev(values_sorted) if n > 1 else 0.0
    print(f"  Count:  {n}")
    print(f"  Min:    {values_sorted[0]}")
    print(f"  Max:    {values_sorted[-1]}")
    print(f"  Mean:   {mean:.2f}")
    print(f"  Median: {median:.1f}")
    print(f"  StdDev: {stdev:.2f}")

    # Simple histogram
    if n >= 5:
        num_bins = min(15, max(5, n // 10))
        lo, hi = values_sorted[0], values_sorted[-1]
        if lo == hi:
            print(f"  (all values are {lo})")
            return
        bin_width = (hi - lo) / num_bins
        bins = [0] * num_bins
        for v in values_sorted:
            idx = min(int((v - lo) / bin_width), num_bins - 1)
            bins[idx] += 1
        max_count = max(bins)
        print(f"\n  Distribution:")
        for i, count in enumerate(bins):
            lo_edge = lo + i * bin_width
            hi_edge = lo + (i + 1) * bin_width
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  {lo_edge:>7.1f}–{hi_edge:<7.1f}  {count:>4d}  {bar}")


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

        # Basic metrics
        metrics["char_counts"].append(char_count(text))
        metrics["word_counts"].append(estimate_word_count(text))

        # Tag presence
        if "<character>" in text and "</character>" in text:
            metrics["has_character_tags"] += 1

        # Section analysis
        sections = extract_sections(text)
        # Normalize section names for matching (strip trailing colons, etc.)
        found_sections = set()
        for s_name in sections:
            # Try to match against expected sections
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

        # Example messages
        metrics["example_msg_counts"].append(count_example_messages(text))

        # Name tracking
        name = meta.get("name", "Unknown")
        name_counts[name] += 1

    # Find duplicates
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
        help="Show additional detail (e.g. all duplicate names)",
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

    print(f"\n  Total personas: {total}")

    # ── Demographics ─────────────────────────────────────────────────────
    ages = []
    regions = Counter()
    subregions = Counter()
    names = Counter()
    models = Counter()
    genders = Counter()

    for record in personas:
        meta = record.get("meta", {})
        age = meta.get("age")
        if age is not None:
            ages.append(age)
        regions[meta.get("region", "Unknown")] += 1
        subregions[meta.get("subregion", "Unknown")] += 1
        names[meta.get("name", "Unknown")] += 1
        models[meta.get("model", "Unknown")] += 1

    print(f"\n{'─' * 70}")
    print(f"  AGE DISTRIBUTION")
    print(f"{'─' * 70}")
    print_numeric_stats(ages, "Age")

    print(f"\n{'─' * 70}")
    print(f"  REGION DISTRIBUTION (top {args.top_n})")
    print(f"{'─' * 70}")
    print_distribution(regions, "Region", top_n=args.top_n, total=total)

    print(f"\n{'─' * 70}")
    print(f"  SUBREGION DISTRIBUTION (top {args.top_n})")
    print(f"{'─' * 70}")
    print_distribution(subregions, "Subregion", top_n=args.top_n, total=total)

    print(f"\n{'─' * 70}")
    print(f"  NAME DISTRIBUTION (top {args.top_n})")
    print(f"{'─' * 70}")
    print_distribution(names, "Name", top_n=args.top_n, total=total)

    # Unique name ratio
    unique_names = len(names)
    print(
        f"\n  Unique names: {unique_names} / {total} ({unique_names / total * 100:.1f}%)"
    )

    print(f"\n{'─' * 70}")
    print(f"  MODEL DISTRIBUTION")
    print(f"{'─' * 70}")
    print_distribution(models, "Model", top_n=10, total=total)

    # ── Text Quality ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  TEXT QUALITY METRICS")
    print(f"{'─' * 70}")

    quality = analyze_text_quality(personas)

    print(f"\n  Character count:")
    print_numeric_stats(quality["char_counts"], "Characters")

    print(f"\n  Word count:")
    print_numeric_stats(quality["word_counts"], "Words")

    print(f"\n  Section count (out of 11 expected):")
    print_numeric_stats(quality["section_counts"], "Sections")

    print(f"\n  Example message blocks (<START> count):")
    print_numeric_stats(quality["example_msg_counts"], "Examples")

    # Tag presence
    tag_pct = quality["has_character_tags"] / total * 100
    print(
        f"\n  Proper <character> tags: {quality['has_character_tags']} / {total} ({tag_pct:.1f}%)"
    )

    # Missing sections
    if quality["missing_sections"]:
        print(f"\n  Frequently missing sections:")
        for section, count in quality["missing_sections"].most_common():
            pct = count / total * 100
            print(f"    {section:>35s}:  {count:>4d} missing ({pct:.1f}%)")

    # ── Potential Issues ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  POTENTIAL ISSUES")
    print(f"{'─' * 70}")

    issues = []

    # Very short personas
    short_count = sum(1 for c in quality["char_counts"] if c < 1500)
    if short_count > 0:
        issues.append(
            f"  ⚠ {short_count} personas under 1500 characters (may be truncated)"
        )

    # Very long personas
    long_count = sum(1 for c in quality["char_counts"] if c > 5000)
    if long_count > 0:
        issues.append(
            f"  ⚠ {long_count} personas over 5000 characters (may be verbose)"
        )

    # Missing tags
    missing_tags = total - quality["has_character_tags"]
    if missing_tags > 0:
        issues.append(f"  ⚠ {missing_tags} personas missing <character> tags")

    # No example messages
    no_examples = sum(1 for c in quality["example_msg_counts"] if c == 0)
    if no_examples > 0:
        issues.append(f"  ⚠ {no_examples} personas have no example message blocks")

    # Duplicate names
    dupes = quality["duplicate_names"]
    if dupes:
        dupe_count = sum(c for _, c in dupes)
        issues.append(
            f"  ⚠ {len(dupes)} duplicate name(s) affecting {dupe_count} personas"
        )
        if args.verbose:
            for name, count in sorted(dupes, key=lambda x: -x[1])[:20]:
                issues.append(f"      {name}: {count}x")

    # Underrepresented regions (less than 0.5% when they should have more)
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
                for r in zero_regions[:10]:
                    issues.append(f"      {r}: {regions[r]}")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No obvious issues detected")

    # ── Region Coverage ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  REGION COVERAGE")
    print(f"{'─' * 70}")

    # Try to load the regions.yaml for expected regions
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
                f"  Coverage:          {len(covered)} / {len(expected_regions)} ({len(covered) / len(expected_regions) * 100:.1f}%)"
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
    print(f"  Analysis complete.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
