#!/usr/bin/env python3
"""
Comprehensive persona dataset statistics.

Analyzes merged or raw persona JSONL files and prints detailed statistics
covering demographic distributions, token usage, text quality metrics,
evaluation scores (overall and per generation model), and potential issues.
A text report is automatically saved alongside the input file.

Usage:
    python scripts/01e_stats_personas.py
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl --verbose
    python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl --report reports/my_report.txt
"""

import argparse
import datetime
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Tee ───────────────────────────────────────────────────────────────────────


class Tee:
    """Duplicate stdout writes to both the terminal and a file."""

    def __init__(self, filepath: Path):
        self._terminal = sys.stdout
        self._file = open(filepath, "w", encoding="utf-8")
        sys.stdout = self

    def write(self, message):
        self._terminal.write(message)
        self._file.write(message)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._terminal
        self._file.close()


# ── I/O helpers ───────────────────────────────────────────────────────────────


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


# ── Text helpers ──────────────────────────────────────────────────────────────


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


# ── Print helpers ─────────────────────────────────────────────────────────────


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


# ── Text quality ──────────────────────────────────────────────────────────────


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


# ── Eval analysis ─────────────────────────────────────────────────────────────

EVAL_CRITERIA = [
    "internal_consistency",
    "archetype_fidelity",
    "show_dont_tell",
    "psychological_realism",
    "roleplay_utility",
    "distinctiveness",
    "format_compliance",
]


def analyze_eval_scores(personas: list) -> dict:
    """
    Collect overall eval_score and per-criterion scores from eval_results.

    Returns a dict with:
        overall      : list[float]          — top-level eval_score values
        criteria     : dict[str, list[int]] — per-criterion score lists
        judge_models : Counter              — frequency of judge_model strings
        with_eval    : int                  — number of records that have eval data
        score_dist   : Counter[str]         — bucketed overall score distribution
        by_model     : dict[str, dict]      — per-generation-model overall + criteria
    """
    result = {
        "overall": [],
        "criteria": {c: [] for c in EVAL_CRITERIA},
        "judge_models": Counter(),
        "with_eval": 0,
        "score_dist": Counter(),
        "by_model": {},
    }

    for record in personas:
        eval_score = record.get("eval_score")
        eval_results = record.get("eval_results")

        if eval_score is None and eval_results is None:
            continue

        result["with_eval"] += 1

        model = record.get("meta", {}).get("model", "Unknown")
        if model not in result["by_model"]:
            result["by_model"][model] = {
                "overall": [],
                "criteria": {c: [] for c in EVAL_CRITERIA},
            }

        if eval_score is not None:
            try:
                score = float(eval_score)
                result["overall"].append(score)
                result["score_dist"][f"{score:.1f}"] += 1
                result["by_model"][model]["overall"].append(score)
            except (TypeError, ValueError):
                pass

        if eval_results and isinstance(eval_results, dict):
            scores = eval_results.get("scores", {})
            if isinstance(scores, dict):
                for criterion in EVAL_CRITERIA:
                    val = scores.get(criterion)
                    if val is not None:
                        try:
                            ival = int(val)
                            result["criteria"][criterion].append(ival)
                            result["by_model"][model]["criteria"][criterion].append(
                                ival
                            )
                        except (TypeError, ValueError):
                            pass

            judge = eval_results.get("judge_model")
            if judge:
                result["judge_models"][judge] += 1

    return result


# ── Eval print helpers ────────────────────────────────────────────────────────


def print_eval_per_model_section(eval_data: dict, verbose: bool):
    """Print per-generation-model evaluation breakdown."""
    by_model = eval_data.get("by_model", {})
    if not by_model:
        return

    model_rows = []
    for model, data in by_model.items():
        overall = data["overall"]
        if not overall:
            continue
        mean_overall = statistics.mean(overall)
        n = len(overall)
        criteria_means = {}
        for c in EVAL_CRITERIA:
            vals = data["criteria"].get(c, [])
            criteria_means[c] = statistics.mean(vals) if vals else None
        model_rows.append((model, n, mean_overall, criteria_means))

    model_rows.sort(key=lambda r: -r[2])

    print_section("EVAL SCORES BY GENERATION MODEL")

    short = {
        "internal_consistency": "consist.",
        "archetype_fidelity": "archtype",
        "show_dont_tell": "show/tell",
        "psychological_realism": "psych.",
        "roleplay_utility": "roleplay",
        "distinctiveness": "distinct.",
        "format_compliance": "format",
    }

    header_criteria = "  ".join(f"{short[c]:>9s}" for c in EVAL_CRITERIA)
    print(f"\n  {'Model':<40s}  {'n':>5s}  {'Mean':>6s}  {header_criteria}")
    print(
        f"  {'─' * (40 + 5 + 6 + 9 * len(EVAL_CRITERIA) + 4 + len(EVAL_CRITERIA) * 2)}"
    )

    for model, n, mean_overall, criteria_means in model_rows:
        criteria_cols = "  ".join(
            f"{criteria_means[c]:>9.2f}"
            if criteria_means[c] is not None
            else f"{'—':>9s}"
            for c in EVAL_CRITERIA
        )
        display_model = model if len(model) <= 40 else model[:37] + "..."
        print(f"  {display_model:<40s}  {n:>5d}  {mean_overall:>6.2f}  {criteria_cols}")

    if len(model_rows) >= 2:
        best = model_rows[0]
        worst = model_rows[-1]
        print(f"\n  Best  model (by mean score): {best[0]}  →  {best[2]:.2f}")
        print(f"  Worst model (by mean score): {worst[0]}  →  {worst[2]:.2f}")

    if verbose:
        print(f"\n  Per-model weakest criterion:")
        for model, n, mean_overall, criteria_means in model_rows:
            valid = {c: v for c, v in criteria_means.items() if v is not None}
            if valid:
                weakest = min(valid, key=lambda c: valid[c])
                print(f"    {model:<40s}  →  {weakest} ({valid[weakest]:.2f})")


def print_eval_section(eval_data: dict, total: int, top_n: int, verbose: bool):
    """Print the full evaluation scores section."""
    with_eval = eval_data["with_eval"]
    eval_pct = with_eval / total * 100 if total else 0

    print_section("EVALUATION SCORES")
    print(f"\n  Evaluated personas: {with_eval} / {total} ({eval_pct:.1f}%)")

    if not eval_data["overall"]:
        print("  (no eval_score data found)")
        print_eval_per_model_section(eval_data, verbose)
        return

    # ── Overall score stats ──────────────────────────────────────────────
    print("\n  Overall eval_score:")
    print_numeric_stats(eval_data["overall"], "eval_score", show_histogram=False)

    score_dist = eval_data["score_dist"]
    if score_dist:
        print("\n  Score distribution:")
        dist_total = sum(score_dist.values())
        for bucket, count in sorted(score_dist.items()):
            pct = count / dist_total * 100
            bar = "█" * int(pct / 2)
            print(f"  {bucket:>6s}  {count:>5d}  ({pct:5.1f}%)  {bar}")

    # ── Per-criterion stats ──────────────────────────────────────────────
    criteria_with_data = {c: vals for c, vals in eval_data["criteria"].items() if vals}
    if criteria_with_data:
        print(f"\n  Per-criterion mean scores:")
        header = (
            f"  {'Criterion':>28s}  {'n':>5s}  {'Mean':>6s}"
            f"  {'Median':>7s}  {'Min':>4s}  {'Max':>4s}  {'StdDev':>7s}"
        )
        print(header)
        print(f"  {'─' * 65}")
        for criterion in EVAL_CRITERIA:
            vals = criteria_with_data.get(criterion)
            if not vals:
                print(f"  {criterion:>28s}  {'—':>5s}")
                continue
            n = len(vals)
            mean = statistics.mean(vals)
            median = statistics.median(vals)
            stdev = statistics.stdev(vals) if n > 1 else 0.0
            lo, hi = min(vals), max(vals)
            print(
                f"  {criterion:>28s}  {n:>5d}  {mean:>6.2f}  {median:>7.1f}"
                f"  {lo:>4d}  {hi:>4d}  {stdev:>7.2f}"
            )

        sorted_criteria = sorted(
            criteria_with_data.items(),
            key=lambda kv: statistics.mean(kv[1]),
        )
        if len(sorted_criteria) >= 2:
            weakest_name, weakest_vals = sorted_criteria[0]
            strongest_name, strongest_vals = sorted_criteria[-1]
            print(
                f"\n  Weakest criterion:   {weakest_name} "
                f"(mean {statistics.mean(weakest_vals):.2f})"
            )
            print(
                f"  Strongest criterion: {strongest_name} "
                f"(mean {statistics.mean(strongest_vals):.2f})"
            )

    # ── Judge model distribution ─────────────────────────────────────────
    if eval_data["judge_models"]:
        print(f"\n  Judge model distribution:")
        print_distribution(
            eval_data["judge_models"], "Judge model", top_n=top_n, total=with_eval
        )

    # ── Low-score breakdown (verbose) ────────────────────────────────────
    if verbose and eval_data["overall"]:
        low_threshold = 3.0
        low_scores = [s for s in eval_data["overall"] if s < low_threshold]
        if low_scores:
            print(
                f"\n  Low scores (< {low_threshold}): {len(low_scores)} "
                f"({len(low_scores) / len(eval_data['overall']) * 100:.1f}%)"
            )

    # ── Per-model breakdown ──────────────────────────────────────────────
    print_eval_per_model_section(eval_data, verbose)


# ── Main ──────────────────────────────────────────────────────────────────────


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
        "--report",
        type=str,
        default=None,
        help=(
            "Path to save the text report. "
            "Defaults to <input_dir>/stats_report_<timestamp>.txt"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail (e.g. all duplicate names, low-rep regions, low eval scores)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top entries to show in distributions (default: 20)",
    )
    args = parser.parse_args()

    # ── Report file setup ─────────────────────────────────────────────────
    input_path = Path(args.input)

    if args.report:
        report_path = Path(args.report)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = input_path.parent / f"stats_report_{timestamp}.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(report_path)

    # ── Validate input ────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        tee.close()
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  PERSONA DATASET STATISTICS")
    print(f"  Source:  {input_path}")
    print(f"  Report:  {report_path}")
    print(f"  Run at:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}")

    personas = load_personas(str(input_path))
    total = len(personas)

    if total == 0:
        print("\n  No personas found in file.")
        tee.close()
        sys.exit(0)

    print(f"\n  Total personas:  {total}")

    with_ids = sum(1 for p in personas if p.get("meta", {}).get("id"))
    print(f"  With unique IDs: {with_ids} / {total}")

    with_eval_quick = sum(
        1
        for p in personas
        if p.get("eval_score") is not None or p.get("eval_results") is not None
    )
    print(f"  With eval data:  {with_eval_quick} / {total}")

    # ── Demographics ──────────────────────────────────────────────────────
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
            source_timestamps[ts[:10]] += 1

    # ── Age ───────────────────────────────────────────────────────────────
    print_section("AGE DISTRIBUTION")
    print_numeric_stats(ages, "Age")

    # ── Region ────────────────────────────────────────────────────────────
    print_section(f"REGION DISTRIBUTION (top {args.top_n})")
    print_distribution(regions, "Region", top_n=args.top_n, total=total)

    print_section(f"SUBREGION DISTRIBUTION (top {args.top_n})")
    print_distribution(subregions, "Subregion", top_n=args.top_n, total=total)

    # ── Names ─────────────────────────────────────────────────────────────
    print_section(f"NAME DISTRIBUTION (top {args.top_n})")
    print_distribution(names, "Name", top_n=args.top_n, total=total)
    unique_names = len(names)
    print(
        f"\n  Unique names: {unique_names} / {total} ({unique_names / total * 100:.1f}%)"
    )

    # ── Models ────────────────────────────────────────────────────────────
    print_section("MODEL DISTRIBUTION")
    print_distribution(models, "Model", top_n=10, total=total)

    # ── Archetypes ────────────────────────────────────────────────────────
    print_section("ARCHETYPE DISTRIBUTION")
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
    unknown_arch = archetypes.get("Unknown", 0)
    if unknown_arch:
        print(f"  {'Unknown (pre-archetype)':>25s}  {unknown_arch:>6d}")

    # ── Regulation Styles ─────────────────────────────────────────────────
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

    # ── Source Timestamps ─────────────────────────────────────────────────
    if source_timestamps:
        print_section("PERSONAS BY SOURCE DATE")
        print_distribution(source_timestamps, "Date", top_n=30, total=total)

    # ── Token Usage ───────────────────────────────────────────────────────
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

    # ── Evaluation Scores ─────────────────────────────────────────────────
    eval_data = analyze_eval_scores(personas)
    print_eval_section(eval_data, total, args.top_n, args.verbose)

    # ── Text Quality ──────────────────────────────────────────────────────
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

    # ── Potential Issues ──────────────────────────────────────────────────
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

    if arch_total >= 30:
        for archetype, base_rate in ARCHETYPE_BASE_RATES.items():
            count = archetypes.get(archetype, 0)
            actual_rate = count / arch_total
            if abs(actual_rate - base_rate) > 0.10:
                issues.append(
                    f"  ⚠ Archetype '{archetype}' is {actual_rate * 100:.1f}% "
                    f"(target {base_rate * 100:.1f}%) — deviation > 10pp"
                )

    if reg_total >= 30:
        for style, base_rate in REGULATION_BASE_RATES.items():
            count = regulation_styles.get(style, 0)
            actual_rate = count / reg_total
            if abs(actual_rate - base_rate) > 0.10:
                issues.append(
                    f"  ⚠ Regulation style '{style}' is {actual_rate * 100:.1f}% "
                    f"(target {base_rate * 100:.1f}%) — deviation > 10pp"
                )

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

    no_token_data = total - len(input_tokens)
    if no_token_data > 0 and no_token_data < total:
        issues.append(
            f"  ⚠ {no_token_data} personas have no token usage data (pre-tracking records)"
        )

    if eval_data["overall"]:
        low_eval_threshold = 3.0
        low_eval_count = sum(1 for s in eval_data["overall"] if s < low_eval_threshold)
        if low_eval_count > 0:
            issues.append(
                f"  ⚠ {low_eval_count} personas with eval_score < {low_eval_threshold} "
                f"(consider filtering or regenerating)"
            )

    unevaluated = total - eval_data["with_eval"]
    if unevaluated > 0:
        issues.append(f"  ⚠ {unevaluated} personas have no evaluation data")

    for criterion, vals in eval_data["criteria"].items():
        if vals and statistics.mean(vals) < 3.0:
            issues.append(
                f"  ⚠ Criterion '{criterion}' has mean score "
                f"{statistics.mean(vals):.2f} — below acceptable floor (3.0)"
            )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No obvious issues detected")

    # ── Region Coverage ───────────────────────────────────────────────────
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
    print(f"  Analysis complete.")
    print(f"  Report saved to: {report_path}")
    print(f"{'═' * 70}\n")

    tee.close()


if __name__ == "__main__":
    main()
