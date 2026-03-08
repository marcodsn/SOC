#!/usr/bin/env python3
"""
Comprehensive conversation dataset statistics.

Analyzes merged or raw conversation JSONL files and prints detailed
statistics covering turn counts, message types, timing patterns,
conversation styles, topic exhaustion, token usage (with means), and
potential issues.

Usage:
    python scripts/03e_stats_conversations.py
    python scripts/03e_stats_conversations.py --input data/conversations/generated/data.jsonl
    python scripts/03e_stats_conversations.py --input data/conversations/generated/data.jsonl --verbose
    python scripts/03e_stats_conversations.py --input "data/conversations/generated/conversations_*.jsonl" --glob
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Data loading ─────────────────────────────────────────────────────────────


def load_conversations(path: str, use_glob: bool = False) -> list:
    """Load conversation records from one or more JSONL files."""
    conversations = []
    if use_glob:
        files = sorted(glob(path))
        if not files:
            print(f"  ⚠ No files matched glob pattern: {path}")
            return []
        for fp in files:
            conversations.extend(_load_single_file(fp))
    else:
        conversations = _load_single_file(path)
    return conversations


def _load_single_file(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping invalid JSON in {path}, line {line_num}: {e}")
    return records


# ── Turn / message parsing ───────────────────────────────────────────────────


def parse_messages_from_turn(turn_xml: str) -> list:
    """Extract all <message> elements from a turn XML block."""
    messages = []
    pattern = (
        r'<message\s+t="([^"]*?)"\s+d="([^"]*?)"\s+type="([^"]*?)">(.*?)</message>'
    )
    for m in re.finditer(pattern, turn_xml, re.DOTALL):
        messages.append(
            {
                "time": m.group(1),
                "date": m.group(2),
                "type": m.group(3),
                "content": m.group(4).strip(),
            }
        )
    return messages


def parse_state_from_turn(turn_xml: str) -> str | None:
    """Extract <state> content from a turn XML block."""
    m = re.search(r"<state>\s*(.*?)\s*</state>", turn_xml, re.DOTALL)
    return m.group(1).strip() if m else None


def has_instant_event(turn_xml: str) -> bool:
    return "<instant_event>" in turn_xml


def is_exhausted(turn_xml: str) -> bool:
    return "<predefined_topics_exhausted" in turn_xml


def message_word_count(content: str) -> int:
    """Word count for a single message, ignoring media descriptions."""
    return len(content.split())


def time_to_minutes(t: str) -> float | None:
    """Convert HH:MM to minutes since midnight."""
    parts = t.split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return None


def date_to_ordinal(d: str) -> int | None:
    """Convert DD.MM to a sortable ordinal (month * 100 + day)."""
    parts = d.split(".")
    if len(parts) != 2:
        return None
    try:
        day, month = int(parts[0]), int(parts[1])
        return month * 100 + day
    except ValueError:
        return None


# ── Printing helpers ─────────────────────────────────────────────────────────


def print_distribution(counter: Counter, top_n: int = 20, total: int = 0):
    if not counter:
        print("  (no data)")
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


def print_numeric_stats(values: list, label: str = "", show_histogram: bool = True):
    if not values:
        print("  (no data)")
        return
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

    if show_histogram and n >= 5:
        num_bins = min(15, max(5, n // 10))
        lo, hi = values_sorted[0], values_sorted[-1]
        if lo == hi:
            print(f"  (all values are {lo})")
            return
        # Handle float and int ranges
        if isinstance(lo, float):
            bin_width = (hi - lo) / num_bins
        else:
            bin_width = max(1, (hi - lo) / num_bins)
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
            if isinstance(lo, int):
                print(f"  {lo_edge:>7.0f}–{hi_edge:<7.0f}  {count:>4d}  {bar}")
            else:
                print(f"  {lo_edge:>7.1f}–{hi_edge:<7.1f}  {count:>4d}  {bar}")


def print_section(title: str, width: int = 70):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Analysis functions ───────────────────────────────────────────────────────


def analyze_conversations(conversations: list, verbose: bool = False, top_n: int = 20):
    total = len(conversations)
    if total == 0:
        print("\n  No conversations found in file.")
        return

    print(f"\n  Total conversations: {total}")

    # Check for IDs
    with_ids = sum(
        1
        for c in conversations
        if c.get("meta", {}).get("experience_meta", {}).get("id")
        or c.get("meta", {}).get("id")
    )
    print(f"  With unique IDs:    {with_ids} / {total}")

    # ── Basic metadata ───────────────────────────────────────────────────
    turn_counts = []
    styles = Counter()
    cadences = Counter()
    exhausted_count = 0
    models = Counter()
    summarizer_models = Counter()
    persona_names_all = Counter()
    source_timestamps = Counter()
    total_input_tokens = []
    total_output_tokens = []
    total_summarizer_input_tokens = []
    total_summarizer_output_tokens = []
    generation_times = []

    # Per-conversation message-level stats
    all_messages_per_conv = []
    all_message_types = Counter()
    all_message_word_counts = []
    all_messages_per_turn = []

    all_turns_with_events = 0

    # Timing analysis
    all_inter_message_gaps = []  # in minutes, within same day
    all_multi_day_convos = 0

    # Topic tracking
    final_states = Counter()

    for conv in conversations:
        meta = conv.get("meta", {})
        turns = conv.get("turns", [])

        n_turns = meta.get("n_turns", len(turns))
        turn_counts.append(n_turns)

        style = meta.get("conversation_style", "unknown")
        styles[style] += 1

        cadence = meta.get("message_cadence", "unknown")
        cadences[cadence] += 1

        if meta.get("exhausted", False):
            exhausted_count += 1

        model = meta.get("model", "unknown")
        models[model] += 1

        s_model = meta.get("summarizer_model", "unknown")
        summarizer_models[s_model] += 1

        # Persona names
        for name in meta.get("persona_names", []):
            persona_names_all[name] += 1

        # Token usage
        if meta.get("total_input_tokens"):
            total_input_tokens.append(meta["total_input_tokens"])
        if meta.get("total_output_tokens"):
            total_output_tokens.append(meta["total_output_tokens"])
        if meta.get("total_summarizer_input_tokens"):
            total_summarizer_input_tokens.append(meta["total_summarizer_input_tokens"])
        if meta.get("total_summarizer_output_tokens"):
            total_summarizer_output_tokens.append(
                meta["total_summarizer_output_tokens"]
            )

        # Generation time
        if meta.get("time_taken"):
            generation_times.append(meta["time_taken"])

        # Source timestamp (from nested experience_meta, by date)
        exp_meta = meta.get("experience_meta", {})
        ts = exp_meta.get("source_timestamp")
        if ts:
            source_timestamps[ts[:10]] += 1

        # Per-turn analysis
        conv_messages = []
        conv_dates = set()
        prev_time_minutes = None
        prev_date_ordinal = None

        for turn_xml in turns:
            if not isinstance(turn_xml, str):
                continue

            messages = parse_messages_from_turn(turn_xml)
            msg_count = len(messages)
            all_messages_per_turn.append(msg_count)

            if has_instant_event(turn_xml):
                all_turns_with_events += 1
                # (all_instant_events counter removed — use all_turns_with_events instead)

            state = parse_state_from_turn(turn_xml)
            if is_exhausted(turn_xml):
                pass  # already counted via meta

            for msg in messages:
                all_message_types[msg["type"]] += 1

                if msg["type"] == "text":
                    wc = message_word_count(msg["content"])
                    all_message_word_counts.append(wc)

                conv_messages.append(msg)

                # Timing
                date_ord = date_to_ordinal(msg["date"])
                time_min = time_to_minutes(msg["time"])

                if date_ord is not None:
                    conv_dates.add(date_ord)

                # Inter-message gap (same day only for simplicity)
                if (
                    prev_time_minutes is not None
                    and time_min is not None
                    and prev_date_ordinal is not None
                    and date_ord is not None
                ):
                    if date_ord == prev_date_ordinal:
                        gap = time_min - prev_time_minutes
                        if gap < 0:
                            gap += 24 * 60  # crossed midnight
                        all_inter_message_gaps.append(gap)
                    else:
                        # Different day — count it but don't compute gap
                        pass

                prev_time_minutes = time_min
                prev_date_ordinal = date_ord

            # Track final state
            if turn_xml == turns[-1] and state:
                final_states[state] += 1

        all_messages_per_conv.append(len(conv_messages))
        if len(conv_dates) > 1:
            all_multi_day_convos += 1

    # ── Print results ────────────────────────────────────────────────────

    # Turn counts
    print_section("TURNS PER CONVERSATION")
    print_numeric_stats(turn_counts)

    # Messages per conversation
    print_section("MESSAGES PER CONVERSATION")
    print_numeric_stats(all_messages_per_conv)

    # Messages per turn
    print_section("MESSAGES PER TURN")
    print_numeric_stats(all_messages_per_turn)

    # Conversation styles
    print_section("CONVERSATION STYLE DISTRIBUTION")
    print_distribution(styles, total=total)

    # Message cadences
    print_section("MESSAGE CADENCE DISTRIBUTION")
    print_distribution(cadences, total=total)

    # Exhaustion
    print_section("TOPIC EXHAUSTION")
    exh_pct = exhausted_count / total * 100 if total else 0
    non_exh = total - exhausted_count
    non_pct = non_exh / total * 100 if total else 0
    print(f"  Exhausted (topics completed):    {exhausted_count:>5d}  ({exh_pct:.1f}%)")
    print(f"  Not exhausted (hit max turns):   {non_exh:>5d}  ({non_pct:.1f}%)")

    # Message types
    print_section("MESSAGE TYPE DISTRIBUTION")
    total_msgs = sum(all_message_types.values())
    print(f"  Total messages: {total_msgs}")
    print_distribution(all_message_types, total=total_msgs)

    # Text message word counts
    if all_message_word_counts:
        print_section("TEXT MESSAGE WORD COUNT")
        print_numeric_stats(all_message_word_counts)

    # Instant events
    print_section("INSTANT EVENTS")
    total_turns = sum(turn_counts)
    event_pct = all_turns_with_events / total_turns * 100 if total_turns else 0
    print(
        f"  Turns with instant events: {all_turns_with_events} / {total_turns} ({event_pct:.1f}%)"
    )

    # Multi-day conversations
    print_section("MULTI-DAY CONVERSATIONS")
    multi_pct = all_multi_day_convos / total * 100 if total else 0
    print(
        f"  Conversations spanning multiple days: {all_multi_day_convos} / {total} ({multi_pct:.1f}%)"
    )

    # Inter-message timing
    if all_inter_message_gaps:
        print_section("INTER-MESSAGE GAP (minutes, same-day only)")
        print_numeric_stats(all_inter_message_gaps)

        # Cadence breakdown
        realtime_gaps = [g for g in all_inter_message_gaps if g <= 2]
        delayed_gaps = [g for g in all_inter_message_gaps if 2 < g <= 60]
        async_gaps = [g for g in all_inter_message_gaps if g > 60]
        g_total = len(all_inter_message_gaps)
        print(f"\n  Gap breakdown:")
        print(
            f"    ≤2 min (realtime):   {len(realtime_gaps):>5d}  ({len(realtime_gaps) / g_total * 100:.1f}%)"
        )
        print(
            f"    2–60 min (delayed):  {len(delayed_gaps):>5d}  ({len(delayed_gaps) / g_total * 100:.1f}%)"
        )
        print(
            f"    >60 min (async):     {len(async_gaps):>5d}  ({len(async_gaps) / g_total * 100:.1f}%)"
        )

    # Models
    print_section("MODEL DISTRIBUTION")
    print_distribution(models, total=total)

    if any(m != "unknown" for m in summarizer_models):
        print_section("SUMMARIZER MODEL DISTRIBUTION")
        print_distribution(summarizer_models, total=total)

    # Source timestamps
    if source_timestamps:
        print_section("CONVERSATIONS BY SOURCE DATE")
        print_distribution(source_timestamps, top_n=30, total=total)

    # Token usage
    if total_input_tokens:
        print_section("TOKEN USAGE — TURN GENERATION (per conversation)")
        print(f"\n  Input tokens  (n={len(total_input_tokens)}):")
        print_numeric_stats(total_input_tokens, show_histogram=False)
        print(f"\n  Output tokens  (n={len(total_output_tokens)}):")
        print_numeric_stats(total_output_tokens, show_histogram=False)

        if total_summarizer_input_tokens and any(
            t > 0 for t in total_summarizer_input_tokens
        ):
            print(
                f"\n  Summarizer input tokens  (n={len(total_summarizer_input_tokens)}):"
            )
            print_numeric_stats(total_summarizer_input_tokens, show_histogram=False)
            print(
                f"\n  Summarizer output tokens  (n={len(total_summarizer_output_tokens)}):"
            )
            print_numeric_stats(total_summarizer_output_tokens, show_histogram=False)

        total_in = sum(total_input_tokens)
        total_out = sum(total_output_tokens)
        total_s_in = (
            sum(total_summarizer_input_tokens) if total_summarizer_input_tokens else 0
        )
        total_s_out = (
            sum(total_summarizer_output_tokens) if total_summarizer_output_tokens else 0
        )
        grand_total = total_in + total_out + total_s_in + total_s_out

        print("\n  Aggregate token usage:")
        print(f"    Turn gen input:        {total_in:>12,d}")
        print(f"    Turn gen output:       {total_out:>12,d}")
        print(f"    Summarizer input:      {total_s_in:>12,d}")
        print(f"    Summarizer output:     {total_s_out:>12,d}")
        print(f"    Grand total:           {grand_total:>12,d}")

        n_tok = len(total_input_tokens)
        print("\n  Mean per conversation:")
        print(f"    Turn gen input:        {total_in / n_tok:>12,.1f}")
        print(f"    Turn gen output:       {total_out / n_tok:>12,.1f}")
        if total_s_in > 0:
            n_s = len(total_summarizer_input_tokens)
            print(f"    Summarizer input:      {total_s_in / n_s:>12,.1f}")
            print(f"    Summarizer output:     {total_s_out / n_s:>12,.1f}")
        print(f"    Total (all):           {grand_total / n_tok:>12,.1f}")

        # Per-turn rates (using turn_counts aligned to token records)
        # turn_counts are in the same order as conversations
        turns_with_tokens = [
            turn_counts[i]
            for i, conv in enumerate(conversations)
            if conv.get("meta", {}).get("total_input_tokens")
        ]
        if turns_with_tokens and len(turns_with_tokens) == n_tok:
            total_turns_for_tok = sum(turns_with_tokens)
            if total_turns_for_tok > 0:
                print("\n  Per-turn rates (turn gen only):")
                print(
                    f"    Input  tokens / turn:  {total_in / total_turns_for_tok:>12,.1f}"
                )
                print(
                    f"    Output tokens / turn:  {total_out / total_turns_for_tok:>12,.1f}"
                )
    else:
        print_section("TOKEN USAGE")
        print("  (no token data found — pre-token-tracking records)")

    # Generation time
    if generation_times:
        print_section("GENERATION TIME (seconds per conversation)")
        print_numeric_stats(generation_times, show_histogram=False)
        total_time = sum(generation_times)
        print(
            f"\n  Total generation time: {total_time:.1f}s ({total_time / 60:.1f} min)"
        )
        if total > 0:
            print(f"  Mean per conversation:  {total_time / total:.1f}s")

    # Persona participation
    print_section(f"PERSONA PARTICIPATION (top {top_n})")
    print_distribution(persona_names_all, top_n=top_n, total=total * 2)
    unique_personas = len(persona_names_all)
    print(f"\n  Unique persona names in conversations: {unique_personas}")

    # ── Potential Issues ─────────────────────────────────────────────────
    print_section("POTENTIAL ISSUES")
    issues = []

    # Very short conversations
    short_convos = sum(1 for tc in turn_counts if tc < 5)
    if short_convos > 0:
        issues.append(
            f"  ⚠ {short_convos} conversations with fewer than 5 turns (may indicate generation failures)"
        )

    # Conversations that hit max turns without exhaustion
    max_turn_hits = sum(1 for tc in turn_counts if tc >= 50)
    if max_turn_hits > 0:
        issues.append(f"  ⚠ {max_turn_hits} conversations hit the 50-turn ceiling")

    # Very wordy messages (knowledge dumping indicator)
    if all_message_word_counts:
        wordy = sum(1 for wc in all_message_word_counts if wc > 80)
        if wordy > 0:
            wordy_pct = wordy / len(all_message_word_counts) * 100
            issues.append(
                f"  ⚠ {wordy} messages exceed 80 words ({wordy_pct:.1f}%) — possible knowledge dumping"
            )

    # Single-message turns dominating (might indicate lack of multi-message variety)
    if all_messages_per_turn:
        single_msg = sum(1 for mpt in all_messages_per_turn if mpt == 1)
        single_pct = single_msg / len(all_messages_per_turn) * 100
        if single_pct > 90:
            issues.append(
                f"  ⚠ {single_pct:.0f}% of turns have exactly 1 message — consider encouraging more multi-message turns"
            )

    # Very few media messages
    non_text = sum(c for t, c in all_message_types.items() if t != "text")
    if total_msgs > 0:
        media_pct = non_text / total_msgs * 100
        if media_pct < 1 and total_msgs > 100:
            issues.append(
                f"  ⚠ Only {media_pct:.1f}% of messages are non-text (images, audio, video, stickers)"
            )

    # Instant event rate check
    if total_turns > 0:
        if event_pct < 1 and total_turns > 200:
            issues.append(
                f"  ⚠ Instant event rate ({event_pct:.1f}%) is very low — check INSTANT_EVENT_PROB"
            )
        elif event_pct > 15:
            issues.append(
                f"  ⚠ Instant event rate ({event_pct:.1f}%) seems high — may disrupt conversation flow"
            )

    # Style imbalance
    if len(styles) > 1 and total >= 20:
        min_style_count = min(styles.values())
        max_style_count = max(styles.values())
        if max_style_count > 4 * min_style_count:
            least = styles.most_common()[-1]
            most = styles.most_common()[0]
            issues.append(
                f"  ⚠ Style imbalance: '{most[0]}' ({most[1]}×) vs '{least[0]}' ({least[1]}×) — "
                f"consider using style diversity nudging"
            )

    # Low exhaustion rate for structured/semi-structured
    structured_total = styles.get("structured", 0) + styles.get("semi-structured", 0)
    if structured_total > 10:
        # Count exhausted conversations that are structured or semi-structured
        structured_exhausted = 0
        for conv in conversations:
            m = conv.get("meta", {})
            if m.get("conversation_style") in (
                "structured",
                "semi-structured",
            ) and m.get("exhausted"):
                structured_exhausted += 1
        exh_rate = structured_exhausted / structured_total * 100
        if exh_rate < 50:
            issues.append(
                f"  ⚠ Only {exh_rate:.0f}% of structured/semi-structured conversations exhausted their topics"
            )

    # Conversations with no token data
    no_token_data = sum(
        1
        for conv in conversations
        if not conv.get("meta", {}).get("total_input_tokens")
    )
    if 0 < no_token_data < total:
        issues.append(
            f"  ⚠ {no_token_data} conversations have no token usage data (pre-tracking records)"
        )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No obvious issues detected")

    # ── Verbose: sample final states ─────────────────────────────────────
    if verbose and final_states:
        print_section(f"FINAL CONVERSATION STATES (top {top_n})")
        print_distribution(final_states, top_n=top_n)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive conversation dataset statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/conversations/generated/data.jsonl",
        help="Path to conversations JSONL file or glob pattern (default: data/conversations/generated/data.jsonl)",
    )
    parser.add_argument(
        "--glob",
        action="store_true",
        help="Treat --input as a glob pattern and merge all matching files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail (final states, etc.)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top entries to show in distributions (default: 20)",
    )
    args = parser.parse_args()

    if not args.glob and not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        print("  Use --glob to treat the path as a glob pattern.")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  CONVERSATION DATASET STATISTICS")
    print(f"  Source: {args.input}{'  (glob)' if args.glob else ''}")
    print(f"{'═' * 70}")

    conversations = load_conversations(args.input, use_glob=args.glob)
    analyze_conversations(conversations, verbose=args.verbose, top_n=args.top_n)

    print(f"\n{'═' * 70}")
    print("  Analysis complete.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
