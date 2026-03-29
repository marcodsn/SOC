#!/usr/bin/env python3
"""
Data validation and quality tests for the SOC pipeline.

Validates personas, experiences, and conversations against structural
and quality expectations. Can be run as a standalone script or imported
as a test module.

Usage:
    python scripts/04_validate_data.py
    python scripts/04_validate_data.py --personas data/personas/generated/data.jsonl
    python scripts/04_validate_data.py --conversations data/conversations/generated/data.jsonl --verbose
    python scripts/04_validate_data.py --skip-conversations
    python scripts/04_validate_data.py --sample 50
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_PERSONA_SECTIONS = [
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
]

VALID_STYLES = {"structured", "semi-structured", "freeform"}
VALID_CADENCES = {"realtime", "delayed", "async"}
VALID_MESSAGE_TYPES = {"text", "audio", "image", "video", "sticker"}

PERSONA_MIN_CHARS = 800
PERSONA_MAX_CHARS = 8000
PERSONA_TARGET_MIN_CHARS = 2000
PERSONA_TARGET_MAX_CHARS = 3000

MESSAGE_MAX_WORDS = 150  # flag as potential knowledge dump
MESSAGE_MAX_WORDS_HARD = 300  # almost certainly wrong

MIN_TURNS_PER_CONVERSATION = 3
MAX_TURNS_PER_CONVERSATION = 60

# ── Result tracking ──────────────────────────────────────────────────────────


class ValidationResult:
    """Accumulates pass/fail/warn results for a validation run."""

    def __init__(self):
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []
        self.info: List[str] = []

    def ok(self, msg: str):
        self.passed.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def fail(self, msg: str):
        self.failures.append(msg)

    def note(self, msg: str):
        self.info.append(msg)

    @property
    def total_checks(self) -> int:
        return len(self.passed) + len(self.warnings) + len(self.failures)

    @property
    def is_healthy(self) -> bool:
        return len(self.failures) == 0

    def print_summary(self, verbose: bool = False):
        print(f"\n{'═' * 70}")
        print(f"  VALIDATION SUMMARY")
        print(f"{'═' * 70}")
        print(f"  ✓ Passed:   {len(self.passed)}")
        print(f"  ⚠ Warnings: {len(self.warnings)}")
        print(f"  ✗ Failures: {len(self.failures)}")
        print(f"  Total checks: {self.total_checks}")

        if self.failures:
            print(f"\n{'─' * 70}")
            print(f"  FAILURES")
            print(f"{'─' * 70}")
            for f in self.failures:
                print(f"  ✗ {f}")

        if self.warnings:
            print(f"\n{'─' * 70}")
            print(f"  WARNINGS")
            print(f"{'─' * 70}")
            for w in self.warnings:
                print(f"  ⚠ {w}")

        if verbose and self.passed:
            print(f"\n{'─' * 70}")
            print(f"  PASSED")
            print(f"{'─' * 70}")
            for p in self.passed:
                print(f"  ✓ {p}")

        if verbose and self.info:
            print(f"\n{'─' * 70}")
            print(f"  INFO")
            print(f"{'─' * 70}")
            for i in self.info:
                print(f"  ℹ {i}")

        status = "HEALTHY" if self.is_healthy else "ISSUES FOUND"
        print(f"\n  Overall: {status}")
        print(f"{'═' * 70}\n")


# ── Loading ──────────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_parse_error": True, "_line": line_num})
    return records


def sample_records(records: list, n: Optional[int]) -> list:
    """Optionally subsample records for faster validation."""
    if n is None or n >= len(records):
        return records
    import random

    return random.sample(records, n)


# ── Persona validation ───────────────────────────────────────────────────────


def validate_persona_record(
    record: Dict[str, Any], index: int
) -> List[Tuple[str, str]]:
    """Validate a single persona record. Returns list of (level, message) tuples."""
    issues: List[Tuple[str, str]] = []

    if record.get("_parse_error"):
        issues.append(
            (
                "fail",
                f"Persona #{index}: JSON parse error at line {record.get('_line')}",
            )
        )
        return issues

    text = record.get("persona_text", "")
    meta = record.get("meta", {})

    # ── Structural checks ────────────────────────────────────────────────

    if not text:
        issues.append(("fail", f"Persona #{index}: empty persona_text"))
        return issues

    if "<character>" not in text or "</character>" not in text:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): missing <character> tags",
            )
        )

    char_count = len(text)
    if char_count < PERSONA_MIN_CHARS:
        issues.append(
            (
                "fail",
                f"Persona #{index} ({meta.get('name', '?')}): too short ({char_count} chars, min {PERSONA_MIN_CHARS})",
            )
        )
    elif char_count < PERSONA_TARGET_MIN_CHARS:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): below target length ({char_count} chars, target ≥{PERSONA_TARGET_MIN_CHARS})",
            )
        )
    elif char_count > PERSONA_MAX_CHARS:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): very long ({char_count} chars, max {PERSONA_MAX_CHARS})",
            )
        )

    # ── Section presence ─────────────────────────────────────────────────

    missing_sections = []
    for section in EXPECTED_PERSONA_SECTIONS:
        # Flexible match: check if section name appears as a bold heading
        if f"**{section}**" not in text and section.lower() not in text.lower():
            missing_sections.append(section)

    if len(missing_sections) > 3:
        issues.append(
            (
                "fail",
                f"Persona #{index} ({meta.get('name', '?')}): missing {len(missing_sections)} sections: {', '.join(missing_sections[:5])}",
            )
        )
    elif missing_sections:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): missing sections: {', '.join(missing_sections)}",
            )
        )

    # ── Example messages ─────────────────────────────────────────────────

    start_count = text.count("<START>")
    if start_count == 0:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): no <START> example message blocks",
            )
        )

    # ── Meta checks ──────────────────────────────────────────────────────

    if not meta:
        issues.append(("warn", f"Persona #{index}: empty meta"))
    else:
        if not meta.get("name"):
            issues.append(("warn", f"Persona #{index}: missing meta.name"))
        if not meta.get("region"):
            issues.append(("warn", f"Persona #{index}: missing meta.region"))
        age = meta.get("age")
        if age is not None:
            if not isinstance(age, (int, float)) or age < 10 or age > 110:
                issues.append(
                    (
                        "warn",
                        f"Persona #{index} ({meta.get('name', '?')}): unusual age: {age}",
                    )
                )

    # ── Content quality heuristics ───────────────────────────────────────

    # Check for common LLM artifacts
    llm_artifacts = [
        "As an AI",
        "as a language model",
        "I cannot",
        "I'm sorry, but",
        "Sure! Here",
        "Certainly!",
        "Here's a",
        "Here is a",
    ]
    for artifact in llm_artifacts:
        if artifact.lower() in text.lower():
            issues.append(
                (
                    "warn",
                    f"Persona #{index} ({meta.get('name', '?')}): possible LLM artifact: '{artifact}'",
                )
            )
            break

    # Check for bullet point / list formatting that shouldn't be there in narrative sections
    bullet_count = len(re.findall(r"^\s*[-•]\s", text, re.MULTILINE))
    if bullet_count > 10:
        issues.append(
            (
                "warn",
                f"Persona #{index} ({meta.get('name', '?')}): {bullet_count} bullet points — persona should be narrative, not listy",
            )
        )

    return issues


def validate_personas(records: list, result: ValidationResult, verbose: bool = False):
    """Run all persona validations."""
    print(f"\n{'─' * 70}")
    print(f"  VALIDATING PERSONAS ({len(records)} records)")
    print(f"{'─' * 70}")

    if not records:
        result.fail("No persona records to validate")
        return

    result.ok(f"Loaded {len(records)} persona records")

    # Per-record validation
    all_issues: List[Tuple[str, str]] = []
    for i, record in enumerate(records):
        all_issues.extend(validate_persona_record(record, i))

    fails = [msg for level, msg in all_issues if level == "fail"]
    warns = [msg for level, msg in all_issues if level == "warn"]

    for f in fails:
        result.fail(f)
    for w in warns:
        result.warn(w)

    # Aggregate checks
    parse_errors = sum(1 for r in records if r.get("_parse_error"))
    if parse_errors > 0:
        result.fail(f"{parse_errors} persona records failed JSON parsing")
    else:
        result.ok("All persona records parsed successfully")

    valid_records = [r for r in records if not r.get("_parse_error")]

    # Name uniqueness
    names = Counter(r.get("meta", {}).get("name", "?") for r in valid_records)
    duplicate_names = {n: c for n, c in names.items() if c > 1 and n != "?"}
    if duplicate_names:
        dup_summary = ", ".join(
            f"{n}({c}x)"
            for n, c in sorted(duplicate_names.items(), key=lambda x: -x[1])[:5]
        )
        if len(duplicate_names) > len(valid_records) * 0.3:
            result.fail(
                f"High name duplication: {len(duplicate_names)} names repeated — {dup_summary}"
            )
        else:
            result.warn(f"{len(duplicate_names)} duplicate name(s): {dup_summary}")
    else:
        result.ok("All persona names are unique")

    # Region coverage
    regions = Counter(r.get("meta", {}).get("region", "?") for r in valid_records)
    region_count = len([r for r in regions if r != "?"])
    if region_count < 5 and len(valid_records) >= 50:
        result.warn(
            f"Low region diversity: only {region_count} distinct regions for {len(valid_records)} personas"
        )
    elif region_count > 0:
        result.ok(f"Region coverage: {region_count} distinct regions")

    # Age distribution sanity
    ages = [
        r.get("meta", {}).get("age")
        for r in valid_records
        if r.get("meta", {}).get("age") is not None
    ]
    if ages:
        import statistics

        mean_age = statistics.mean(ages)
        if mean_age < 15 or mean_age > 60:
            result.warn(f"Unusual mean age: {mean_age:.1f} (expected ~25)")
        else:
            result.ok(f"Mean age: {mean_age:.1f}")

    # Character count distribution
    char_counts = [
        len(r.get("persona_text", "")) for r in valid_records if r.get("persona_text")
    ]
    if char_counts:
        mean_chars = sum(char_counts) / len(char_counts)
        result.note(
            f"Mean persona length: {mean_chars:.0f} chars (target: {PERSONA_TARGET_MIN_CHARS}–{PERSONA_TARGET_MAX_CHARS})"
        )

    ok_count = len(valid_records) - len(fails)
    result.note(
        f"Persona validation: {ok_count}/{len(valid_records)} records passed without failures"
    )


# ── Experience validation ────────────────────────────────────────────────────


def validate_experience_record(
    record: Dict[str, Any], index: int
) -> List[Tuple[str, str]]:
    """Validate a single experience record."""
    issues: List[Tuple[str, str]] = []

    if record.get("_parse_error"):
        issues.append(
            (
                "fail",
                f"Experience #{index}: JSON parse error at line {record.get('_line')}",
            )
        )
        return issues

    text = record.get("experience_text", "")
    meta = record.get("meta", {})

    if not text:
        issues.append(("fail", f"Experience #{index}: empty experience_text"))
        return issues

    # ── Tag structure ────────────────────────────────────────────────────

    if "<experience>" not in text or "</experience>" not in text:
        issues.append(("warn", f"Experience #{index}: missing <experience> tags"))

    # ── Required fields ──────────────────────────────────────────────────

    style = meta.get("conversation_style")
    if style and style not in VALID_STYLES:
        issues.append(
            (
                "warn",
                f"Experience #{index}: unusual style '{style}' (expected: {VALID_STYLES})",
            )
        )
    elif not style:
        # Try to parse from text
        style_match = re.search(r"Conversation style:\s*(\S+)", text, re.IGNORECASE)
        if not style_match:
            issues.append(
                ("warn", f"Experience #{index}: cannot determine conversation style")
            )

    cadence = meta.get("message_cadence")
    if cadence and cadence not in VALID_CADENCES:
        issues.append(
            (
                "warn",
                f"Experience #{index}: unusual cadence '{cadence}' (expected: {VALID_CADENCES})",
            )
        )
    elif not cadence:
        cadence_match = re.search(r"Message cadence:\s*(\S+)", text, re.IGNORECASE)
        if not cadence_match:
            issues.append(
                ("warn", f"Experience #{index}: cannot determine message cadence")
            )

    initial_state = meta.get("initial_state")
    if not initial_state:
        state_match = re.search(r"Initial state:\s*(.+)", text, re.IGNORECASE)
        if not state_match:
            issues.append(("warn", f"Experience #{index}: cannot find initial state"))

    # ── Instant events ───────────────────────────────────────────────────

    events = meta.get("instant_events", [])
    if not events:
        # Try to find in text
        events_block = re.search(r"Possible instant events:", text, re.IGNORECASE)
        if not events_block:
            issues.append(
                ("warn", f"Experience #{index}: no instant events section found")
            )

    # ── Topic structure for structured/semi-structured ────────────────────

    effective_style = style
    if not effective_style:
        style_match = re.search(r"Conversation style:\s*(\S+)", text, re.IGNORECASE)
        if style_match:
            effective_style = style_match.group(1).strip().lower()

    if effective_style in ("structured", "semi-structured"):
        topic_matches = re.findall(r"[-•]\s*(.+?)\(\d+\s*turns?\)", text)
        if not topic_matches:
            issues.append(
                (
                    "warn",
                    f"Experience #{index}: structured/semi-structured but no topic list with turn counts found",
                )
            )

    # ── Content quality ──────────────────────────────────────────────────

    if len(text) < 200:
        issues.append(("warn", f"Experience #{index}: very short ({len(text)} chars)"))

    # Check for persona references
    persona_names = meta.get("persona_names", [])
    if persona_names:
        for name in persona_names:
            if name and name not in text:
                issues.append(
                    (
                        "warn",
                        f"Experience #{index}: persona name '{name}' not mentioned in experience text",
                    )
                )

    return issues


def validate_experiences(
    records: list, result: ValidationResult, verbose: bool = False
):
    """Run all experience validations."""
    print(f"\n{'─' * 70}")
    print(f"  VALIDATING EXPERIENCES ({len(records)} records)")
    print(f"{'─' * 70}")

    if not records:
        result.fail("No experience records to validate")
        return

    result.ok(f"Loaded {len(records)} experience records")

    all_issues: List[Tuple[str, str]] = []
    for i, record in enumerate(records):
        all_issues.extend(validate_experience_record(record, i))

    fails = [msg for level, msg in all_issues if level == "fail"]
    warns = [msg for level, msg in all_issues if level == "warn"]

    for f in fails:
        result.fail(f)
    for w in warns:
        result.warn(w)

    parse_errors = sum(1 for r in records if r.get("_parse_error"))
    if parse_errors > 0:
        result.fail(f"{parse_errors} experience records failed JSON parsing")
    else:
        result.ok("All experience records parsed successfully")

    valid_records = [r for r in records if not r.get("_parse_error")]

    # Style distribution
    styles = Counter(
        r.get("meta", {}).get("conversation_style", "unknown") for r in valid_records
    )
    result.note(f"Style distribution: {dict(styles)}")

    if len(valid_records) >= 10:
        for style_name in VALID_STYLES:
            count = styles.get(style_name, 0)
            pct = count / len(valid_records) * 100
            if pct < 5:
                result.warn(
                    f"Style '{style_name}' underrepresented: {count} ({pct:.1f}%)"
                )

    # Cadence distribution
    cadences = Counter(
        r.get("meta", {}).get("message_cadence", "unknown") for r in valid_records
    )
    result.note(f"Cadence distribution: {dict(cadences)}")

    # Persona pairing diversity
    persona_pairs = set()
    for r in valid_records:
        names = tuple(sorted(r.get("meta", {}).get("persona_names", [])))
        if names:
            persona_pairs.add(names)
    if persona_pairs:
        result.note(f"Unique persona pairings: {len(persona_pairs)}")
        if len(persona_pairs) < len(valid_records) * 0.5 and len(valid_records) >= 20:
            result.warn(
                f"Low pairing diversity: {len(persona_pairs)} unique pairs for {len(valid_records)} experiences"
            )

    ok_count = len(valid_records) - len(fails)
    result.note(
        f"Experience validation: {ok_count}/{len(valid_records)} records passed without failures"
    )


# ── Conversation validation ──────────────────────────────────────────────────


def validate_conversation_record(
    record: Dict[str, Any], index: int
) -> List[Tuple[str, str]]:
    """Validate a single conversation record."""
    issues: List[Tuple[str, str]] = []

    if record.get("_parse_error"):
        issues.append(
            (
                "fail",
                f"Conversation #{index}: JSON parse error at line {record.get('_line')}",
            )
        )
        return issues

    turns = record.get("turns", [])
    meta = record.get("meta", {})
    names = meta.get("persona_names", ["?", "?"])
    label = f"Conv #{index} ({' ↔ '.join(names)})"

    # ── Turn count ───────────────────────────────────────────────────────

    n_turns = len(turns)
    if n_turns == 0:
        issues.append(("fail", f"{label}: no turns"))
        return issues

    if n_turns < MIN_TURNS_PER_CONVERSATION:
        issues.append(("warn", f"{label}: very few turns ({n_turns})"))

    if n_turns > MAX_TURNS_PER_CONVERSATION:
        issues.append(("warn", f"{label}: exceeded max turns ({n_turns})"))

    # ── Turn structure ───────────────────────────────────────────────────

    empty_turns = 0
    malformed_turns = 0
    total_messages = 0
    message_types = Counter()
    word_counts = []
    turn_message_counts = []
    instant_event_count = 0
    has_exhaustion_tag = False

    all_timestamps = []  # (date_ordinal, time_minutes) for ordering check

    for t_idx, turn_xml in enumerate(turns):
        if not isinstance(turn_xml, str):
            malformed_turns += 1
            continue

        if not turn_xml.strip():
            empty_turns += 1
            continue

        # Check for <turn> tags
        if "<turn>" not in turn_xml:
            malformed_turns += 1
            issues.append(("warn", f"{label} turn {t_idx}: missing <turn> tag"))
            continue

        # Parse messages
        msg_pattern = (
            r'<message\s+t="([^"]*?)"\s+d="([^"]*?)"\s+type="([^"]*?)">(.*?)</message>'
        )
        messages = list(re.finditer(msg_pattern, turn_xml, re.DOTALL))
        msg_count = len(messages)
        turn_message_counts.append(msg_count)

        if msg_count == 0:
            issues.append(("warn", f"{label} turn {t_idx}: no messages in turn"))

        if msg_count > 5:
            issues.append(
                (
                    "warn",
                    f"{label} turn {t_idx}: {msg_count} messages in one turn (max expected: 3)",
                )
            )

        total_messages += msg_count

        for msg in messages:
            time_str, date_str, msg_type, content = (
                msg.group(1),
                msg.group(2),
                msg.group(3),
                msg.group(4).strip(),
            )

            # Type check
            if msg_type not in VALID_MESSAGE_TYPES:
                issues.append(
                    ("warn", f"{label} turn {t_idx}: unknown message type '{msg_type}'")
                )
            message_types[msg_type] += 1

            # Word count for text messages
            if msg_type == "text":
                wc = len(content.split())
                word_counts.append(wc)
                if wc > MESSAGE_MAX_WORDS_HARD:
                    issues.append(
                        (
                            "fail",
                            f"{label} turn {t_idx}: text message has {wc} words (hard limit: {MESSAGE_MAX_WORDS_HARD})",
                        )
                    )
                elif wc > MESSAGE_MAX_WORDS:
                    issues.append(
                        (
                            "warn",
                            f"{label} turn {t_idx}: text message has {wc} words — possible knowledge dump",
                        )
                    )

            # Empty content check
            if not content:
                issues.append(
                    ("warn", f"{label} turn {t_idx}: empty {msg_type} message")
                )

            # Timestamp validation
            time_parts = time_str.split(":")
            if len(time_parts) == 2:
                try:
                    h, m = int(time_parts[0]), int(time_parts[1])
                    if h < 0 or h > 23 or m < 0 or m > 59:
                        issues.append(
                            ("warn", f"{label} turn {t_idx}: invalid time '{time_str}'")
                        )
                except ValueError:
                    issues.append(
                        ("warn", f"{label} turn {t_idx}: unparseable time '{time_str}'")
                    )
            else:
                issues.append(
                    ("warn", f"{label} turn {t_idx}: malformed time '{time_str}'")
                )

            # Date validation
            date_parts = date_str.split(".")
            if len(date_parts) == 2:
                try:
                    d, mo = int(date_parts[0]), int(date_parts[1])
                    if d < 1 or d > 31 or mo < 1 or mo > 12:
                        issues.append(
                            ("warn", f"{label} turn {t_idx}: invalid date '{date_str}'")
                        )
                except ValueError:
                    issues.append(
                        ("warn", f"{label} turn {t_idx}: unparseable date '{date_str}'")
                    )

            # Collect for ordering
            try:
                time_min = int(time_str.split(":")[0]) * 60 + int(
                    time_str.split(":")[1]
                )
                date_ord = int(date_str.split(".")[1]) * 100 + int(
                    date_str.split(".")[0]
                )
                all_timestamps.append((date_ord, time_min))
            except (ValueError, IndexError):
                pass

            # Content quality: check for em dashes in messages (explicitly forbidden)
            if "—" in content and msg_type == "text":
                issues.append(
                    (
                        "warn",
                        f"{label} turn {t_idx}: em dash (—) in text message (forbidden by prompt)",
                    )
                )

            # Check for LLM artifacts in messages
            artifact_patterns = [
                r"^(Sure|Certainly|Of course)[,!]",
                r"^As an AI",
                r"^I('m| am) (sorry|happy to help)",
                r"\*[^*]+\*",  # action narration *does something*
            ]
            for pattern in artifact_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(
                        (
                            "warn",
                            f"{label} turn {t_idx}: possible LLM artifact in message",
                        )
                    )
                    break

        # Check for instant event
        if "<instant_event>" in turn_xml:
            instant_event_count += 1

        # Check for exhaustion
        if "<predefined_topics_exhausted" in turn_xml:
            has_exhaustion_tag = True

        # Check for <state> tag
        state_match = re.search(r"<state>\s*(.*?)\s*</state>", turn_xml, re.DOTALL)
        if not state_match:
            issues.append(("warn", f"{label} turn {t_idx}: missing <state> tag"))

    if empty_turns > 0:
        issues.append(("warn", f"{label}: {empty_turns} empty turns"))

    if malformed_turns > 0:
        issues.append(
            ("fail", f"{label}: {malformed_turns} malformed turns (missing <turn> tag)")
        )

    # ── Timestamp ordering ───────────────────────────────────────────────

    if len(all_timestamps) >= 2:
        out_of_order = 0
        for i in range(1, len(all_timestamps)):
            prev_date, prev_time = all_timestamps[i - 1]
            curr_date, curr_time = all_timestamps[i]
            if curr_date < prev_date:
                out_of_order += 1
            elif curr_date == prev_date and curr_time < prev_time:
                out_of_order += 1
        if out_of_order > 0:
            issues.append(
                (
                    "warn",
                    f"{label}: {out_of_order} timestamp(s) out of chronological order",
                )
            )

    # ── Meta consistency ─────────────────────────────────────────────────

    meta_n_turns = meta.get("n_turns", n_turns)
    if meta_n_turns != n_turns:
        issues.append(
            (
                "warn",
                f"{label}: meta.n_turns ({meta_n_turns}) != actual turn count ({n_turns})",
            )
        )

    meta_exhausted = meta.get("exhausted", False)
    if meta_exhausted and not has_exhaustion_tag:
        issues.append(
            (
                "warn",
                f"{label}: meta says exhausted but no <predefined_topics_exhausted/> tag found",
            )
        )

    style = meta.get("conversation_style")
    if style and style not in VALID_STYLES:
        issues.append(("warn", f"{label}: unusual conversation_style '{style}'"))

    cadence = meta.get("message_cadence")
    if cadence and cadence not in VALID_CADENCES:
        issues.append(("warn", f"{label}: unusual message_cadence '{cadence}'"))

    return issues


def validate_conversations(
    records: list, result: ValidationResult, verbose: bool = False
):
    """Run all conversation validations."""
    print(f"\n{'─' * 70}")
    print(f"  VALIDATING CONVERSATIONS ({len(records)} records)")
    print(f"{'─' * 70}")

    if not records:
        result.fail("No conversation records to validate")
        return

    result.ok(f"Loaded {len(records)} conversation records")

    all_issues: List[Tuple[str, str]] = []
    for i, record in enumerate(records):
        all_issues.extend(validate_conversation_record(record, i))

    fails = [msg for level, msg in all_issues if level == "fail"]
    warns = [msg for level, msg in all_issues if level == "warn"]

    for f in fails:
        result.fail(f)
    for w in warns:
        result.warn(w)

    parse_errors = sum(1 for r in records if r.get("_parse_error"))
    if parse_errors > 0:
        result.fail(f"{parse_errors} conversation records failed JSON parsing")
    else:
        result.ok("All conversation records parsed successfully")

    valid_records = [r for r in records if not r.get("_parse_error")]

    # Aggregate stats
    turn_counts = [
        r.get("meta", {}).get("n_turns", len(r.get("turns", []))) for r in valid_records
    ]
    if turn_counts:
        import statistics

        mean_turns = statistics.mean(turn_counts)
        result.note(f"Mean turns per conversation: {mean_turns:.1f}")

    styles = Counter(
        r.get("meta", {}).get("conversation_style", "unknown") for r in valid_records
    )
    result.note(f"Style distribution: {dict(styles)}")

    exhausted_count = sum(
        1 for r in valid_records if r.get("meta", {}).get("exhausted")
    )
    result.note(
        f"Exhaustion rate: {exhausted_count}/{len(valid_records)} ({exhausted_count / len(valid_records) * 100:.1f}%)"
        if valid_records
        else ""
    )

    ok_count = len(valid_records) - len(fails)
    result.note(
        f"Conversation validation: {ok_count}/{len(valid_records)} records passed without failures"
    )


# ── Seed data validation ─────────────────────────────────────────────────────


def validate_seed_data(result: ValidationResult):
    """Validate that seed data files exist and are well-formed."""
    print(f"\n{'─' * 70}")
    print(f"  VALIDATING SEED DATA")
    print(f"{'─' * 70}")

    # Persona seeds
    persona_seed_dir = Path("data/personas/seed")
    if persona_seed_dir.exists():
        seed_files = list(persona_seed_dir.glob("*.txt"))
        if seed_files:
            result.ok(f"Found {len(seed_files)} persona seed file(s)")
            for sf in seed_files:
                content = sf.read_text(encoding="utf-8")
                if "<character>" not in content:
                    result.warn(f"Seed persona '{sf.name}' missing <character> tag")
                if len(content) < 500:
                    result.warn(
                        f"Seed persona '{sf.name}' is very short ({len(content)} chars)"
                    )
        else:
            result.warn("No persona seed files found in data/personas/seed/")
    else:
        result.warn("Persona seed directory not found: data/personas/seed/")

    # Experience seeds
    exp_seed_dir = Path("data/experiences/seed")
    if exp_seed_dir.exists():
        seed_files = list(exp_seed_dir.glob("*.txt"))
        if seed_files:
            result.ok(f"Found {len(seed_files)} experience seed file(s)")
            for sf in seed_files:
                content = sf.read_text(encoding="utf-8")
                if "<experience>" not in content:
                    result.warn(f"Seed experience '{sf.name}' missing <experience> tag")
        else:
            result.warn("No experience seed files found in data/experiences/seed/")
    else:
        result.warn("Experience seed directory not found: data/experiences/seed/")

    # Name CSVs
    names_dir = Path("data/stats/names")
    if names_dir.exists():
        csv_files = list(names_dir.glob("*.csv"))
        result.ok(f"Found {len(csv_files)} name CSV file(s)")
        if csv_files:
            # Spot-check one file
            import csv

            test_file = csv_files[0]
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames or []
                    if "name" not in fields or "probability" not in fields:
                        result.warn(
                            f"Name CSV '{test_file.name}' missing expected columns (name, probability)"
                        )
                    else:
                        result.ok(f"Name CSV format verified ({test_file.name})")
            except Exception as e:
                result.warn(f"Could not read name CSV '{test_file.name}': {e}")
    else:
        result.warn("Name stats directory not found: data/stats/names/")

    # Regions YAML
    regions_path = Path("data/stats/demographics/regions.yaml")
    if regions_path.exists():
        try:
            import yaml

            with open(regions_path, "r") as f:
                data = yaml.safe_load(f)
            regions = data.get("regions", {})
            subregions = data.get("subregions", {})
            result.ok(
                f"Regions YAML: {len(regions)} regions, {len(subregions)} subregion mappings"
            )

            # Check that every region has subregions
            missing_subregions = [r for r in regions if r not in subregions]
            if missing_subregions:
                result.warn(
                    f"{len(missing_subregions)} region(s) missing subregion data: {', '.join(missing_subregions[:5])}"
                )
        except ImportError:
            result.note("PyYAML not available, skipping regions.yaml validation")
        except Exception as e:
            result.warn(f"Could not parse regions.yaml: {e}")
    else:
        result.fail("Regions YAML not found: data/stats/demographics/regions.yaml")

    # Config files
    for config_file in ["conf/config.yaml"]:
        if Path(config_file).exists():
            result.ok(f"Config file exists: {config_file}")
        else:
            result.warn(f"Config file missing: {config_file}")

    # Prompt templates
    prompts_dir = Path("conf/prompts")
    expected_prompts = [
        "persona_generation.j2",
        "experience_generation.j2",
        "turn_generation.j2",
        "summarization.j2",
    ]
    if prompts_dir.exists():
        for prompt_name in expected_prompts:
            prompt_path = prompts_dir / prompt_name
            if prompt_path.exists():
                content = prompt_path.read_text(encoding="utf-8")
                if len(content) < 50:
                    result.warn(
                        f"Prompt template '{prompt_name}' is suspiciously short ({len(content)} chars)"
                    )
                else:
                    result.ok(f"Prompt template exists: {prompt_name}")
            else:
                result.fail(f"Missing prompt template: {prompt_name}")
    else:
        result.fail("Prompts directory not found: conf/prompts/")


# ── Cross-referencing ────────────────────────────────────────────────────────


def validate_cross_references(
    personas_path: Optional[str],
    experiences_path: Optional[str],
    result: ValidationResult,
):
    """Check that experiences reference valid persona IDs."""
    print(f"\n{'─' * 70}")
    print(f"  CROSS-REFERENCE VALIDATION")
    print(f"{'─' * 70}")

    if not personas_path or not Path(personas_path).exists():
        result.note("Skipping cross-reference check (no personas file)")
        return
    if not experiences_path or not Path(experiences_path).exists():
        result.note("Skipping cross-reference check (no experiences file)")
        return

    personas = load_jsonl(personas_path)
    experiences = load_jsonl(experiences_path)

    persona_ids = set()
    persona_names = set()
    for p in personas:
        if p.get("_parse_error"):
            continue
        pid = p.get("meta", {}).get("id")
        if pid:
            persona_ids.add(pid)
        name = p.get("meta", {}).get("name")
        if name:
            persona_names.add(name)

    if not persona_ids:
        result.warn(
            "No persona IDs found — personas may not be merged yet (run 01e_merge_personas.py)"
        )
        return

    result.note(
        f"Loaded {len(persona_ids)} persona IDs, {len(persona_names)} unique names"
    )

    missing_ids = 0
    matched_ids = 0
    for exp in experiences:
        if exp.get("_parse_error"):
            continue
        exp_persona_ids = exp.get("meta", {}).get("persona_ids", [])
        for pid in exp_persona_ids:
            if pid in persona_ids:
                matched_ids += 1
            else:
                missing_ids += 1

    if missing_ids > 0:
        result.warn(
            f"{missing_ids} persona ID reference(s) in experiences not found in persona file"
        )
    else:
        result.ok(f"All {matched_ids} persona references in experiences are valid")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate SOC pipeline data quality")
    parser.add_argument(
        "--personas",
        type=str,
        default="data/personas/generated/data.jsonl",
        help="Path to personas JSONL file",
    )
    parser.add_argument(
        "--experiences",
        type=str,
        default="data/experiences/generated/data.jsonl",
        help="Path to experiences JSONL file",
    )
    parser.add_argument(
        "--conversations",
        type=str,
        default="data/conversations/generated/data.jsonl",
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--skip-personas",
        action="store_true",
        help="Skip persona validation",
    )
    parser.add_argument(
        "--skip-experiences",
        action="store_true",
        help="Skip experience validation",
    )
    parser.add_argument(
        "--skip-conversations",
        action="store_true",
        help="Skip conversation validation",
    )
    parser.add_argument(
        "--skip-seeds",
        action="store_true",
        help="Skip seed data validation",
    )
    parser.add_argument(
        "--skip-crossref",
        action="store_true",
        help="Skip cross-reference validation",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N records from each file (for faster validation)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all passed checks and info notes",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any failures are found",
    )
    args = parser.parse_args()

    result = ValidationResult()

    print(f"\n{'═' * 70}")
    print(f"  SOC DATA VALIDATION")
    print(f"{'═' * 70}")

    # ── Seed data ────────────────────────────────────────────────────────
    if not args.skip_seeds:
        validate_seed_data(result)

    # ── Personas ─────────────────────────────────────────────────────────
    if not args.skip_personas:
        personas_path = Path(args.personas)
        if personas_path.exists():
            personas = load_jsonl(str(personas_path))
            personas = sample_records(personas, args.sample)
            validate_personas(personas, result, verbose=args.verbose)
        else:
            result.note(f"Personas file not found: {personas_path} (skipping)")

    # ── Experiences ──────────────────────────────────────────────────────
    if not args.skip_experiences:
        experiences_path = Path(args.experiences)
        if experiences_path.exists():
            experiences = load_jsonl(str(experiences_path))
            experiences = sample_records(experiences, args.sample)
            validate_experiences(experiences, result, verbose=args.verbose)
        else:
            result.note(f"Experiences file not found: {experiences_path} (skipping)")

    # ── Conversations ────────────────────────────────────────────────────
    if not args.skip_conversations:
        conversations_path = Path(args.conversations)
        if conversations_path.exists():
            conversations = load_jsonl(str(conversations_path))
            conversations = sample_records(conversations, args.sample)
            validate_conversations(conversations, result, verbose=args.verbose)
        else:
            result.note(
                f"Conversations file not found: {conversations_path} (skipping)"
            )

    # ── Cross-references ─────────────────────────────────────────────────
    if not args.skip_crossref:
        validate_cross_references(
            personas_path=args.personas if not args.skip_personas else None,
            experiences_path=args.experiences if not args.skip_experiences else None,
            result=result,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    result.print_summary(verbose=args.verbose)

    if args.strict and not result.is_healthy:
        sys.exit(1)


if __name__ == "__main__":
    main()
