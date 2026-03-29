#!/usr/bin/env python3
"""
Evaluate generated personas using LLM-as-a-judge.

Reads personas from data.jsonl, scores each on 7 criteria using an LLM judge,
writes per-criterion results into an `eval_results` column and a weighted mean
into an `eval_score` column, then saves the updated file back.

Usage:
    python scripts/01e_evaluate_personas.py
    python scripts/01e_evaluate_personas.py --preset kimi_k2
    python scripts/01e_evaluate_personas.py --model "moonshotai/Kimi-K2.5:fireworks-ai" --provider huggingface
    python scripts/01e_evaluate_personas.py --batch-size 8 --max-retries 3
    python scripts/01e_evaluate_personas.py --input data/personas/generated/data.jsonl
    python scripts/01e_evaluate_personas.py --force  # re-evaluate already-scored personas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jinja2 import Environment, FileSystemLoader

from soc_2602.llm.client import LLMClient
from soc_2602.utils.config import get_section, load_config

# ── Evaluation criteria & weights ────────────────────────────────────────────

CRITERIA = [
    "internal_consistency",
    "archetype_fidelity",
    "show_dont_tell",
    "psychological_realism",
    "roleplay_utility",
    "distinctiveness",
    "format_compliance",
]

# Weights reflect relative importance — higher-weighted criteria matter more
# for the final score.  The weighted mean is normalised to a 1–5 scale.
CRITERIA_WEIGHTS: Dict[str, float] = {
    "internal_consistency": 1.0,
    "archetype_fidelity": 1.5,
    "show_dont_tell": 1.5,
    "psychological_realism": 2.0,
    "roleplay_utility": 1.0,
    "distinctiveness": 1.0,
    "format_compliance": 0.5,
}

# ── Static system prompt (cache-friendly — identical across all eval calls) ──

_EVAL_SYSTEM_PROMPT = """\
## Your Role

You are a persona quality evaluator. You will be given a generated persona and the
parameters it was generated from. Your job is to score it on the criteria below.

You must respond with ONLY the `<eval>` XML block — no preamble, no commentary outside it.

---

## Evaluation Criteria

Score each criterion from **1 to 5** using the rubrics below.

### `internal_consistency` — No contradictions
Does the persona's traits, backstory, habits, and speech patterns form a coherent whole?
- **5** No contradictions; every detail reinforces the same person
- **4** Minor tension that could be read as realistic human complexity
- **3** One notable contradiction that slightly undermines believability
- **2** Multiple contradictions that make the character feel incoherent
- **1** The persona fundamentally contradicts itself

### `archetype_fidelity` — Faithful to the assigned psychological direction
Does the persona embody the assigned archetype authentically, without naming it?
Note: the archetype label must never appear in the persona text.
- **5** The archetype is unmistakably present and textures every dimension of the character
- **4** Clearly recognisable; minor moments where the archetype is underplayed
- **3** Present but inconsistent — some sections fit, others feel off-archetype
- **2** The archetype barely registers; the character feels generic or misaligned
- **1** The persona contradicts the archetype or names it directly

### `show_dont_tell` — Reveals traits through behavior, not labels
Does the persona use specific scenes, habits, and quoted speech rather than adjective-labels?
- **5** All major traits are shown through concrete action, dialogue, or habit — zero labels
- **4** Mostly shown; one or two adjective-label slips
- **3** Mixed: some good scenes, but several traits stated directly
- **2** Mostly labels with only superficial behavioral detail
- **1** Entirely label-based ("she is kind", "he is cynical")

### `psychological_realism` — Feels like a real person
Does this feel like a plausible human being rather than a flat archetype or wish-fulfilment figure?
- **5** Feels fully human: mundane flaws, contradictory desires, specific history
- **4** Mostly believable; one dimension feels slightly constructed
- **3** Realistic in parts, but at least one trait or backstory element feels contrived
- **2** Noticeably artificial; feels like a character "type" rather than a person
- **1** Implausible or stereotyped throughout

### `roleplay_utility` — Usable as a roleplay partner
Does the persona give enough handles for a model to convincingly play this character?
Consider: named relationships, specific hobbies, quoted speech patterns, distinctive opinions or reactions.
- **5** Rich with roleplay handles — voice, relationships, opinions, edge cases all specified
- **4** Good coverage; one dimension (e.g. speech style or relationships) is thin
- **3** Adequate but generic; a model would produce a flat performance from it
- **2** Sparse: mostly abstract traits with few concrete anchors
- **1** Not usable — a model couldn't derive a distinct voice from it

### `distinctiveness` — Different from typical LLM-generated personas
Does the new persona feel like a genuinely distinct individual rather than a generic template?
- **5** Differs in multiple major dimensions (age/region/voice/psychology/life situation)
- **4** Clearly different from typical outputs, though shares one predictable dimension
- **3** Superficially different (name/age changed) but structurally similar to generic outputs
- **2** Feels like a direct variant or echo of a common template
- **1** Near-identical to what any LLM would produce by default

### `format_compliance` — Follows the required format
Does the output correctly use the XML + markdown structure and stay within the character budget?

**Required format:**
- Must be wrapped in `<character>...</character>` XML tags
- Must contain these markdown sections: **Basic Information** (Name, Age, Location, Pronouns), **Physical & Lifestyle** (2-3 paragraphs), **Personality Overview** (2-3 paragraphs), **Core Traits** (1-2 paragraphs), **Emotional Profile** (2-3 paragraphs), **Relationships** (2-4 paragraphs), **Values, Motivations & Fears** (2-3 paragraphs), **Behavioral Patterns** (1-2 paragraphs), **Communication Style** (2 paragraphs), **Example Messages** (2-3 exchanges with `<START>` markers), **Summary** (1 paragraph)

**Scoring:**
- **5** Perfect compliance: correct tags, all required sections, within length bounds
- **4** Minor deviation (e.g. a section slightly short, one tag misformatted)
- **3** Some structural issues but the core format is recognisable
- **2** Significant format violations that would break downstream parsing
- **1** Does not follow the required format at all

---

## Output Format

Respond with ONLY this XML block — nothing before or after it:

```xml
<eval>
  <reasoning>
    Walk through each criterion in order. For each one, cite a specific passage
    from the persona text (quote or paraphrase) that justifies the score.
    Be concise but precise — one or two sentences per criterion.
  </reasoning>
  <eval_result criteria="internal_consistency">1–5</eval_result>
  <eval_result criteria="archetype_fidelity">1–5</eval_result>
  <eval_result criteria="show_dont_tell">1–5</eval_result>
  <eval_result criteria="psychological_realism">1–5</eval_result>
  <eval_result criteria="roleplay_utility">1–5</eval_result>
  <eval_result criteria="distinctiveness">1–5</eval_result>
  <eval_result criteria="format_compliance">1–5</eval_result>
</eval>
```

Replace `1–5` with an actual integer score (1, 2, 3, 4, or 5) for each criterion."""


# ── XML parsing ──────────────────────────────────────────────────────────────

_EVAL_RESULT_RE = re.compile(
    r'<eval_result\s+criteria="([^"]+)">\s*(\d)\s*</eval_result>'
)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


def parse_eval_xml(text: str) -> Optional[Dict[str, Any]]:
    """Parse the <eval>...</eval> block returned by the judge.

    Returns a dict like::

        {
            "reasoning": "...",
            "scores": {
                "internal_consistency": 4,
                "archetype_fidelity": 5,
                ...
            }
        }

    Returns ``None`` if parsing fails.
    """
    # Find the <eval> block
    eval_start = text.find("<eval>")
    eval_end = text.find("</eval>")
    if eval_start == -1 or eval_end == -1:
        return None

    eval_block = text[eval_start : eval_end + len("</eval>")]

    # Extract reasoning
    reasoning_match = _REASONING_RE.search(eval_block)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract scores
    scores: Dict[str, int] = {}
    for match in _EVAL_RESULT_RE.finditer(eval_block):
        criterion = match.group(1)
        score = int(match.group(2))
        if criterion in CRITERIA and 1 <= score <= 5:
            scores[criterion] = score

    if len(scores) != len(CRITERIA):
        return None

    return {"reasoning": reasoning, "scores": scores}


def compute_weighted_mean(scores: Dict[str, int]) -> float:
    """Compute the weighted mean score across all criteria.

    Returns a float rounded to 2 decimal places on the 1–5 scale.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for criterion, score in scores.items():
        w = CRITERIA_WEIGHTS.get(criterion, 1.0)
        weighted_sum += score * w
        total_weight += w
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 2)


# ── Async evaluation ─────────────────────────────────────────────────────────


async def evaluate_one(
    llm_client: LLMClient,
    template: Any,
    persona_record: Dict[str, Any],
    index: int,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single persona. Returns the parsed eval dict or None."""
    meta = persona_record.get("meta", {})
    persona_text = persona_record.get("persona_text", "")

    persona_info = {
        "name": meta.get("name", "Unknown"),
        "age": meta.get("age", "Unknown"),
        "region": meta.get("region", "Unknown"),
        "subregion": meta.get("subregion", "Unknown"),
        "archetype": meta.get("archetype", "Unknown"),
        "regulation_style": meta.get("regulation_style", "Unknown"),
    }
    if meta.get("lingua_franca"):
        persona_info["lingua_franca"] = meta["lingua_franca"]

    # Render user message from template
    user_text = template.render(
        persona_info=persona_info,
        persona=persona_text,
    )

    messages = LLMClient.build_messages(
        system_text=_EVAL_SYSTEM_PROMPT,
        user_text=user_text,
    )

    persona_id = meta.get("id", f"idx-{index}")
    persona_name = meta.get("name", "???")

    for attempt in range(1, max_retries + 1):
        try:
            start = time.monotonic()
            response = await llm_client.generate_async(
                messages,
                temperature=0.3,  # low temperature for consistent judging
                max_tokens=2048,
            )
            elapsed = time.monotonic() - start

            raw = response.choices[0].message.content.strip()
            parsed = parse_eval_xml(raw)

            if parsed is None:
                print(
                    f"  [#{index} {persona_name}] Attempt {attempt}/{max_retries}: "
                    f"failed to parse eval XML, retrying..."
                )
                continue

            score = compute_weighted_mean(parsed["scores"])
            print(
                f"  [#{index} {persona_name}] score={score:.2f}  "
                f"({', '.join(f'{c[:3]}={s}' for c, s in parsed['scores'].items())})  "
                f"[{elapsed:.1f}s]"
            )

            return {
                "reasoning": parsed["reasoning"],
                "scores": parsed["scores"],
                "weighted_mean": score,
                "judge_model": llm_client.model_id,
            }

        except Exception as e:
            print(
                f"  [#{index} {persona_name}] Attempt {attempt}/{max_retries}: "
                f"error — {type(e).__name__}: {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

    print(f"  [#{index} {persona_name}] FAILED after {max_retries} attempts")
    return None


async def evaluate_batch(
    llm_client: LLMClient,
    template: Any,
    personas: List[Dict[str, Any]],
    indices: List[int],
    max_retries: int = 2,
) -> List[Optional[Dict[str, Any]]]:
    """Evaluate a batch of personas concurrently."""
    tasks = [
        evaluate_one(llm_client, template, persona, idx, max_retries)
        for persona, idx in zip(personas, indices)
    ]
    return await asyncio.gather(*tasks)


# ── Main ─────────────────────────────────────────────────────────────────────


def load_personas(path: str) -> List[Dict[str, Any]]:
    """Load all persona records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at line {line_num}: {e}")
    return records


def save_personas(records: List[Dict[str, Any]], path: str) -> None:
    """Write persona records back to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate persona quality using LLM-as-a-judge"
    )

    # ── Input/output ─────────────────────────────────────────────────────
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to personas JSONL file (default: from config paths.merged_personas)",
    )

    # ── Model selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use for evaluation (overrides config and preset)",
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

    # ── Evaluation parameters ────────────────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent evaluation requests (default: 4)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retry attempts per persona on parse failure (default: 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate personas that already have eval_score",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N unevaluated personas (useful for testing)",
    )

    # ── Config ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: conf/config.yaml)",
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

    # Resolve paths
    input_path = args.input or cfg.get("paths", {}).get(
        "merged_personas", "data/personas/generated/data.jsonl"
    )
    prompt_dir = args.prompt_dir or persona_cfg.get("prompt_dir", "conf/prompts")

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run the merge script first: python scripts/01e_merge_personas.py")
        sys.exit(1)

    # ── Load Jinja template ──────────────────────────────────────────────
    jinja_env = Environment(
        loader=FileSystemLoader(prompt_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = jinja_env.get_template("persona_evaluation.j2")

    # ── Build LLM client ─────────────────────────────────────────────────
    llm_client = LLMClient.from_config(
        cfg,
        preset_name=args.preset,
        model_id=args.model,
        provider=args.provider,
    )
    print(f"Judge model: {llm_client.model_id} (provider: {llm_client.provider})")

    # ── Load personas ────────────────────────────────────────────────────
    records = load_personas(input_path)
    print(f"Loaded {len(records)} personas from {input_path}")

    # ── Filter to unevaluated (unless --force) ───────────────────────────
    if args.force:
        to_evaluate = list(enumerate(records))
    else:
        to_evaluate = [(i, r) for i, r in enumerate(records) if "eval_score" not in r]

    if not to_evaluate:
        print("All personas already evaluated. Use --force to re-evaluate.")
        sys.exit(0)

    if args.limit is not None:
        to_evaluate = to_evaluate[: args.limit]

    already_done = len(records) - len(to_evaluate)
    print(
        f"Evaluating {len(to_evaluate)} personas "
        f"({already_done} already scored, skipped)"
    )
    print(f"Batch size: {args.batch_size}, max retries: {args.max_retries}")
    print()

    # ── Run evaluation in batches ────────────────────────────────────────
    total_start = time.monotonic()
    evaluated = 0
    failed = 0

    async def run_all():
        nonlocal evaluated, failed

        for batch_start in range(0, len(to_evaluate), args.batch_size):
            batch = to_evaluate[batch_start : batch_start + args.batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_records = [rec for _, rec in batch]

            batch_num = batch_start // args.batch_size + 1
            total_batches = (len(to_evaluate) + args.batch_size - 1) // args.batch_size
            print(
                f"Batch {batch_num}/{total_batches} "
                f"(personas {batch_start + 1}–{batch_start + len(batch)} "
                f"of {len(to_evaluate)})"
            )

            results = await evaluate_batch(
                llm_client,
                template,
                batch_records,
                batch_indices,
                max_retries=args.max_retries,
            )

            for (global_idx, _), result in zip(batch, results):
                if result is not None:
                    records[global_idx]["eval_score"] = result["weighted_mean"]
                    records[global_idx]["eval_results"] = {
                        "scores": result["scores"],
                        "reasoning": result["reasoning"],
                        "judge_model": result["judge_model"],
                    }
                    evaluated += 1
                else:
                    failed += 1

            save_personas(records, input_path)
            print()

    asyncio.run(run_all())

    total_elapsed = time.monotonic() - total_start

    # ── Save updated file ────────────────────────────────────────────────
    save_personas(records, input_path)

    # ── Summary ──────────────────────────────────────────────────────────
    scored = [r["eval_score"] for r in records if "eval_score" in r]
    if scored:
        mean_score = sum(scored) / len(scored)
        min_score = min(scored)
        max_score = max(scored)
    else:
        mean_score = min_score = max_score = 0.0

    print("=" * 60)
    print("Evaluation complete")
    print(f"  Evaluated:  {evaluated}")
    print(f"  Failed:     {failed}")
    print(f"  Total time: {total_elapsed:.1f}s")
    if evaluated > 0:
        print(f"  Avg time:   {total_elapsed / evaluated:.1f}s per persona")
    print()
    print(f"  Total scored: {len(scored)} / {len(records)}")
    print(f"  Mean score:   {mean_score:.2f}")
    print(f"  Min score:    {min_score:.2f}")
    print(f"  Max score:    {max_score:.2f}")
    print()
    print(f"Results saved to {input_path}")

    # Per-criterion averages
    if scored:
        print()
        print("Per-criterion averages:")
        for criterion in CRITERIA:
            vals = [
                r["eval_results"]["scores"][criterion]
                for r in records
                if "eval_results" in r
                and criterion in r.get("eval_results", {}).get("scores", {})
            ]
            if vals:
                avg = sum(vals) / len(vals)
                weight = CRITERIA_WEIGHTS.get(criterion, 1.0)
                print(f"  {criterion:<28s} {avg:.2f}  (weight: {weight})")


if __name__ == "__main__":
    main()
