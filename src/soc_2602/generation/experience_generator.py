import asyncio
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from soc_2602.llm.client import LLMClient

# ── Static system prompt ─────────────────────────────────────────────────────
# This block is identical across ALL experience generation calls, maximising
# the byte-identical prefix for providers with automatic prompt caching.
# The variable content (personas, few-shot examples, diversity hints) goes
# into the user message rendered from the Jinja template.

_EXPERIENCE_SYSTEM_PROMPT = """\
You are an experience generation system for realistic conversational AI.
Your output is shared with both agents, who CANNOT see each other's persona.

## Task

Generate ONE realistic experience bringing two personas into conversation.
Vary the style widely: experiences can be structured, semi-structured, or
completely freeform (like two friends who chat indefinitely with no agenda).

## Output Format

<experience>
[2–4 paragraphs: relationship history, current situation, emotional states,
and environmental context including the text-based communication channel used]

Conversation style: [structured | semi-structured | freeform]
Message cadence: [realtime (seconds–1 min) | delayed (minutes–hours) | async (hours–days)]

[IF structured or semi-structured:]
The conversation will likely cover the following topics:
- [Topic 1] (X turns) ← PRIMARY
- [Topic 2] (Y turns)
- [Topic 3] (Z turns)

Initial state: [Topic 1]; [X turns remaining]; [Topic 2]

[IF freeform:]
Conversational energy: [2–3 sentences describing the tone, what these two
tend to talk about, recurring themes, shared references, and how they
typically drift from subject to subject. No turn counts.]

Initial state: freeform; —; —

Possible instant events:
- [Plausible event affecting Persona 1 given their situation]
- [Plausible event affecting Persona 2 given their situation]
- [Optional shared or environmental event]
[2–5 events. For async conversations, events can include things like a
memory resurfacing, a news item, a song coming on, not just physical accidents.]
</experience>

## Requirements

1. Vary relationship type: close friends, distant friends, colleagues,
   family, romantic partners, acquaintances, online communities
2. Vary cadence: don't always make conversations real-time
3. Freeform conversations have NO topic list — they just drift naturally
4. Instant events must fit the cadence (async chats → no physical accidents;
   use mood shifts, something they saw, a memory, a notification instead)
5. Psychological depth: current emotional states and recent context for each persona
6. Strict 1-on-1 Channels Only: The conversation MUST take place in a private,
   text-based, direct message setting (e.g., WhatsApp DM, iMessage, Telegram DM,
   Tinder, Discord DM). Do NOT place them in group chats, public Slack channels,
   or forum threads.
7. Shared Language (No "Babel Fish"): Explicitly establish a shared lingua franca.
   If personas have different native backgrounds, note that they communicate in
   the shared language, only occasionally sprinkling in a native slang word or
   greeting for flavor. Do NOT set up scenarios where they speak entirely in
   different languages.

## Output

Output ONLY the content between <experience> and </experience> tags (inclusive). No additional commentary."""


class ExperienceGenerator:
    """Generates realistic conversational experiences pairing two personas.

    Supports region profiles (via the persona pool it draws from),
    language configuration for the shared lingua franca, and style
    diversity nudging to prevent over-representation of any single
    conversation style.

    Parameters
    ----------
    llm_client : LLMClient
        Client for LLM API calls.
    persona_file : str
        Path to the merged personas JSONL file.
    output_file : str | None
        Path to the output JSONL file.  Auto-generated if ``None``.
    seed_dir : str
        Directory containing seed experience text files.
    prompt_dir : str
        Directory containing Jinja2 prompt templates.
    language_settings : dict | None
        Language configuration from ``config.yaml``.  Keys:
        ``lingua_franca``, ``native_flavor``, ``max_native_words_per_message``.
    same_region_probability : float
        Probability that both personas are drawn from the same region.
    """

    # Pool management
    POOL_MAX_SIZE = 20

    # Valid field values for validation and diversity tracking
    VALID_STYLES = {"structured", "semi-structured", "freeform"}
    VALID_CADENCES = {"realtime", "delayed", "async"}

    def __init__(
        self,
        llm_client: LLMClient,
        persona_file: str,
        output_file: Optional[str] = None,
        seed_dir: str = "data/experiences/seed",
        prompt_dir: str = "conf/prompts",
        language_settings: Optional[Dict[str, Any]] = None,
        same_region_probability: float = 0.5,
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id

        self.persona_file = Path(persona_file)
        self.seed_dir = Path(seed_dir)

        if output_file is None:
            output_file = f"data/experiences/generated/experiences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("experience_generation.j2")

        self.personas = self._load_personas()
        self.seed_experiences = self._load_seed_experiences()
        self.generated_experiences: List[str] = []
        self.n_generated = 0
        self.mean_time_per_experience = 0.0
        self.total_wall_time = 0.0

        # Tracks style distribution across the run for diversity nudging
        self.style_counts: Counter = Counter()

        # Language settings
        self.language_settings = language_settings or {
            "lingua_franca": "English",
            "native_flavor": True,
            "max_native_words_per_message": 2,
        }

        # Persona pairing
        self.same_region_probability = same_region_probability

        print(f"Loaded {len(self.personas)} personas")
        print(f"Loaded {len(self.seed_experiences)} seed experiences")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_personas(self) -> List[Dict[str, Any]]:
        """Load personas from JSONL file."""
        personas = []
        if not self.persona_file.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_file}")

        with open(self.persona_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        personas.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return personas

    def _load_seed_experiences(self) -> List[str]:
        """Load seed experience files, sorted for determinism."""
        experiences = []
        if self.seed_dir.exists():
            for file_path in sorted(self.seed_dir.glob("*.txt")):
                with open(file_path, "r", encoding="utf-8") as f:
                    experiences.append(f.read().strip())
        return experiences

    # -------------------------------------------------------------------------
    # Selection helpers
    # -------------------------------------------------------------------------

    def _select_shots(self, iteration: int, num_shots: int = 1) -> List[str]:
        """
        Select few-shot examples.
        Seeds only for the first 10 iterations (warmup),
        then a 50/50 mix of seeds and recently generated experiences.
        """
        if iteration < 10 or not self.generated_experiences:
            return random.sample(
                self.seed_experiences, min(num_shots, len(self.seed_experiences))
            )

        pool = self.seed_experiences.copy()
        if random.random() > 0.5:
            pool.extend(self.generated_experiences)

        return random.sample(pool, min(num_shots, len(pool)))

    def _select_personas(self, n: int = 2) -> List[Dict[str, Any]]:
        """
        Select n personas with a configurable chance of same-region pairing
        to encourage geographical coherence where it matters.
        """
        if len(self.personas) < n:
            return self.personas

        first_persona = random.choice(self.personas)
        if random.random() < self.same_region_probability:
            same_region = [
                p
                for p in self.personas
                if p.get("meta", {}).get("region")
                == first_persona.get("meta", {}).get("region")
                and p != first_persona
            ]
            second_persona = (
                random.choice(same_region)
                if same_region
                else random.choice(self.personas)
            )
        else:
            second_persona = random.choice(self.personas)

        return [first_persona, second_persona]

    def _underrepresented_style(self) -> Optional[str]:
        """
        Return the least-used conversation style if one style is more than
        2x more common than the least-used one, else None.
        Used to nudge the template toward style diversity.
        Activates only after 6 generations to let the distribution stabilise.
        """
        if self.n_generated < 6:
            return None

        least = min(self.VALID_STYLES, key=lambda s: self.style_counts.get(s, 0))
        most_count = max(self.style_counts.values(), default=0)
        least_count = self.style_counts.get(least, 0)

        return least if most_count > 2 * (least_count + 1) else None

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_experience_metadata(text: str) -> Dict[str, Any]:
        """
        Extract structured fields from the raw experience block via regex.
        All fields fall back gracefully to None / [] if absent.
        """
        meta: Dict[str, Any] = {
            "conversation_style": None,
            "message_cadence": None,
            "initial_state": None,
            "instant_events": [],
        }

        # Conversation style
        m = re.search(r"Conversation style:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower()
            meta["conversation_style"] = (
                raw if raw in ExperienceGenerator.VALID_STYLES else raw
            )

        # Message cadence — normalise to one of three canonical values
        m = re.search(r"Message cadence:\s*([^\n(]+)", text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower()
            if "realtime" in raw or "real-time" in raw or "real time" in raw:
                meta["message_cadence"] = "realtime"
            elif "async" in raw:
                meta["message_cadence"] = "async"
            elif "delay" in raw:
                meta["message_cadence"] = "delayed"
            else:
                meta["message_cadence"] = raw

        # Initial state
        m = re.search(r"Initial state:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            meta["initial_state"] = m.group(1).strip()

        # Instant events — collect bullet items under the events section
        events_block = re.search(
            r"Possible instant events:(.*?)(?:\n\n|\Z|</experience)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if events_block:
            events = re.findall(r"[-•*]\s*(.+)", events_block.group(1))
            meta["instant_events"] = [e.strip() for e in events if e.strip()]

        return meta

    # -------------------------------------------------------------------------
    # Pool management
    # -------------------------------------------------------------------------

    def _update_pool(self, experience_text: str) -> None:
        """
        Rolling window pool: append new experience, then pop the oldest entries
        once POOL_MAX_SIZE is exceeded.
        """
        self.generated_experiences.append(experience_text)
        if len(self.generated_experiences) > self.POOL_MAX_SIZE:
            self.generated_experiences.pop(0)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    async def generate_experience_async(self, iteration: int = 0) -> Dict[str, Any]:
        """Generate one experience asynchronously and return it with metadata."""
        selected_personas = self._select_personas(n=2)
        shots = self._select_shots(iteration)
        style_hint = self._underrepresented_style()

        # Build language hint from settings
        lingua_franca = self.language_settings.get("lingua_franca", "English")
        language_hint = lingua_franca if lingua_franca else None

        # Render the user message (variable part only)
        user_text = self.template.render(
            shots=shots,
            personas=selected_personas,
            style_hint=style_hint,
            language_hint=language_hint,
        )

        # Build messages with static system prompt for cache-friendliness
        messages = LLMClient.build_messages(
            system_text=_EXPERIENCE_SYSTEM_PROMPT,
            user_text=user_text,
        )

        names = [
            p["meta"].get("name", "Unknown") for p in selected_personas if "meta" in p
        ]
        hint_label = f" [nudge → {style_hint}]" if style_hint else ""
        print(
            f"\n[Iteration {iteration}] Generating experience for {names}{hint_label}..."
        )

        start_time = datetime.now()
        response = await self.llm_client.generate_async(messages)
        generated_text = response.choices[0].message.content.strip()

        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)

        time_taken = (datetime.now() - start_time).total_seconds()

        # Extract <experience> block
        if "<experience>" in generated_text:
            start_idx = generated_text.find("<experience>") + len("<experience>")
            end_idx = generated_text.find("</experience>")
            if end_idx > start_idx:
                generated_text = generated_text[start_idx:end_idx].strip()
                generated_text = f"<experience>\n{generated_text}\n</experience>"
        else:
            if not generated_text:
                return {}

        parsed = self._parse_experience_metadata(generated_text)

        persona_ids = [
            p["meta"].get("id")
            for p in selected_personas
            if "meta" in p and "id" in p["meta"]
        ]

        return {
            "experience_text": generated_text,
            "meta": {
                "iteration": iteration,
                "model": self.model_name,
                "persona_ids": persona_ids,
                "persona_names": names,
                "conversation_style": parsed["conversation_style"],
                "message_cadence": parsed["message_cadence"],
                "initial_state": parsed["initial_state"],
                "instant_events": parsed["instant_events"],
                "time_taken": time_taken,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def _append_to_jsonl(self, data: Dict[str, Any]) -> None:
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # Batch runner
    # -------------------------------------------------------------------------

    async def _run_batch_async(
        self, num_experiences: int, start_iteration: int, batch_size: int
    ) -> None:
        print(f"\n{'=' * 60}")
        print(f"Generating {num_experiences} experiences → {self.output_file}")
        print(f"Batch size: {batch_size}")
        print(f"{'=' * 60}")

        iterations = range(start_iteration, start_iteration + num_experiences)

        for i in range(0, len(iterations), batch_size):
            batch_iterations = list(iterations[i : i + batch_size])
            tasks = [self.generate_experience_async(it) for it in batch_iterations]

            batch_start = datetime.now()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_wall_time = (datetime.now() - batch_start).total_seconds()
            self.total_wall_time += batch_wall_time

            for j, result in enumerate(results):
                iteration = batch_iterations[j]

                if isinstance(result, Exception):
                    print(f"✗ Error at iteration {iteration}: {result}")
                    continue

                if not result or not result.get("experience_text"):
                    print(
                        f"✗ No experience generated at iteration {iteration}, skipping."
                    )
                    continue

                time_taken = result["meta"].pop(
                    "time_taken", 0.0
                )  # per-call latency kept for info

                self._append_to_jsonl(result)

                style = result["meta"].get("conversation_style", "?")
                cadence = result["meta"].get("message_cadence", "?")
                n_events = len(result["meta"].get("instant_events", []))
                print(
                    f"✓ Saved record {iteration} | "
                    f"style={style}  cadence={cadence}  events={n_events} | "
                    f"call latency {time_taken:.2f}s"
                )

                self._update_pool(result["experience_text"])
                if style in self.VALID_STYLES:
                    self.style_counts[style] += 1
                self.n_generated += 1

            # Update mean once per batch — wall time already includes all parallel calls
            if self.n_generated > 0:
                self.mean_time_per_experience = self.total_wall_time / self.n_generated
                print(
                    f"  → batch wall time {batch_wall_time:.2f}s | avg throughput {self.mean_time_per_experience:.2f}s/experience"
                )

        print(f"\n{'=' * 60}")
        print(f"Generation complete! Total: {self.n_generated}")
        print(f"Style distribution: {dict(self.style_counts)}")
        print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_batch(
        self, num_experiences: int, start_iteration: int = 0, batch_size: int = 4
    ) -> None:
        """Generate experiences in async batches."""
        asyncio.run(self._run_batch_async(num_experiences, start_iteration, batch_size))

    def generate_experience(self, iteration: int = 0) -> Dict[str, Any]:
        """Synchronous wrapper for single experience generation."""
        return asyncio.run(self.generate_experience_async(iteration))
