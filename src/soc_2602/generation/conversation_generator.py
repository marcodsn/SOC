"""
Conversation generator for the SOC pipeline.

Generates multi-turn conversations from paired personas and experiences,
with rolling summarization, instant event injection, and topic state
tracking.

Prompt Caching Strategy
-----------------------
The turn generation system prompt is a large, static instruction block
that is identical across ALL turn generation calls within a conversation
(and across conversations).  By placing it in the system message and
keeping only the variable per-turn context (persona, experience, memory,
state) in the user message, we maximise the byte-identical prefix for
providers with automatic prompt caching (vLLM prefix caching, Anthropic,
OpenAI, DeepSeek, etc.).
"""

import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from soc_2602.llm.client import LLMClient

# ── Static system prompt ─────────────────────────────────────────────────────
# This block is identical across ALL turn generation calls, maximising
# the byte-identical prefix for providers with automatic prompt caching.
# The variable content (persona, experience, conversation memory, state)
# goes into the user message rendered from the Jinja template.

_TURN_SYSTEM_PROMPT = """\
You are roleplaying a realistic character in an ongoing conversation.
Stay fully in character. You can only perceive your own persona and the shared context — not the other person's private details.

Your task: generate the next turn in the conversation. Follow these rules exactly.

## Texting Realism (CRITICAL — READ CAREFULLY, and keep in mind these depend heavily on your persona)

These are casual phone texts, NOT emails, NOT essays, and NOT helpful assistant responses. Also avoid meta-commentary.

- **NO Formatting:** Absolutely NO bullet points, NO numbered lists, and NO numbered emojis (1️⃣, 2️⃣, etc.). Do not structure information perfectly.
- **Tone & Mechanics:** Use abbreviations (rn, tbh, lol, wtf, omg) if appropriate for your persona and context. Drop trailing punctuation if appropriate for your persona. Do NOT over-explain why an idea is good (no corporate speak or "therapy speak").
- **Shared Language Only:** The conversation MUST be in a single, shared lingua franca. Do NOT have one person speak Arabic and the other reply in Chinese. You may (but preferably do not) use 1 or 2 native slang words (e.g., "Inshallah", "Gracias"), only when the other person would plausibly understand them from context, and very sparingly.
- **Do NOT narrate actions:** Never put actions or environment descriptions in the texts.
- **Punctuation & Syntax:** Strictly avoid using em dashes (—) in <message> tags. If needed, use an ellipsis (...), but don't overuse them.

## Message Cadence

Cadence types:
- realtime: messages usually seconds apart, quick back-and-forth
- delayed: minutes between messages, occasional gaps
- async: messages hours or days apart; often just one message per turn

Timestamp every message accurately relative to the ones already sent. Delays are allowed. Some messages may require more time to compose, and that's okay. If your persona is busy or distracted, it's also okay for there to be a long gap before they reply. Just make sure the timestamps reflect that.

**Remember:** The timestamps need to be realistic; this rule has priority over the expected cadence.

## Multi-Message Turns

- Send 1–3 messages per turn when it feels authentic
- Never fragment one thought into multiple messages, split only when a real person would (e.g. a sticker after a text, a voice note following up, a photo mid-story)
- For async cadence: usually 1 message; short bursts of 2–3 are fine when excited

## Content & Topic Flow

**For structured / semi-structured conversations:**
- If turns remaining is 2, start weaving the next topic in organically — no hard cuts
- If turns remaining reaches 0, transition fully; the next topic becomes current
- Decrement turns remaining by 1 in your <state> block
- If all topics are exhausted and the conversation feels naturally complete, add <predefined_topics_exhausted/> with an optional closing message

**For freeform conversations:**
- No topic list to follow — just drift naturally
- Write as the character actually would: random thoughts, questions mid-sentence, long silences, a sudden photo, drifting to something unrelated
- Keep <state> as "freeform; —; —" throughout
- Add <predefined_topics_exhausted/> only if the exchange has clearly and naturally wound down

## Coincidence Budget

Do not let both personas share the same hidden desire, secret, or parallel situation in the same conversation unless your persona explicitly states that it is a shared experience.

## Endings

Conversations do not need to resolve. If turns are exhausted and no natural closing has emerged, it is valid for the last message to be mid-thought, unanswered, or simply a one-word reaction. Avoid engineering emotional catharsis at the end. Do not exhaust topics before the state tracker expects it.

## Output Format

Output ONLY the <turn> block. Nothing outside it.

<turn>
<state>
current topic; turns remaining; next topic
</state>
<message t="HH:MM" d="DD.MM" type="text">Actual text message typed on a phone</message>
<message t="HH:MM" d="DD.MM" type="audio">voice note transcription (spoken naturally)</message>
<message t="HH:MM" d="DD.MM" type="image">brief, realistic description of image sent (e.g., "blurry photo of a cat")</message>
<message t="HH:MM" d="DD.MM" type="video">brief, realistic description of video sent</message>
<message t="HH:MM" d="DD.MM" type="sticker">brief description of a standard chat sticker</message>
</turn>

When an instant event is present, include an <instant_event> block inside <turn> before the messages:
<instant_event>
Write your raw internal reaction here. This is never spoken aloud.
</instant_event>

Add <predefined_topics_exhausted/> inside the <turn> block only when the conversation has wound down."""


class ConversationGenerator:
    """Generates multi-turn conversations from paired personas and experiences.

    Conversations within a batch run concurrently; turns within a single
    conversation are strictly sequential to maintain coherence.

    Parameters
    ----------
    llm_client : LLMClient
        Client for turn generation API calls.
    persona_file : str
        Path to the merged personas JSONL file.
    experience_file : str
        Path to the merged experiences JSONL file.
    output_file : str | None
        Path to the output JSONL file.  Auto-generated if ``None``.
    prompt_dir : str
        Directory containing Jinja2 prompt templates.
    summarizer_client : LLMClient | None
        Optional cheaper/faster client for rolling summarization.
        Falls back to *llm_client* if ``None``.
    language_settings : dict | None
        Language configuration from ``config.yaml``.
    max_turns : int
        Hard safety ceiling on turns per conversation.
    instant_event_prob : float
        Probability of injecting an instant event on any given turn.
    max_active_turns : int
        Number of recent turns to keep in the prompt window.
    retire_batch : int
        Number of turns to retire into the summary at once.
    max_consecutive_failures : int
        Abort the conversation after this many consecutive turn failures.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        persona_file: str,
        experience_file: str,
        output_file: Optional[str] = None,
        prompt_dir: str = "conf/prompts",
        summarizer_client: Optional[LLMClient] = None,
        language_settings: Optional[Dict[str, Any]] = None,
        max_turns: int = 50,
        instant_event_prob: float = 0.05,
        max_active_turns: int = 20,
        retire_batch: int = 8,
        max_consecutive_failures: int = 3,
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id

        # Allow a separate (cheaper/faster) client for summarization
        self.summarizer_client = summarizer_client or llm_client
        self.summarizer_model_name = self.summarizer_client.model_id

        self.persona_file = Path(persona_file)
        self.experience_file = Path(experience_file)

        if output_file is None:
            output_file = f"data/conversations/generated/conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.turn_template = self.jinja_env.get_template("turn_generation.j2")
        self.summarizer_template = self.jinja_env.get_template("summarization.j2")

        # Language settings
        self.language_settings = language_settings or {
            "lingua_franca": "English",
            "native_flavor": True,
            "max_native_words_per_message": 2,
        }

        # Conversation parameters
        self.MAX_TURNS = max_turns
        self.INSTANT_EVENT_PROB = instant_event_prob
        self.MAX_ACTIVE_TURNS = max_active_turns
        self.RETIRE_BATCH = retire_batch
        self.MAX_CONSECUTIVE_FAILURES = max_consecutive_failures

        # Data loading
        self.personas: Dict[str, Dict[str, Any]] = self._load_personas()
        self.experiences: List[Dict[str, Any]] = self._load_experiences()
        self.n_generated: int = 0
        self.mean_time_per_conversation: float = 0.0

        print(f"Loaded {len(self.personas)} personas")
        print(f"Loaded {len(self.experiences)} experiences")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_personas(self) -> Dict[str, Dict[str, Any]]:
        """Load personas from JSONL, keyed by persona ID."""
        personas = {}
        if not self.persona_file.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_file}")

        with open(self.persona_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pid = record.get("meta", {}).get("id")
                    if pid:
                        personas[pid] = record
                except json.JSONDecodeError:
                    continue
        return personas

    def _load_experiences(self) -> List[Dict[str, Any]]:
        """Load experiences from JSONL file."""
        experiences = []
        if not self.experience_file.exists():
            raise FileNotFoundError(
                f"Experience file not found: {self.experience_file}"
            )

        with open(self.experience_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    experiences.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return experiences

    # -------------------------------------------------------------------------
    # Parsing helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_state(turn_xml: str) -> Optional[str]:
        """Extract the <state> content from a turn XML block."""
        m = re.search(r"<state>\s*(.*?)\s*</state>", turn_xml, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _is_exhausted(turn_xml: str) -> bool:
        """Check whether the turn signals that all predefined topics are exhausted."""
        return "<predefined_topics_exhausted" in turn_xml

    @staticmethod
    def _parse_persona_ids_from_experience(experience_text: str) -> List[str]:
        """Fallback: extract ordered persona IDs from <persona id="..."> tags."""
        return re.findall(r'<persona\s+id=["\']([^"\']+)["\']', experience_text)

    # -------------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------------

    async def _summarize_async(
        self, prior_summary: str, turns_to_retire: List[str]
    ) -> Tuple[str, int, int]:
        """Summarize retired turns into a rolling conversation summary."""
        prompt_text = self.summarizer_template.render(
            prior_summary=prior_summary or None,
            messages_to_retire="\n\n".join(turns_to_retire),
        )
        messages = LLMClient.build_messages(
            system_text=(
                "You are a conversation memory system. Produce concise, "
                "accurate summaries that preserve key facts, emotional beats, "
                "and unresolved threads."
            ),
            user_text=prompt_text,
        )
        response = await self.summarizer_client.generate_async(
            messages, max_tokens=2048
        )
        raw = response.choices[0].message.content.strip()
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)

        # Programmatic extraction — never trust the LLM to output only the summary
        m = re.search(r"<summary>\s*(.*?)\s*</summary>", raw, re.DOTALL)
        if m:
            return m.group(1).strip(), input_tokens, output_tokens

        # Graceful fallback: if the model forgot the tags, use the full output
        # rather than silently losing the summary
        print("  ⚠ summarizer did not wrap output in <summary> tags, using raw output")
        return raw, input_tokens, output_tokens

    async def _maybe_summarize(
        self, summary: str, active_turns: List[str]
    ) -> Tuple[str, List[str], int, int]:
        """
        If the active window is full, retire the oldest RETIRE_BATCH turns
        into the rolling summary and return the trimmed active window.
        """
        if len(active_turns) <= self.MAX_ACTIVE_TURNS:
            return summary, active_turns, 0, 0

        to_retire = active_turns[: self.RETIRE_BATCH]
        keep = active_turns[self.RETIRE_BATCH :]
        new_summary, inp, out = await self._summarize_async(summary, to_retire)
        return new_summary, keep, inp, out

    # -------------------------------------------------------------------------
    # Turn generation
    # -------------------------------------------------------------------------

    async def _generate_turn_async(
        self,
        persona: Dict[str, Any],
        experience_text: str,
        summary: str,
        active_turns: List[str],
        current_state: str,
        instant_event: Optional[str],
        conversation_style: str,
        message_cadence: str,
    ) -> Tuple[str, int, int]:
        """Generate one agent's turn and return the raw XML string."""
        user_text = self.turn_template.render(
            persona=persona,
            experience=experience_text,
            summary=summary or None,
            active_messages="\n\n".join(active_turns),
            current_state=current_state,
            instant_event=instant_event,
            conversation_style=conversation_style,
            message_cadence=message_cadence,
        )

        # Build messages with static system prompt for cache-friendliness
        messages = LLMClient.build_messages(
            system_text=_TURN_SYSTEM_PROMPT,
            user_text=user_text,
        )

        response = await self.llm_client.generate_async(messages)
        raw = response.choices[0].message.content.strip()
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)

        # Ensure we always return a well-formed <turn> block
        if "<turn>" in raw and "</turn>" in raw:
            start = raw.find("<turn>")
            end = raw.find("</turn>") + len("</turn>")
            return raw[start:end], input_tokens, output_tokens
        return raw, input_tokens, output_tokens

    # -------------------------------------------------------------------------
    # Conversation loop
    # -------------------------------------------------------------------------

    async def generate_conversation_async(
        self,
        experience_record: Dict[str, Any],
        record_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the full turn-by-turn conversation loop for one experience record.
        Conversations within a batch run concurrently; turns within a
        conversation are strictly sequential.
        """
        experience_text = experience_record.get("experience_text", "")
        exp_meta = experience_record.get("meta", {})

        # Resolve persona IDs: prefer stored meta, fall back to parsing the XML
        persona_ids = exp_meta.get(
            "persona_ids"
        ) or self._parse_persona_ids_from_experience(experience_text)
        if len(persona_ids) < 2:
            print(
                f"  ✗ Record {record_index}: fewer than 2 persona IDs found, skipping."
            )
            return {}

        agents: List[Dict[str, Any]] = []
        for pid in persona_ids[:2]:
            persona = self.personas.get(pid)
            if not persona:
                print(
                    f"  ✗ Record {record_index}: persona '{pid}' not in persona map, skipping."
                )
                return {}
            agents.append(persona)

        conversation_style = exp_meta.get("conversation_style", "semi-structured")
        message_cadence = exp_meta.get("message_cadence", "delayed")
        initial_state = exp_meta.get("initial_state", "freeform; —; —")
        instant_events: List[str] = exp_meta.get("instant_events", [])

        names = [a["meta"].get("name", f"Agent{i}") for i, a in enumerate(agents)]
        print(
            f"\n[Record {record_index}] {names[0]} ↔ {names[1]} | "
            f"style={conversation_style} | cadence={message_cadence}"
        )

        # Initialise state
        current_state = initial_state
        summary = ""
        active_turns: List[str] = []  # sliding window for the prompt
        all_turns: List[str] = []  # full record — never trimmed
        exhausted = False
        consecutive_failures = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_summarizer_input_tokens = 0
        total_summarizer_output_tokens = 0

        start_time = datetime.now()

        for turn_index in range(self.MAX_TURNS):
            agent = agents[turn_index % 2]
            agent_name = agent["meta"].get("name", f"Agent{turn_index % 2}")

            # Instant event injection (per-agent, not shared)
            instant_event: Optional[str] = (
                random.choice(instant_events)
                if instant_events and random.random() < self.INSTANT_EVENT_PROB
                else None
            )

            try:
                turn_xml, inp, out = await self._generate_turn_async(
                    persona=agent,
                    experience_text=experience_text,
                    summary=summary,
                    active_turns=active_turns,
                    current_state=current_state,
                    instant_event=instant_event,
                    conversation_style=conversation_style,
                    message_cadence=message_cadence,
                )
                total_input_tokens += inp
                total_output_tokens += out
                consecutive_failures = 0
            except Exception as e:
                print(f"  ✗ Turn {turn_index} ({agent_name}) failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"  ✗ {self.MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting conversation."
                    )
                    break
                continue

            # Parse and advance state
            new_state = self._parse_state(turn_xml)
            if new_state:
                current_state = new_state

            all_turns.append(turn_xml)
            active_turns.append(turn_xml)

            # Rolling summarization (async, inline)
            summary, active_turns, inp, out = await self._maybe_summarize(
                summary, active_turns
            )
            total_summarizer_input_tokens += inp
            total_summarizer_output_tokens += out

            # Termination check
            if self._is_exhausted(turn_xml):
                exhausted = True
                print(f"  ✓ Topics exhausted at turn {turn_index + 1}")
                break

        time_taken = (datetime.now() - start_time).total_seconds()
        print(
            f"  ✓ Done: {len(all_turns)} turns | "
            f"{time_taken:.1f}s | exhausted={exhausted}"
        )

        return {
            "turns": all_turns,
            "final_summary": summary,
            "meta": {
                "record_index": record_index,
                "experience_meta": exp_meta,
                "persona_ids": persona_ids[:2],
                "persona_names": names,
                "n_turns": len(all_turns),
                "conversation_style": conversation_style,
                "message_cadence": message_cadence,
                "exhausted": exhausted,
                "model": self.model_name,
                "summarizer_model": self.summarizer_model_name,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_summarizer_input_tokens": total_summarizer_input_tokens,
                "total_summarizer_output_tokens": total_summarizer_output_tokens,
                "time_taken": time_taken,
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
        self,
        experience_records: List[Dict[str, Any]],
        batch_size: int,
        start_index: int,
    ) -> None:
        print(f"\n{'=' * 60}")
        print(
            f"Generating {len(experience_records)} conversations → {self.output_file}"
        )
        print(f"Conversations per batch: {batch_size}")
        print(f"{'=' * 60}")

        for i in range(0, len(experience_records), batch_size):
            batch = experience_records[i : i + batch_size]
            tasks = [
                self.generate_conversation_async(rec, record_index=start_index + i + j)
                for j, rec in enumerate(batch)
            ]
            # Conversations in the same batch run concurrently;
            # turns within each conversation remain sequential
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(results):
                global_idx = start_index + i + j

                if isinstance(result, Exception):
                    print(f"✗ Error at record {global_idx}: {result}")
                    continue

                if not result or not result.get("turns"):
                    print(f"✗ No turns at record {global_idx}, skipping.")
                    continue

                time_taken = result["meta"].pop("time_taken", 0.0)
                self.mean_time_per_conversation = (
                    self.mean_time_per_conversation * self.n_generated + time_taken
                ) / (self.n_generated + 1)

                self._append_to_jsonl(result)

                n_turns = result["meta"]["n_turns"]
                style = result["meta"]["conversation_style"]
                exhausted = result["meta"]["exhausted"]
                print(
                    f"✓ Saved record {global_idx} | "
                    f"{n_turns} turns | style={style} | exhausted={exhausted} | "
                    f"avg {self.mean_time_per_conversation:.1f}s/conv"
                )
                self.n_generated += 1

        print(f"\n{'=' * 60}")
        print(f"Generation complete! Total: {self.n_generated}")
        print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_batch(
        self,
        experience_records: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 4,
        start_index: int = 0,
    ) -> None:
        """
        Generate conversations for a list of experience records in async batches.
        Defaults to all loaded experiences if no records are provided.
        """
        records = (
            experience_records if experience_records is not None else self.experiences
        )
        asyncio.run(self._run_batch_async(records, batch_size, start_index))

    def generate_conversation(
        self,
        experience_record: Dict[str, Any],
        record_index: int = 0,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for single conversation generation."""
        return asyncio.run(
            self.generate_conversation_async(experience_record, record_index)
        )
