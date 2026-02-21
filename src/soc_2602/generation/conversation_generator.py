import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader


class ConversationGenerator:
    # Instant event injection probability per turn
    INSTANT_EVENT_PROB = 0.05

    # Summarization: how many active turns to keep in the prompt,
    # and how many to retire into the summary at once
    MAX_ACTIVE_TURNS = 20
    RETIRE_BATCH = 8

    # Hard safety ceiling on turns per conversation
    MAX_TURNS = 100

    def __init__(
        self,
        llm_client,
        persona_file: str,
        experience_file: str,
        output_file: str = f"data/conversations/generated/conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        prompt_dir: str = "conf/prompts",
        summarizer_client=None,  # optional cheaper model for summarization
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id
        # Allow a separate (cheaper/faster) client for summarization
        self.summarizer_client = summarizer_client or llm_client

        self.persona_file = Path(persona_file)
        self.experience_file = Path(experience_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.turn_template = self.jinja_env.get_template("turn_generation.j2")
        self.summarizer_template = self.jinja_env.get_template("summarization.j2")

        self.personas: Dict[str, Dict[str, Any]] = self._load_personas()
        self.experiences: List[Dict[str, Any]] = self._load_experiences()
        self.n_generated = 0
        self.mean_time_per_conversation = 0.0

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
    ) -> str:
        prompt_text = self.summarizer_template.render(
            prior_summary=prior_summary or None,
            messages_to_retire="\n\n".join(turns_to_retire),
        )
        messages = [
            {
                "role": "system",
                "content": "You are a conversation memory system. Produce concise, accurate summaries.",
            },
            {"role": "user", "content": prompt_text},
        ]
        response = await self.summarizer_client.generate_async(messages)
        raw = response.choices[0].message.content.strip()

        # Programmatic extraction — never trust the LLM to output only the summary
        m = re.search(r"<summary>\s*(.*?)\s*</summary>", raw, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Graceful fallback: if the model forgot the tags, use the full output
        # rather than silently losing the summary
        print("  ⚠ summarizer did not wrap output in <summary> tags, using raw output")
        return raw

    async def _maybe_summarize(
        self, summary: str, active_turns: List[str]
    ) -> Tuple[str, List[str]]:
        """
        If the active window is full, retire the oldest RETIRE_BATCH turns
        into the rolling summary and return the trimmed active window.
        """
        if len(active_turns) <= self.MAX_ACTIVE_TURNS:
            return summary, active_turns

        to_retire = active_turns[: self.RETIRE_BATCH]
        keep = active_turns[self.RETIRE_BATCH :]
        new_summary = await self._summarize_async(summary, to_retire)
        return new_summary, keep

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
    ) -> str:
        """Generate one agent's turn and return the raw XML string."""
        prompt_text = self.turn_template.render(
            persona=persona,
            experience=experience_text,
            summary=summary or None,
            active_messages="\n\n".join(active_turns),
            current_state=current_state,
            instant_event=instant_event,
            conversation_style=conversation_style,
            message_cadence=message_cadence,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are roleplaying a realistic character in an ongoing conversation. "
                    "Stay fully in character. Output only the <turn> XML block."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        response = await self.llm_client.generate_async(messages)
        raw = response.choices[0].message.content.strip()

        # Ensure we always return a well-formed <turn> block
        if "<turn>" in raw and "</turn>" in raw:
            start = raw.find("<turn>")
            end = raw.find("</turn>") + len("</turn>")
            return raw[start:end]
        return raw

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
        MAX_CONSECUTIVE_FAILURES = 3

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
                turn_xml = await self._generate_turn_async(
                    persona=agent,
                    experience_text=experience_text,
                    summary=summary,
                    active_turns=active_turns,
                    current_state=current_state,
                    instant_event=instant_event,
                    conversation_style=conversation_style,
                    message_cadence=message_cadence,
                )
                consecutive_failures = 0
            except Exception as e:
                print(f"  ✗ Turn {turn_index} ({agent_name}) failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"  ✗ {MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting conversation."
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
            summary, active_turns = await self._maybe_summarize(summary, active_turns)

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
