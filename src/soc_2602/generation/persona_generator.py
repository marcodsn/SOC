import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from soc_2602.llm.client import LLMClient
from soc_2602.utils.sampling import StatsEngine

# ── Static system prompt ─────────────────────────────────────────────────────
# This block is identical across ALL persona generation calls, maximising
# the byte-identical prefix for providers with automatic prompt caching.
# The variable content (few-shot examples, target metadata) goes into the
# user message rendered from the Jinja template.

_PERSONA_SYSTEM_PROMPT = """\
You are a persona generation system. Your task is to create realistic, psychologically grounded character profiles for conversational AI applications.

## Instructions

Generate ONE detailed persona following the exact format shown in the examples the user will provide. The persona must be:

1. **Realistic and grounded**: Based on real human psychology, not stereotypes or caricatures
2. **Internally consistent**: Traits, behaviors, and history should align logically
3. **Conversationally rich**: Include speech patterns, communication style, and typical phrases
4. **Diverse**: Different from the example(s) in age, background, personality, life circumstances
5. **Normal people**: Represent "average" or "mean" population — not extreme or exceptional cases
6. **Psychologically deep**: Include emotional patterns, values, fears, relationships, coping mechanisms
7. **Show, don't tell**: Imply traits through narrative situations, history, and behavior rather than bare keyword labels. Instead of `Trait: "extremely loyal"`, write: "She has driven four hours in a snowstorm when a friend called in crisis; it's not something she would question."
8. **Positive framing**: Express all behavioral tendencies as what the character *does* or *values*, not as prohibitions. Write "She speaks her mind candidly" rather than "She never lies."

## Format Requirements

Use this exact XML structure with markdown formatting inside:

<character>
**Basic Information**
**Name:** [Name]
**Age:** [Age]
**Location:** [Location]
**Pronouns:** [Pronouns]

**Physical & Lifestyle**
[2-3 paragraphs describing appearance, daily routines, hobbies, living situation, aesthetic choices]

**Personality Overview**
[2-3 paragraphs covering core personality, self-perception, contradictions, developmental stage]

**Core Traits**
[1-2 paragraphs listing and explaining defining characteristics]

**Emotional Profile**
[2-3 paragraphs on emotional patterns, regulation strategies, common expressions, anxiety/depression/joy manifestations]

**Relationships**
[2-4 paragraphs covering family, friends, romantic relationships, work relationships — specific people and dynamics]

**Values, Motivations & Fears**
[2-3 paragraphs on core values, what drives them, what terrifies them]

**Behavioral Patterns**
[1-2 paragraphs covering both adaptive and maladaptive coping strategies]

**Communication Style**
[2 paragraphs on how they speak and text; pacing, vocabulary level, formality, use of humor or silence]

**Example Messages**
[2-3 short back-and-forth exchanges showing how the persona actually sounds in conversation. Precede each block with <START>.]

**Summary**
[1 paragraph synthesizing their journey/current life stage]
</character>

## Diversity Guidelines

Vary across these dimensions:
- **Occupation**: Blue collar, white collar, creative, service, unemployed, retired, student
- **Personality**: Introvert/extrovert, optimistic/pessimistic, stable/volatile, organized/chaotic
- **Life circumstances**: Single, partnered, divorced, widowed, with/without children, different family structures
- **Mental health**: Generally stable, managing conditions, in therapy, undiagnosed struggles
- **Socioeconomic**: Working class, middle class, comfortable (avoid extremes)
- **Life stage**: Student, early career, established, midlife, retirement, crisis, transition
- **Self-awareness level**: Highly introspective, moderately self-aware, unreflective/practically minded (most common in real populations)

## Output

Output ONLY the content between <character> and </character> tags (inclusive). No additional commentary."""


class PersonaGenerator:
    """Generates diverse, psychologically grounded character personas.

    Supports region profiles for scoped generation (e.g. "west", "east_asia")
    and language configuration for the lingua franca used in example messages.

    Parameters
    ----------
    llm_client : LLMClient
        Client for LLM API calls.
    seed_dir : str
        Directory containing seed persona text files.
    output_file : str
        Path to the output JSONL file.
    prompt_dir : str
        Directory containing Jinja2 prompt templates.
    stats_engine : StatsEngine | None
        Pre-configured sampling engine. If ``None``, a default global
        engine is created from ``data/stats``.
    language_settings : dict | None
        Language configuration from ``config.yaml``. Keys:
        ``lingua_franca``, ``native_flavor``, ``max_native_words_per_message``.
    age_mean : float
        Mean for the age normal distribution.
    age_std : float
        Standard deviation for the age normal distribution.
    """

    # Pool management
    POOL_MAX_SIZE = 20

    def __init__(
        self,
        llm_client: LLMClient,
        seed_dir: str = "data/personas/seed",
        output_file: Optional[str] = None,
        prompt_dir: str = "conf/prompts",
        stats_engine: Optional[StatsEngine] = None,
        language_settings: Optional[Dict[str, Any]] = None,
        age_mean: float = 25,
        age_std: float = 5,
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id
        self.seed_dir = Path(seed_dir)

        if output_file is None:
            output_file = f"data/personas/generated/personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("persona_generation.j2")

        self.seed_personas: List[str] = self._load_seed_personas()
        self.generated_personas: List[str] = []
        self.n_generated: int = 0
        self.mean_time_per_persona: float = 0.0
        self.total_wall_time: float = 0.0

        # Stats engine — may be pre-configured with a region profile
        self.stats_engine = stats_engine or StatsEngine(stats_dir=Path("data/stats"))

        # Language settings
        self.language_settings = language_settings or {
            "lingua_franca": "English",
            "native_flavor": True,
            "max_native_words_per_message": 2,
        }

        # Age distribution parameters
        self.age_mean = age_mean
        self.age_std = age_std

        print(f"Loaded {len(self.seed_personas)} seed personas")
        print(f"Active regions: {len(self.stats_engine.available_regions)} codes")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_seed_personas(self) -> List[str]:
        """Load all seed persona files, sorted for determinism."""
        personas = []
        for file_path in sorted(self.seed_dir.glob("*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                personas.append(f.read().strip())
        return personas

    # -------------------------------------------------------------------------
    # Selection helpers
    # -------------------------------------------------------------------------

    def _select_shots(self, iteration: int, num_shots: int = 1) -> List[str]:
        """
        Select few-shot examples.
        Seeds only for the first 10 iterations (warmup),
        then a 50/50 mix of seeds and recently generated personas.
        """
        if iteration < 10 or not self.generated_personas:
            return random.sample(
                self.seed_personas, min(num_shots, len(self.seed_personas))
            )

        pool = self.seed_personas.copy()
        if random.random() > 0.5:
            pool.extend(self.generated_personas)

        return random.sample(pool, min(num_shots, len(pool)))

    # -------------------------------------------------------------------------
    # Pool management
    # -------------------------------------------------------------------------

    def _update_pool(self, persona_text: str) -> None:
        """
        Rolling window pool: append new persona, then drop the oldest entry
        once POOL_MAX_SIZE is exceeded. No cliff resets.
        """
        self.generated_personas.append(persona_text)
        if len(self.generated_personas) > self.POOL_MAX_SIZE:
            self.generated_personas.pop(0)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    async def generate_persona_async(self, iteration: int = 0) -> Dict[str, Any]:
        """Generate a persona asynchronously and return a dictionary with metadata."""
        shots = self._select_shots(iteration)

        region = self.stats_engine.get_random_region()
        persona_info: Dict[str, Any] = {
            "name": self.stats_engine.get_random_name(region),
            "age": StatsEngine.get_random_age(self.age_mean, self.age_std),
            "region": region,
            "subregion": self.stats_engine.gen_random_subregion(region),
        }

        # Pass language settings into the template context
        lingua_franca = self.language_settings.get("lingua_franca", "English")
        if lingua_franca and lingua_franca.lower() != "english":
            persona_info["lingua_franca"] = lingua_franca

        # Render the user message (variable part only)
        user_text = self.template.render(
            shots=shots,
            persona_info=persona_info,
        )

        # Build messages with static system prompt for cache-friendliness
        messages = LLMClient.build_messages(
            system_text=_PERSONA_SYSTEM_PROMPT,
            user_text=user_text,
        )

        print(
            f"\n[Iteration {iteration}] Generating persona ({persona_info['region']})..."
        )

        start_time = datetime.now()
        response = await self.llm_client.generate_async(messages)
        generated_text = response.choices[0].message.content.strip()
        time_taken = (datetime.now() - start_time).total_seconds()

        # Track token usage when available
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)

        if "<character>" in generated_text:
            start_idx = generated_text.find("<character>") + len("<character>")
            end_idx = generated_text.find("</character>")
            if end_idx > start_idx:
                generated_text = generated_text[start_idx:end_idx].strip()
                generated_text = f"<character>\n{generated_text}\n</character>"
        else:
            if not generated_text:
                return {}

        return {
            "persona_text": generated_text,
            "meta": {
                "iteration": iteration,
                "model": self.model_name,
                "region": persona_info["region"],
                "subregion": persona_info["subregion"],
                "name": persona_info["name"],
                "age": persona_info["age"],
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
        self, num_personas: int, start_iteration: int, batch_size: int
    ) -> None:
        print(f"\n{'=' * 60}")
        print(f"Generating {num_personas} personas → {self.output_file}")
        print(f"Batch size: {batch_size}")
        print(f"Regions: {len(self.stats_engine.available_regions)} active codes")
        print(f"{'=' * 60}")

        iterations = range(start_iteration, start_iteration + num_personas)

        for i in range(0, len(iterations), batch_size):
            batch_iterations = list(iterations[i : i + batch_size])
            tasks = [self.generate_persona_async(it) for it in batch_iterations]

            batch_start = datetime.now()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_wall_time = (datetime.now() - batch_start).total_seconds()
            self.total_wall_time += batch_wall_time

            for j, result in enumerate(results):
                iteration = batch_iterations[j]

                if isinstance(result, Exception):
                    print(f"✗ Error at iteration {iteration}: {result}")
                    continue

                if not result or not result.get("persona_text"):
                    print(f"✗ No persona generated at iteration {iteration}, skipping.")
                    continue

                time_taken = result["meta"].pop("time_taken", 0.0)

                self._append_to_jsonl(result)
                print(
                    f"✓ Saved record {iteration} | "
                    f"region={result['meta']['region']} | "
                    f"call latency {time_taken:.2f}s"
                )

                self._update_pool(result["persona_text"])
                self.n_generated += 1

            if self.n_generated > 0:
                self.mean_time_per_persona = self.total_wall_time / self.n_generated
                print(
                    f"  → batch wall time {batch_wall_time:.2f}s | "
                    f"avg throughput {self.mean_time_per_persona:.2f}s/persona"
                )

        print(f"\n{'=' * 60}")
        print(f"Generation complete! Total: {self.n_generated}")
        print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_batch(
        self, num_personas: int, start_iteration: int = 0, batch_size: int = 4
    ) -> None:
        """Generate personas in async batches."""
        asyncio.run(self._run_batch_async(num_personas, start_iteration, batch_size))

    def generate_persona(self, iteration: int = 0) -> Dict[str, Any]:
        """Synchronous wrapper for single persona generation."""
        return asyncio.run(self.generate_persona_async(iteration))
