import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

from soc_2602.utils.sampling import StatsEngine


class PersonaGenerator:
    # Pool management
    POOL_MAX_SIZE = 20

    def __init__(
        self,
        llm_client,
        seed_dir: str = "data/personas/seed",
        output_file: str = f"data/personas/generated/personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        prompt_dir: str = "prompts",
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id
        self.seed_dir = Path(seed_dir)

        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("persona_generation.j2")

        self.seed_personas = self._load_seed_personas()
        self.generated_personas: List[str] = []
        self.n_generated = 0
        self.mean_time_per_persona = 0.0
        self.total_wall_time = 0.0

        self.stats_engine = StatsEngine(stats_dir=Path("data/stats"))

        print(f"Loaded {len(self.seed_personas)} seed personas")

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
        persona_info = {
            "name": self.stats_engine.get_random_name(region),
            "age": self.stats_engine.get_random_age(25, 5),
            "region": region,
            "subregion": self.stats_engine.gen_random_subregion(region),
        }

        prompt_text = self.template.render(
            shots=shots,
            persona_info=persona_info,
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating detailed, realistic character personas for conversational AI applications.",
            },
            {"role": "user", "content": prompt_text},
        ]

        print(
            f"\n[Iteration {iteration}] Generating persona ({persona_info['region']})..."
        )

        start_time = datetime.now()
        response = await self.llm_client.generate_async(messages)
        generated_text = response.choices[0].message.content.strip()
        time_taken = (datetime.now() - start_time).total_seconds()

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
