import asyncio
import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader


class ExperienceGenerator:
    def __init__(
        self,
        llm_client,
        persona_file: str,
        output_file: str = f"data/experiences/generated/experiences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        seed_dir: str = "data/experiences/seed",
        prompt_dir: str = "conf/prompts",
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id

        self.persona_file = Path(persona_file)
        self.seed_dir = Path(seed_dir)

        # Setup output file
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("experience_generation.j2")

        # Load data
        self.personas = self._load_personas()
        self.seed_experiences = self._load_seed_experiences()
        self.generated_experiences = []
        self.n_generated = 0
        self.mean_time_per_experience = 0.0

        print(f"Loaded {len(self.personas)} personas")
        print(f"Loaded {len(self.seed_experiences)} seed experiences")

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
        """Load seed experience files."""
        experiences = []
        if self.seed_dir.exists():
            for file_path in self.seed_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    experiences.append(f.read().strip())
        return experiences

    def _select_shots(self, num_shots: int = 2) -> List[str]:
        """Select shots for few-shot learning."""
        # Mix seeds and generated if available
        pool = self.seed_experiences.copy()
        if random.random() > 0.5 and self.generated_experiences:
            # Add some recently generated ones to the pool
            pool.extend(self.generated_experiences)

        if not pool:
            return []

        return random.sample(pool, min(num_shots, len(pool)))

    def _select_personas(self, n: int = 2) -> List[Dict[str, Any]]:
        """Select n random personas."""
        if len(self.personas) < n:
            return self.personas
        else:
            # We first select a random persona, then with a 50% chance we can select another one
            # from the same region to increase coherence, or otherwise one from a random region
            first_persona = random.choice(self.personas)
            if random.random() > 0.5:
                # Try to find another persona from the same region
                same_region_personas = [
                    p
                    for p in self.personas
                    if p.get("meta", {}).get("region")
                    == first_persona.get("meta", {}).get("region")
                    and p != first_persona
                ]
                if same_region_personas:
                    second_persona = random.choice(same_region_personas)
                else:
                    second_persona = random.choice(self.personas)
            else:
                second_persona = random.choice(self.personas)
            return [first_persona, second_persona]

    async def generate_experience_async(self, iteration: int = 0) -> Dict[str, Any]:
        """Generate an experience asynchronously and return a dictionary with metadata."""
        # Select personas
        selected_personas = self._select_personas(n=2)

        # Select shots
        shots = self._select_shots()

        # Render prompt
        # The template expects 'personas' list where each item has 'persona_text'
        # It iterates: {% for persona in personas %}{{ persona.persona_text }}{% endfor %}
        prompt_text = self.template.render(
            shots=shots,
            personas=selected_personas,
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating realistic scenarios and conversation contexts for AI personas.",
            },
            {"role": "user", "content": prompt_text},
        ]

        names = [
            p["meta"].get("name", "Unknown") for p in selected_personas if "meta" in p
        ]
        print(f"\n[Iteration {iteration}] Generating experience for {names}...")

        # Call LLM
        start_time = datetime.now()
        response = await self.llm_client.generate_async(messages)
        generated_text = response.choices[0].message.content.strip()
        end_time = datetime.now()

        time_taken = (end_time - start_time).total_seconds()

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

        # Collect persona IDs for metadata
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
                "time_taken": time_taken,
                # "timestamp": datetime.now().isoformat(),
            },
        }

    def _append_to_jsonl(self, data: Dict[str, Any]):
        """Append a single record to the JSONL file."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def _run_batch_async(
        self, num_experiences: int, start_iteration: int, batch_size: int
    ):
        print(f"\n{'=' * 60}")
        print(f"Generating {num_experiences} experiences to {self.output_file}")
        print(f"Batch size: {batch_size}")
        print(f"{'=' * 60}")

        iterations = range(start_iteration, start_iteration + num_experiences)

        for i in range(0, len(iterations), batch_size):
            batch_iterations = iterations[i : i + batch_size]
            tasks = [
                self.generate_experience_async(iteration)
                for iteration in batch_iterations
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(results):
                iteration = batch_iterations[j]

                if isinstance(result, Exception):
                    print(f"✗ Error at iteration {iteration}: {result}")
                    # traceback.print_exception(type(result), result, result.__traceback__)
                    continue

                if not result or not result.get("experience_text"):
                    print(
                        f"✗ No experience generated at iteration {iteration}, skipping."
                    )
                    continue

                time_taken = result["meta"].pop("time_taken", 0.0)
                self.mean_time_per_experience = (
                    self.mean_time_per_experience * self.n_generated + time_taken
                ) / (self.n_generated + 1)

                self._append_to_jsonl(result)
                print(
                    f"✓ Saved record {iteration}, Avg time: {self.mean_time_per_experience:.2f}s"
                )

                self.generated_experiences.append(result["experience_text"])
                self.n_generated += 1

                if len(self.generated_experiences) > 20:
                    self.generated_experiences = []  # Keep pool fresh

        print(f"\n{'=' * 60}")
        print(f"Generation complete! Total: {self.n_generated}")
        print(f"{'=' * 60}")

    def generate_batch(
        self, num_experiences: int, start_iteration: int = 0, batch_size: int = 4
    ):
        """Generate experiences in batches asynchronously."""
        asyncio.run(self._run_batch_async(num_experiences, start_iteration, batch_size))

    def generate_experience(self, iteration: int = 0) -> Dict[str, Any]:
        """Synchronous wrapper for single experience generation."""
        return asyncio.run(self.generate_experience_async(iteration))
