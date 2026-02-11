import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

# Assuming this import exists in your project structure
from soc_2602.utils.sampling import StatsEngine


class PersonaGenerator:
    def __init__(
        self,
        llm_client,
        seed_dir: str = "data/personas/seed",
        output_file: str = f"data/personas/generated/b_personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        prompt_dir: str = "prompts",
    ):
        self.llm_client = llm_client
        self.model_name = self.llm_client.model_id
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
        self.template = self.jinja_env.get_template("persona_generation.j2")

        # Load seed personas
        self.seed_personas = self._load_seed_personas()
        self.generated_personas = []  # Keep only text here for few-shot prompting
        self.n_generated = 0
        self.mean_time_per_persona = 0.0

        # Instantiate stats engine
        self.stats_engine = StatsEngine(stats_dir=Path("data/stats"))

        print(f"Loaded {len(self.seed_personas)} seed personas")

    def _load_seed_personas(self) -> List[str]:
        """Load all seed persona files."""
        personas = []
        for file_path in self.seed_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                personas.append(f.read().strip())
        return personas

    def _select_shots(self, iteration: int, num_shots: int = 2) -> List[str]:
        """Select shots for few-shot learning."""
        # First 10 iterations: only use seeds
        if iteration < 10 or len(self.generated_personas) == 0:
            return random.sample(
                self.seed_personas, min(num_shots, len(self.seed_personas))
            )

        # Later: 50% chance to use generated personas
        pool = self.seed_personas.copy()
        if random.random() > 0.5 and self.generated_personas:
            pool.extend(self.generated_personas)

        return random.sample(pool, min(num_shots, len(pool)))

    def generate_persona(self, iteration: int = 0) -> Dict[str, Any]:
        """Generate a persona and return a dictionary with metadata."""
        # Select shots
        shots = self._select_shots(iteration)

        # Generate metadata using StatsEngine
        region = self.stats_engine.get_random_region()
        persona_info = {
            "name": self.stats_engine.get_random_name(region),
            "age": self.stats_engine.get_random_age(25, 5),
            "region": region,
            "subregion": self.stats_engine.gen_random_subregion(region),
        }

        # Render prompt
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

        # Call LLM and measure time
        start_time = datetime.now()

        response = self.llm_client.generate(messages)
        generated_text = response.choices[0].message.content.strip()

        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        self.mean_time_per_persona = (
            self.mean_time_per_persona * self.n_generated + time_taken
        ) / (self.n_generated + 1)

        # Extract <character> block
        if "<character>" in generated_text:
            start_idx = generated_text.find("<character>") + len("<character>")
            end_idx = generated_text.find("</character>")
            if end_idx > start_idx:
                generated_text = generated_text[start_idx:end_idx].strip()
                generated_text = f"<character>\n{generated_text}\n</character>"
        else:
            # Fallback or empty if parsing fails
            if not generated_text:
                return {}

        # Construct the full data object
        return {
            "persona_text": generated_text,
            "meta": {
                "iteration": iteration,
                # "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "region": persona_info["region"],
                "subregion": persona_info["subregion"],
                "name": persona_info["name"],
                "age": persona_info["age"],
            },
        }

    def _append_to_jsonl(self, data: Dict[str, Any]):
        """Append a single record to the JSONL file."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def generate_batch(self, num_personas: int, start_iteration: int = 0):
        print(f"\n{'=' * 60}")
        print(f"Generating {num_personas} personas to {self.output_file}")
        print(f"{'=' * 60}")

        for i in range(num_personas):
            iteration = start_iteration + i

            try:
                result = self.generate_persona(iteration)

                # Check if generation failed (empty dict or empty text)
                if not result or not result.get("persona_text"):
                    print(f"✗ No persona generated at iteration {iteration}, skipping.")
                    continue

                # Save to JSONL
                self._append_to_jsonl(result)
                print(
                    f"✓ Saved record {iteration}, Avg time: {self.mean_time_per_persona:.2f}s"
                )

                # Update pool for few-shot
                self.generated_personas.append(result["persona_text"])
                self.n_generated += 1

                # Reset pool occasionally
                if len(self.generated_personas) > 20:
                    self.generated_personas = []

            except Exception as e:
                print(f"✗ Error at iteration {iteration}: {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"Generation complete! Total: {self.n_generated}")
        print(f"{'=' * 60}")
