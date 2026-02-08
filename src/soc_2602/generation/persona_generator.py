import random
from datetime import datetime
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from soc_2602.utils.sampling import StatsEngine


class PersonaGenerator:
    def __init__(
        self,
        llm_client,
        seed_dir: str = "data/personas/seed",
        output_dir: str = "data/personas/generated",
        prompt_dir: str = "prompts",
    ):
        self.llm_client = llm_client
        self.seed_dir = Path(seed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("persona_generation.j2")

        # Load seed personas
        self.seed_personas = self._load_seed_personas()
        self.generated_personas = []
        self.n_generated = 0

        # Instantiate stats engine for persona sampling
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
        """
        Select shots for few-shot learning.
        Early iterations use only seed personas.
        Later iterations sample from both seed and generated.
        """
        # First 10 iterations: only use seeds
        if iteration < 10 or len(self.generated_personas) == 0:
            return random.sample(
                self.seed_personas, min(num_shots, len(self.seed_personas))
            )

        # Later iterations: 50% chance to use generated personas
        pool = self.seed_personas.copy()
        if random.random() > 0.5 and self.generated_personas:
            pool.extend(self.generated_personas)

        return random.sample(pool, min(num_shots, len(pool)))

    def generate_persona(self, iteration: int = 0) -> str:
        """Generate a single persona using iterative sampling."""
        # Select shots
        shots = self._select_shots(iteration)

        persona_info = {
            "name": "",
            "age": -1,
            "region": self.stats_engine.get_random_region(),
            "subregion": "",
        }

        persona_info["subregion"] = self.stats_engine.gen_random_subregion(
            persona_info["region"]
        )
        persona_info["name"] = self.stats_engine.get_random_name(persona_info["region"])
        persona_info["age"] = self.stats_engine.get_random_age(25, 5)
        # print(f"Selected persona info for iteration {iteration}: {persona_info}")

        # Render prompt with Jinja
        prompt_text = self.template.render(
            shots=shots,
            persona_info=persona_info,
        )

        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating detailed, realistic character personas for conversational AI applications.",
            },
            {"role": "user", "content": prompt_text},
        ]

        # Generate
        print(
            f"\n[Iteration {iteration}] Generating persona with {len(shots)} shot(s)..."
        )
        response = self.llm_client.generate(messages)

        # Extract generated text
        generated_text = response.choices[0].message.content.strip()

        # Find and extract the persona block if it exists
        if "<character>" in generated_text:
            start_idx = generated_text.find("<character>") + len("<character>")
            end_idx = generated_text.find("</character>")
            if end_idx > start_idx:
                generated_text = generated_text[start_idx:end_idx].strip()
                generated_text = "<character>" + generated_text + "</character>"
        else:
            generated_text = ""

        return generated_text

    def save_persona(self, persona: str, iteration: int):
        """Save generated persona to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"b_oss120_persona_{iteration:04d}_{timestamp}.txt"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(persona)

        print(f"✓ Saved to {filepath}")
        return filepath

    def generate_batch(self, num_personas: int, start_iteration: int = 0):
        """Generate multiple personas with iterative sampling."""
        print(f"\n{'=' * 60}")
        print(f"Generating {num_personas} personas")
        print(f"{'=' * 60}")

        for i in range(num_personas):
            iteration = start_iteration + i

            try:
                # Generate
                persona = self.generate_persona(iteration)
                if persona == "":
                    print(f"✗ No persona generated at iteration {iteration}, skipping.")
                    continue

                # Save
                _ = self.save_persona(persona, iteration)

                # Add to generated pool for future iterations
                self.generated_personas.append(persona)
                self.n_generated += 1

                # Reset generated pool every 20 iterations to maintain diversity
                if len(self.generated_personas) > 20:
                    self.generated_personas = []

                print(f"Progress: {i + 1}/{num_personas}")

            except Exception as e:
                print(f"✗ Error at iteration {iteration}: {e}")
                continue

        print(f"\n{'=' * 60}")
        print("Generation complete!")
        print(f"Total generated: {self.n_generated}")
        print(f"{'=' * 60}")
