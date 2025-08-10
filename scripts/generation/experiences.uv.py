#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub",
#   "pydantic",
#   "datasets" # Added datasets as an explicit dependency
# ]
# ///

import os
import json
import time
import uuid
import random
import asyncio
from typing import List, Set, Dict, Any, Tuple
from huggingface_hub import AsyncInferenceClient
from datasets import load_dataset
from pydantic import BaseModel, Field

# --- Configuration ---
PERSONAS_FILE = ""  # "data/personas/processed_personas_1754226515.jsonl"
PERSONAS_DATASET = "marcodsn/SPB-2508"
SEED_EXPERIENCES_FILE = "data/experiences/experiences_Qwen3-235B-A22B-Instruct-2507_1754377597.jsonl"
COMPONENTS_FILE = "data/seed/experience_components.json"
TARGET_N = 950
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CONCURRENCY = 20
CHECKPOINT_EVERY = 50
RESET_EVERY = 100

# Ensure the Hugging Face token is available
if "HF_TOKEN" not in os.environ:
    raise ValueError("Missing Hugging Face token. Please set the HF_TOKEN environment variable.")

# Use the Asynchronous client for parallel execution
client = AsyncInferenceClient(
    provider="together",
    api_key=os.environ.get("HF_TOKEN")
)

# --- Pydantic Models ---
class Persona(BaseModel):
    name: str
    username: str | None = None
    age: int
    traits: list[str]
    background: str
    chatting_style: str
    model: str
    id: str

class GeneratedScenario(BaseModel):
    situation: str = Field(description="A specific, creative online situation where the two personas might start talking.")
    topic: str = Field(description="A concise, natural, and engaging opening conversation topic that fits the situation and personas.")

class ConversationContext(BaseModel):
    persona1: Persona
    persona2: Persona
    relationship: str
    situation: str
    topic: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

# --- Component Loading ---
def load_components(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Components file not found at '{filepath}'. Please create it.")
        exit(1)

def weighted_choice(items: List[Dict[str, Any]]) -> str:
    values = [item["value"] for item in items]
    weights = [item["weight"] for item in items]
    return random.choices(values, weights=weights, k=1)[0]

# --- Dynamic Component System ---
class ComponentManager:
    def __init__(self, components_data: Dict[str, Any]):
        self.components = components_data

    def get_platform(self) -> str:
        return weighted_choice(self.components["online_platforms"])

    def get_community(self) -> str:
        return weighted_choice(self.components["online_communities"])

    def get_fandom(self) -> str:
        return weighted_choice(self.components["fandoms"])

    def get_gaming_context(self) -> str:
        return weighted_choice(self.components["gaming_contexts"])

    def get_conversation_trigger(self) -> str:
        return weighted_choice(self.components["conversation_triggers"])

    def generate_relationship(self) -> str:
        relationship_template = weighted_choice(self.components["relationship_types"])
        if "{platform}" in relationship_template:
            return relationship_template.format(platform=self.get_platform())
        elif "{community}" in relationship_template:
            return relationship_template.format(community=self.get_community())
        elif "{fandom}" in relationship_template:
            return relationship_template.format(fandom=self.get_fandom())
        elif "{gaming}" in relationship_template:
            return relationship_template.format(gaming=self.get_gaming_context())
        return relationship_template

# Define the structured JSON output format
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "GeneratedScenario",
        "schema": GeneratedScenario.model_json_schema(),
        "strict": True,
    },
}

# --- Helper Functions ---
def load_personas_from_jsonl(filepath: str) -> list[Persona]:
    personas = []
    print(f"Loading personas from {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        personas.append(Persona(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"⚠️ Skipping invalid JSON on line {i+1}: {e}")
        print(f"✅ Successfully loaded {len(personas)} personas.")
        return personas
    except FileNotFoundError:
        print(f"❌ ERROR: Persona file not found at '{filepath}'. Please check the path.")
        exit(1)

def load_seed_experiences(filepath: str) -> list[ConversationContext]:
    experiences = []
    print(f"Loading seed experiences from {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        experiences.append(ConversationContext(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"⚠️ Skipping invalid seed experience on line {i+1}: {e}")
        print(f"✅ Successfully loaded {len(experiences)} seed experiences.")
        return experiences
    except FileNotFoundError:
        print(f"❌ ERROR: Seed experience file not found at '{filepath}'.")
        exit(1)

def save_to_jsonl(data: dict, filepath: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def format_experience_examples(examples: list[ConversationContext]) -> str:
    if not examples:
        return ""

    formatted_string = "Here are some examples of high-quality online conversation scenarios we have already generated. AVOID repeating their specific themes:\n"
    for ex in examples:
        p1_json = json.dumps(ex.persona1.model_dump(exclude={'username', 'model', 'id'}), indent=2)
        p2_json = json.dumps(ex.persona2.model_dump(exclude={'username', 'model', 'id'}), indent=2)
        formatted_string += (
            f"\n--- EXAMPLE START ---\n"
            f"PERSONA 1: {p1_json}\n"
            f"PERSONA 2: {p2_json}\n"
            f"RELATIONSHIP: \"{ex.relationship}\"\n"
            f"GENERATED SITUATION: \"{ex.situation}\"\n"
            f"GENERATED TOPIC: \"{ex.topic}\"\n"
            f"--- EXAMPLE END ---\n"
        )
    return formatted_string + "\n---\n"


def calculate_age_similarity_weight(age1: int, age2: int, similarity_strength: float = 2.0) -> float:
    """Calculate probability weight based on age similarity."""
    age_diff = abs(age1 - age2)
    return (1 / (age_diff + 1)) ** similarity_strength

def precompute_weighted_pairs(
    persona_pool: list[Persona], similarity_strength: float = 2.0
) -> Tuple[List[Tuple[Persona, Persona]], List[float]]:
    """
    Pre-calculates all persona pairs and their age-similarity weights.
    This is an O(n^2) operation that should only be run ONCE.
    """
    if len(persona_pool) < 2:
        raise ValueError("Need at least 2 personas to create pairs.")

    print(f"\n⏳ Pre-computing age-weighted pairs for {len(persona_pool)} personas... (This may take a moment)")
    start_time = time.time()

    pairs = []
    weights = []

    for i in range(len(persona_pool)):
        for j in range(i + 1, len(persona_pool)):
            p1 = persona_pool[i]
            p2 = persona_pool[j]
            pairs.append((p1, p2))
            weight = calculate_age_similarity_weight(p1.age, p2.age, similarity_strength)
            weights.append(weight)

    end_time = time.time()
    print(f"✅ Pre-computation complete. Generated {len(pairs)} pairs in {end_time - start_time:.2f} seconds.")
    return pairs, weights

def select_precomputed_pair(
    precomputed_pairs: List[Tuple[Persona, Persona]],
    precomputed_weights: List[float]
) -> tuple[Persona, Persona]:
    """Selects a random pair from the pre-computed lists."""
    return random.choices(precomputed_pairs, weights=precomputed_weights, k=1)[0]

# --- Main Generation Logic ---
async def generate_one_experience(
    semaphore: asyncio.Semaphore,
    precomputed_pairs: List[Tuple[Persona, Persona]],
    precomputed_weights: List[float],
    reference_experiences: list[ConversationContext],
    component_manager: ComponentManager
) -> ConversationContext | None:
    async with semaphore:
        try:
            if not reference_experiences:
                print("⚠️ Reference experience pool is empty. Cannot generate an example.")
                return None

            references = random.sample(reference_experiences, min(len(reference_experiences), 2))
            examples_str = format_experience_examples(references)

            p1_data, p2_data = select_precomputed_pair(precomputed_pairs, precomputed_weights)

            relationship = component_manager.generate_relationship()
            conversation_trigger = component_manager.get_conversation_trigger()

            instruction = (
                f"{examples_str}"
                "Your task is to generate a NEW, unique ONLINE conversation scenario by creatively combining the two personas and their suggested relationship. "
                "Focus on digital communication (texting, social media, messaging apps, etc.). Create a believable online situation and a natural starting topic.\n\n"
                "--- NEW PERSONA 1 ---\n"
                f"{json.dumps(p1_data.model_dump(exclude={'username', 'model', 'id'}), indent=2, ensure_ascii=False)}\n\n"
                "--- NEW PERSONA 2 ---\n"
                f"{json.dumps(p2_data.model_dump(exclude={'username', 'model', 'id'}), indent=2, ensure_ascii=False)}\n\n"
                "--- SUGGESTED RELATIONSHIP ---\n"
                f"{relationship}\n\n"
                "--- CONVERSATION TRIGGER INSPIRATION ---\n"
                f"{conversation_trigger}\n\n"
                "NOTE: Use these as inspiration but feel free to adapt them creatively to fit the personas better. "
                "The situation should focus on WHY they're messaging each other right now in this online context.\n\n"
                "--- YOUR TASK ---\n"
                "Generate a JSON object with:\n"
                "- 'situation': The specific online context/reason they're chatting (≤300 chars, no actual messages)\n"
                "- 'topic': The natural conversation starter that fits perfectly (1-2 sentences, open-ended)\n\n"
                "Output ONLY the strict JSON object, nothing else."
            )

            messages = [
                {"role": "system", "content": "You are an expert at creating realistic online conversation scenarios. Output only a JSON object containing 'situation' and 'topic' fields for digital communication contexts. No extra text or formatting."},
                {"role": "user", "content": instruction}
            ]

            response = await client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                response_format=response_format,
                max_tokens=768,
                temperature=0.8
            )

            scenario_data = GeneratedScenario.model_validate_json(response.choices[0].message.content)

            return ConversationContext(
                persona1=p1_data,
                persona2=p2_data,
                relationship=relationship,
                situation=scenario_data.situation,
                topic=scenario_data.topic,
            )

        except Exception as e:
            print(f"⚠️ Error generating an experience: {e}. Retrying with another task.")
            await asyncio.sleep(2)
            return None

async def main():
    # Load all components
    components_data = load_components(COMPONENTS_FILE)
    component_manager = ComponentManager(components_data)

    if len(PERSONAS_FILE) > 0:
        print(f"Loading personas from file: {PERSONAS_FILE}...")
        persona_pool = load_personas_from_jsonl(PERSONAS_FILE)
    else:
        print(f"Loading personas from dataset: {PERSONAS_DATASET}...")
        dataset = load_dataset(PERSONAS_DATASET, split="train")
        persona_pool = [Persona(**item) for item in dataset]
        print(f"✅ Successfully loaded {len(persona_pool)} personas from dataset.")
    seed_experiences = load_seed_experiences(SEED_EXPERIENCES_FILE)

    experiences_pool = seed_experiences.copy()

    if len(persona_pool) < 2:
        print("⚠️ Not enough personas to generate experiences. Need at least 2.")
        return
    if not seed_experiences:
        print("⚠️ No seed experiences loaded. Cannot proceed with few-shot prompting.")
        return

    precomputed_pairs, precomputed_weights = precompute_weighted_pairs(persona_pool)

    timestamp = int(time.time())
    output_filename = f"data/experiences/experiences_{MODEL_NAME.split('/')[-1]}_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\n🚀 Starting ONLINE experience generation. Target: {TARGET_N} experiences.")
    print(f"Concurrency: {CONCURRENCY}, Checkpoint: {CHECKPOINT_EVERY}, Anti-Drift Reset: {RESET_EVERY}")
    print(f"Output will be saved to: {output_filename}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks: Set[asyncio.Task] = set()
    results_buffer = []
    successful_generations = 0

    while successful_generations < TARGET_N:
        while len(tasks) < CONCURRENCY and successful_generations + len(tasks) < TARGET_N:
            current_iteration = successful_generations + len(tasks)
            if RESET_EVERY > 0 and current_iteration > 0 and current_iteration % RESET_EVERY == 0:
                reference_pool = seed_experiences
                print(f"--- 🔄 Iteration {current_iteration}: Resetting reference pool to seeds to prevent drift ---")
            else:
                reference_pool = experiences_pool

            task = asyncio.create_task(
                generate_one_experience(
                    semaphore,
                    precomputed_pairs,
                    precomputed_weights,
                    reference_pool,
                    component_manager
                )
            )
            tasks.add(task)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for future in done:
            result = await future
            tasks.remove(future)

            if result:
                successful_generations += 1
                experiences_pool.append(result)
                results_buffer.append(result)
                print(f"✅ ({successful_generations}/{TARGET_N}) Generated: {result.persona1.name} & {result.persona2.name} - {result.relationship}")

                if successful_generations % CHECKPOINT_EVERY == 0 and results_buffer:
                    print(f"\n--- CHECKPOINT: Saving {len(results_buffer)} experiences to {output_filename}... ---")
                    for context in results_buffer:
                        save_to_jsonl(context.model_dump(), output_filename)
                    results_buffer.clear()
                    print("--- ✅ CHECKPOINT Complete. ---\n")

    if results_buffer:
        print(f"\n--- FINAL SAVE: Saving {len(results_buffer)} remaining experiences to file... ---")
        for context in results_buffer:
            save_to_jsonl(context.model_dump(), output_filename)
        print("--- ✅ FINAL SAVE Complete. ---")

    print("\n-----------------------------------------")
    print(f"✅ Target of {TARGET_N} completed. {successful_generations} online experiences successfully generated.")
    print(f"Final data saved in {output_filename}")
    print("-----------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
