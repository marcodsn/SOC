#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub",
#   "pydantic"
# ]
# ///

"""
Full, self-contained chat-generation pipeline
---------------------------------------------

• Loads "experiences" (scenario records) from a `.jsonl` file.
• Uses an LLM via the HF Inference Providers to iteratively generate a
  back-and-forth chat between the two personas in each experience.
• Streams the finished `Chat` objects to an output `.jsonl` file, one
  line per chat, checkpointing regularly so you can resume safely.

Environment variables
---------------------
HF_TOKEN   HuggingFace access token with permissions for the model.
"""

# ───────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

class Config:
    """All script settings grouped in a single class."""
    EXPERIENCES_FILES = [
        "data/experiences/experiences_Qwen3-235B-A22B-Instruct-2507_1754636647.jsonl",
        "data/experiences/experiences_Qwen3-235B-A22B-Instruct-2507_1754637814.jsonl",
    ]
    MODEL_NAME       = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    OUTPUT_FILE      = f"data/chats/chats_{MODEL_NAME.split('/')[-1]}_{int(time.time())}.jsonl"
    PROCESSED_FILE   = "data/chats/cache/processed_experiences.json"

    CONCURRENCY      = 10
    CHECKPOINT_EVERY = 1
    TARGET_CHATS     = 1200
    GLOBAL_MAX_TOKENS = 32_000
    GLOBAL_MAX_TURNS  = 200

    ADDITIONAL_INSTRUCTIONS = """
Additional instructions:
1. You can generate more than one message in a single turn (like in a natural conversation where a person can write 1, 2 or more consecutive messages before getting a reply), but you must output them as a single JSON object with a list of messages, and do not abuse this feature! Aim for 1 to 3 messages.
2. You can use XML-like tags to simulate multimedia content, such as <audio>audio transcription/description</audio>, <image>image description</image>, <gif>gif description</gif>, etc.
3. A message containing a multimedia tag shall not contain any other text; you can add a caption after closing the tag for images, gifs and videos, but not for audio. (ok example: ["Hi!", "<image>a cute kitten</image> This is my cat!"]; not ok example: ["Hi! <image>a cute kitten</image> This is my cat!"])
4. If the conversation naturally comes to an end (e.g. the topic has been exhausted, and you are not able to continue the conversation with a new topic), you can indicate this by adding <end/> at the end of a message. This will signal the end of the conversations.
5. As in natural conversations it may happen that a person cannot reply immediately sometimes, you can simulate this by adding a delay like <delay minutes="30" hours="6" days="1"/> at the start of a message, indicating that the person took some time to reply (delay params are additive).
"""

    REALISM_INSTRUCTIONS = """
Keep the conversation natural and **imperfectly human**. Real chats are often messy, contain typos, change topics abruptly, and show varying levels of effort. Allow for these imperfections:
- **Distractions:** Acknowledge the persona's background. Are they at work? Dealing with kids? Studying? Out with friends? Let that reality bleed into the conversation.
- **Topic Drift:** Don't feel forced to stick to the main topic. A character might remember a funny story, ask a random question, or comment on their immediate surroundings.
- **Variable Effort:** Not every reply needs to be a thoughtful paragraph. Sometimes a short reply, a typo-filled rush of words, or just a GIF is more realistic. Embrace the ebb and flow.
"""

    CONFLICT_INSTRUCTIONS = """
When the situation involves a disagreement or fight, let the argument breathe and develop realistically. Avoid immediate de-escalation or easy resolutions. Real arguments are fueled by emotion, misunderstanding, and ego.
- Focus on the Emotional Core: What is the persona really upset about beneath the surface? Is it pride, insecurity, feeling disrespected, a clash of values, or just exhaustion? Let their replies be driven by this underlying emotion, not just the surface-level topic.
- Embrace Defensiveness and Stubbornness: Instead of immediately conceding a point, a persona might double down, justify their actions, or deflect criticism. People rarely enjoy admitting they're wrong, especially mid-argument. Let them resist.
- Introduce Misunderstanding: A persona might read into a neutral comment, take a joke personally, or filter a message through their own anxieties (e.g., a stressed person sees an attack where none was intended). The argument can escalate from a simple miscommunication.
- Utilize Passive Aggression: Conflict isn't always a direct shouting match. It can manifest as sarcasm, backhanded compliments ('Well, it's great that you have time for that...'), pointed silence (a long delay before a terse reply), or guilt-tripping.
- Allow for Messy Resolutions: An argument doesn't have to end with a perfect apology and mutual understanding. It might just fizzle out, end with a begrudging 'fine, whatever,' or leave a lingering tension that affects the next message. The goal is a realistic outcome, not always a happy one.
"""


# Schema definitions (Pydantic)
class Persona(BaseModel):
    name: str
    username: str | None = None
    age: int
    traits: list[str]
    background: str
    chatting_style: str
    model: str
    id: str

class Experience(BaseModel):
    persona1: Persona
    persona2: Persona
    relationship: str
    situation: str
    topic: str
    id: str

class LLMChatPart(BaseModel):
    messages: List[str] = Field(
        default_factory=list,
        description="Messages written by the sender in this turn. A single message is most common. Example: [\"Hey, what's up?\"]"
    )

class ChatPart(BaseModel):
    sender: str  # Persona's id
    messages: List[str]

class Chat(BaseModel):
    chat_id: str
    experience: Experience
    chat_parts: List[ChatPart]
    model: str


# Chat Generation Pipeline
class ChatGenerator:
    """Orchestrates the entire chat generation process."""

    def __init__(self, config: Config):
        self.config = config
        self.client = self._initialize_client()
        self.response_format = self._build_response_format()

    def _initialize_client(self) -> AsyncInferenceClient:
        if "HF_TOKEN" not in os.environ:
            raise RuntimeError("Please set the HF_TOKEN env variable.")
        return AsyncInferenceClient(
            provider="together", api_key=os.environ["HF_TOKEN"]
        )

    def _build_response_format(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "GeneratedScenario",
                "schema": LLMChatPart.model_json_schema(),
                "strict": True,
            },
        }

    async def __aenter__(self) -> "ChatGenerator":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logging.info("Gracefully closing client connections...")
        await self.client.close()
        logging.info("Client closed.")

    def _load_experiences(self) -> list[Experience]:
        """Load all experiences from the configured files."""
        all_experiences = []
        for file_path in self.config.EXPERIENCES_FILES:
            path = Path(file_path)
            if not path.exists():
                logging.warning(f"Experiences file not found: {path}")
                continue
            with path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        all_experiences.append(Experience(**json.loads(line)))
                    except Exception as err:
                        logging.warning(f"Skipping bad line {idx} in {path}: {err}")
            logging.info(f"Loaded {len(all_experiences)} experiences from {path}")
        return all_experiences

    def _load_processed_ids(self) -> set[str]:
        """Load the set of already processed experience IDs."""
        path = Path(self.config.PROCESSED_FILE)
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                processed_ids = set(data.get("processed_experience_ids", []))
                logging.info(f"Loaded {len(processed_ids)} processed IDs from {path}")
                return processed_ids
        except Exception as err:
            logging.warning(f"Error loading processed experiences file: {err}")
            return set()

    def _save_processed_ids(self, processed_ids: set[str]) -> None:
        """Save the set of processed experience IDs."""
        path = Path(self.config.PROCESSED_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {"processed_experience_ids": sorted(list(processed_ids))}
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as err:
            logging.error(f"Error saving processed experiences: {err}")

    def _save_chat(self, chat: Chat) -> None:
        """Append a single finished chat to the output file."""
        path = Path(self.config.OUTPUT_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(chat.model_dump_json() + "\n")

    def _build_prompt_messages(self, chat: Chat, sender: Persona) -> list[dict]:
        """Construct the prompt for the LLM."""
        personas_names = {p.id: p.name for p in [chat.experience.persona1, chat.experience.persona2]}
        chat_state = "\n".join(
            f"{personas_names[part.sender]}: {msg}"
            for part in chat.chat_parts for msg in part.messages
        ) or "The chat is empty."

        # Helper to create a lean representation of the experience
        def lean_experience(exp: Experience) -> dict:
            return {
                "persona1": exp.persona1.model_dump(exclude={"username", "model", "id"}),
                "persona2": exp.persona2.model_dump(exclude={"username", "model", "id"}),
                "relationship": exp.relationship,
                "situation": exp.situation,
                "topic": exp.topic,
            }

        return [
            {"role": "system", "content": "You are a creative conversation generator. Return ONLY a single JSON object that matches the provided schema. No additional commentary."},
            {"role": "user", "content": (
                f"Generate the next chat part from {sender.name}.\n\n"
                f"Experience:\n{json.dumps(lean_experience(chat.experience), indent=2)}\n\n"
                f"Current chat state:\n{chat_state}\n\n"
                f"{self.config.ADDITIONAL_INSTRUCTIONS}\n\n"
                f"{self.config.REALISM_INSTRUCTIONS}\n\n"
                f"{self.config.CONFLICT_INSTRUCTIONS}\n\n"
                "If your persona is a nerd but your interlocutor is not, do not use overly complex language or jargon, and viceversa if your interlocutor uses jargon or info that you are not supposed to know, ask for clarifications.\n"
                "Remember: output must be the strict JSON object and nothing else."
            )},
        ]

    async def _generate_turn(self, semaphore: asyncio.Semaphore, chat: Chat, sender: Persona) -> ChatPart | None:
        """Generate one turn of the conversation."""
        async with semaphore:
            messages = self._build_prompt_messages(chat, sender)
            turn_n = len(chat.chat_parts) + 1
            try:
                response = await self.client.chat_completion(
                    messages=messages,
                    model=self.config.MODEL_NAME,
                    response_format=self.response_format,
                    max_tokens=512,
                    temperature=0.8,
                )
                content = response.choices[0].message.content

                if turn_n <= 4 and "<end/>" in content:
                    logging.warning(f"Removing premature <end/> tag in turn {turn_n} for chat {chat.chat_id}")
                    content = content.replace("<end/>", "")

                llm_part = LLMChatPart.model_validate_json(content)
                # Filter out any accidentally generated empty messages
                valid_messages = [msg for msg in llm_part.messages if msg.strip()]
                return ChatPart(sender=sender.id, messages=valid_messages)
            except Exception as err:
                logging.error(f"API error on turn {turn_n} for chat {chat.chat_id}: {err}")
                await asyncio.sleep(2)  # Back-off on API error
                return None

    async def _generate_chat(self, semaphore: asyncio.Semaphore, experience: Experience) -> Chat:
        """Orchestrate the generation of a full chat for one experience."""
        chat = Chat(
            chat_id=f"{experience.persona1.id}_{experience.persona2.id}_{int(time.time())}",
            experience=experience,
            chat_parts=[],
            model=self.config.MODEL_NAME.split("/")[-1]
        )

        speakers = itertools.cycle([experience.persona1, experience.persona2])
        total_tokens = 0

        for turn_no in range(1, self.config.GLOBAL_MAX_TURNS + 1):
            sender = next(speakers)
            part = await self._generate_turn(semaphore, chat, sender)

            if part is None:  # Unrecoverable API failure for this turn
                logging.error(f"Stopping chat {chat.chat_id} due to generation failure.")
                break

            if not part.messages: # LLM returned an empty message list
                logging.warning(f"Turn {turn_no} for chat {chat.chat_id} resulted in no messages. Ending chat.")
                break

            chat.chat_parts.append(part)
            logging.info(f"Turn {turn_no} complete for chat {chat.chat_id}")

            joined_messages = " ".join(part.messages)
            total_tokens += int(len(joined_messages.split()) * 0.75) # Crude token estimation

            if "<end/>" in joined_messages or total_tokens >= self.config.GLOBAL_MAX_TOKENS:
                break

        return chat

    async def run(self):
        """The main execution loop for the generation pipeline."""
        all_experiences = self._load_experiences()
        processed_ids = self._load_processed_ids()
        experiences_to_process = [exp for exp in all_experiences if exp.id not in processed_ids]

        logging.info(f"Total experiences loaded: {len(all_experiences)}")
        logging.info(f"Already processed: {len(processed_ids)}")
        logging.info(f"Remaining to process: {len(experiences_to_process)}")

        if not experiences_to_process:
            logging.info("All experiences have already been processed!")
            return

        target_count = min(len(experiences_to_process), self.config.TARGET_CHATS)
        if target_count <= 0:
            logging.info(f"Target chats is {self.config.TARGET_CHATS}, nothing to generate.")
            return

        logging.info(f"Creating {target_count} generation tasks with concurrency={self.config.CONCURRENCY}.")
        semaphore = asyncio.Semaphore(self.config.CONCURRENCY)
        tasks = [
            self._generate_chat(semaphore, exp)
            for exp in experiences_to_process[:target_count]
        ]

        completed_count = 0
        for future in asyncio.as_completed(tasks):
            try:
                chat = await future
                completed_count += 1
                total_messages = sum(len(part.messages) for part in chat.chat_parts)

                logging.info(
                    f"✅ ({completed_count}/{target_count}) Finished chat {chat.chat_id} with "
                    f"{total_messages} messages in {len(chat.chat_parts)} turns."
                )

                self._save_chat(chat)
                processed_ids.add(chat.experience.id)

                if completed_count % self.config.CHECKPOINT_EVERY == 0 or completed_count == target_count:
                    logging.info(f"💾 Checkpoint: {completed_count}/{target_count} chats saved.")
                    self._save_processed_ids(processed_ids)

            except Exception as e:
                logging.error(f"A chat generation task failed unexpectedly: {e}", exc_info=True)


async def main() -> None:
    """Initializes and runs the chat generation pipeline."""
    config = Config()
    try:
        async with ChatGenerator(config) as generator:
            await generator.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.warning("\nInterrupted by user. Exiting gracefully.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
