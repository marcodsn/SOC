# Synthetic Online Conversations (SOC)

This repository contains the generation code and information for the **Synthetic Online Conversations (SOC)** dataset.

SOC is a collection of over 1,180 synthetically generated, multi-turn online conversations. Each dialogue is a complete interaction between two fictional personas drawn from the [Synthetic Persona Bank (SPB-2508)](https://huggingface.co/datasets/marcodsn/SPB-2508) dataset.

The dataset was created using a multi-stage programmatic pipeline driven by the [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) large language model. This process was designed to produce natural, context-aware, and stylistically consistent dialogues that include human-like imperfections, realistic conflict, and simulated multimedia elements via special tags (e.g., `<image>`, `<delay>`).

This dataset is ideal for training and evaluating language models on tasks requiring:
- Long-form, context-aware dialogue generation
- Persona consistency and role-playing
- Understanding nuanced online interactions

**Explore the conversations visually with the [SOC Visualizer](https://huggingface.co/spaces/marcodsn/SOC_Visualizer) on Hugging Face Spaces!**

## Key Features

-   **Persona-Grounded:** Every conversation is based on two detailed, pre-defined personas, ensuring deep consistency.
-   **Rich Context:** Dialogues are initiated from a specific `relationship`, `situation`, and `topic`, providing a full backstory for the interaction.
-   **Realistic Dialogue:** The generation process encouraged human-like imperfections such as typos, topic drift, and varying effort levels.
-   **Simulated Multimedia & Actions:** Special XML-like tags like `<image>`, `<gif>`, `<delay>`, and `<end/>` are used to simulate a richer chat environment.

## Getting Started

You can easily load and use this dataset with the 🤗 `datasets` library.

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
dataset = load_dataset("marcodsn/SOC-2508")

# The dataset has a single 'train' split
train_dataset = dataset['train']

# Print the first conversation
print(train_dataset[0])
```

## Dataset Structure

The dataset is a single JSONL file (`data.jsonl`), where each line is a JSON object representing a complete conversation.

### Data Schema

Each JSON object has the following structure:

```json
{
  "chat_id": "4436437d368e4325a7c1c6f7092c2d9e_f8e1b2a3c4d5e6f7g8h9i0j1k2l3m4n5_1754636647",
  "experience": {
    "persona1": {
      "name": "Elias Vance",
      "username": "quantum_scribe",
      "age": 42,
      "traits": ["analytical", "introspective", "witty", "reserved"],
      "background": "A theoretical physicist who, after a breakthrough, left academia to write science fiction novels from a secluded cabin. He's currently grappling with a severe case of writer's block for his second book.",
      "chatting_style": "Uses precise language and often employs metaphors from physics. Tends to write in well-structured, complete sentences, even in casual chat.",
      "model": "Qwen3-235B-A22B-Instruct-2507",
      "id": "4436437d368e4325a7c1c6f7092c2d9e"
    },
    "persona2": {
      "name": "Luna Reyes",
      "username": "StardustSketcher",
      "age": 28,
      "traits": ["creative", "optimistic", "daydreamer", "empathetic"],
      "background": "A freelance digital artist who illustrates children's books and streams her drawing process online. She finds inspiration in mythology and the night sky.",
      "chatting_style": "Uses a lot of emojis and kaomoji (´｡• ᵕ •｡`). Her messages are often short, enthusiastic, and full of creative typos.",
      "model": "Qwen3-235B-A22B-Instruct-2507",
      "id": "f8e1b2a3c4d5e6f7g8h9i0j1k2l3m4n5"
    },
    "relationship": "Strangers who met in a 'Vintage Sci-Fi Book Club' Discord server.",
    "situation": "Elias posted a message asking for recommendations to overcome writer's block, and Luna, a fellow member, decided to DM him directly to offer some creative, non-traditional advice.",
    "topic": "I saw your post in the #writing- woes channel and had a few weird ideas that might help! Mind if I share?",
    "id": "c1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6"
  },
  "chat_parts": [
    {
      "sender": "f8e1b2a3c4d5e6f7g8h9i0j1k2l3m4n5",
      "messages": [
        "Hiii Elias! Saw your post in #writing-woes. I know the feeling (art block is the wooooorst 😭).",
        "Had a few maybe-weird ideas if you're open to them? ✨"
      ]
    },
    {
      "sender": "4436437d368e4325a7c1c6f7092c2d9e",
      "messages": [
        "<delay minutes=\"5\"/>",
        "Hello, Luna. I appreciate the outreach. At this point, I am receptive to any and all suggestions, regardless of their position on the conventionality spectrum."
      ]
    }
  ],
  "model": "Qwen3-235B-A22B-Instruct-2507"
}
```

### Field Descriptions

-   `chat_id` (string): A unique identifier for the conversation.
-   `experience` (object): The full context for the conversation.
    -   `persona1` & `persona2` (object): Complete persona objects from the SPB-2508 dataset.
    -   `relationship` (string): How the two personas know each other.
    -   `situation` (string): The specific context or reason for the conversation starting.
    -   `topic` (string): The opening line or subject that kicks off the dialogue.
-   `chat_parts` (list of objects): A list of turns in the conversation.
    -   `sender` (string): The ID of the persona sending the messages in this turn.
    -   `messages` (list of strings): The messages sent in this turn. Can include special tags.
-   `model` (string): The model used to generate the conversation.

## Generation Process

The conversations were generated using a three-stage pipeline inspired by [ConvoGen](https://huggingface.co/papers/2503.17460):

1.  **Stage 1: Experience Generation**
    -   Two personas were selected from the `SPB-2508` pool, with a weighting system to favor pairing personas of similar age.
    -   A `relationship` context (e.g., "old friends," "strangers in a gaming lobby") was dynamically constructed.
    -   The LLM generated a unique `situation` and starting `topic` based on the personas and their relationship, using few-shot examples to encourage novelty.

2.  **Stage 2: Conversational Rollout**
    -   The generated "experience" was used as the master prompt for a turn-by-turn dialogue generation.
    -   The LLM was instructed to alternate between personas, maintaining consistency with their background and chatting style.
    -   Rich instructions were provided to encourage realism, including:
        -   **Human Imperfection:** Allowing for typos, grammar mistakes, and topic drift.
        -   **Realistic Conflict:** Handling disagreements without immediate, clean resolutions.
        -   **Special Tags:** Using tags like `<image>`, `<gif>`, `<delay>`, and `<end/>` to simulate real chat behavior.

3.  **Stage 3: Post-Processing**
    -   Raw generated chats were collected and cleaned.
    -   Scripts were used to remove duplicates, filter out conversations that were too short, and scrub generation artifacts (like injected speaker names).

The generation scripts and seed data can be found in this repository: [github.com/marcodsn/SOC/tree/2508](https://github.com/marcodsn/SOC/tree/2508).

## Known Limitations

-   **Synthetic Nature**: The dialogues are designed for realism but may not capture the full chaotic unpredictability of genuine human interaction.
-   **Inherited Bias**: Biases or stereotypes from the source `SPB-2508` dataset or the base LLM may be present and amplified.
-   **Tag Frequency**: The use of special tags (`<image>`, `<delay>`, etc.) is not uniform, as their inclusion was left to the model's discretion.
-   **Conversation Endings**: The `<end/>` tag might lead to some conversations concluding more formulaically than they would in the wild.
-   **Instruction Following**: The LLM is not perfect, and some issues from poor instruction following may exist. The `<end/>` tag is sometimes used prematurely.

## Licensing

-   The **dataset** is released under the [**Creative Commons Attribution 4.0 International (CC BY 4.0)**](https://creativecommons.org/licenses/by/4.0/) license.
-   The **generation code** in this repository is released under the [**Apache 2.0 License**](LICENSE).

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@misc{marcodsn_2025_SOC2508,
  title     = {Synthetic Online Conversations},
  author    = {Marco De Santis},
  year      = {2025},
  month     = {August},
  url       = {https://huggingface.co/datasets/marcodsn/SOC-2508},
}
```
