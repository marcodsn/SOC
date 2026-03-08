# SOC — Synthetic Online Conversations

A pipeline for generating realistic, persona-grounded synthetic dialogue datasets using large language models. SOC produces multi-turn conversations that feel like real text-message exchanges between people with distinct personalities, backgrounds, and communication styles.

## Overview

SOC builds conversations bottom-up: from **people**, to **situations**, to **chats**, with each stage grounding the next.

```text
Seed data
  → Persona generation     (iterative + diversity resets)
  → Experience generation  (pairing + relationship + situation + trigger)
  → Chat generation        (multi-turn, media tags, timestamp pacing)
```

The pipeline takes heavy inspiration from [ConvoGen](https://huggingface.co/papers/2503.17460) (Gody et al.), adopting its experience-first architecture and iterative sampling, while extending it with:

- **Timestamps and multi-message turns** baked directly into the generation loop
- **Typed media attachments** (images, audio, video, stickers) reflecting real online conversations
- **Rolling summarization memory** to manage long-context drift
- **Region profiles** for scoped generation (e.g. "West", "East Asia", "Global")
- **Prompt-caching optimisation** via stable system-message prefixes

## Setup

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone and enter the project
git clone https://github.com/marcodsn/SOC.git
cd SOC

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Variables

Create a `.env` file in the project root with your API keys:

```env
# At least one of these is needed, depending on your provider
HF_TOKEN=hf_...              # HuggingFace / local vLLM
NVIDIA_API_KEY=nvapi-...     # NVIDIA NIM
MODAL_TOKEN=...              # Modal
MISTRAL_API_KEY=...          # Mistral
```

## Configuration

All pipeline parameters are centralised in two YAML files:

| File | Purpose |
|------|---------|
| `conf/config.yaml` | Region profiles, language settings, generation parameters, paths |
| `conf/models.yaml` | Named model presets, provider endpoints |

CLI flags always override config file values. See the files for full documentation of every option.

### Region Profiles

Region profiles restrict persona sampling to a geographic subset. Available profiles:

| Profile | Description |
|---------|-------------|
| `global` | All 42 regions, population-weighted (default) |
| `west` | North America + Europe |
| `east_asia` | East and Southeast Asia |
| `south_asia` | South Asia and Iran |
| `middle_east` | Middle East and North Africa |
| `africa` | Sub-Saharan Africa |
| `latam` | Latin America |
| `english_speaking` | Primarily English-speaking countries |

Custom profiles can be added directly to `conf/config.yaml`.

### Language Settings

The `language` section in `config.yaml` controls:

- **`lingua_franca`**: The shared language all conversations are conducted in (default: English)
- **`native_flavor`**: Whether personas may sprinkle in native-language words
- **`max_native_words_per_message`**: Cap on native words per message (default: 2)

## Usage

### 1. Generate Personas

```bash
# Basic — 50 personas using config defaults
python scripts/01_gen_personas.py --num 50

# West-only personas with a specific model preset
python scripts/01_gen_personas.py --num 100 --region-profile west --preset kimi_k2

# Custom age distribution for younger demographics
python scripts/01_gen_personas.py --num 50 --age-mean 22 --age-std 3

# Using a specific provider and model directly
python scripts/01_gen_personas.py --num 50 --model "moonshotai/Kimi-K2.5:fireworks-ai" --provider huggingface
```

### 2. Merge Personas

```bash
python scripts/01e_merge_personas.py
```

This assigns unique IDs and merges all individual persona files into `data/personas/generated/data.jsonl`.

### 3. Generate Experiences

```bash
# Basic — 50 experiences from the merged persona pool
python scripts/02_gen_experiences.py --num 50

# With higher same-region pairing probability
python scripts/02_gen_experiences.py --num 50 --same-region-prob 0.7

# With a specific preset
python scripts/02_gen_experiences.py --num 100 --preset glm5
```

### 4. Merge Experiences

```bash
python scripts/02e_merge_experiences.py
```

### 5. Generate Conversations

```bash
# Basic — process all experiences
python scripts/03_gen_conversations.py

# First 20 experiences, freeform style only
python scripts/03_gen_conversations.py --num 20 --style freeform

# With a separate cheaper model for summarization
python scripts/03_gen_conversations.py --preset kimi_k25 --summarizer-preset local_small

# Higher turn ceiling for longer conversations
python scripts/03_gen_conversations.py --max-turns 80
```

### 6. Merge Conversations

```bash
python scripts/03e_merge_conversations.py
```

## Analysis & Validation

### Dataset Statistics

```bash
# Persona statistics (demographics, text quality, region coverage)
python scripts/01e_stats_personas.py
python scripts/01e_stats_personas.py --input data/personas/generated/data.jsonl --verbose

# Conversation statistics (turns, timing, styles, token usage, quality flags)
python scripts/03e_stats_conversations.py
python scripts/03e_stats_conversations.py --input "data/conversations/generated/conversations_*.jsonl" --glob
```

### Data Validation

The validation script checks structural integrity, content quality, and cross-references across all pipeline stages:

```bash
# Full validation
python scripts/04_validate_data.py

# Quick validation on a sample
python scripts/04_validate_data.py --sample 50

# Verbose output with all passed checks
python scripts/04_validate_data.py --verbose

# Strict mode — exits with code 1 on any failure (for CI)
python scripts/04_validate_data.py --strict

# Skip specific stages
python scripts/04_validate_data.py --skip-conversations --skip-crossref
```

The validator checks for:
- JSON parse errors and structural integrity
- Missing sections in persona character cards
- LLM artifacts ("As an AI", action narration, etc.)
- Knowledge dumping (messages exceeding word count thresholds)
- Timestamp ordering and validity
- Em dashes in messages (forbidden by prompt)
- Style and cadence distribution balance
- Cross-reference integrity between personas and experiences

## Architecture

### Prompt Caching Strategy

Most providers with automatic prompt caching (vLLM prefix caching, Anthropic, OpenAI, DeepSeek) operate on shared byte-identical prefixes. SOC maximises cache hits by:

1. **Static system messages** — all instruction text (format specs, rules, style guides) lives in the `system` message, which is identical across every call within a generation stage
2. **Variable user messages** — only per-call content (few-shot examples, persona data, conversation history) goes in the `user` message
3. **`LLMClient.build_messages()`** helper enforces this split so callers can't accidentally interleave static and dynamic content

This means the first call in a batch pays the full prompt processing cost, but subsequent calls in the same stage can reuse the cached KV prefix.

### Directory Structure

```
SOC/
├── conf/
│   ├── config.yaml              # Pipeline configuration
│   ├── models.yaml              # Model presets and provider endpoints
│   └── prompts/
│       ├── persona_generation.j2    # Persona user-message template
│       ├── experience_generation.j2 # Experience user-message template
│       ├── turn_generation.j2       # Turn user-message template
│       └── summarization.j2         # Rolling summary template
├── data/
│   ├── personas/
│   │   ├── seed/                # Hand-written seed personas (*.txt)
│   │   └── generated/           # Generated persona JSONL files
│   ├── experiences/
│   │   ├── seed/                # Hand-written seed experiences (*.txt)
│   │   └── generated/           # Generated experience JSONL files
│   ├── conversations/
│   │   └── generated/           # Generated conversation JSONL files
│   └── stats/
│       ├── demographics/
│       │   └── regions.yaml     # Population-weighted region + subregion data
│       └── names/               # Per-region name frequency CSVs
├── scripts/
│   ├── 01_gen_personas.py       # Step 1: generate personas
│   ├── 01e_merge_personas.py    # Merge persona files, assign IDs
│   ├── 01e_stats_personas.py    # Persona dataset statistics
│   ├── 02_gen_experiences.py    # Step 2: generate experiences
│   ├── 02e_merge_experiences.py # Merge experience files, assign IDs
│   ├── 03_gen_conversations.py  # Step 3: generate conversations
│   ├── 03e_merge_conversations.py # Merge conversation files
│   ├── 03e_stats_conversations.py # Conversation dataset statistics
│   └── 04_validate_data.py      # Data validation and quality tests
└── src/soc_2602/
    ├── generation/
    │   ├── persona_generator.py
    │   ├── experience_generator.py
    │   └── conversation_generator.py
    ├── llm/
    │   └── client.py            # Multi-provider LLM client
    └── utils/
        ├── config.py            # YAML config loader
        └── sampling.py          # StatsEngine (demographics, names, ages)
```

### Key Design Decisions

- **Experience-first architecture**: Conversations are grounded in a generated experience that specifies the relationship, situation, emotional states, and topic roadmap — words follow from context, not the other way around
- **Rolling few-shot window**: Both persona and experience generation use a sliding window of recent outputs as negative examples to prevent style collapse, seeded with hand-written examples during warmup
- **Turn-level state tracking**: A `<state>` block in each turn tracks the current topic, turns remaining, and next topic, giving the model explicit structure to follow without making conversations feel robotic
- **Instant event injection**: Background events (a notification, a memory, a roommate interrupting) are injected with low probability (~5%) to simulate real-life friction
- **Rolling summarization**: When the active conversation window exceeds a threshold, older turns are compressed into a rolling summary by a (optionally cheaper) summarizer model
- **Media-aware turns**: Each message has a `type` attribute (`text`, `image`, `audio`, `video`, `sticker`) reflecting how real online conversations mix media types

## Datasets

- [SOC-2602](https://huggingface.co/datasets/marcodsn/SOC-2602) — Synthetic Online Conversations
- [SPB-2602](https://huggingface.co/datasets/marcodsn/SPB-2602) — Synthetic Persona Bank

## Citation

```bibtex
@misc{marcodsn_2026_SOC2602,
  title  = {Synthetic Online Conversations},
  author = {Marco De Santis},
  year   = {2026},
  month  = {February},
  url    = {https://huggingface.co/datasets/marcodsn/SOC-2602},
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)