# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "torch",
#     "transformers>=4.53.0",
#     "accelerate",
# ]
# ///

import os
import sys
import logging
import argparse
import json
import re
from datetime import datetime
from typing import Optional, Dict, List, Any

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import DatasetCard, get_token, login
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Language configurations with ISO 639 codes
LANGUAGE_CONFIGS = {
    "es": {"name": "Spanish", "native_name": "Español"},
    "fr": {"name": "French", "native_name": "Français"},
    "de": {"name": "German", "native_name": "Deutsch"},
    "it": {"name": "Italian", "native_name": "Italiano"},
    "pt": {"name": "Portuguese", "native_name": "Português"},
    "zh": {"name": "Chinese", "native_name": "中文"},
    "ja": {"name": "Japanese", "native_name": "日本語"},
    "ko": {"name": "Korean", "native_name": "한국어"},
    "ru": {"name": "Russian", "native_name": "Русский"},
    "ar": {"name": "Arabic", "native_name": "العربية"},
    "hi": {"name": "Hindi", "native_name": "हिन्दी"},
    "pl": {"name": "Polish", "native_name": "Polski"},
    "nl": {"name": "Dutch", "native_name": "Nederlands"},
    "sv": {"name": "Swedish", "native_name": "Svenska"},
    "da": {"name": "Danish", "native_name": "Dansk"},
    "no": {"name": "Norwegian", "native_name": "Norsk"},
    "fi": {"name": "Finnish", "native_name": "Suomi"},
    "tr": {"name": "Turkish", "native_name": "Türkçe"},
    "cs": {"name": "Czech", "native_name": "Čeština"},
    "hu": {"name": "Hungarian", "native_name": "Magyar"}
}

class Config:
    """All script settings grouped in a single class."""
    SOURCE_DATASET = "marcodsn/SOC-2508"
    OUTPUT_DATASET = "marcodsn/SOC-2508-MULTI"
    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
    TARGET_LANGUAGES = ["fr", "es", "de", "it", "pt"]  # Default languages to translate to


def check_gpu_availability() -> int:
    """Check if CUDA is available and return the number of GPUs."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Please run on a machine with NVIDIA GPU or use HF Jobs with GPU flavor."
        )
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


def create_translation_prompt(data: Dict[str, Any], target_language: str) -> str:
    """Create a translation prompt for a single language."""
    lang_info = LANGUAGE_CONFIGS[target_language]

    # Extract text fields that need translation
    persona1 = data["experience"]["persona1"]
    persona2 = data["experience"]["persona2"]
    experience = data["experience"]

    # Build the prompt
    prompt = f"""You are a professional translator. Translate the following English text into {lang_info['name']} ({lang_info['native_name']}).

IMPORTANT RULES:
1. Preserve ALL formatting, including <image>, <video>, <audio>, <gif>, <delay>, and <end/> tags exactly as they appear
2. Only translate the actual text content, NOT the tags or special formatting
3. Maintain the natural conversational tone and personality
4. Keep cultural context appropriate for the target language
5. Respond ONLY with a valid JSON object in the exact format shown

Input data to translate:

```
{json.dumps(data, indent=2, ensure_ascii=False)}
```

Translate to {lang_info['name']} and respond with a JSON object containing ONLY the translated fields in this exact structure:

{{
  "persona1": {{
    "traits": [list of translated traits],
    "background": "translated background",
    "chatting_style": "translated chatting style"
  }},
  "persona2": {{
    "traits": [list of translated traits],
    "background": "translated background",
    "chatting_style": "translated chatting style"
  }},
  "relationship": "translated relationship",
  "situation": "translated situation",
  "topic": "translated topic",
  "messages": [
    [list of translated messages for first chat_part, preserving <image>, <video>, <audio>, <gif>, <delay>, <end/> tags],
    [list of translated messages for second chat_part, preserving tags],
    ... (one array per chat_part)
  ]
}}

Remember: Preserve ALL , , and  tags exactly. Only translate the text content."""

    return prompt


def parse_translation_response(response: str, target_language: str) -> Dict[str, Any]:
    """Parse the translation response and extract the translated content."""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # If no JSON found, try parsing the entire response
            return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse translation response for {target_language}: {e}")
        logger.error(f"Response: {response[:500]}...")
        return None


def merge_translation(original: Dict[str, Any], translation: Dict[str, Any], language_code: str) -> Dict[str, Any]:
    """Merge a single language translation into the multilingual object."""
    if translation is None:
        return original

    # Initialize multilingual structure if this is the first language
    if "en" not in original["experience"]["persona1"]["traits"]:
        # Convert original to multilingual format
        result = json.loads(json.dumps(original))  # Deep copy

        # Convert persona1 fields
        result["experience"]["persona1"]["traits"] = {"en": original["experience"]["persona1"]["traits"]}
        result["experience"]["persona1"]["background"] = {"en": original["experience"]["persona1"]["background"]}
        result["experience"]["persona1"]["chatting_style"] = {"en": original["experience"]["persona1"]["chatting_style"]}

        # Convert persona2 fields
        result["experience"]["persona2"]["traits"] = {"en": original["experience"]["persona2"]["traits"]}
        result["experience"]["persona2"]["background"] = {"en": original["experience"]["persona2"]["background"]}
        result["experience"]["persona2"]["chatting_style"] = {"en": original["experience"]["persona2"]["chatting_style"]}

        # Convert experience fields
        result["experience"]["relationship"] = {"en": original["experience"]["relationship"]}
        result["experience"]["situation"] = {"en": original["experience"]["situation"]}
        result["experience"]["topic"] = {"en": original["experience"]["topic"]}

        # Convert chat messages
        for i, chat_part in enumerate(result["chat_parts"]):
            new_messages = []
            for msg in chat_part["messages"]:
                if isinstance(msg, str) and not msg.startswith("<image>") and not msg.startswith("<delay"):
                    new_messages.append({"en": msg})
                else:
                    new_messages.append(msg)  # Keep image/delay tags as-is
            result["chat_parts"][i]["messages"] = new_messages
    else:
        result = original

    # Add the new language translation
    if "persona1" in translation:
        result["experience"]["persona1"]["traits"][language_code] = translation["persona1"]["traits"]
        result["experience"]["persona1"]["background"][language_code] = translation["persona1"]["background"]
        result["experience"]["persona1"]["chatting_style"][language_code] = translation["persona1"]["chatting_style"]

    if "persona2" in translation:
        result["experience"]["persona2"]["traits"][language_code] = translation["persona2"]["traits"]
        result["experience"]["persona2"]["background"][language_code] = translation["persona2"]["background"]
        result["experience"]["persona2"]["chatting_style"][language_code] = translation["persona2"]["chatting_style"]

    if "relationship" in translation:
        result["experience"]["relationship"][language_code] = translation["relationship"]

    if "situation" in translation:
        result["experience"]["situation"][language_code] = translation["situation"]

    if "topic" in translation:
        result["experience"]["topic"][language_code] = translation["topic"]

    # Add translated messages
    if "messages" in translation:
        for chat_idx, translated_msgs in enumerate(translation["messages"]):
            if chat_idx < len(result["chat_parts"]):
                for msg_idx, translated_msg in enumerate(translated_msgs):
                    if msg_idx < len(result["chat_parts"][chat_idx]["messages"]):
                        current_msg = result["chat_parts"][chat_idx]["messages"][msg_idx]
                        if isinstance(current_msg, dict) and "en" in current_msg:
                            current_msg[language_code] = translated_msg
                        elif isinstance(current_msg, str) and not current_msg.startswith(("<image>", "<delay")):
                            # Convert to multilingual format
                            result["chat_parts"][chat_idx]["messages"][msg_idx] = {
                                "en": current_msg,
                                language_code: translated_msg
                            }

    return result


def create_dataset_card(
    source_dataset: str,
    model_id: str,
    target_languages: List[str],
    generation_config: GenerationConfig,
    num_examples: int,
    generation_time: str,
) -> str:
    """Create a dataset card for the translated dataset."""

    languages_list = ", ".join([f"{code} ({LANGUAGE_CONFIGS[code]['name']})" for code in target_languages])

    return f"""
# Multilingual Chat Dataset

This dataset contains multilingual translations of conversational data, translated using {model_id} and HF Jobs.

## Dataset Details

- **Source Dataset**: {source_dataset}
- **Translation Model**: {model_id}
- **Languages**: English (original) + {languages_list}
- **Total Examples**: {num_examples:,}
- **Generation Date**: {generation_time}

## Translation Parameters

- **Temperature**: {generation_config.temperature}
- **Top-p**: {generation_config.top_p}
- **Max Tokens**: {generation_config.max_new_tokens}

## Dataset Structure

Each example contains:
- `chat_id`: Unique identifier for the conversation
- `experience`: Multilingual persona and context information
- `chat_parts`: Multilingual conversation messages
- `model`: Original model used for generation

### Multilingual Fields

The following fields are available in multiple languages:
- `experience.persona1/persona2.traits`: Character traits
- `experience.persona1/persona2.background`: Character background
- `experience.persona1/persona2.chatting_style`: Communication style
- `experience.relationship`: Relationship description
- `experience.situation`: Scenario description
- `experience.topic`: Conversation topic
- `chat_parts[].messages[]`: Individual messages (preserving image/delay tags)

## Language Codes

This dataset uses ISO 639 language codes:
{chr(10).join([f"- `{code}`: {LANGUAGE_CONFIGS[code]['name']} ({LANGUAGE_CONFIGS[code]['native_name']})" for code in target_languages])}

## Usage

```
from datasets import load_dataset

dataset = load_dataset("marcodsn/SOC-2508-MULTI")

# Access English version
english_background = dataset["experience"]["persona1"]["background"]["en"]

# Access Spanish version
spanish_background = dataset["experience"]["persona1"]["background"]["es"]
```

## Citation

If you use this dataset, please cite the original source dataset and mention the translation methodology.
"""


def main(
    src_dataset_hub_id: str = "marcodsn/SOC-2508",
    output_dataset_hub_id: str = "marcodsn/SOC-2508-MULTI",
    model_id: str = "HuggingFaceTB/SmolLM3-3B",
    target_languages: List[str] = None,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    device: str = "auto",
    torch_dtype: str = "auto",
    hf_token: Optional[str] = None,
    batch_size: int = 4,
):
    """
    Main translation pipeline.
    """
    if target_languages is None:
        target_languages = Config.TARGET_LANGUAGES

    generation_start_time = datetime.now().isoformat()

    # Validate language codes
    invalid_langs = [lang for lang in target_languages if lang not in LANGUAGE_CONFIGS]
    if invalid_langs:
        logger.error(f"Invalid language codes: {invalid_langs}")
        logger.error(f"Available languages: {list(LANGUAGE_CONFIGS.keys())}")
        sys.exit(1)

    # GPU check and configuration
    num_gpus = check_gpu_availability()

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set torch dtype
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Authentication
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not HF_TOKEN:
        logger.error("No HuggingFace token found. Please provide token.")
        sys.exit(1)

    logger.info("HuggingFace token found, authenticating...")
    login(token=HF_TOKEN)

    # Initialize model and tokenizer
    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if num_gpus > 1 else device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create generation configuration
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load dataset
    logger.info(f"Loading dataset: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")
    total_examples = len(dataset)
    logger.info(f"Dataset loaded with {total_examples:,} examples")

    # Process translations language by language
    multilingual_data = []

    for example in tqdm(dataset, desc="Initializing multilingual data"):
        multilingual_data.append(dict(example))

    for target_lang in target_languages:
        logger.info(f"Translating to {LANGUAGE_CONFIGS[target_lang]['name']} ({target_lang})...")

        # Create translation prompts
        prompts = []
        for i, example in enumerate(multilingual_data):
            prompt = create_translation_prompt(example, target_lang)

            # Apply chat template
            messages = [
                {"role": "system", "content": "/no_think"},  # Disable thinking for SmolLM3
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted_prompt)

        # Generate translations in batches
        logger.info(f"Generating {len(prompts)} translations for {target_lang}...")

        # Process in batches to manage memory
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_indices = range(i, min(i+batch_size, len(prompts)))

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings or 8192,
            ).to(model.device)

            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                )

            # Decode responses
            responses = []
            for j, output in enumerate(outputs):
                # Remove input tokens from output
                input_length = inputs.input_ids[j].shape[0]
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                responses.append(response)

            # Parse responses and merge translations
            for j, response in enumerate(responses):
                example_idx = list(batch_indices)[j]

                translation = parse_translation_response(response, target_lang)
                multilingual_data[example_idx] = merge_translation(
                    multilingual_data[example_idx], translation, target_lang
                )

    # Create final dataset
    logger.info("Creating final multilingual dataset...")
    final_dataset = Dataset.from_list(multilingual_data)

    # Create dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=src_dataset_hub_id,
        model_id=model_id,
        target_languages=target_languages,
        generation_config=generation_config,
        num_examples=total_examples,
        generation_time=generation_start_time,
    )

    # Push dataset to hub
    logger.info(f"Pushing dataset to: {output_dataset_hub_id}")
    final_dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    # Push dataset card
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    logger.info("✅ Translation complete!")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset_hub_id}")
    logger.info(f"Languages: English + {', '.join(target_languages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate chat dataset to multiple languages using Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "src_dataset_hub_id",
        help="Input dataset on Hugging Face Hub",
    )
    parser.add_argument(
        "output_dataset_hub_id",
        help="Output dataset name on Hugging Face Hub"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolLM3-3B",
        help="Model to use for translation",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["fr", "es", "de", "it", "pt"],
        help="Target language codes (ISO 639)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto, default: auto)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype (auto/float16/bfloat16/float32, default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available language codes and exit",
    )

    args = parser.parse_args()

    if args.list_languages:
        print("Available language codes:")
        for code, info in LANGUAGE_CONFIGS.items():
            print(f"  {code}: {info['name']} ({info['native_name']})")
        sys.exit(0)

    # Convert torch_dtype string to actual dtype
    torch_dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(args.torch_dtype, "auto")

    main(
        src_dataset_hub_id=args.src_dataset_hub_id,
        output_dataset_hub_id=args.output_dataset_hub_id,
        model_id=args.model_id,
        target_languages=args.languages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=torch_dtype,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
    )
