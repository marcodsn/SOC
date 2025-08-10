# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "torch",
#     "transformers>=4.53.0",
#     "flashinfer-python",
#     "vllm",
# ]
# ///

import os
import sys
import logging
import argparse
import json
from typing import Optional, List

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import get_token, login
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Language configurations with ISO 639 codes
LANGUAGE_CONFIGS = {
    "en": {"name": "English", "native_name": "English"},
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
    MODEL_NAME = "google/gemma-3n-E4B-it"
    TARGET_LANGUAGES = ["fr", "es", "de", "it", "pt"]  # Default languages to translate to


def check_gpu_availability() -> int:
    """Check if CUDA is available and return the number of GPUs."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU and vLLM requires CUDA.")
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

def create_simple_translation_prompt(text: str, target_language: str) -> str:
    """Creates a simple, direct prompt for translating a single piece of text."""
    lang_info = LANGUAGE_CONFIGS[target_language]
    prompt = f"""You are a professional translator. Translate the following English text to {lang_info['name']} ({lang_info['native_name']}).

IMPORTANT RULES:
1. Preserve special tags like <image>, <video>, <audio>, <gif>, <delay>, and <end/> exactly as they appear.
2. Only translate the text content.
3. Respond ONLY with the translated text, and nothing else.

English Text:
"{text}"

Translated to {lang_info['name']}:
"""
    return prompt


def translate_batch_of_texts(
    texts: List[str],
    target_language: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
) -> List[str]:
    """Translates a list of text strings using vLLM for high-throughput inference."""
    # Filter out empty or tag-only texts to avoid unnecessary API calls
    original_indices = [i for i, text in enumerate(texts) if text and text.strip() and not text.strip().startswith("<")]
    texts_to_translate = [texts[i] for i in original_indices]

    if not texts_to_translate:
        return texts # Return original if nothing to translate

    # Create prompts
    prompts = [create_simple_translation_prompt(text, target_language) for text in texts_to_translate]

    # Apply chat template
    chat_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in prompts]

    # Generate responses in a single call to vLLM
    logger.info(f"Generating {len(chat_prompts)} translations for {target_language} with vLLM...")
    outputs = model.generate(chat_prompts, sampling_params)

    # Extract translated text from the output
    translations = [output.outputs[0].text.strip() for output in outputs]

    # Reconstruct the full list, putting back non-translated parts
    full_translations = list(texts) # Start with a copy of the original
    for i, original_idx in enumerate(original_indices):
        full_translations[original_idx] = translations[i]

    return full_translations


def main(
    src_dataset_hub_id: str = "marcodsn/SOC-2508",
    output_dataset_hub_id: str = "marcodsn/SOC-2508-MULTI",
    model_id: str = "google/gemma-3n-E4B-it",
    target_languages: List[str] = None,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    torch_dtype: str = "auto",
    hf_token: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Main translation pipeline.
    """
    if target_languages is None:
        target_languages = Config.TARGET_LANGUAGES

    # Validate language codes
    invalid_langs = [lang for lang in target_languages if lang not in LANGUAGE_CONFIGS]
    if invalid_langs:
        logger.error(f"Invalid language codes: {invalid_langs}")
        logger.error(f"Available languages: {list(LANGUAGE_CONFIGS.keys())}")
        sys.exit(1)

    # GPU check and configuration
    num_gpus = check_gpu_availability()

    # Set torch dtype for vLLM
    if torch_dtype == "auto":
        torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    # Authentication
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not HF_TOKEN:
        logger.error("No HuggingFace token found. Please provide token via --hf-token or HUGGING_FACE_HUB_TOKEN env var.")
        sys.exit(1)
    logger.info("HuggingFace token found, authenticating...")
    login(token=HF_TOKEN)

    # Load dataset
    logger.info(f"Loading source dataset: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")
    total_examples = len(dataset)
    logger.info(f"Dataset loaded with {total_examples:,} examples")

    if not total_examples > 0:
        logger.warning("Source dataset is empty. Exiting.")
        sys.exit(0)

    # Create a mutable copy of the data
    multilingual_data = [json.loads(json.dumps(ex)) for ex in tqdm(dataset, desc="Cloning data")]

    # --- DATA PREPARATION AND VALIDATION ---

    # 1. Check if the data is already multilingual by inspecting the first record.
    sample_field = multilingual_data[0].get("experience", {}).get("persona1", {}).get("background")
    is_multilingual = isinstance(sample_field, dict)

    if not is_multilingual:
        logger.info("Source dataset is monolingual. Converting to multilingual structure with 'en' key...")
        for example in tqdm(multilingual_data, desc="Structuring data"):
            # Personas
            for p_key in ["persona1", "persona2"]:
                for field in ["traits", "background", "chatting_style"]:
                    example["experience"][p_key][field] = {"en": example["experience"][p_key][field]}
            # Experience
            for field in ["relationship", "situation", "topic"]:
                example["experience"][field] = {"en": example["experience"][field]}
            # Messages
            for chat_part in example["chat_parts"]:
                new_messages = []
                for msg in chat_part["messages"]:
                    # Ensure every message is a dictionary (struct) for Arrow compatibility
                    if isinstance(msg, str):
                        if msg.startswith("<"):
                            # This is a special, non-translatable tag. Wrap it.
                            new_messages.append({"tag": msg})
                        else:
                            # This is a translatable text message.
                            new_messages.append({"en": msg})
                    else:
                        # If it's already a dict (from a previous run), keep it.
                        new_messages.append(msg)
                chat_part["messages"] = new_messages
    else:
        logger.info("Source dataset is already in multilingual format.")

    # 2. Check for language conflicts if not overwriting.
    if not overwrite:
        # We check the first record as a representative sample of the dataset's languages.
        existing_langs = set()
        sample_record = multilingual_data[0]
        for field in [
            sample_record.get("experience", {}).get("persona1", {}).get("background", {}),
            sample_record.get("experience", {}).get("topic", {}),
        ]:
            if isinstance(field, dict):
                existing_langs.update(field.keys())

        conflicting_langs = set(target_languages) & existing_langs
        if 'en' in conflicting_langs:
            conflicting_langs.remove('en')  # 'en' is the source, not a conflict.

        if conflicting_langs:
            logger.error(f"The following languages already exist in the dataset: {list(conflicting_langs)}.")
            logger.error("To replace them, run the script again with the --overwrite flag.")
            logger.error("To add other languages, remove the conflicting ones from your --languages list.")
            sys.exit(1)

    # --- INITIALIZE MODEL ---
    logger.info(f"Loading model: {model_id} with vLLM")
    llm = LLM(model=model_id, tensor_parallel_size=num_gpus, dtype=torch_dtype, trust_remote_code=True, gpu_memory_utilization=0.90)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop_token_ids=[tokenizer.eos_token_id])

    # --- TRANSLATION LOOP ---
    for target_lang in target_languages:
        logger.info(f"--- Starting translation to {LANGUAGE_CONFIGS[target_lang]['name']} ({target_lang}) ---")

        # 1. Gather all unique text fields that need translation (always from English).
        texts_to_translate = []
        text_map = []  # To map translations back to their original place

        for i, example in enumerate(tqdm(multilingual_data, desc=f"Gathering texts for {target_lang}")):
            exp = example["experience"]
            # Persona fields
            for p_key in ["persona1", "persona2"]:
                texts_to_translate.extend(exp[p_key]["traits"]["en"])
                text_map.extend([(i, p_key, "traits", j) for j in range(len(exp[p_key]["traits"]["en"]))])
                texts_to_translate.append(exp[p_key]["background"]["en"])
                text_map.append((i, p_key, "background", None))
                texts_to_translate.append(exp[p_key]["chatting_style"]["en"])
                text_map.append((i, p_key, "chatting_style", None))

            # Experience fields
            for field in ["relationship", "situation", "topic"]:
                texts_to_translate.append(exp[field]["en"])
                text_map.append((i, "experience", field, None))

            # Message fields
            for j, chat_part in enumerate(example["chat_parts"]):
                for k, msg in enumerate(chat_part["messages"]):
                    if isinstance(msg, dict) and "en" in msg:
                        texts_to_translate.append(msg["en"])
                        text_map.append((i, "message", j, k))

        # 2. Translate all texts in a single, batched call using vLLM
        logger.info(f"Collected {len(texts_to_translate):,} text segments to translate for {target_lang}.")
        translated_texts = translate_batch_of_texts(
            texts_to_translate, target_lang, llm, tokenizer, sampling_params
        )

        # 3. Re-integrate the translations back into the data structure
        logger.info(f"Integrating {len(translated_texts):,} translations for {target_lang}...")
        for idx, translation in enumerate(tqdm(translated_texts, desc=f"Integrating for {target_lang}")):
            map_info = text_map[idx]
            ex_idx, key1, key2, key3 = map_info

            if key1 == "persona1" or key1 == "persona2":
                if key2 == "traits":
                    if target_lang not in multilingual_data[ex_idx]["experience"][key1][key2]:
                        multilingual_data[ex_idx]["experience"][key1][key2][target_lang] = [""] * len(multilingual_data[ex_idx]["experience"][key1][key2]["en"])
                    multilingual_data[ex_idx]["experience"][key1][key2][target_lang][key3] = translation
                else:  # background, chatting_style
                    multilingual_data[ex_idx]["experience"][key1][key2][target_lang] = translation
            elif key1 == "experience":  # relationship, situation, topic
                multilingual_data[ex_idx]["experience"][key2][target_lang] = translation
            elif key1 == "message":  # chat message
                multilingual_data[ex_idx]["chat_parts"][key2]["messages"][key3][target_lang] = translation

    # --- FINALIZATION AND PUSH TO HUB ---
    logger.info("Creating final multilingual dataset...")
    final_dataset = Dataset.from_list(multilingual_data)

    logger.info(f"Pushing dataset to: {output_dataset_hub_id}")
    final_dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN, private=False)

    logger.info("✅ Translation complete!")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset_hub_id}")
    logger.info(f"Languages added/updated: {', '.join(target_languages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate a chat dataset to multiple languages using vLLM. Can handle monolingual or existing multilingual datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "src_dataset_hub_id",
        help="Input dataset on Hugging Face Hub (e.g., 'marcodsn/SOC-2508' or 'marcodsn/SOC-2508-MULTI').",
    )
    parser.add_argument(
        "output_dataset_hub_id",
        help="Output dataset name on Hugging Face Hub (e.g., 'marcodsn/SOC-2508-MULTI').",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3n-E4B-it",
        help="Model to use for translation (must be supported by vLLM).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["fr", "es", "de", "it", "pt"],
        help="Target language codes (ISO 639).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p sampling parameter (default: 0.95)."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for the model (auto/float16/bfloat16/float32, default: auto).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token (can also be set via HUGGING_FACE_HUB_TOKEN env var).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translations for the specified languages if the input is a multilingual dataset.",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available language codes and exit.",
    )

    args = parser.parse_args()

    if args.list_languages:
        print("Available language codes:")
        for code, info in LANGUAGE_CONFIGS.items():
            print(f"  {code}: {info['name']} ({info['native_name']})")
        sys.exit(0)

    main(
        src_dataset_hub_id=args.src_dataset_hub_id,
        output_dataset_hub_id=args.output_dataset_hub_id,
        model_id=args.model_id,
        target_languages=args.languages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        overwrite=args.overwrite,
    )
