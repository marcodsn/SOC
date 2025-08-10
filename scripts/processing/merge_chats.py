import json
import argparse
import glob
import sys
import time
import copy
import logging
import random
import re
from pydantic import BaseModel

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

def setup_logging():
    """
    Configure logging for the script.
    """
    timestamp = int(time.time())
    log_file = f"logs/merge_chats_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return timestamp

def split_message_by_tags(message: str) -> list[str]:
    """
    Splits a single message string based on multimedia tags.

    - For <audio> and <delay>, the tag is isolated into its own message.
      e.g., "Hi <delay/> wait" -> ["Hi", "<delay/>", "wait"]
    - For <image>, <gif>, <video>, the tag and any subsequent text are
      kept together, but split from any preceding text.
      e.g., "Look! <image>cat</image> my cat" -> ["Look!", "<image>cat</image> my cat"]
    """
    # Pattern for tags that require a "hard break" (split before and after)
    # The parentheses capture the delimiter, so re.split() keeps the tag in the list.
    hard_break_pattern = re.compile(r'(<audio.*?</audio>|<delay.*?>)')

    # Pattern for tags that require a "soft break" (split only before)
    # A positive lookahead `(?=...)` splits without consuming the tag.
    soft_break_pattern = re.compile(r'(?=<image.*?</image>|<gif.*?</gif>|<video.*?</video>)')

    final_messages = []

    # Stage 1: Split by hard-break tags first.
    # This gives us chunks, some of which are the tags themselves.
    chunks = hard_break_pattern.split(message)

    # Stage 2: Process each chunk for soft-break tags.
    for chunk in chunks:
        if not chunk:
            continue

        # If the chunk is a hard-break tag itself, just add it.
        if hard_break_pattern.fullmatch(chunk):
            final_messages.append(chunk.strip())
        else:
            # Otherwise, split this chunk by the soft-break tags.
            sub_chunks = soft_break_pattern.split(chunk)
            # Add the resulting non-empty, stripped sub-chunks to the final list.
            final_messages.extend([sub.strip() for sub in sub_chunks if sub and sub.strip()])

    return final_messages

def clean(data):
    """
    Cleaning logic.
    """
    chat_id = data.get('chat_id', 'unknown')

    # Check if we have a valid data structure
    if not isinstance(data, dict):
        logging.warning(f"Invalid data structure received for cleaning - chat_id: {chat_id}")
        return data

    # Remove personas names from messages
    persona1 = data.get("experience", {}).get("persona1", {})
    persona2 = data.get("experience", {}).get("persona2", {})

    # Check if we have valid personas
    if not persona1 or not persona2:
        logging.warning(f"Missing persona information in chat {chat_id}")
        return data

    logging.info(f"Cleaning chat {chat_id} with personas: {persona1.get('name', 'Unknown')} and {persona2.get('name', 'Unknown')}")

    # Create a deep copy to avoid modifying the original data
    data = copy.deepcopy(data)

    # Get persona names for replacement
    persona1_name = persona1.get('name', '')
    persona2_name = persona2.get('name', '')

    # Get usernames too if available
    persona1_username = persona1.get('username', '')
    persona2_username = persona2.get('username', '')

    # Skip processing if no persona names to remove
    if not persona1_name and not persona2_name:
        logging.warning(f"No persona names found to clean in chat {chat_id}")
        # We might still need to split tags, so we don't return here.

    # Process chat parts
    chat_parts = data.get("chat_parts", [])
    if not chat_parts:
        logging.warning(f"No chat parts found in chat {chat_id}")
        return data

    for part_idx, part in enumerate(chat_parts):
        messages = part.get("messages", [])
        if not messages:
            logging.warning(f"No messages found in part {part_idx} of chat {chat_id}")
            continue

        # --- Rebuild the messages list to handle splitting ---
        new_messages = []
        for i, message in enumerate(messages):
            if not isinstance(message, str):
                logging.warning(f"Non-string message found in part {part_idx}, message {i} of chat {chat_id}")
                new_messages.append(message) # Keep non-string data as-is
                continue

            # 1. First, clean the persona names from the message
            cleaned_message = message
            original_message = message  # Save original for logging

            # Define possible name patterns to remove
            name_patterns = []
            if persona1_name:
                name_patterns.extend([f"{persona1_name}:", f"{persona1_name.lower()}:", f"{persona1_name.upper()}:", f"{persona1_name} :", f" {persona1_name}:"])
            if persona2_name:
                name_patterns.extend([f"{persona2_name}:", f"{persona2_name.lower()}:", f"{persona2_name.upper()}:", f"{persona2_name} :", f" {persona2_name}:"])
            if persona1_username:
                name_patterns.extend([f"{persona1_username}:", f"@{persona1_username}:"])
            if persona2_username:
                name_patterns.extend([f"{persona2_username}:", f"@{persona2_username}:"])

            for pattern in name_patterns:
                if pattern in cleaned_message:
                    cleaned_message = cleaned_message.replace(pattern, "")

            if persona1_name and cleaned_message.strip().startswith(persona1_name):
                cleaned_message = cleaned_message.replace(persona1_name, "", 1).strip()
            if persona2_name and cleaned_message.strip().startswith(persona2_name):
                cleaned_message = cleaned_message.replace(persona2_name, "", 1).strip()

            cleaned_message = cleaned_message.strip()

            # 2. Now, split the cleaned message by tags
            split_messages = split_message_by_tags(cleaned_message)

            # Log if a split occurred
            if len(split_messages) > 1:
                logging.info(f"Chat {chat_id}, message split from: '{original_message}' -> {split_messages}")

            # Add the resulting messages (which could be one or more) to our new list
            new_messages.extend(split_messages)

        # Replace the old messages list with the new, processed one
        part["messages"] = new_messages

    return data

def main():
    """
    Main function to clean and merge JSONL files.
    """
    timestamp = setup_logging()

    parser = argparse.ArgumentParser(
        description="Clean and merge multiple JSONL files into a single file. "
                    "This script removes duplicates based on 'chat_id' and filters out "
                    "entries with an empty 'chat_parts' array.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input-pattern",
        default="data/chats/chats_*.jsonl",
        help="Glob pattern for input JSONL files (e.g., 'data/chats_*.jsonl').\n"
             "Be sure to wrap in quotes if your shell expands it automatically."
    )

    parser.add_argument(
        "-o", "--output",
        default=f"data/chats/merged_chats_{timestamp}.jsonl",
        help="Name of the output file (default: merged_clean.jsonl)."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with additional validation tests"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of output lines"
    )


    args = parser.parse_args()

    # Find all files matching the input pattern
    input_files = glob.glob(args.input_pattern)

    if not input_files:
        print(f"Error: No files found matching pattern '{args.input_pattern}'")
        sys.exit(1)

    print(f"Found {len(input_files)} files to process. Merging into '{args.output}'...")

    # Use a set for efficient tracking of seen chat_ids
    seen_chat_ids = set()

    # Counters for statistics
    total_lines_read = 0
    chats_written = 0
    duplicates_skipped = 0
    empty_chats_skipped = 0
    malformed_lines = 0

    # Collect all valid chats in a list
    valid_chats = []

    for file_path in sorted(input_files): # Sort for deterministic order
        try:
            with open(file_path, 'r', encoding='utf-8') as input_f:
                logging.info(f"Processing {file_path}...")
                for line in input_f:
                    total_lines_read += 1

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        malformed_lines += 1
                        continue # Skip malformed lines

                    # --- Cleaning and Filtering Logic ---

                    # 1. Check for chat_id presence
                    chat_id = data.get("chat_id")
                    if not chat_id:
                        malformed_lines += 1
                        continue

                    # 2. Check for duplicates
                    if chat_id in seen_chat_ids:
                        duplicates_skipped += 1
                        continue

                    # 3. Check for non-empty chat_parts (i.e., actual conversations)
                    chat_parts = data.get("chat_parts")
                    if not isinstance(chat_parts, list) or not chat_parts:
                        empty_chats_skipped += 1
                        continue

                    # 4. Skip single turn conversations
                    if len(data.get("chat_parts")) < 2:
                        empty_chats_skipped += 1
                        continue

                    # 5. Clean the data
                    cleaned_data = clean(data)

                    # --- If all checks pass, collect the cleaned data ---
                    valid_chats.append(cleaned_data)
                    seen_chat_ids.add(chat_id)
                    chats_written += 1
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}", exc_info=True)

    # Shuffle the collected chats if not disabled
    if not args.no_shuffle:
        logging.info(f"Shuffling {len(valid_chats)} valid chats...")
        random.shuffle(valid_chats)
    else:
        logging.info("Shuffling disabled, preserving original order...")

    # Write the shuffled chats to the output file
    with open(args.output, 'w', encoding='utf-8') as output_f:
        for chat in valid_chats:
            output_f.write(json.dumps(chat) + '\n')



    # Print the final summary report
    summary = [
        "\n--- Merge Complete ---",
        f"Total files processed: {len(input_files)}",
        f"Total lines read:      {total_lines_read}",
        f"Clean chats written:   {chats_written}",
        "------------------------",
        f"Duplicates skipped:    {duplicates_skipped}",
        f"Empty chats skipped:   {empty_chats_skipped} (no conversation)",
        f"Malformed lines:       {malformed_lines}",
        "------------------------",
        f"Merged data saved to:  '{args.output}'"
    ]

    summary_text = "\n".join(summary)
    print(summary_text)
    logging.info(summary_text)

if __name__ == "__main__":
    main()
