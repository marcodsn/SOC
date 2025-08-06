#!/usr/bin/env python3

import os
import glob
import argparse
from huggingface_hub import HfApi, HfFolder, login

def get_latest_processed_file(processed_dir="data/chats"):
    """Finds the most recently created .jsonl file in the processed directory."""
    list_of_files = glob.glob(os.path.join(processed_dir, 'merged_*.jsonl'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def main():
    parser = argparse.ArgumentParser(description="Upload a dataset to the Hugging Face Hub.")
    parser.add_argument(
        "repo_name",
        type=str,
        help="The name of the repository on Hugging Face Hub (e.g., 'SOC-2508' in this case)."
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=None,
        help="Your Hugging Face username or organization name. If not provided, will try to infer from login."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to the .jsonl file to upload. If not provided, the latest file from 'data/processed' will be used."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Your Hugging Face API token. If not provided, it will try to use the cached token or prompt for login."
    )

    args = parser.parse_args()

    # --- 1. Authenticate ---
    # You can either pass the token as an argument or log in via the CLI: `huggingface-cli login`
    if args.token:
        print("Logging in with provided token.")
        login(token=args.token)
    else:
        # Tries to use cached token
        token = HfFolder.get_token()
        if token is None:
            print("Hugging Face token not found. Please log in.")
            # This will open a prompt to enter your token
            login()
        else:
            print("Using cached Hugging Face token.")

    api = HfApi()

    # --- 2. Determine Repo ID ---
    if args.hf_username:
        repo_id = f"{args.hf_username}/{args.repo_name}"
    else:
        # Try to get the username from the API after login
        try:
            current_user = api.whoami()['name']
            repo_id = f"{current_user}/{args.repo_name}"
        except Exception as e:
            print(f"Could not automatically determine username: {e}")
            print("Please specify your username with the --hf_username flag.")
            return

    print(f"Preparing to upload to repository: {repo_id}")

    # --- 3. Find the file to upload ---
    file_to_upload = args.file_path
    if not file_to_upload:
        print("No file path provided. Searching for the latest processed file...")
        file_to_upload = get_latest_processed_file()
        if not file_to_upload:
            print("❌ Error: No .jsonl file found in 'data/processed/'.")
            print("Please run the processing script first or provide a file path with --file_path.")
            return

    if not os.path.exists(file_to_upload):
        print(f"❌ Error: The specified file does not exist: {file_to_upload}")
        return

    print(f"Found file to upload: {file_to_upload}")

    # --- 4. Create the repository (if it doesn't exist) ---
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True, # Don't error if the repo already exists
    )
    print(f"Dataset repository '{repo_id}' created or already exists.")

    # --- 5. Upload the file ---
    print("Uploading file to the Hub... this may take a moment.")
    try:
        api.upload_file(
            path_or_fileobj=file_to_upload,
            path_in_repo="data.jsonl",  # Recommended standard name for the main data file
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✅ Upload complete!")
        print(f"Check out your dataset at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ An error occurred during upload: {e}")


if __name__ == "__main__":
    main()
