"""Headless run against a vault, without the Streamlit UI.

    python run_local.py                 # uses OBSIDIAN_VAULT_DIR
    python run_local.py "C:/path/vault"
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import run_pipeline

vault_dir = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBSIDIAN_VAULT_DIR", "")
if not vault_dir:
    vault_dir = input("Please enter your Obsidian vault directory path: ").strip()

print(f"Running the Constellation pipeline on: {vault_dir}")
state = run_pipeline(vault_dir)
print(f"\nFinished. Notes updated: {state.get('notes_written', 0)}")
