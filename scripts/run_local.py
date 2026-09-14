"""Headless run against a vault, without the Streamlit UI.

    python run_local.py                 # uses OBSIDIAN_VAULT_DIR
    python run_local.py "C:/path/vault"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import cli

if __name__ == "__main__":
    cli()
