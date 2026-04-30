"""Push the Streamlit app and its shared modules to the HuggingFace Space.

Usage: ``python scripts/push_app_to_space.py``
Required env: ``HF_TOKEN`` (write scope), ``HF_USERNAME`` (optional, defaults
to Osman-Ozcanli).

Triggered automatically by the deploy_app.yml GitHub Actions workflow when
either ``app/app.py`` or any file under ``common/`` changes.
"""
import os
from datetime import datetime, timezone

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
SPACE_ID = f"{HF_USERNAME}/car_price_prediction_space"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(ROOT, "app", "app.py")
COMMON_DIR = os.path.join(ROOT, "common")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is missing.")

api = HfApi(token=HF_TOKEN)
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

api.upload_file(
    path_or_fileobj=APP_PATH,
    path_in_repo="app.py",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message=f"deploy: app.py updated ({stamp})",
)
print(f"app.py pushed to HF Space: {SPACE_ID}")

# The Space runs app.py from its root. ``common`` is a sibling package the
# app imports from, so we mirror it at the Space root as well.
for fname in sorted(os.listdir(COMMON_DIR)):
    if not fname.endswith(".py"):
        continue
    api.upload_file(
        path_or_fileobj=os.path.join(COMMON_DIR, fname),
        path_in_repo=f"common/{fname}",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message=f"deploy: common/{fname} updated ({stamp})",
    )
    print(f"common/{fname} pushed.")
