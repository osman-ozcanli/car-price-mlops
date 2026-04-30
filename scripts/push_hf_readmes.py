"""Upload the HF Model / Dataset / Space README files from hf_assets/.

Usage: ``python scripts/push_hf_readmes.py``
Required env: ``HF_TOKEN`` (write scope), ``HF_USERNAME`` (default: Osman-Ozcanli).

Run-once helper: HF READMEs rarely change, so this script is invoked manually
after edits in ``hf_assets/`` rather than wired to a workflow.
"""
import os

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is missing.")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(ROOT, "hf_assets")

TARGETS = [
    ("MODEL_README.md", f"{HF_USERNAME}/car_price_prediction", "model"),
    ("DATASET_README.md", f"{HF_USERNAME}/car_price_prediction_feedback", "dataset"),
    ("SPACE_README.md", f"{HF_USERNAME}/car_price_prediction_space", "space"),
]

api = HfApi(token=HF_TOKEN)

for filename, repo_id, repo_type in TARGETS:
    src = os.path.join(ASSETS, filename)
    api.upload_file(
        path_or_fileobj=src,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"docs: refresh README from hf_assets/{filename}",
    )
    print(f"  → {repo_type:8s}  {repo_id}")

print("Done.")
