"""
HF Space'e app.py'yi push eder.
Kullanım: python push_app_to_space.py
HF_TOKEN environment variable gerekli (write scope).
"""
import os
from huggingface_hub import HfApi

HF_TOKEN    = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
SPACE_ID    = f"{HF_USERNAME}/car_price_prediction_space"
APP_PATH    = os.path.join(os.path.dirname(__file__), "app", "app.py")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable eksik.")

api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=APP_PATH,
    path_in_repo="app.py",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="fix: load AddInteractions directly, skip interactions.pkl (module error fix)",
)
print(f"app.py basariyla HF Space'e push edildi: {SPACE_ID}")
