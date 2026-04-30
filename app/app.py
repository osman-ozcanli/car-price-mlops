"""Streamlit UI for the car price prediction MLOps demo.

This file runs both locally and on a HuggingFace Space. It is mirrored to the
Space by the deploy_app.yml GitHub Actions workflow on every push that touches
this file.

The Space runs each file as ``__main__``. Shared modules (common/) are bundled
alongside via the same workflow.
"""
import os
import sys
import json
import random
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

# Ensure ``common`` package is importable when this file runs as a script
# inside an HF Space (where the working directory is /home/user/app).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.transformers import AddInteractions  # noqa: E402
from common.constants import (  # noqa: E402
    MODEL_REPO, DATASET_REPO, INFLATION_MULTIPLIER, THRESHOLD,
    GITHUB_OWNER, GITHUB_REPO, GITHUB_WORKFLOW,
)
from common.validation import validate_feedback  # noqa: E402

FEEDBACK_FILE = "feedback.parquet"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── Categorical option lists (US market, 2015-era auction taxonomy) ─────────
BODY_TYPES = ["Sedan", "SUV", "Hatchback", "Minivan",
              "Coupe", "Wagon", "Convertible", "Van", "Unknown"]
TRANSMISSIONS = ["automatic", "manual"]
COLORS = ["beige", "black", "blue", "brown", "burgundy", "charcoal",
          "gold", "gray", "green", "lime", "off-white", "orange",
          "pink", "purple", "red", "silver", "turquoise", "white", "yellow"]
INTERIORS = ["beige", "black", "blue", "brown", "burgundy", "gold",
             "gray", "green", "off-white", "orange", "purple", "red",
             "silver", "tan", "white", "yellow"]
STATES = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
          "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
          "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
          "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
          "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"]


# ── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_hierarchy():
    path = hf_hub_download(repo_id=MODEL_REPO, filename="car_hierarchy.json")
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_models():
    """Download and cache the active and previous models plus preprocessing.

    ``interactions.pkl`` is intentionally not loaded — ``AddInteractions`` is
    stateless, so it is constructed fresh in :func:`predict`. This avoids the
    pickle module-path coupling that previously caused load failures.
    """
    files = ["lgbm_tuned.pkl", "preprocessor.pkl", "power_transformer.pkl"]
    loaded = {}
    for fname in files:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=fname)
        loaded[fname] = joblib.load(path)

    try:
        prev_path = hf_hub_download(repo_id=MODEL_REPO, filename="lgbm_tuned_prev.pkl")
        loaded["lgbm_tuned_prev.pkl"] = joblib.load(prev_path)
    except Exception:
        # First deploy or prev artifact missing — fall back to current. The
        # session_state still records the assigned label for honest A/B logging.
        loaded["lgbm_tuned_prev.pkl"] = loaded["lgbm_tuned.pkl"]
    return loaded


# ── Inference ────────────────────────────────────────────────────────────────
def predict(input_dict: dict, artifacts: dict, model_version: str = "v_current") -> float:
    df_input = pd.DataFrame([input_dict])
    interact = AddInteractions()
    preproc = artifacts["preprocessor.pkl"]
    model_key = "lgbm_tuned_prev.pkl" if model_version == "v_previous" else "lgbm_tuned.pkl"
    model = artifacts[model_key]
    pt = artifacts["power_transformer.pkl"]

    X_inter = interact.transform(df_input)
    X_proc = preproc.transform(X_inter)
    y_yj = model.predict(X_proc)
    y_pred = pt.inverse_transform(y_yj.reshape(-1, 1)).ravel()
    return float(np.clip(y_pred * INFLATION_MULTIPLIER, 500, None)[0])


# ── Feedback persistence ─────────────────────────────────────────────────────
def save_feedback(row: dict) -> int:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO, filename=FEEDBACK_FILE,
            repo_type="dataset", token=token,
        )
        df_existing = pd.read_parquet(path)
    except Exception:
        df_existing = pd.DataFrame()

    df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    tmp_path = os.path.join(tempfile.gettempdir(), FEEDBACK_FILE)
    df_new.to_parquet(tmp_path, index=False)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=FEEDBACK_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=token,
    )
    return len(df_new)


def trigger_retraining_workflow() -> None:
    """Fire-and-forget dispatch of the GitHub Actions retraining workflow."""
    import requests
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    url = (f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
           f"/actions/workflows/{GITHUB_WORKFLOW}/dispatches")
    requests.post(
        url,
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github.v3+json"},
        json={"ref": "main"},
        timeout=10,
    )


# ── A/B assignment (per session) ─────────────────────────────────────────────
if "model_version" not in st.session_state:
    st.session_state.model_version = random.choice(["v_current", "v_previous"])


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🚗 Car Price Predictor")
st.caption("US used-car market · price range $500–$78,000")

with st.spinner("Loading model..."):
    artifacts = load_models()
    hierarchy = load_hierarchy()

st.divider()

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Make", sorted(hierarchy.keys()))
    model_list = sorted(hierarchy.get(make, {}).keys())
    model_name = st.selectbox("Model", model_list)
    trim_list = hierarchy.get(make, {}).get(model_name, [])
    trim = st.selectbox("Trim", trim_list) if trim_list else "unknown"
    if not trim_list:
        st.caption("No trim data for this model — defaulting to 'unknown'.")
    body = st.selectbox("Body type", BODY_TYPES)
    transmission = st.selectbox("Transmission", TRANSMISSIONS)

with col2:
    state = st.selectbox(
        "State", STATES,
        help="States with little training data fall back to the national average; "
             "regional pricing nuances may not be fully reflected.",
    )
    color = st.selectbox("Exterior color", COLORS)
    interior = st.selectbox("Interior color", INTERIORS)
    age = st.number_input("Vehicle age (years)", min_value=0, max_value=30,
                          value=5, step=1)
    odometer = st.number_input("Odometer (miles)", min_value=0, max_value=300_000,
                               value=50_000, step=1_000)
    condition = st.slider("Condition (1 = poor · 5 = excellent)",
                          min_value=1.0, max_value=5.0, value=3.0, step=0.1)

st.divider()

if st.button("💰 Estimate price", use_container_width=True, type="primary"):
    input_dict = {
        "make": make, "model": model_name,
        "trim": trim if trim else "unknown",
        "body": body, "transmission": transmission, "state": state,
        "color": color, "interior": interior,
        "age": int(age), "odometer": int(odometer), "condition": float(condition),
        # ``seller`` is dropped by the current preprocessor (remainder='drop');
        # we still provide a placeholder for backwards compatibility with any
        # legacy artifact that references the column.
        "seller": "unknown",
    }
    price = predict(input_dict, artifacts, st.session_state.model_version)
    st.session_state["last_prediction"] = price
    st.session_state["last_input"] = input_dict
    st.session_state["feedback_given"] = False
    st.session_state["show_feedback_form"] = False

if "last_prediction" in st.session_state and not st.session_state.get("feedback_given", True):
    pred = st.session_state["last_prediction"]
    # Wider, more honest confidence band: ±5% or ±$500, whichever is larger.
    band = max(0.05 * pred, 500)
    low, high = max(500, pred - band), pred + band
    st.success(f"### Estimated price: ${pred:,.0f}")
    st.caption(f"Indicative range: ${low:,.0f} – ${high:,.0f}")
    st.markdown("**Was this estimate useful?**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Yes", use_container_width=True):
            st.session_state["show_feedback_form"] = True
    with col_b:
        if st.button("❌ No", use_container_width=True):
            st.session_state["feedback_given"] = True
            st.info("Thanks for the feedback.")

if st.session_state.get("show_feedback_form") and not st.session_state.get("feedback_given", False):
    with st.form("feedback_form"):
        st.markdown("**Enter the actual sale price ($):**")
        real_price = st.number_input(
            "Actual price ($)", min_value=500, max_value=78_000,
            value=int(st.session_state["last_prediction"]), step=100,
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            result = validate_feedback(
                price=real_price,
                odometer=st.session_state["last_input"]["odometer"],
                predicted=st.session_state.get("last_prediction"),
            )
            if not result.accepted:
                st.error(f"Feedback rejected ({result.layer}/{result.reason_code}): "
                         f"{result.reason_message}")
            else:
                row = {
                    **st.session_state["last_input"],
                    "sellingprice": real_price,
                    # model_version is recorded for downstream A/B analysis but
                    # not surfaced to the user — keeps internal state private.
                    "model_version": st.session_state["model_version"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    total_new = save_feedback(row)
                    st.success("Thanks! Your feedback was recorded.")
                    st.session_state["feedback_given"] = True
                    st.session_state["show_feedback_form"] = False
                    # Trigger retraining silently every THRESHOLD-th feedback.
                    # No user-facing message — internal pipeline state must not
                    # leak into the UI.
                    if total_new >= THRESHOLD and total_new % THRESHOLD == 0:
                        trigger_retraining_workflow()
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
