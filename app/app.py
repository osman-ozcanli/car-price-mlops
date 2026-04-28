import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import json
import os
import tempfile
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi
from sklearn.base import BaseEstimator, TransformerMixin

# ── Sayfa ayarı ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Araba Fiyat Tahmini",
    page_icon="🚗",
    layout="centered"
)

# ── Sabitler ─────────────────────────────────────────────────────────────────
HF_USERNAME     = "Osman-Ozcanli"
MODEL_REPO      = f"{HF_USERNAME}/car_price_prediction"
DATASET_REPO    = f"{HF_USERNAME}/car_price_prediction_feedback"
FEEDBACK_FILE   = "feedback.parquet"
THRESHOLD       = 10
GITHUB_OWNER    = "Osman-Ozcanli"
GITHUB_REPO     = "car-price-mlops"
GITHUB_WORKFLOW = "retrain.yml"

# ── Seçenek listeleri ─────────────────────────────────────────────────────────
BODY_TYPES    = ["Sedan", "SUV", "Hatchback", "Minivan",
                 "Coupe", "Wagon", "Convertible", "Van", "Unknown"]
TRANSMISSIONS = ["automatic", "manual"]
COLORS        = ["beige", "black", "blue", "brown", "burgundy", "charcoal",
                 "gold", "gray", "green", "lime", "off-white", "orange",
                 "pink", "purple", "red", "silver", "turquoise", "white", "yellow"]
INTERIORS     = ["beige", "black", "blue", "brown", "burgundy", "gold",
                 "gray", "green", "off-white", "orange", "purple", "red",
                 "silver", "tan", "white", "yellow"]
STATES        = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
                 "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
                 "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
                 "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
                 "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"]

# ── AddInteractions (pkl deserialize için burada tanımlı olmalı) ──────────────
class AddInteractions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["age_x_odo"] = X["age"] * X["odometer"]
        return X

# ── Hiyerarşi JSON yükle ──────────────────────────────────────────────────────
@st.cache_data
def load_hierarchy():
    path = hf_hub_download(repo_id=MODEL_REPO, filename="car_hierarchy.json")
    with open(path) as f:
        return json.load(f)

# ── Model yükleme ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # interactions.pkl yüklenmez — AddInteractions stateless, direkt oluşturulur (module uyumsuzluğunu önler)
    files = ["lgbm_tuned.pkl", "preprocessor.pkl", "power_transformer.pkl"]
    loaded = {}
    for fname in files:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=fname)
        loaded[fname] = joblib.load(path)

    # v_previous: önceki model varsa yükle, yoksa v_current ile aynı
    try:
        prev_path = hf_hub_download(repo_id=MODEL_REPO, filename="lgbm_tuned_prev.pkl")
        loaded["lgbm_tuned_prev.pkl"] = joblib.load(prev_path)
    except Exception:
        loaded["lgbm_tuned_prev.pkl"] = loaded["lgbm_tuned.pkl"]

    return loaded

# ── Tahmin ────────────────────────────────────────────────────────────────────
def predict(input_dict, artifacts, model_version="v_current"):
    df_input = pd.DataFrame([input_dict])
    # AddInteractions stateless olduğu için pkl'den yüklenmez, direkt oluşturulur
    interact = AddInteractions()
    preproc  = artifacts["preprocessor.pkl"]
    model    = artifacts["lgbm_tuned_prev.pkl"] if model_version == "v_previous" else artifacts["lgbm_tuned.pkl"]
    pt       = artifacts["power_transformer.pkl"]
    X_inter  = interact.transform(df_input)
    X_proc   = preproc.transform(X_inter)
    y_yj     = model.predict(X_proc)
    y_pred   = pt.inverse_transform(y_yj.reshape(-1, 1)).ravel()
    INFLATION_MULTIPLIER = 1.38
    return float(np.clip(y_pred * INFLATION_MULTIPLIER, 500, None)[0])

# ── Validasyon ────────────────────────────────────────────────────────────────
def is_valid(price, odometer, predicted_price=None):
    # Defense-in-depth: Streamlit input zaten kısıtlıyor, ama bypass edilirse diye
    if odometer > 300_000:
        return False, "Kilometre değeri kabul sınırı dışında (>300,000 mil)."
    if price < 500 or price > 78_000:
        return False, "Fiyat kabul edilebilir aralık dışında ($500–$78,000)."
    # Saçma feedback filtresi: gerçek fiyat tahminin 5x üstünde veya 1/5'i altında ise şüpheli
    if predicted_price and predicted_price > 0:
        ratio = price / predicted_price
        if ratio > 5 or ratio < 0.2:
            return False, (f"Bildirilen fiyat (${price:,}) tahminden (${predicted_price:,.0f}) "
                           f"çok uzak. Lütfen kontrol et.")
    return True, "OK"

# ── Feedback kaydet ───────────────────────────────────────────────────────────
def save_feedback(row: dict):
    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO, filename=FEEDBACK_FILE,
            repo_type="dataset", token=token
        )
        df_existing = pd.read_parquet(path)
    except Exception:
        df_existing = pd.DataFrame()

    df_new   = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    tmp_path = os.path.join(tempfile.gettempdir(), FEEDBACK_FILE)
    df_new.to_parquet(tmp_path, index=False)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=FEEDBACK_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=token
    )
    return len(df_new)

# ── GitHub Actions tetikle ────────────────────────────────────────────────────
def trigger_github_actions():
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
        json={"ref": "main"}
    )

# ── A/B Testing ───────────────────────────────────────────────────────────────
if "model_version" not in st.session_state:
    st.session_state.model_version = random.choice(["v_current", "v_previous"])

# ── Yükle ─────────────────────────────────────────────────────────────────────
st.title("🚗 Araba Fiyat Tahmini")
st.caption("Amerika ikinci el araç piyasası · $500–$78,000 aralığı")

with st.spinner("Model yükleniyor..."):
    artifacts = load_models()
    hierarchy = load_hierarchy()

st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    make      = st.selectbox("Marka", sorted(hierarchy.keys()))
    model_list = sorted(hierarchy.get(make, {}).keys())
    model_name = st.selectbox("Model", model_list)
    trim_list  = hierarchy.get(make, {}).get(model_name, [])
    trim       = st.selectbox("Trim / Versiyon", trim_list) if trim_list else "unknown"
    if not trim_list:
        st.caption("Bu model için trim verisi yok, otomatik atlandı.")
    body         = st.selectbox("Kasa tipi", BODY_TYPES)
    transmission = st.selectbox("Vites", TRANSMISSIONS)

with col2:
    state    = st.selectbox("Eyalet", STATES,
                            help="Az veri bulunan eyaletler için tahmin ulusal ortalamaya yaklaşır; "
                                 "eyaletinize özgü piyasa farklılıkları tam yansımayabilir.")
    color    = st.selectbox("Dış renk", COLORS)
    interior = st.selectbox("İç renk", INTERIORS)
    age      = st.number_input("Araç yaşı (yıl)", min_value=0, max_value=30,
                                value=5, step=1)
    odometer = st.number_input("Kilometre (mil)", min_value=0, max_value=300_000,
                                value=50_000, step=1_000)
    condition = st.slider("Kondisyon (1=kötü · 5=mükemmel)",
                          min_value=1.0, max_value=5.0, value=3.0, step=0.1)

st.divider()

# ── Tahmin butonu ─────────────────────────────────────────────────────────────
if st.button("💰 Fiyat Tahmin Et", use_container_width=True, type="primary"):
    input_dict = {
        "make": make, "model": model_name,
        "trim": trim if trim else "unknown",
        "body": body, "transmission": transmission, "state": state,
        "color": color, "interior": interior,
        "age": int(age), "odometer": int(odometer), "condition": float(condition),
        "seller": "unknown"
    }
    price = predict(input_dict, artifacts, st.session_state.model_version)
    st.session_state["last_prediction"]    = price
    st.session_state["last_input"]         = input_dict
    st.session_state["feedback_given"]     = False
    st.session_state["show_feedback_form"] = False

# ── Sonuç ─────────────────────────────────────────────────────────────────────
if "last_prediction" in st.session_state and not st.session_state.get("feedback_given", True):
    pred = st.session_state["last_prediction"]
    st.success(f"### Tahmini Fiyat: ${pred:,.0f}")
    st.markdown("**Bu tahmin işine yaradı mı?**")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("✅ Memnunum", use_container_width=True):
            st.session_state["show_feedback_form"] = True
    with col_b:
        if st.button("❌ Hayır", use_container_width=True):
            st.session_state["feedback_given"] = True
            st.info("Geri bildirim için teşekkürler.")

# ── Feedback formu ────────────────────────────────────────────────────────────
if st.session_state.get("show_feedback_form") and not st.session_state.get("feedback_given", False):
    with st.form("feedback_form"):
        st.markdown("**Gerçek satış fiyatını gir ($):**")
        real_price = st.number_input(
            "Gerçek fiyat ($)", min_value=500, max_value=78_000,
            value=int(st.session_state["last_prediction"]), step=100
        )
        submitted = st.form_submit_button("Gönder")

        if submitted:
            valid, reason = is_valid(
                real_price,
                st.session_state["last_input"]["odometer"],
                predicted_price=st.session_state.get("last_prediction"),
            )
            if not valid:
                st.error(f"Veri reddedildi: {reason}")
            else:
                row = {
                    **st.session_state["last_input"],
                    "sellingprice": real_price,
                    "model_version": st.session_state["model_version"],
                    "timestamp": datetime.utcnow().isoformat()
                }
                try:
                    total_new = save_feedback(row)
                    st.success("Teşekkürler! Veriniz kaydedildi.")
                    st.session_state["feedback_given"]     = True
                    st.session_state["show_feedback_form"] = False
                    # Her THRESHOLD'un katında bir tetikle (10, 20, 30...) — sürekli tetiklemeyi önler
                    if total_new >= THRESHOLD and total_new % THRESHOLD == 0:
                        trigger_github_actions()
                        st.info("🔄 Yeterli veri birikti, model yeniden eğitim kuyruğuna alındı.")
                except Exception as e:
                    st.error(f"Kayıt hatası: {e}")
