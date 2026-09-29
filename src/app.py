# ═══════════════════════════════════════════════════════════════════════════════
# SMART CROP AND NUTRIENT ADVISORY SYSTEM
# AI-powered agricultural guidance using Gemini + ML Models
# ═══════════════════════════════════════════════════════════════════════════════

import csv
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from gtts import gTTS
from deep_translator import GoogleTranslator
import uuid

# Page Configuration
st.set_page_config(
    page_title="Smart Crop Advisory System",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
<style>
div[data-baseweb="input"] input {
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────────
# GEMINI API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────────
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-2.5-flash')

# ─────────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────────
SOIL_TYPES = ["Sandy", "Clay", "Loamy", "Silt", "Peaty", "Chalky", "Saline"]
SEASONS = [
    # Tamil Nadu Seasons
    "Kuruvai (Jun – Sep) – TN Early Kharif",
    "Samba (Aug – Jan) – TN Main Season",
    "Navarai (Jan – Mar) – TN Summer Crop",
    # Andhra Pradesh / Telangana / Karnataka Seasons
    "Kharif / SW Monsoon (Jun – Oct) – AP / Telangana / Karnataka",
    "Rabi / Winter (Nov – Feb) – AP / Telangana / Karnataka",
    "Summer / Zaid (Mar – May) – AP / Telangana / Karnataka",
    # Kerala Seasons
    "Virippu / SW Monsoon (Jun – Aug) – Kerala",
    "Mundakan / NE Monsoon (Sep – Nov) – Kerala",
    "Puncha / Winter Paddy (Dec – Feb) – Kerala",
]

REGIONS = [
    # Tamil Nadu
    "Tamil Nadu – Cauvery Delta",
    "Tamil Nadu – Coastal Tamil Nadu",
    "Tamil Nadu – Nilgiris / High Altitude Zone",
    "Tamil Nadu – Southern Dry Zone",
    # Andhra Pradesh
    "Andhra Pradesh – Godavari Delta",
    "Andhra Pradesh – Krishna Delta",
    "Andhra Pradesh – Coastal Andhra",
    "Andhra Pradesh – Rayalaseema",
    # Telangana
    "Telangana – Northern Telangana Dry Zone",
    "Telangana – Eastern Ghats Upland",
    # Karnataka
    "Karnataka – Deccan Plateau",
    "Karnataka – Krishna–Tungabhadra Basin",
    "Karnataka – Northern Dry Zone",
    "Karnataka – Malnad (Western Ghats)",
    # Kerala
    "Kerala – Palakkad Gap",
    "Kerala – Western Ghats (High Rainfall Zone)",
    "Kerala – Coastal Kerala",
]

HIGH_DEMAND_CROPS = {
    # ── Tamil Nadu seasons ──────────────────────────────────────────────────
    "Kuruvai (Jun – Sep) – TN Early Kharif": {
        "Tamil Nadu – Cauvery Delta":        ["Short-duration Rice (ADT-43, CR-1009)", "Sesame", "Groundnut"],
        "Tamil Nadu – Coastal Tamil Nadu":   ["Rice", "Blackgram", "Sesame"],
        "Tamil Nadu – Southern Dry Zone":    ["Pearl Millet (Kambu)", "Groundnut", "Sesame"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Potato", "Carrot", "Cabbage"],
    },
    "Samba (Aug – Jan) – TN Main Season": {
        "Tamil Nadu – Cauvery Delta":        ["Samba Rice (BPT-5204, ADT-36)", "Sugarcane", "Banana"],
        "Tamil Nadu – Coastal Tamil Nadu":   ["Rice", "Sugarcane", "Groundnut"],
        "Tamil Nadu – Southern Dry Zone":    ["Sorghum (Cholam)", "Cotton", "Castor"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Tea", "Potato", "Tomato"],
    },
    "Navarai (Jan – Mar) – TN Summer Crop": {
        "Tamil Nadu – Cauvery Delta":        ["Short-duration Rice", "Groundnut", "Sunflower"],
        "Tamil Nadu – Coastal Tamil Nadu":   ["Groundnut", "Watermelon", "Cucumber"],
        "Tamil Nadu – Southern Dry Zone":    ["Sunflower", "Groundnut", "Watermelon"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Potato", "Beans", "Peas"],
    },
    # ── AP / Telangana / Karnataka seasons ─────────────────────────────────
    "Kharif / SW Monsoon (Jun – Oct) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta":   ["Rice (BPT-5204, Swarna)", "Maize", "Sugarcane"],
        "Andhra Pradesh – Krishna Delta":    ["Rice", "Sugarcane", "Cotton"],
        "Andhra Pradesh – Coastal Andhra":   ["Rice", "Groundnut", "Blackgram"],
        "Andhra Pradesh – Rayalaseema":      ["Cotton", "Groundnut", "Castor"],
        "Telangana – Northern Telangana Dry Zone": ["Cotton", "Soybean", "Maize"],
        "Telangana – Eastern Ghats Upland":  ["Maize", "Redgram (Tur)", "Sorghum"],
        "Karnataka – Deccan Plateau":        ["Jowar (Sorghum)", "Cotton", "Maize"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Rice", "Maize", "Sunflower"],
        "Karnataka – Northern Dry Zone":     ["Cotton", "Soybean", "Pigeonpea"],
        "Karnataka – Malnad (Western Ghats)": ["Rice", "Arecanut", "Pepper"],
    },
    "Rabi / Winter (Nov – Feb) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta":   ["Rabi Rice", "Maize", "Bengal Gram"],
        "Andhra Pradesh – Krishna Delta":    ["Rabi Rice", "Groundnut", "Bengalgram"],
        "Andhra Pradesh – Coastal Andhra":   ["Groundnut", "Bengalgram", "Sunflower"],
        "Andhra Pradesh – Rayalaseema":      ["Bengalgram (Chickpea)", "Sunflower", "Safflower"],
        "Telangana – Northern Telangana Dry Zone": ["Sorghum (Rabi)", "Chickpea", "Sunflower"],
        "Telangana – Eastern Ghats Upland":  ["Chickpea", "Linseed", "Safflower"],
        "Karnataka – Deccan Plateau":        ["Chickpea", "Sunflower", "Safflower"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Sunflower", "Wheat", "Maize"],
        "Karnataka – Northern Dry Zone":     ["Chickpea", "Wheat", "Safflower"],
        "Karnataka – Malnad (Western Ghats)": ["Arecanut (off-season care)", "Pepper", "Ginger"],
    },
    "Summer / Zaid (Mar – May) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta":   ["Watermelon", "Muskmelon", "Moong (Greengram)"],
        "Andhra Pradesh – Krishna Delta":    ["Greengram (Moong)", "Watermelon", "Groundnut"],
        "Andhra Pradesh – Coastal Andhra":   ["Greengram", "Sesame", "Watermelon"],
        "Andhra Pradesh – Rayalaseema":      ["Groundnut", "Sesame", "Watermelon"],
        "Telangana – Northern Telangana Dry Zone": ["Greengram", "Sesame", "Watermelon"],
        "Telangana – Eastern Ghats Upland":  ["Greengram", "Cowpea", "Vegetables"],
        "Karnataka – Deccan Plateau":        ["Sunflower", "Greengram", "Watermelon"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Greengram", "Sunflower", "Vegetables"],
        "Karnataka – Northern Dry Zone":     ["Greengram", "Watermelon", "Sesame"],
        "Karnataka – Malnad (Western Ghats)": ["Vegetables", "Banana", "Turmeric"],
    },
    # ── Kerala seasons ──────────────────────────────────────────────────────
    "Virippu / SW Monsoon (Jun – Aug) – Kerala": {
        "Kerala – Palakkad Gap":             ["Rice (Jyothi, Uma)", "Banana", "Vegetables"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Rubber", "Cardamom", "Pepper"],
        "Kerala – Coastal Kerala":           ["Coconut", "Tapioca", "Banana"],
    },
    "Mundakan / NE Monsoon (Sep – Nov) – Kerala": {
        "Kerala – Palakkad Gap":             ["Rice (Mundakan)", "Cowpea", "Sesame"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Ginger", "Turmeric", "Pepper"],
        "Kerala – Coastal Kerala":           ["Tapioca", "Banana", "Vegetables"],
    },
    "Puncha / Winter Paddy (Dec – Feb) – Kerala": {
        "Kerala – Palakkad Gap":             ["Puncha Rice", "Vegetables", "Watermelon"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Coffee", "Cardamom", "Tea"],
        "Kerala – Coastal Kerala":           ["Coconut", "Vegetables", "Banana"],
    },
}
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml"
}
# ─────────────────────────────────────────────────────────────────────────────────
# ML MODEL CONFIGURATION (UPDATED – 5 MODELS)
# ─────────────────────────────────────────────────────────────────────────────────
ML_MODEL_DIR     = "models"
ML_NUMERIC_FEATS = ["nitrogen", "phosphorus", "potassium",
                    "ph", "moisture", "temperature", "humidity", "rainfall"]
ML_CAT_FEATS     = ["soil_type", "season", "region"]


@st.cache_resource
def load_ml_models():
    """Load all 5 trained classification models from disk."""
    def _load(name):
        path = os.path.join(ML_MODEL_DIR, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    return {
        "rf":  _load("random_forest_model.pkl"),
        "log": _load("logistic_regression_model.pkl"),
        "svm": _load("svm_model.pkl"),
        "xgb": _load("xgboost_model.pkl"),
        "cat": _load("catboost_model.pkl"),
        "le":  _load("label_encoder.pkl"),
        "fe":  _load("feature_encoder.pkl"),
    }


def ml_build_input(inputs: dict, fe) -> np.ndarray:
    """Encode a user input dict into the feature vector expected by all models."""
    cat_df = pd.DataFrame([[inputs[c] for c in ML_CAT_FEATS]], columns=ML_CAT_FEATS)
    cat_encoded = fe.transform(cat_df)
    num_values = np.array([[inputs[c] for c in ML_NUMERIC_FEATS]])
    return np.hstack([num_values, cat_encoded])


def ml_get_top5_probs(model, X, label_encoder):
    """Return predicted crop and top-5 probabilities."""
    probs = model.predict_proba(X)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = {label_encoder.classes_[i]: round(float(probs[i]) * 100, 2) for i in top5_idx}
    predicted = label_encoder.classes_[np.argmax(probs)]
    return predicted, top5


def ml_predict_rf(X, models):
    return ml_get_top5_probs(models["rf"], X, models["le"])


def ml_predict_log(X, models):
    return ml_get_top5_probs(models["log"], X, models["le"])


def ml_predict_svm(X, models):
    return ml_get_top5_probs(models["svm"], X, models["le"])


def ml_predict_xgb(X, models):
    return ml_get_top5_probs(models["xgb"], X, models["le"])


def ml_predict_cat(X, models):
    return ml_get_top5_probs(models["cat"], X, models["le"])


def ml_crop_badge(crop: str, color="#155724", bg="#d4edda", border="#c3e6cb"):
    st.markdown(
        f"""
        <div style="text-align:center;margin:12px 0;">
            <span style="background:{bg};color:{color};border:2px solid {border};
                border-radius:16px;padding:10px 28px;font-size:1.3rem;
                font-weight:700;display:inline-block;">
                🌱 {crop}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ml_confidence_bar(crop: str, pct: float, is_top: bool = False):
    bar_color = "#28a745" if is_top else "#6c757d"
    st.markdown(
        f"""
        <div style="margin:4px 0;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="width:180px;font-weight:{'700' if is_top else '400'};
                    font-size:0.9rem;">{crop}</span>
                <div style="flex:1;background:#e9ecef;border-radius:6px;height:18px;">
                    <div style="width:{pct}%;background:{bar_color};height:18px;
                        border-radius:6px;"></div>
                </div>
                <span style="width:55px;text-align:right;font-size:0.9rem;
                    font-weight:{'700' if is_top else '400'};">{pct:.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ml_show_model_result(title, predicted, top5, badge_color="#155724", badge_bg="#d4edda", badge_border="#c3e6cb"):
    st.markdown(f"### {title}")
    ml_crop_badge(predicted, color=badge_color, bg=badge_bg, border=badge_border)

    st.markdown("**Top 5 Crop Probabilities:**")
    top_crop = list(top5.keys())[0]
    for crop, pct in top5.items():
        ml_confidence_bar(crop, pct, is_top=(crop == top_crop))


def ml_show_comparison_table(predictions):
    st.markdown("### 📊 Model Comparison Summary")

    df = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Predicted Crop": list(predictions.values())
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    crop_counts = df["Predicted Crop"].value_counts()
    majority_crop = crop_counts.idxmax()
    majority_count = crop_counts.max()

    if majority_count >= 3:
        st.success(f"✅ Majority Recommendation: **{majority_crop}** ({majority_count}/5 models agree)")
    else:
        st.info("📌 Best performing models based on validation: XGBoost, Random Forest, and CatBoost.")
# ─────────────────────────────────────────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────────────────────────────────────────
CSV_FILE = "predictions.csv"
_CSV_COLUMNS = [
    "timestamp", "prediction_type",
    "nitrogen", "phosphorus", "potassium", "ph", "moisture", "soil_type",
    "temperature", "humidity", "rainfall", "season", "region",
    "crop", "question", "context", "ai_output",
]


def save_to_csv(prediction_type, **kwargs):
    """Append one prediction row to the shared predictions.csv file."""
    file_exists = os.path.isfile(CSV_FILE)
    row = {col: "" for col in _CSV_COLUMNS}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["prediction_type"] = prediction_type
    row.update(kwargs)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────────
# AI FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────────
def get_crop_recommendation(soil_data):
    """Get AI-powered crop recommendations based on soil parameters."""
    prompt = f"""You are an agricultural advisor. Recommend the best crops based on these conditions:

Soil: N={soil_data['nitrogen']} kg/ha, P={soil_data['phosphorus']} kg/ha, K={soil_data['potassium']} kg/ha, pH={soil_data['ph']}, Moisture={soil_data['moisture']}%, Type={soil_data['soil_type']}

Environment: Temp={soil_data['temperature']}°C, Humidity={soil_data['humidity']}%, Rainfall={soil_data['rainfall']}mm, Season={soil_data['season']}, Region={soil_data['region']}

Provide:
1. Top 3 Recommended Crops with suitability score
2. Why each crop suits these conditions
3. Expected yield
4. Best planting time

Use clear formatting with bullet points."""

    try:
        response = model.generate_content(prompt)

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text

        if hasattr(response, 'text') and response.text:
            return response.text

        return "Unable to generate recommendations. Please try again."

    except Exception as e:
        return f"Error: {str(e)}. Please try again."


def get_nutrient_advisory(soil_data, selected_crop):
    """Get nutrient recommendations for a specific crop."""
    prompt = f"""You are an agricultural advisor. Provide fertilizer recommendations for growing {selected_crop}.

Soil Data:
- Nitrogen: {soil_data['nitrogen']} kg/ha
- Phosphorus: {soil_data['phosphorus']} kg/ha
- Potassium: {soil_data['potassium']} kg/ha
- pH: {soil_data['ph']}
- Soil Type: {soil_data['soil_type']}

Provide:
1. Nutrient Analysis - What is deficient or excess?
2. Fertilizer Recommendations - Which fertilizers and how much per hectare?
3. Application Schedule - When to apply?
4. Organic Options - Natural alternatives?
5. pH Correction - If needed, how to adjust?

Give practical advice for farmers."""

    try:
        response = model.generate_content(prompt)

        # Check if response has valid parts
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text

        # Fallback to response.text
        if hasattr(response, 'text') and response.text:
            return response.text

        return "Unable to generate advisory. Please try again with different parameters."

    except Exception as e:
        return f"Error: {str(e)}. Please try again."


def get_farming_advice(question, context=""):
    """Get AI response for general farming questions."""
    prompt = f"""You are an agricultural expert. Answer this farming question clearly and practically.

{f"Context: {context}" if context else ""}

Question: {question}

Provide helpful advice that farmers can easily understand and implement."""

    try:
        response = model.generate_content(prompt)

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text

        if hasattr(response, 'text') and response.text:
            return response.text

        return "Unable to get advice. Please try again."

    except Exception as e:
        return f"Error: {str(e)}. Please try again."
def generate_summary(text):

    prompt = f"""
Summarize the following agricultural explanation for farmers.

Rules:
• Only 5 to 7 bullet points
• Very simple language
• Practical advice

Text:
{text}
"""

    try:
        response = model.generate_content(prompt)

        if response.candidates:
            return response.candidates[0].content.parts[0].text

        return "Summary unavailable"

    except:
        return "Summary generation failed"
#translation
def translate_text(text, language_code):

    if language_code == "en":
        return text

    try:
        translated = GoogleTranslator(
            source='auto',
            target=language_code
        ).translate(text)

        return translated

    except:
        return text
#clean
import re

def clean_text_for_audio(text):

    # Remove markdown symbols
    text = re.sub(r'[#*`]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove bullet characters
    text = text.replace("•", "")

    # Remove dashes used as bullets
    text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)

    return text.strip()
#audio generation
def generate_audio(text, language_code):

    try:

        # CLEAN TEXT BEFORE AUDIO
        clean_text = clean_text_for_audio(text)

        filename = f"audio_{uuid.uuid4().hex}.mp3"

        tts = gTTS(
            text=clean_text,
            lang=language_code
        )

        tts.save(filename)

        return filename

    except:
        return None
#display
def show_summary_audio(full_text, tab_key):

    st.markdown("---")

    language = st.selectbox(
        "🌐 Select Output Language",
        list(LANGUAGES.keys()),
        key=f"lang_{tab_key}"
    )

    lang_code = LANGUAGES[language]

    summary = generate_summary(full_text)

    translated_full = translate_text(full_text, lang_code)
    translated_summary = translate_text(summary, lang_code)

    st.markdown("### 📖 Detailed Explanation")
    st.markdown(translated_full)

    st.markdown("### 📌 Quick Summary for Farmers")
    st.markdown(translated_summary)

    audio_file = generate_audio(translated_summary, lang_code)

    if audio_file:
        st.audio(audio_file)   
# ─────────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────────

# ── Synced Slider + Number Input ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# SYNCED SLIDER + NUMBER INPUT (NO SESSION STATE ERROR)
# ─────────────────────────────────────────────────────────────────
def synced_slider_input(label, min_val, max_val, default, step=1, key="param"):

    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    # Initialize
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    if input_key not in st.session_state:
        st.session_state[input_key] = default

    # Callback functions
    def update_from_slider():
        st.session_state[input_key] = st.session_state[slider_key]

    def update_from_input():
        st.session_state[slider_key] = st.session_state[input_key]

    col1, col2 = st.columns([3,1])

    with col1:
        st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=slider_key,
            on_change=update_from_slider
        )

    with col2:
        st.number_input(
            " ",
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=input_key,
            on_change=update_from_input
        )

    return st.session_state[slider_key]

# ─────────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────────
def show_header():
    st.title("🌾 Smart Crop & Nutrient Advisory System")
    st.markdown("### AI-Powered Agricultural Guidance for Farmers · 🤖 ML Crop Prediction")
    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────────
# SOIL INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────────
def show_soil_input_form(key_prefix="tab1"):

    st.markdown("""
### 📊 Soil & Environment Parameters
<div style="
background:#ffffff;
padding:20px;
border-radius:10px;
box-shadow:0 2px 10px rgba(0,0,0,0.05);
margin-bottom:20px;
">
Adjust the parameters below based on your field conditions.
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)


    # ── Soil Nutrients ──────────────────────────────────────────────────────────
    with col1:

        st.markdown("""
<div style="
background:#e8f5e9;
padding:10px;
border-radius:6px;
font-weight:600;
margin-bottom:10px;">
🧪 Soil Nutrients
</div>
""", unsafe_allow_html=True)

        nitrogen = synced_slider_input(
            "Nitrogen (N) kg/ha",
            0,
            150,
            50,
            key=f"{key_prefix}_nitrogen"
        )

        phosphorus = synced_slider_input(
            "Phosphorus (P) kg/ha",
            0,
            150,
            40,
            key=f"{key_prefix}_phosphorus"
        )

        potassium = synced_slider_input(
            "Potassium (K) kg/ha",
            0,
            250,
            50,
            key=f"{key_prefix}_potassium"
        )

    # ── Soil Properties ─────────────────────────────────────────────────────────
    with col2:

        st.markdown("""
<div style="
background:#e3f2fd;
padding:10px;
border-radius:6px;
font-weight:600;
margin-bottom:10px;">
🌱 Soil Properties
</div>
""", unsafe_allow_html=True)

        ph = synced_slider_input(
            "pH Level",
            3.0,
            10.0,
            6.5,
            step=0.1,
            key=f"{key_prefix}_ph"
        )

        moisture = synced_slider_input(
            "Soil Moisture %",
            0,
            100,
            50,
            key=f"{key_prefix}_moisture"
        )

        soil_type = st.selectbox(
            "Soil Type",
            SOIL_TYPES,
            key=f"{key_prefix}_soil_type"
        )

    # ── Environmental Conditions ───────────────────────────────────────────────
    with col3:

        st.markdown("""
<div style="
background:#fff8e1;
padding:10px;
border-radius:6px;
font-weight:600;
margin-bottom:10px;">
🌤 Environmental Conditions
</div>
""", unsafe_allow_html=True)

        temperature = synced_slider_input(
            "Temperature (°C)",
            5,
            50,
            28,
            key=f"{key_prefix}_temperature"
        )

        humidity = synced_slider_input(
            "Humidity %",
            10,
            100,
            65,
            key=f"{key_prefix}_humidity"
        )

        rainfall = synced_slider_input(
            "Rainfall (mm/year)",
            0,
            3000,
            1000,
            key=f"{key_prefix}_rainfall"
        )

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        season = st.selectbox(
            "🗓️ Cropping Season",
            SEASONS,
            key=f"{key_prefix}_season"
        )

    with col5:
        region = st.selectbox(
            "📍 Region",
            REGIONS,
            key=f"{key_prefix}_region"
        )

    return {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "moisture": moisture,
        "soil_type": soil_type,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "season": season,
        "region": region,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────────
def display_recommendation(recommendation):

    st.markdown("### 🌱 Crop Recommendations")
    st.markdown(recommendation)


def display_nutrient_advisory(advisory):

    st.markdown("### 🧪 Nutrient Advisory")
    st.markdown(advisory)


# ─────────────────────────────────────────────────────────────────────────────────
# HIGH DEMAND CROPS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────────
def show_high_demand_crops(season, region):

    crops = HIGH_DEMAND_CROPS.get(season, {}).get(region)

    if crops:

        badges = " ".join(
            f'<span style="background:#d4edda;color:#155724;border:1px solid #c3e6cb;'
            f'border-radius:12px;padding:4px 12px;margin:3px;display:inline-block;'
            f'font-weight:600;font-size:0.9rem;">{crop}</span>'
            for crop in crops
        )

        st.markdown(
            f"""
            <div style="background:#f0fff4;border:1px solid #b2dfdb;border-radius:8px;
                        padding:16px 20px;margin-bottom:16px;">
                <p style="margin:0 0 8px 0;font-weight:700;color:#1b5e20;font-size:1rem;">
                    📈 High-Demand Crops
                </p>
                <p style="margin:0 0 10px 0;color:#4a4a4a;font-size:0.88rem;">
                    <strong>Season:</strong> {season}&nbsp;&nbsp;|&nbsp;&nbsp;
                    <strong>Region:</strong> {region}
                </p>
                <div>{badges}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "No high-demand crop data available for this season–region combination. "
            "Showing AI recommendations only.",
            icon="ℹ️",
        )


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────────
def main():
    show_header()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Crop Recommendation",
        "🧪 Nutrient Advisory",
        "💬 Ask Expert",
        "🤖 ML Crop Prediction",
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 1: CROP RECOMMENDATION
    # ─────────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Get personalized crop recommendations based on your soil and environmental conditions")

        soil_data = show_soil_input_form("tab1")

        # Store soil data in session state for other tabs
        st.session_state.soil_data = soil_data

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            get_recommendation = st.button("🔍 Get Crop Recommendations", type="primary", use_container_width=True)

        if get_recommendation:
            with st.spinner("🌱 Analyzing soil parameters and generating recommendations..."):
                recommendation = get_crop_recommendation(soil_data)
                st.session_state.crop_recommendation = recommendation
                st.session_state.last_season = soil_data["season"]
                st.session_state.last_region = soil_data["region"]
                save_to_csv(
                    "crop_recommendation",
                    nitrogen=soil_data["nitrogen"],
                    phosphorus=soil_data["phosphorus"],
                    potassium=soil_data["potassium"],
                    ph=soil_data["ph"],
                    moisture=soil_data["moisture"],
                    soil_type=soil_data["soil_type"],
                    temperature=soil_data["temperature"],
                    humidity=soil_data["humidity"],
                    rainfall=soil_data["rainfall"],
                    season=soil_data["season"],
                    region=soil_data["region"],
                    ai_output=recommendation,
                )

        if "crop_recommendation" in st.session_state:
            st.markdown("---")
            st.success("✅ Recommendations generated successfully!")
            show_high_demand_crops(
                st.session_state.get("last_season", ""),
                st.session_state.get("last_region", ""),
            )
            show_summary_audio(
    st.session_state.crop_recommendation,
    "tab1"
)

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 2: NUTRIENT ADVISORY
    # ─────────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Get detailed nutrient and fertilizer recommendations for your selected crop")

        # Editable soil parameters
        st.markdown("**🧪 Soil Parameters:**")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            n_nutrient = st.number_input("Nitrogen (N) kg/ha", 0, 150, 50, key="n_tab2")
        with col2:
            p_nutrient = st.number_input("Phosphorus (P) kg/ha", 0, 150, 40, key="p_tab2")
        with col3:
            k_nutrient = st.number_input("Potassium (K) kg/ha", 0, 250, 50, key="k_tab2")
        with col4:
            ph_nutrient = st.number_input("pH Level", 3.0, 10.0, 6.5, 0.1, key="ph_tab2")
        with col5:
            soil_type_nutrient = st.selectbox("Soil Type", SOIL_TYPES, key="soil_tab2")

        # Store for use in advisory
        st.session_state.nutrient_soil_data = {
            "nitrogen": n_nutrient,
            "phosphorus": p_nutrient,
            "potassium": k_nutrient,
            "ph": ph_nutrient,
            "soil_type": soil_type_nutrient
        }

        st.markdown("---")

        # Crop selection
        common_crops = [
            "Rice", "Wheat", "Maize (Corn)", "Cotton", "Sugarcane",
            "Soybean", "Groundnut", "Mustard", "Chickpea (Gram)", "Pigeon Pea (Tur)",
            "Potato", "Tomato", "Onion", "Chilli", "Banana",
            "Mango", "Tea", "Coffee", "Coconut", "Other"
        ]

        selected_crop = st.selectbox("🌿 Select Crop for Nutrient Advisory", common_crops)

        if selected_crop == "Other":
            selected_crop = st.text_input("Enter crop name:")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            get_advisory = st.button("🧪 Get Nutrient Advisory", type="primary", use_container_width=True)

        if get_advisory:
            if selected_crop:
                with st.spinner("🔬 Analyzing nutrient requirements..."):
                    advisory = get_nutrient_advisory(st.session_state.nutrient_soil_data, selected_crop)
                    st.session_state.nutrient_advisory = advisory
                    save_to_csv(
                        "nutrient_advisory",
                        nitrogen=st.session_state.nutrient_soil_data["nitrogen"],
                        phosphorus=st.session_state.nutrient_soil_data["phosphorus"],
                        potassium=st.session_state.nutrient_soil_data["potassium"],
                        ph=st.session_state.nutrient_soil_data["ph"],
                        soil_type=st.session_state.nutrient_soil_data["soil_type"],
                        crop=selected_crop,
                        ai_output=advisory,
                    )
            else:
                st.warning("Please select a crop")

        if "nutrient_advisory" in st.session_state:
            st.markdown("---")
            st.success("✅ Nutrient advisory generated successfully!")
            show_summary_audio(
    st.session_state.nutrient_advisory,
    "tab2"
)
    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 3: ASK EXPERT (AI CHAT)
    # ─────────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Ask any farming-related question to our AI Expert")

        # Sample questions
        st.markdown("**💡 Sample Questions:**")
        sample_questions = [
            "What is the best time to plant rice in North India?",
            "How to control pest attacks on cotton crops?",
            "What are the signs of nitrogen deficiency in plants?",
            "How to improve soil fertility naturally?",
            "What is crop rotation and why is it important?"
        ]

        cols = st.columns(3)
        for i, q in enumerate(sample_questions):
            with cols[i % 3]:
                if st.button(f"📌 {q[:40]}...", key=f"sample_{i}", use_container_width=True):
                    st.session_state.user_question = q

        st.markdown("---")

        # Question input
        user_question = st.text_area(
            "🤔 Your Question:",
            value=st.session_state.get("user_question", ""),
            placeholder="Ask any question about farming, crops, fertilizers, pest control, irrigation, etc.",
            height=100
        )

        # Include soil data context option
        include_context = st.checkbox("Include my soil parameters in the question context", value=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_expert = st.button("🎯 Get Expert Advice", type="primary", use_container_width=True)

        if ask_expert:
            if user_question:
                context = ""
                if include_context and "soil_data" in st.session_state:
                    data = st.session_state.soil_data
                    context = f"N={data['nitrogen']}, P={data['phosphorus']}, K={data['potassium']}, pH={data['ph']}, Region={data['region']}"

                with st.spinner("🤖 Getting expert advice..."):
                    advice = get_farming_advice(user_question, context)
                    st.session_state.expert_advice = advice
                    st.session_state.user_question = ""  # Clear after asking
                    save_to_csv(
                        "expert_advice",
                        question=user_question,
                        context=context,
                        ai_output=advice,
                    )
            else:
                st.warning("Please enter a question")

        if "expert_advice" in st.session_state:
            st.markdown("---")
            st.success("✅ Expert advice ready!")
            show_summary_audio(
    st.session_state.expert_advice,
    "tab3"
)

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 4: ML CROP PREDICTION (UPDATED – 5 MODELS)
    # ─────────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Predict the best crop using trained **5 ML classification models**")
        st.markdown("---")

        # Load models
        try:
            ml_models = load_ml_models()
        except FileNotFoundError:
            st.error("ML models not found in the `models/` folder.")
            st.info(
                "Run the following commands to generate the dataset and train the models:\n\n"
                "```bash\npython generate_dataset.py\npython train_model.py\n```"
            )
            st.stop()

        # Input form
        ml_soil = show_soil_input_form("tab4")
        st.markdown("---")

        # Prediction mode
        st.markdown("**🔧 Choose Prediction Mode:**")
        pred_mode = st.radio(
            "Mode",
            [
                "Compare All 5 Models",
                "Random Forest Only",
                "Logistic Regression Only",
                "SVM Only",
                "XGBoost Only",
                "CatBoost Only"
            ],
            horizontal=False,
            label_visibility="collapsed",
        )

        col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
        with col_b2:
            run_ml = st.button("🚀 Run ML Prediction", type="primary", use_container_width=True)

        if run_ml:
            X_ml = ml_build_input(ml_soil, ml_models["fe"])
            le = ml_models["le"]

            rf_crop, rf_top5 = ml_predict_rf(X_ml, ml_models)
            log_crop, log_top5 = ml_predict_log(X_ml, ml_models)
            svm_crop, svm_top5 = ml_predict_svm(X_ml, ml_models)
            xgb_crop, xgb_top5 = ml_predict_xgb(X_ml, ml_models)
            cat_crop, cat_top5 = ml_predict_cat(X_ml, ml_models)

            st.markdown("---")

            predictions = {
                "Random Forest": rf_crop,
                "Logistic Regression": log_crop,
                "SVM": svm_crop,
                "XGBoost": xgb_crop,
                "CatBoost": cat_crop,
            }

            # ── ALL MODELS COMPARISON ───────────────────────────────────────
            if pred_mode == "Compare All 5 Models":
                ml_show_comparison_table(predictions)

                st.markdown("### 🔍 Individual Model Outputs")

                col1, col2 = st.columns(2)

                with col1:
                    with st.container(border=True):
                        ml_show_model_result("🌲 Random Forest", rf_crop, rf_top5)

                    with st.container(border=True):
                        ml_show_model_result("🎯 SVM", svm_crop, svm_top5, "#5a189a", "#f3e8ff", "#d8b4fe")

                    with st.container(border=True):
                        ml_show_model_result("🧠 CatBoost", cat_crop, cat_top5, "#7c2d12", "#ffedd5", "#fdba74")

                with col2:
                    with st.container(border=True):
                        ml_show_model_result("📘 Logistic Regression", log_crop, log_top5, "#0c3547", "#cce5ff", "#b8daff")

                    with st.container(border=True):
                        ml_show_model_result("⚡ XGBoost", xgb_crop, xgb_top5, "#1f2937", "#e5e7eb", "#9ca3af")

            # ── SINGLE MODEL DISPLAY ────────────────────────────────────────
            model_outputs = {
                "Random Forest Only": ("🌲 Random Forest", rf_crop, rf_top5, "rf"),
                "Logistic Regression Only": ("📘 Logistic Regression", log_crop, log_top5, "log"),
                "SVM Only": ("🎯 SVM", svm_crop, svm_top5, "svm"),
                "XGBoost Only": ("⚡ XGBoost", xgb_crop, xgb_top5, "xgb"),
                "CatBoost Only": ("🧠 CatBoost", cat_crop, cat_top5, "cat"),
            }

            if pred_mode in model_outputs:
                title, pred_crop, top5, model_key = model_outputs[pred_mode]

                st.success(f"✅ Recommended Crop: **{pred_crop}**")

                with st.container(border=True):
                    ml_show_model_result(title, pred_crop, top5)

                model = ml_models[model_key]
                all_probs = model.predict_proba(X_ml)[0]

                prob_df = pd.DataFrame({
                    "Crop": le.classes_,
                    "Probability (%)": (all_probs * 100).round(2),
                }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

                with st.expander("📊 Full Probability Table (all crops)"):
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # ── Input summary ────────────────────────────────────────────────
            with st.expander("📋 Input Parameters Used"):
                param_df = pd.DataFrame([ml_soil]).T.reset_index()
                param_df.columns = ["Parameter", "Value"]
                st.dataframe(param_df, use_container_width=True, hide_index=True)

            # ── CSV Log ──────────────────────────────────────────────────────
            save_to_csv(
                "ml_prediction",
                nitrogen=ml_soil["nitrogen"],
                phosphorus=ml_soil["phosphorus"],
                potassium=ml_soil["potassium"],
                ph=ml_soil["ph"],
                moisture=ml_soil["moisture"],
                soil_type=ml_soil["soil_type"],
                temperature=ml_soil["temperature"],
                humidity=ml_soil["humidity"],
                rainfall=ml_soil["rainfall"],
                season=ml_soil["season"],
                region=ml_soil["region"],
                ai_output=(
                    f"RF:{rf_crop} | LOG:{log_crop} | SVM:{svm_crop} | "
                    f"XGB:{xgb_crop} | CAT:{cat_crop}"
                ),
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 10px;">
            🌾 Smart Crop & Nutrient Advisory System | Powered by AI + ML<br/>
            <small>Helping farmers make informed decisions for better yields</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
