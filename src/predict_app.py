"""
Crop Prediction App – Smart Crop Advisory System
=================================================
Streamlit UI that loads the trained ML models and predicts the best crop.

Models used:
  1. Random Forest
  2. Logistic Regression
  3. Support Vector Machine (SVM)
  4. XGBoost
  5. CatBoost

Run:
    streamlit run predict_app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Prediction – ML Models",
    page_icon="🌿",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS (must match generate_dataset.py / train_model.py)
# ──────────────────────────────────────────────────────────────────────────────
SOIL_TYPES = ["Sandy", "Clay", "Loamy", "Silt", "Peaty", "Chalky", "Saline"]

SEASONS = [
    "Kuruvai (Jun – Sep) – TN Early Kharif",
    "Samba (Aug – Jan) – TN Main Season",
    "Navarai (Jan – Mar) – TN Summer Crop",
    "Kharif / SW Monsoon (Jun – Oct) – AP / Telangana / Karnataka",
    "Rabi / Winter (Nov – Feb) – AP / Telangana / Karnataka",
    "Summer / Zaid (Mar – May) – AP / Telangana / Karnataka",
    "Virippu / SW Monsoon (Jun – Aug) – Kerala",
    "Mundakan / NE Monsoon (Sep – Nov) – Kerala",
    "Puncha / Winter Paddy (Dec – Feb) – Kerala",
]

REGIONS = [
    "Tamil Nadu – Cauvery Delta",
    "Tamil Nadu – Coastal Tamil Nadu",
    "Tamil Nadu – Nilgiris / High Altitude Zone",
    "Tamil Nadu – Southern Dry Zone",
    "Andhra Pradesh – Godavari Delta",
    "Andhra Pradesh – Krishna Delta",
    "Andhra Pradesh – Coastal Andhra",
    "Andhra Pradesh – Rayalaseema",
    "Telangana – Northern Telangana Dry Zone",
    "Telangana – Eastern Ghats Upland",
    "Karnataka – Deccan Plateau",
    "Karnataka – Krishna–Tungabhadra Basin",
    "Karnataka – Northern Dry Zone",
    "Karnataka – Malnad (Western Ghats)",
    "Kerala – Palakkad Gap",
    "Kerala – Western Ghats (High Rainfall Zone)",
    "Kerala – Coastal Kerala",
]

NUMERIC_FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "ph", "moisture", "temperature", "humidity", "rainfall"
]
CATEGORICAL_FEATURES = ["soil_type", "season", "region"]
MODEL_DIR = "models"

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    def load(name):
        path = os.path.join(MODEL_DIR, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    return {
        "rf": load("random_forest_model.pkl"),
        "log": load("logistic_regression_model.pkl"),
        "svm": load("svm_model.pkl"),
        "xgb": load("xgboost_model.pkl"),
        "cat": load("catboost_model.pkl"),
        "le": load("label_encoder.pkl"),
        "fe": load("feature_encoder.pkl"),
        "scaler": load("scaler.pkl"),   # IMPORTANT
    }

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_row(inputs: dict, feature_encoder) -> np.ndarray:
    """Encode a single input dict into a feature vector."""
    cat_df = pd.DataFrame(
        [[inputs[c] for c in CATEGORICAL_FEATURES]],
        columns=CATEGORICAL_FEATURES
    )
    cat_encoded = feature_encoder.transform(cat_df)
    num_values = np.array([[inputs[c] for c in NUMERIC_FEATURES]], dtype=float)
    return np.hstack([num_values, cat_encoded])


def get_top5_probs(model, X, label_encoder):
    """Return predicted crop and top-5 probabilities."""
    probs = model.predict_proba(X)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = {
        label_encoder.classes_[i]: round(float(probs[i]) * 100, 2)
        for i in top5_idx
    }
    predicted = label_encoder.classes_[np.argmax(probs)]
    return predicted, top5


def get_scaled_input(X, scaler):
    return scaler.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL-SPECIFIC PREDICTION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def predict_random_forest(X, models):
    return get_top5_probs(models["rf"], X, models["le"])


def predict_logistic(X, models):
    X_scaled = get_scaled_input(X, models["scaler"])
    return get_top5_probs(models["log"], X_scaled, models["le"])


def predict_svm(X, models):
    X_scaled = get_scaled_input(X, models["scaler"])
    return get_top5_probs(models["svm"], X_scaled, models["le"])


def predict_xgboost(X, models):
    return get_top5_probs(models["xgb"], X, models["le"])


def predict_catboost(X, models):
    return get_top5_probs(models["cat"], X, models["le"])


def get_model_probability_dataframe(model_key, X, models):
    """
    Returns full probability dataframe for the selected model.
    Applies scaling only where needed.
    """
    model = models[model_key]

    if model_key in ["log", "svm"]:
        X_input = get_scaled_input(X, models["scaler"])
    else:
        X_input = X

    all_probs = model.predict_proba(X_input)[0]

    prob_df = pd.DataFrame({
        "Crop": models["le"].classes_,
        "Probability (%)": (all_probs * 100).round(2),
    }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

    return prob_df


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def crop_badge(crop: str, color: str = "#155724", bg: str = "#d4edda", border: str = "#c3e6cb"):
    st.markdown(
        f"""
        <div style="text-align:center;margin:12px 0;">
            <span style="background:{bg};color:{color};border:2px solid {border};
                border-radius:16px;padding:10px 28px;font-size:1.4rem;
                font-weight:700;display:inline-block;">
                🌱 {crop}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_bar(crop: str, pct: float, is_top: bool = False):
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


def show_model_result(title, predicted, top5, badge_color="#155724", badge_bg="#d4edda", badge_border="#c3e6cb"):
    st.markdown(f"### {title}")
    crop_badge(predicted, color=badge_color, bg=badge_bg, border=badge_border)

    st.markdown("**Top 5 Crop Probabilities:**")
    top_crop = list(top5.keys())[0]
    for crop, pct in top5.items():
        confidence_bar(crop, pct, is_top=(crop == top_crop))


def show_comparison_table(predictions):
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
        st.warning("⚠️ No strong consensus among models. Consider XGBoost / Random Forest / CatBoost predictions more heavily.")


# ──────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────────────────────────
def show_input_form():
    st.subheader("📊 Enter Soil & Environmental Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧪 Soil Nutrients**")
        nitrogen = st.slider("Nitrogen (N) kg/ha", 0, 150, 80)
        phosphorus = st.slider("Phosphorus (P) kg/ha", 0, 150, 50)
        potassium = st.slider("Potassium (K) kg/ha", 0, 250, 60)

    with col2:
        st.markdown("**🌱 Soil Properties**")
        ph = st.slider("pH Level", 3.0, 10.0, 6.5, 0.1)
        moisture = st.slider("Soil Moisture %", 0, 100, 55)
        soil_type = st.selectbox("Soil Type", SOIL_TYPES)

    with col3:
        st.markdown("**🌤️ Environment**")
        temperature = st.slider("Temperature (°C)", 5, 50, 28)
        humidity = st.slider("Humidity %", 10, 100, 65)
        rainfall = st.slider("Rainfall (mm/yr)", 0, 3000, 1000)

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        season = st.selectbox("🗓️ Cropping Season", SEASONS)
    with col5:
        region = st.selectbox("📍 Region", REGIONS)

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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🌿 Crop Prediction – 5 ML Model Comparison")
    st.markdown("#### Predict the best crop using **5 machine learning algorithms**")
    st.markdown("---")

    try:
        models = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python generate_dataset.py` then `python train_model.py` to create the models.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Compare All Models",
        "🌲 Random Forest",
        "📘 Logistic Regression",
        "🎯 SVM",
        "⚡ XGBoost",
        "🧠 CatBoost",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – ALL MODELS
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("##### Fill in the parameters and compare predictions from all 5 models.")
        inputs = show_input_form()

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            run_all = st.button("🚀 Predict with All Models", type="primary", use_container_width=True)

        if run_all:
            X = build_feature_row(inputs, models["fe"])

            rf_crop, rf_top5 = predict_random_forest(X, models)
            log_crop, log_top5 = predict_logistic(X, models)
            svm_crop, svm_top5 = predict_svm(X, models)
            xgb_crop, xgb_top5 = predict_xgboost(X, models)
            cat_crop, cat_top5 = predict_catboost(X, models)

            predictions = {
                "Random Forest": rf_crop,
                "Logistic Regression": log_crop,
                "SVM": svm_crop,
                "XGBoost": xgb_crop,
                "CatBoost": cat_crop,
            }

            st.markdown("---")
            show_comparison_table(predictions)

            st.markdown("### 🔍 Individual Model Outputs")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    show_model_result("🌲 Random Forest", rf_crop, rf_top5)
                with st.container(border=True):
                    show_model_result("🎯 SVM", svm_crop, svm_top5, "#5a189a", "#f3e8ff", "#d8b4fe")
                with st.container(border=True):
                    show_model_result("🧠 CatBoost", cat_crop, cat_top5, "#7c2d12", "#ffedd5", "#fdba74")
            with col2:
                with st.container(border=True):
                    show_model_result("📘 Logistic Regression", log_crop, log_top5, "#0c3547", "#cce5ff", "#b8daff")
                with st.container(border=True):
                    show_model_result("⚡ XGBoost", xgb_crop, xgb_top5, "#1f2937", "#e5e7eb", "#9ca3af")

            with st.expander("📋 Input Parameters Used"):
                param_df = pd.DataFrame([inputs]).T.reset_index()
                param_df.columns = ["Parameter", "Value"]
                st.dataframe(param_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL MODEL TABS
    # ════════════════════════════════════════════════════════════════════════
    model_tabs = [
        (tab2, "🌲 Random Forest", predict_random_forest, "rf"),
        (tab3, "📘 Logistic Regression", predict_logistic, "log"),
        (tab4, "🎯 SVM", predict_svm, "svm"),
        (tab5, "⚡ XGBoost", predict_xgboost, "xgb"),
        (tab6, "🧠 CatBoost", predict_catboost, "cat"),
    ]

    for tab, model_name, predict_func, model_key in model_tabs:
        with tab:
            st.markdown(f"##### Predict crop using **{model_name}**")
            inputs_model = show_input_form()

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_model = st.button(f"🚀 Predict with {model_name}", type="primary", use_container_width=True)

            if run_model:
                X = build_feature_row(inputs_model, models["fe"])
                pred_crop, top5 = predict_func(X, models)

                st.markdown("---")
                st.success(f"✅ Recommended Crop: **{pred_crop}**")

                with st.container(border=True):
                    show_model_result(model_name, pred_crop, top5)

                prob_df = get_model_probability_dataframe(model_key, X, models)

                with st.expander("📊 Full Probability Table (all crops)"):
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#888;font-size:0.85rem;padding:8px;">
            🌾 Smart Crop Advisory System &nbsp;|&nbsp; ML Prediction Engine<br/>
            <small>Models: Random Forest · Logistic Regression · SVM · XGBoost · CatBoost<br/>
            Run <code>python train_model.py</code> to retrain</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()