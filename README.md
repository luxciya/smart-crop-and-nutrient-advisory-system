# 🌱 Smart Crop & Nutrient Advisory System

An AI and machine learning-based application that provides crop and fertilizer recommendations based on agricultural and regional information. The system also provides an AI-powered chat assistant and market-demand information to support farmers in making better decisions.

## 📌 Problem

Farmers need to consider different factors such as region, season, crop conditions, soil-related information, and fertilizer requirements when making agricultural decisions.

This project aims to provide data-driven recommendations through a simple web application.

## 💡 What I Built

I developed a **Smart Crop & Nutrient Advisory System** using Python, machine learning, Streamlit, Gemini.

The application provides:

* 🌾 Crop recommendations
* 🧪 Fertilizer recommendations
* 🤖 AI-powered agricultural chat assistant
* 📊 Market-demand information
* 💾 Storage of user inputs and recommendations
* 📄 Data collection and analysis for improving recommendations

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Machine Learning**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Gemini API**
* **CSV**
* **Jupyter Notebook**

## 🤖 Machine Learning

Different machine learning algorithms were explored and evaluated for the recommendation system.

Models included:

* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* XGBoost
* CatBoost

The Random Forest model achieved approximately **93.95% accuracy** on the evaluated dataset.

Other evaluated results included:

| Model               | Result |
| ------------------- | -----: |
| Random Forest       | 93.95% |
| XGBoost             | 93.09% |
| CatBoost            | 92.72% |
| SVM                 | 83.21% |
| Logistic Regression | 66.23% |

> Results depend on the dataset, preprocessing, train/test split, and evaluation method used.

## 📊 Dataset

The project uses agricultural data covering regions from:

* Tamil Nadu
* Andhra Pradesh
* Telangana
* Karnataka
* Kerala

The dataset contains information across:

* **17 regions**
* **9 seasons**
* **45 crops**
* Approximately **7,650 records**

## 🏗️ Project Workflow

```text
User Input
    ↓
Agricultural Data Processing
    ↓
Machine Learning Model
    ↓
Crop / Fertilizer Recommendation
    ↓
AI Assistant
    ↓
Recommendation & Information
```

## 🌐 Application

The application was developed using **Streamlit** to provide an interactive web interface.

Users can enter relevant agricultural information and receive recommendations through the application.

## 🎯 My Role

I worked on:

* Data preprocessing
* Machine learning model development
* Model comparison
* Recommendation system
* Streamlit application
* Gemini API integration
* Local Data storage and processing
* Testing and evaluation

## 📷 Screenshots

Add your project screenshots here.

Example:

```text
screenshots/
├── home.png
├── crop-recommendation.png
├── fertilizer-recommendation.png
├── ai-assistant.png
└── market-demand.png
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-crop-nutrient-advisory-system.git
cd smart-crop-nutrient-advisory-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Add your required API credentials using environment variables or your preferred secure configuration method.

Do **not** upload API keys or passwords to GitHub.

### 4. Run the application

```bash
streamlit run app.py
```

## 🔐 Security

API keys, database credentials, and other private credentials should not be included in this repository.

Use environment variables or a `.env` file locally and add `.env` to `.gitignore`.

## 🔮 Future Improvements

* Add real-time weather information
* Improve regional recommendations
* Add voice-based agricultural assistance
* Add more agricultural datasets
* Improve model evaluation
* Add multilingual support
* Deploy the application for public use

## 👩‍💻 Project

**Smart Crop & Nutrient Advisory System**

Built as an academic/final-year AI and machine learning project.
