# 🏠 Bengaluru House Price Predictor

An end-to-end machine learning project that predicts house prices in Bengaluru based on location, square footage, number of bedrooms, and bathrooms.

**Live demo:** _[Add your Streamlit Cloud URL here once deployed]_

![App screenshot](screenshot.png)

---

## What it does

Enter a location, total square feet, BHK, and bathroom count — the app returns an estimated price in lakhs and rupees.

Under the hood, the app loads a trained Linear Regression model and constructs a 244-dimension feature vector (3 numeric features + 241 one-hot encoded locations) to make the prediction.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas, NumPy |
| ML | scikit-learn (Linear Regression, Random Forest) |
| App | Streamlit |
| Serialization | pickle, JSON |
| Dev | VS Code, Git, GitHub |

---

## Dataset

**Bengaluru House Price Data** — 13,320 rows, 9 columns, sourced from Kaggle.

Raw features: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`.

---

## Project Structure
