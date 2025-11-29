# 🍦 DemandSense: AI Inventory Optimizer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Model-Facebook%20Prophet-orange)](https://facebook.github.io/prophet/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-green)](https://plotly.com/)

A multivariate Time Series forecasting application designed to optimize retail inventory levels. It uses **Facebook Prophet** to predict daily demand based on historical sales, weekly seasonality, and external weather factors.

![Dashboard Preview]([INSERT YOUR DASHBOARD SCREENSHOT LINK HERE])

---

## 💼 Business Value
Stockouts (running out of inventory) cost retailers billions in lost revenue every year. Static forecasting methods (like Excel averages) fail to account for external variables like heatwaves or holidays.

**DemandSense** solves this by:
1.  **Preventing Stockouts:** Calculating the exact date inventory will hit zero based on predicted demand.
2.  **Scenario Planning:** Allowing supply chain managers to simulate "What-If" weather scenarios (e.g., "What if next week is 10° hotter?") to adjust procurement.
3.  **Reducing Waste:** improving forecast accuracy by accounting for weekly seasonality (weekend spikes).

---

## 🌟 Key Features

### 1. 🔮 Multivariate Forecasting (Prophet)
* Uses **Facebook Prophet** additive regression model.
* Incorporates **External Regressors** (Temperature) to model the correlation between weather and product demand.
* Models **Weekly Seasonality** to capture weekend sales spikes automatically.

### 2. 🎛️ Interactive Scenario Planner
* A real-time interface where users can adjust the **Average Expected Temperature** for the next 30 days.
* The model re-runs inference on the fly, updating the demand curve instantly.

### 3. 📦 Inventory Intelligence
* **Stockout Calculator:** dynamic logic that combines *Current Stock* (User Input) with *Predicted Demand* (Model Output) to flag critical low-stock dates.
* **Alert System:** Visual warning system for "High Heat" or "Low Stock" events.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interactive web dashboard for scenario planning |
| **Model** | Facebook Prophet | Time-series forecasting with additive regressors |
| **Visualization** | Plotly | Interactive charts with zoom/pan and confidence intervals |
| **Data Processing** | Pandas / NumPy | Temporal data manipulation and aggregation |
| **Metrics** | Scikit-Learn | Calculation of Mean Absolute Error (MAE) for validation |

---

## 🚀 How to Run Locally

**1. Clone the Repository**
```bash
git clone https://github.com/Shreyasc0de/DemandSense.git
cd DemandSense
```

**2. Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the App**
```bash
streamlit run forecast_app.py
```

---

## 📸 Dashboard Preview
Add a screenshot of your dashboard here for visual reference.

---

## 📄 License
MIT
