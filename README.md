# TransferIQ ⚽
### AI-Powered Football Scouting and Transfer Intelligence Platform

TransferIQ is an AI-powered football scouting and transfer intelligence platform designed to support data-driven player evaluation and recruitment decisions.

It uses machine learning to analyze player performance, identify similar players, estimate market values, and discover potential transfer bargains.

## 🚀 Key Features

### 1. KNN Player Similarity
Finds the top 7 players who are statistically similar to a selected player using the K-Nearest Neighbors (KNN) algorithm.

### 2. Market Value Prediction
Uses a trained Random Forest regression model to estimate a football player's market value based on relevant player attributes and statistics.

### 3. Bargain Finder
Helps identify potentially undervalued players by comparing their predicted market value with their transfer-market value.

### 4. Performance Analytics Dashboard
Provides visual insights into player performance, including:
- Top goal scorers
- Top assist providers
- Highest-rated players
- Other player performance statistics

## 🛠️ Tech Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |

## 🏗️ Project Architecture

TransferIQ follows a frontend-backend architecture:

1. **Frontend:** React provides the user interface and interactive scouting dashboard.
2. **Backend:** FastAPI handles requests and connects the frontend to the machine learning functionality.
3. **ML Models:** Trained models perform player similarity analysis and market value prediction.
4. **Data Processing:** Python scripts prepare and process football player datasets.

## 📂 Project Structure

```text
TransferIQ/
│
├── frontend/             # React + Vite frontend
│
├── models/               # Trained ML models
│   ├── market_value_model.joblib
│   └── transfer_fee_model.joblib
│
├── scripts/              # Data processing, training and analysis scripts
│
├── api.py                # FastAPI backend
├── README.md
└── .gitignore
```

## ⚙️ Running the Project Locally

### Prerequisites
- Python
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/dattasistla0102-ai/TransferIQ.git
cd TransferIQ
```

### 2. Set up the Python backend

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

Start the backend:

```bash
python -m uvicorn api:app --reload
```

### 3. Set up the React frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your terminal.

> **Note:** The raw and processed datasets are not included in this public repository. Some data-processing scripts and features may require the corresponding datasets to be available locally.

## 🎯 Project Objective

The objective of TransferIQ is to explore how machine learning and data analytics can improve football scouting and transfer evaluation by turning player statistics into actionable insights.

## 👨‍💻 Author

**Kartikeya Datta**

GitHub: https://github.com/dattasistla0102-ai

---

*TransferIQ — Smarter Scouting. Data-Driven Decisions.*
