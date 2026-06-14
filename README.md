# AIDEOM-VN-1-12
# AIDEOM-VN — Mô hình Ra Quyết định Phát triển Kinh tế Việt Nam trong Kỉ nguyên AI
*Bộ bài tập thực hành** | Môn: Các Mô hình Ra Quyết định  
> Trường Đại học Kinh tế — Viện Quản trị Kinh doanh  
> Dữ liệu thực tế Việt Nam 2020–2025

> **AI-powered Decision Support System | Operations Research + Stochastic Optimization + Policy Analytics**

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Pyomo](https://img.shields.io/badge/Optimization-Pyomo-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

---

## 🧠 Overview

**AIDEOM-VN** is a decision intelligence system designed to model and solve **Vietnam’s economic optimization problems (2020–2025)** using AI-assisted operations research techniques.

The system combines:

- 📊 Mathematical Optimization (LP / MIP)
- 🎲 Stochastic Programming (2-stage models)
- 📈 Multi-criteria Decision Analysis (MCDA)
- 🤖 AI-assisted policy simulation
- 🧮 Real-world Vietnam macroeconomic datasets

---

## 🏗️ System Vision

> Transform economic planning into a **data-driven, AI-supported decision system**
> 
## 📦 Project Scope (AIDEOM-VN 1–12)

| Level | Module | Topic | Methods |
|------|--------|-------|--------|
| Easy | 1 | Cobb-Douglas Production Model | NumPy |
|  | 2 | Budget Allocation LP | PuLP |
|  | 3 | Sector Scoring Model | MCDA |
| Medium | 4 | Regional Optimization | CVXPY |
|  | 5 | Project Selection MIP | CBC Solver |
|  | 6 | TOPSIS + AHP Ranking | MCDA |
| Advanced | 7 | Multi-objective Optimization | NSGA-II |
|  | 8 | Dynamic Optimization | CVXPY |
|  | 9 | Labor Market AI Impact | Simulation |
| Hard | 10 | 2-Stage Stochastic Programming | Pyomo |
|  | 11 | Reinforcement Learning Model | DQN |
| Capstone | 12 | Integrated Decision Dashboard | Streamlit |

---

## 🚀 Key Features

### 📊 Optimization Engine
- Deterministic + Stochastic modeling
- Scenario-based economic simulation
- Constraint-driven resource allocation

### 📈 Decision Analytics
- Expected Value (EV)
- Value of Stochastic Solution (VSS)
- Expected Value of Perfect Information (EVPI)

### ⚙️ Interactive Dashboard
- Streamlit-based UI
- Module-by-module navigation
- Real-time optimization results

---

## 🧮 Example: Bài 10 (Stochastic Programming)

### Problem Structure:
- Stage 1: Allocate **65,000 billion VND**
- Stage 2: Scenario-based allocation (**15,000 per scenario**)
- Objective: Maximize expected economic return under uncertainty

### Outputs:
- Optimal allocation (x*)
- Scenario decisions (y_s)
- EV / VSS / EVPI metrics

---

## 🏛️ Data Sources (Vietnam 2020–2025)

- General Statistics Office of Vietnam (GSO)
- Ministry of Planning and Investment (MPI)
- Ministry of Science & Technology (MOST)
- World Bank Vietnam Reports
- WIPO Global Innovation Index

---

## 🧠 Technologies

- Streamlit (Dashboard UI)
- Pyomo (Mathematical Optimization)
- PuLP (Linear Programming)
- CVXPY (Convex Optimization)
- NumPy / Pandas (Data processing)
- PyTorch (Reinforcement Learning - Bài 11)

---
## 📂 Project Structure

```bash
AIDEOM-VN/
│── app.py
│── modules/
│   ├── bai01.py
│   ├── bai02.py
│   ├── bai10.py
│   └── bai11.py
│
│── dashboard/
│   └── m6_dashboard.py
│
│── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_regions_2024.csv
│   └── vietnam_sectors_2024.csv
│
└── README.md
```
## ▶️ How to Run

```bash
git clone https://github.com/your-repo/aideom-vn
cd aideom-vn
pip install -r requirements.txt
streamlit run app.py
🌐 Live Demo

🚧 Streamlit Cloud Deployment (Coming Soon)

🎯 Research Domain

This project sits at the intersection of:

Operations Research
AI-driven Decision Systems
Economic Modeling
Applied Machine Learning
Policy Simulation for Vietnam Economy
⚖️ Academic Note

This project is developed for educational purposes in the course:

Decision Models in Economic Systems — Vietnam University

AI tools were used to assist in coding, modeling, and documentation.

🚀 Future Roadmap
 Deploy as SaaS decision platform
 Add real-time economic data API
 Upgrade solver to Gurobi / HiGHS
 Add scenario generator UI
 Add export PDF policy reports
👨‍💻 Author

AIDEOM-VN Research Team

Decision Intelligence for Vietnam’s digital economy transformation.
