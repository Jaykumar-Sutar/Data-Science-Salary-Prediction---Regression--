 """# 📊 Data Science Salary Analytics & Prediction Engine

An end-to-end cloud-hosted web application engineered to transform a serialized machine learning backend into an interactive, functional consumer product. This application bypasses basic flat-file parsing by managing an in-memory relational database runtime alongside a dynamically patched predictive interface.

🚀 **Live Application Link:** [[Launch the Streamlit Web Dashboard](https://9g63mvxmy4mt49h93bxbec.streamlit.app/)]

---

## 🏗️ Core Project Architecture

The system is decoupled into isolated functional layout components to maximize processing efficiency and maintain modular boundaries:

Use code with caution.[ Application Entry (app.py) ]│├───► [ Tab 1: Market Analytics Dashboard ] ───► Visualizes distributions (Matplotlib/Seaborn)│├───► [ Tab 2: AI Salary Predictor ]        ───► Deserializes binary engine & runs inferences│                                                 └─► Runtime patch: NumPy 2.x & Categorical Type-Casting│└───► [ Tab 3: Under the Hood Pipeline ]   ───► Executes Relational Database Queries (SQLite3)
---

## 💾 Data Infrastructure & Feature Space

The application processes historical workplace arrays from the **Data Science Job Salaries Dataset** to output dynamic compensation curves. 

* **The Regression Task:** The model targets `salary_in_usd` — a continuous, numeric target vector representing standardized annual compensation metrics.
* **Operational Vectors Evaluated:** Models inferences using exactly 8 features: `work_year`, `experience_level` (Encoded), `employment_type`, `job_title`, `employee_residence`, `remote_ratio`, `company_location`, and `company_size`.
* **Evaluation Metrics:** The background modeling logic was mathematically optimized and scored utilizing **Root Mean Squared Error (RMSE)** and **R-squared (R²) Score** variance tracking.

---

## 🛠️ Key Technical Implementations

### 1. In-Memory Relational Database Pipeline
Instead of executing standard flat-file reading loops, the engine dynamically instantiates a temporary, secure **SQLite3 Relational Database Layer** in cache memory at startup. Raw source records are structured into tables, allowing the visual layout components to slice and aggregate metrics utilizing real relational query syntax:
```sql
SELECT 
    experience_level,
    employment_type,
    salary_in_usd
FROM salary_records
WHERE salary_in_usd IS NOT NULL AND salary_in_usd > 0
ORDER BY salary_in_usd DESC;
```

### 2. Object Deserialization & Binary Integration
The system stages an external pre-trained **XGBoost Regression model** binary object (`models/model.pkl`), extracting structural layout logic dynamically inside a server-side prediction block using native Python file stream workflows (`pickle`).

### 3. Production Runtime Optimization & Patching
To ensure absolute production readiness on modern remote cloud containers, the codebase includes advanced runtime environment optimizations:
* **NumPy 2.x Array Support:** Implements unified structure flattening metrics via `np.ravel()` to comfortably circumvent strict element-to-scalar type restrictions imposed by updated array processing libraries.
* **Dynamic Categorical Feature Mapping:** Natively handles ordinal label string arrays (`Entry-level`, `Mid-level`, `Senior`, `Executive`) by mapping choices directly into specific category indices (`astype('category')`) required by XGBoost matrices.
* **Server Dependency Manifest:** Incorporates a systematic deployment layout via `requirements.txt` to align background execution tools cleanly.

---

## 💻 Local Workspace Configuration

To pull down the pipeline files and launch the analytics interface within a local development workspace, run the following commands sequentially:

### 1. Clone the Code Repository
```bash
git clone https://github.com
cd Data-Science-Salary-Prediction---Regression--
```

### 2. Configure Local System Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Server Staging Environment
```bash
streamlit run app.py
```

---

## 🛠️ Application Stack Summary
* **Interface Staging Framework:** Streamlit Cloud Deployment Runtime
* **Data Processing Infrastructure:** Pandas DataFrames, In-Memory SQLite3 Engine
* **Predictive Architecture Backend:** Serialized XGBoost Regressor Object Binary (`.pkl`)
* **Visual Representation Arrays:** Matplotlib Graph Renderers, Seaborn Distribution Frameworks
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
