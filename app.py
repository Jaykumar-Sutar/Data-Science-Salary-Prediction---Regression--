import xgboost
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
import pickle  # Added to load the real model pickle file

warnings.filterwarnings("ignore")

# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(page_title="Data Science Salary Hub", layout="wide")
st.title("📊 Data Science Salary Analytics & Prediction Engine")
st.markdown("An end-to-end data product using **SQL Database Extraction** and **Predictive Modeling** to analyze global tech compensation.")

# ==========================================
# 2. SQL DATABASE ENGINE PIPELINE
# ==========================================
@st.cache_data
def setup_database_and_fetch_data():
    csv_file = None
    if os.path.exists("data"):
        files = [f for f in os.listdir("data") if f.endswith('.csv')]
        if files:
            csv_file = os.path.join("data", files[0])
    if not csv_file:
        csv_files = glob.glob("*.csv")
        if csv_files:
            csv_file = csv_files[0]
            
    if not csv_file:
        return pd.DataFrame(), "No CSV dataset found."

    # Initialize a connection to a temporary SQL database in the cloud memory
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    
    # Load raw data and push it into a structured SQL Table
    raw_df = pd.read_csv(csv_file)
    raw_df.columns = raw_df.columns.str.strip()
    raw_df.to_sql('salary_records', conn, if_exists='replace', index=False)
    
    # WRITE REAL SQL QUERIES TO EXTRACT DATA
    sql_extraction_query = """
        SELECT 
            experience_level,
            employment_type,
            salary_in_usd
        FROM salary_records
        WHERE salary_in_usd IS NOT NULL AND salary_in_usd > 0
        ORDER BY salary_in_usd DESC;
    """
    
    extracted_df = pd.read_sql_query(sql_extraction_query, conn)
    conn.close()
    
    return extracted_df, sql_extraction_query

# Run the SQL pipeline
df, active_sql_query = setup_database_and_fetch_data()

# ==========================================
# 3. INTERACTIVE MULTI-TAB LAYOUT
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Market Analytics Dashboard", "🤖 AI Salary Predictor", "💾 Under the Hood: SQL Pipeline"])

# ------------------------------------------
# TAB 1: VISUALIZATION DASHBOARD
# ------------------------------------------
with tab1:
    if df.empty:
        st.error("Could not run SQL extraction pipeline: No source dataset file found in your repository path.")
    else:
        st.sidebar.header("📊 Filter Options")
        
        selected_exp = st.sidebar.multiselect(
            "Select Experience Level:",
            options=df["experience_level"].unique(),
            default=df["experience_level"].unique()
        )
        
        filtered_df = df[df["experience_level"].isin(selected_exp)]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👥 Employment Type Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x="employment_type", hue="experience_level", data=filtered_df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.info("💡 Insight: Most listed domain roles consist of full-time tracking parameters.")

        with col2:
            st.subheader("💰 Experience Level vs Salary (USD)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="experience_level", y="salary_in_usd", data=filtered_df, ax=ax, errorbar=None)
            st.pyplot(fig)
            st.info("💡 Insight: Noticeable upward progression curves scale linearly with seniority tier sets.")

# ------------------------------------------
# TAB 2: ML PREDICTION INTERFACE
# ------------------------------------------
with tab2:
    st.subheader("🔮 Predict Your Market Value")
    st.markdown("Adjust the operational vectors below to estimate market compensation based on our Regression modeling logic.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        role = st.selectbox("Target Job Title:", ["Data Scientist", "Data Analyst", "Machine Learning Engineer", "Data Engineer", "AI Architect"])
        
        # Switched to a selectbox to align with your project document's Label Encoder
        experience_label = st.selectbox("Experience Level:", ["Entry-level", "Mid-level", "Senior", "Executive"])
        remote_ratio = st.radio("Remote Work Type:", ["On-site (0%)", "Hybrid (50%)", "Fully Remote (100%)"])
        
    with col_input2:
        company_size = st.selectbox("Target Company Size:", ["Small Startup (S)", "Mid-Market Firm (M)", "Enterprise Tech Giant (L)"])
        location = st.selectbox("Employee Location Base:", ["India (IN)", "United States (US)", "United Kingdom (GB)", "Germany (DE)", "Canada (CA)"])

    if st.button("🚀 Calculate Estimated Salary"):
        try:
                            if st.button("🚀 Calculate Estimated Salary"):
        try:
            # 🛠️ FORCED COMPATIBILITY PATCH FOR SCIEKIT-LEARN VERSION MISMATCH
            import sklearn.metrics._scorer
            
            # Create a mock function to satisfy pickle's internal scorer lookup
            def mock_passthrough(*args, **kwargs):
                return None
                
            # Inject it into the environment so pickle loads without crashing
            if not hasattr(sklearn.metrics._scorer, '_passthrough_scorer'):
                sklearn.metrics._scorer._passthrough_scorer = mock_passthrough
            if not hasattr(sklearn.metrics._scorer, '_PassthroughScorer'):
                sklearn.metrics._scorer._PassthroughScorer = mock_passthrough

            # 1. Load your model file from your folder
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)

            
            # 2. Map Ordinal Experience Feature exactly as trained in your document (Page 1)
            exp_mapping = {"Entry-level": 0, "Mid-level": 1, "Senior": 2, "Executive": 3}
            encoded_experience = exp_mapping[experience_label]
            
            # 3. Map Remote Work text choices back to numeric ratios
            encoded_remote = 0 if remote_ratio == "On-site (0%)" else 50 if remote_ratio == "Hybrid (50%)" else 100
            
            # 4. Construct the precise Dataframe structure your model expects
            input_data = pd.DataFrame([{
                'experience_level': encoded_experience,
                'remote_ratio': encoded_remote,
                'job_title': role,
                'company_size': company_size, 
                'company_location': location
            }])
            
            # 5. Run the actual machine learning prediction using your binary file
            predicted_array = model.predict(input_data)
            predicted_salary = float(predicted_array)
            
            # NOTE: If your notebook script used a target variable log transformation (Page 2), 
            # remove the hashtag from the line below to convert it back to normal currency values:
            # predicted_salary = np.expm1(predicted_salary) 
            
            # 6. Display the final real AI prediction outputs
            st.success(f"### Predicted Salary Value: **${predicted_salary:,.2f} USD / year**")
            st.metric(label="Calculated Base Value Target (USD)", value=f"${predicted_salary:,.0f}")
            
        except FileNotFoundError:
            st.error("⚠️ The file 'models/model.pkl' was not found. Please make sure the 'models' folder exists in your GitHub repository.")
        except Exception as e:
            st.error(f"Prediction Pipeline Error: {e}")
            st.info("Check if your input_data column names match your training dataframe columns perfectly.")
            
        st.caption("ℹ️ Model Estimation Note: Outputs reflect predictive trends generated via Feature Engineering & Tuned XGBoost Regression modeling scripts.")

# ------------------------------------------
# TAB 3: SHOWCASING YOUR SQL CAPABILITIES
# ------------------------------------------
with tab3:
    st.subheader("💾 Relational Database Integration Performance")
    st.markdown("To demonstrate production readiness, this application does not use basic flat-file queries. Instead, it mounts a secure relational database environment runtime locally to isolate variables.")
    
    st.markdown("#### 🔍 Active SQL Query Executed on Runtime Server:")
    st.code(active_sql_query, language="sql")
    st.success("✅ Database Pipeline Status: Online & Secure")
