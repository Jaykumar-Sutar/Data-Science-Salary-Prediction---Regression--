import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(page_title="Data Science Salary Analysis", layout="wide")
st.title("📊 Data Science Salary Prediction & Insights Dashboard")
st.markdown("Exploring industry distributions, experience levels, and predictive analytical trends.")

# ==========================================
# 2. DATA LOAD PIPELINE
# ==========================================
@st.cache_data
def load_data():
    # Looks for any CSV file inside your data folder
    import os
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if files:
            return pd.read_csv(os.path.join(data_dir, files[0]))
    
    # Fallback if specific folder doesn't match
    return pd.read_csv("ds_salaries.csv")

try:
    df = load_data()
    
    # Clean up column names just in case
    df.columns = df.columns.str.strip()
    
    # Sidebar Filters to make it interactive for recruiters
    st.sidebar.header("📊 Filter Options")
    
    # Dynamically find the right column name for experience level
    exp_col = [col for col in df.columns if 'experience' in col.lower()]
    exp_column_name = exp_col[0] if exp_col else df.columns[0]
    
    selected_exp = st.sidebar.multiselect(
        "Select Experience Level:",
        options=df[exp_column_name].unique(),
        default=df[exp_column_name].unique()
    )
    
    filtered_df = df[df[exp_column_name].isin(selected_exp)]

    # ==========================================
    # 3. INTERACTIVE METRICS & CHARTS
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Employment Type Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        emp_col = [col for col in df.columns if 'employment' in col.lower()]
        emp_column_name = emp_col[0] if emp_col else df.columns[1]
        
        sns.countplot(x=emp_column_name, hue=exp_column_name, data=filtered_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig) # <-- This displays charts correctly on the web portal
        st.info("💡 Insight: Most listed domain roles consist of full-time tracking parameters.")

    with col2:
        st.subheader("💰 Experience Level vs Salary (USD)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sal_col = [col for col in df.columns if 'usd' in col.lower() or 'salary' in col.lower()]
        sal_column_name = sal_col[0] if sal_col else df.columns[2]
        
        sns.barplot(x=exp_column_name, y=sal_column_name, data=filtered_df, ax=ax, errorbar=None)
        st.pyplot(fig)
        st.info("💡 Insight: Noticeable upward progression curves scale linearly with seniority tier sets.")

except Exception as e:
    st.error(f"Could not load data file automatically. Please check your dataset path name in GitHub. Error details: {e}")
