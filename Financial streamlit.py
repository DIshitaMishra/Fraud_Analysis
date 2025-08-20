import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/ishita mishra/Downloads/archive (1)/PS_20174392719_1491204439457_log.csv")
    return df

df = load_data()

st.title("💳 Financial Transaction Fraud Analysis Dashboard")

# ==============================
# 🔹 KPI SECTION
# ==============================
st.subheader("📌 Key Performance Indicators (KPIs)")

total_tx = len(df)
total_fraud = df['isFraud'].sum()
fraud_percentage = round((total_fraud / total_tx) * 100, 2)
avg_fraud_amount = round(df[df['isFraud'] == 1]['amount'].mean(), 2)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Transactions", f"{total_tx:,}")
kpi2.metric("Fraud Cases", f"{total_fraud:,}")
kpi3.metric("Fraud %", f"{fraud_percentage}%")
kpi4.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")

st.markdown("---")

# ==============================
# Dataset Overview
# ==============================
st.subheader("Dataset Overview")
st.write(df.head())

# Transaction Type Distribution
st.subheader("Transaction Type")
fig, ax = plt.subplots()
sns.countplot(data=df, x="type", order=df["type"].value_counts().index, ax=ax)
st.pyplot(fig)

st.markdown("""
    **Insights:**  
    - `CASH_OUT` and `PAYMENT` dominate the dataset (>2M transactions each).  
    - `CASH_IN` is moderately frequent (~1.4M).  
    - `TRANSFER` transactions are fewer but riskier for fraud.  
    - `DEBIT` transactions are very rare.  
    """)

# Fraud vs Non-Fraud
st.subheader("Fraud vs Non-Fraud")
fraud_count = df['isFraud'].value_counts()
st.write(fraud_count)

fig, ax = plt.subplots()
fraud_count.plot.pie(autopct ="%1.2f%%", labels = ['Not Fraud', 'Fraud'], colors = ['skyblue', 'red'], ax=ax)
st.pyplot(fig)

st.markdown("""
    **Insights:**  
    - Fraudulent transactions are extremely rare (<0.2%).  
    - Highlights severe **class imbalance**, which is a major ML challenge.  
    - Oversampling/undersampling techniques may be needed.  
    """)

# Amount Analysis
st.subheader("Transaction Amounts (Fraud vs Non-Fraud)")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='isFraud', y='amount', ax=ax)
ax.set_ylim(0,500000)
st.pyplot(fig)

st.markdown("""
    **Insights:**  
    - Fraudulent transactions generally involve **higher amounts** than non-fraud.  
    - Extreme outliers exist (multi-million transactions).  
    - Fraud detection should consider transaction size as a key factor.  
    """)

# Sidebar Filters
st.sidebar.subheader("Filters")
tx_type = st.sidebar.multiselect("Select Transaction Type", df['type'].unique())
if tx_type:
    df = df[df['type'].isin(tx_type)]

# Correlation Heatmap
st.subheader("📈 Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("""
    **Insights:**  
    - Strong correlation between `oldbalanceOrg` & `newbalanceOrig`.  
    - Fraud (`isFraud`) shows some relation with `amount` and destination balances.  
    - Helps in selecting features for fraud detection models.  
    """)

st.title("🔎 Deep Insights on Fraud Data")

# Step 1: Fraud % by Transaction Type
st.subheader("Fraud % by Transaction Type")
fraud_by_type = df.groupby("type")["isFraud"].mean() * 100
st.bar_chart(fraud_by_type)
st.write("👉 Fraud happens mostly in TRANSFER and CASH_OUT transactions.")

# Step 2: Amount Distribution
st.subheader("Fraud vs Non-Fraud Amounts")
fraud_amount = df[df['isFraud']==1]['amount']
nonfraud_amount = df[df['isFraud']==0]['amount']

fig, ax = plt.subplots()
sns.boxplot(x='isFraud', y = 'amount',data = df, ax=ax)
st.pyplot(fig)
st.write("👉 Fraud transactions usually involve much higher amounts than normal ones.")

# Step 3: Balance Behavior
st.subheader("Balance Differences (Sender Account)")
df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

fig, ax = plt.subplots()
sns.boxplot(x="isFraud", y="balance_diff", data=df, ax=ax)
st.pyplot(fig)
st.write("👉 Fraud transactions show unusual balance changes compared to non-fraud.")

# Step 4: Flagged Fraud vs Real Fraud
st.subheader("System Flagged vs Actual Fraud")
fraud_count = df["isFraud"].sum()
flagged_count = df["isFlaggedFraud"].sum()

st.metric("Actual Fraud Cases", fraud_count)
st.metric("Flagged Fraud Cases", flagged_count)
st.write("👉 The system flags very few frauds compared to actual fraud cases.")
