import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from datetime import datetime


st.set_page_config(layout="wide")

# Set page layout to wide

def compare_swift_xrp(df):
    """Compares SWIFT and XRP transaction performance with more EDA."""

    st.title("SWIFT vs. XRP Transaction Comparison")

    # --- Columns for SWIFT and XRP ---
    col1, col2 = st.columns(2)

    # --- SWIFT Visualizations (Column 1) ---
    with col1:
        st.header("SWIFT Transactions", divider="green")

        # Average Transaction Time
        avg_swift_time = df["swift_time"].mean() / (60 * 60 * 24)  # Convert seconds to days
        st.metric("Average Time (Days)", f"{avg_swift_time:.2f}", delta=None, delta_color="normal")

        # Average Transaction Fees
        avg_swift_fees = df["swift_fees"].mean()
        st.metric("Average Fees", f"{avg_swift_fees:.2f}", delta=None, delta_color="normal")

        # Transaction Time Distribution
        st.subheader("Transaction Time Distribution (Days)")
        fig_swift_time, ax_swift_time = plt.subplots(figsize=(6, 4))
        sns.histplot(df["swift_time"] / (60 * 60 * 24), bins=20, ax=ax_swift_time, color="green")
        st.pyplot(fig_swift_time)

        # Fee Distribution
        st.subheader("Fee Distribution")
        fig_swift_fees, ax_swift_fees = plt.subplots(figsize=(6, 4))
        sns.histplot(df["swift_fees"], bins=20, ax=ax_swift_fees, color="green")
        st.pyplot(fig_swift_fees)

        # Failure Rate
        avg_swift_failure = df["swift_failure_rate"].mean() * 100
        st.metric("Failure Rate (%)", f"{avg_swift_failure:.2f}", delta=None, delta_color="normal")

        # SWIFT Time vs. Amount
        st.subheader("SWIFT Time vs. Transaction Amount")
        fig_swift_time_amount, ax_swift_time_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["swift_time"] / (60 * 60 * 24), ax=ax_swift_time_amount, color="green")
        st.pyplot(fig_swift_time_amount)

        # SWIFT Fees vs. Amount
        st.subheader("SWIFT Fees vs. Transaction Amount")
        fig_swift_fees_amount, ax_swift_fees_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["swift_fees"], ax=ax_swift_fees_amount, color="green")
        st.pyplot(fig_swift_fees_amount)
        
        st.subheader("Time Difference vs. Transaction Amount")
        fig_time_diff_amount, ax_time_diff_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["time_difference"] / (60 * 60), ax=ax_time_diff_amount)
        st.pyplot(fig_time_diff_amount)
        
        

    # --- XRP Visualizations (Column 2) ---
    with col2:
        st.header("XRP Transactions", divider="blue")

        # Average Transaction Time
        avg_xrp_time = df["xrp_time"].mean()
        st.metric("Average Time (Seconds)", f"{avg_xrp_time:.2f}", delta=None, delta_color="normal")

        # Average Transaction Fees
        avg_xrp_fees = df["xrp_fees"].mean()
        st.metric("Average Fees", f"{avg_xrp_fees:.6f}", delta=None, delta_color="normal")

        # Transaction Time Distribution
        st.subheader("Transaction Time Distribution (Seconds)")
        fig_xrp_time, ax_xrp_time = plt.subplots(figsize=(6, 4))
        sns.histplot(df["xrp_time"], bins=20, ax=ax_xrp_time, color="blue")
        st.pyplot(fig_xrp_time)

        # Fee Distribution
        st.subheader("Fee Distribution")
        fig_xrp_fees, ax_xrp_fees = plt.subplots(figsize=(6, 4))
        sns.histplot(df["xrp_fees"], bins=20, ax=ax_xrp_fees, color="blue")
        st.pyplot(fig_xrp_fees)

        # Failure Rate
        avg_xrp_failure = df["xrp_failure_rate"].mean() * 100
        st.metric("Failure Rate (%)", f"{avg_xrp_failure:.2f}", delta=None, delta_color="normal")

        # XRP Time vs. Amount
        st.subheader("XRP Time vs. Transaction Amount")
        fig_xrp_time_amount, ax_xrp_time_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["xrp_time"], ax=ax_xrp_time_amount, color="blue")
        st.pyplot(fig_xrp_time_amount)

        # XRP Fees vs. Amount
        st.subheader("XRP Fees vs. Transaction Amount")
        fig_xrp_fees_amount, ax_xrp_fees_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["xrp_fees"], ax=ax_xrp_fees_amount, color="blue")
        st.pyplot(fig_xrp_fees_amount)
        
        st.subheader("Fee Difference vs. Transaction Amount")
        fig_fee_diff_amount, ax_fee_diff_amount = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["amount"], y=df["fee_difference"], ax=ax_fee_diff_amount)
        st.pyplot(fig_fee_diff_amount)
        

    # --- Comparison Metrics ---
    st.header("Comparison Metrics")

    # Use columns with smaller width ratios
    col_time_diff, col_fee_diff, col_outperforms = st.columns([1, 1, 1])  # Adjust these ratios

    # Time Difference
    with col_time_diff:
        avg_time_diff = df["time_difference"].mean() / (60 * 60) # difference in hours
        st.metric("Average Time Difference (Hours)", f"{avg_time_diff:.2f}", delta=None, delta_color="normal")

    # Fee Difference
    with col_fee_diff:
        avg_fee_diff = df["fee_difference"].mean()
        st.metric("Average Fee Difference", f"{avg_fee_diff:.2f}", delta=None, delta_color="normal")

    # XRP Outperforms SWIFT Percentage
    with col_outperforms:
        xrp_outperforms_percentage = df["xrp_outperforms_swift"].mean() * 100
        st.metric("XRP Outperforms SWIFT (%)", f"{xrp_outperforms_percentage:.2f}", delta=None, delta_color="normal")
        


# Load the model
# Load the model
model = joblib.load('xrp_outperforms_model.joblib')

def calculate_swift_fees(amount, currency, source_country, destination_country):
    """Calculates realistic SWIFT fees."""
    base_fee = random.uniform(20, 100)
    currency_markup = amount * random.uniform(0.005, 0.03)  # 0.5% - 3% markup
    return base_fee + currency_markup

def calculate_sepa_fees(amount, is_instant):
    """Calculates SEPA fees."""
    return 0 if amount < 5000 else random.uniform(0, 5)

def calculate_ach_fees(amount, is_same_day):
    """Calculates ACH fees."""
    return 0 if not is_same_day else random.uniform(5, 15)

def calculate_xrp_fees():
    """Calculates XRP fees."""
    return random.uniform(0.001, 0.01)

def estimate_swift_time():
    """Estimates SWIFT processing time."""
    return random.randint(1, 5) * 86400  # 1-5 business days

def estimate_sepa_time(is_instant):
    """Estimates SEPA processing time."""
    return 10 if is_instant else 86400  # 10 seconds or 1 day

def estimate_ach_time(is_same_day):
    """Estimates ACH processing time."""
    return 0 if is_same_day else random.randint(1, 3) * 86400  # Same-day or 1-3 business days

def estimate_xrp_time():
    """Estimates XRP processing time."""
    return random.uniform(4, 5)

def predict_xrp_outperforms(amount, currency, source_country, destination_country, is_sepa_instant, is_ach_same_day):
    """Predicts whether XRP outperforms SWIFT and provides details."""

    swift_time = estimate_swift_time()
    swift_fees = calculate_swift_fees(amount, currency, source_country, destination_country)

    sepa_time = estimate_sepa_time(is_sepa_instant)
    sepa_fees = calculate_sepa_fees(amount, is_sepa_instant)

    ach_time = estimate_ach_time(is_ach_same_day)
    ach_fees = calculate_ach_fees(amount, is_ach_same_day)

    xrp_time = estimate_xrp_time()
    xrp_fees = calculate_xrp_fees()

    input_data = pd.DataFrame({
        'amount': [amount],
        'currency': [currency],
        'source_country': [source_country],
        'destination_country': [destination_country],
        'swift_time': [swift_time],
        'swift_fees': [swift_fees],
        'xrp_time': [xrp_time],
        'xrp_fees': [xrp_fees]
    })

    prediction = model.predict(input_data)[0]

    if prediction:
        time_difference = swift_time - xrp_time
        fee_difference = swift_fees - xrp_fees
        return prediction, time_difference, fee_difference, swift_fees, xrp_fees, sepa_fees, sepa_time, ach_fees, ach_time
    else:
        return prediction, None, None, swift_fees, xrp_fees, sepa_fees, sepa_time, ach_fees, ach_time

# Streamlit app

st.title("XRP vs. Traditional Bank Transfer Comparison")

# Input fields
amount = st.number_input("Transaction Amount", value=1000.0)
currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CAD"])
source_country = st.selectbox("Source Country", ["US", "GB", "DE", "JP", "CA", "AU", "CN", "IN", "BR", "ZA"])
destination_country = st.selectbox("Destination Country", ["US", "GB", "DE", "JP", "CA", "AU", "CN", "IN", "BR", "ZA"])
is_sepa_instant = st.checkbox("SEPA Instant (EU/EEA)")
is_ach_same_day = st.checkbox("ACH Same Day (US)")

# Prediction button
if st.button("Compare"):
    prediction, time_diff, fee_diff, swift_fees, xrp_fees, sepa_fees, sepa_time, ach_fees, ach_time = predict_xrp_outperforms(amount, currency, source_country, destination_country, is_sepa_instant, is_ach_same_day)

    st.header("Comparison Report")

    st.subheader("Transaction Details")
    st.write(f"**Amount:** {amount} {currency}")
    st.write(f"**Source Country:** {source_country}")
    st.write(f"**Destination Country:** {destination_country}")

    st.subheader("XRP Transfer")
    st.write(f"**Estimated Time:** {estimate_xrp_time():.2f} seconds")
    st.write(f"**Estimated Fees:** ${xrp_fees:.4f}")

    st.subheader("SWIFT Transfer")
    st.write(f"**Estimated Time:** {estimate_swift_time() / 86400:.2f} days")
    st.write(f"**Estimated Fees:** ${swift_fees:.2f}")

    if currency == "EUR" and source_country in ["AT", "BE", "CY", "EE", "FI", "FR", "DE", "GR", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PT", "SK", "SI", "ES"] and destination_country in ["AT", "BE", "CY", "EE", "FI", "FR", "DE", "GR", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PT", "SK", "SI", "ES"]:
        st.subheader("SEPA Transfer")
        st.write(f"**Estimated Time:** {sepa_time} {'seconds' if is_sepa_instant else 'day(s)'}")
        st.write(f"**Estimated Fees:** €{sepa_fees:.2f}")

    if currency == "USD" and source_country == "US" and destination_country == "US":
        st.subheader("ACH Transfer")
        st.write(f"**Estimated Time:** {ach_time / 86400:.2f} {'day(s)' if not is_ach_same_day else 'same-day'}")
        st.write(f"**Estimated Fees:** ${ach_fees:.2f}")

    if prediction:
        st.success("XRP is predicted to outperform SWIFT.")
        st.subheader("Savings Analysis")
        st.write(f"**Time Savings:** {time_diff / 86400:.2f} days")
        st.write(f"**Fee Savings:** ${fee_diff:.2f}")
    else:
        st.warning("SWIFT is predicted to be better or equivalent.")
# Main Streamlit App
st.title("XRP Integration Pipeline")

# Load data
df = pd.read_parquet("transactions_comparison.parquet")

# Run the comparison function
compare_swift_xrp(df)

# Add other functionalities later...