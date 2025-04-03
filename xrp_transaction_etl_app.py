import streamlit as st
from xrp_transaction_etl import run_transaction_etl

st.title("XRP Transaction Data ETL Pipeline")

# File Uploads
transaction_file = st.file_uploader("Upload Transaction Data (Parquet, CSV, JSON)", type=['parquet', 'csv', 'json'])

if transaction_file:
    transaction_filepath = transaction_file.name
    transaction_filetype = transaction_file.name.split('.')[-1]

    transactions_df, transaction_schema = run_transaction_etl(transaction_filepath, transaction_filetype)

    if transactions_df is not None:
        st.subheader("1. Processed Transaction Data")
        st.dataframe(transactions_df)

        st.subheader("2. Transaction Data Schema")
        st.json(transaction_schema)

        # Download options
        if st.button("Download Processed Transaction Data (CSV)"):
            csv_data = transactions_df.to_csv(index=False)
            st.download_button("Download CSV", csv_data, "processed_transactions.csv", "text/csv")
    else:
        st.error("Error during ETL process.")