import streamlit as st
from xrp_data_etl import run_customer_etl

st.title("XRP Customer Data ETL Pipeline")

# File Uploads
customer_file = st.file_uploader("Upload Customer Data (Parquet, CSV, JSON)", type=['parquet', 'csv', 'json'])

if customer_file:
    customer_filepath = customer_file.name
    customer_filetype = customer_file.name.split('.')[-1]

    customers_df, customer_schema = run_customer_etl(customer_filepath, customer_filetype)

    if customers_df is not None:
        st.subheader("1. Processed Customer Data")
        st.dataframe(customers_df)

        st.subheader("2. Customer Data Schema")
        st.json(customer_schema)

        # Download options
        if st.button("Download Processed Customer Data (CSV)"):
            csv_data = customers_df.to_csv(index=False)
            st.download_button("Download CSV", csv_data, "processed_customers.csv", "text/csv")
    else:
        st.error("Error during ETL process.")