import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath, file_type='parquet'):
    """Loads transaction data from various file types."""
    try:
        if file_type == 'parquet':
            table = pq.read_table(filepath)
            return table.to_pandas()
        elif file_type == 'csv':
            return pd.read_csv(filepath)
        elif file_type == 'json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        st.error(f"Error loading {filepath}: {e}")
        return None

def validate_data(df, schema):
    """Validates transaction data against a schema with advanced checks."""
    valid_df = df.copy()
    for col, dtype in schema.items():
        if col in valid_df.columns:
            try:
                if dtype == 'datetime64[ns]':
                    valid_df[col] = pd.to_datetime(valid_df[col], errors='coerce')
                elif dtype == 'int64':
                    valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce').astype('Int64')
                elif dtype == 'float64':
                    valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
                elif dtype == 'str':
                    valid_df[col] = valid_df[col].astype(str)
                # Advanced checks
                if col == 'Transaction Amount (XRP)':
                    valid_df = valid_df[valid_df[col] >= 0] # Ensure Amount is positive.
                if col == 'Sender XRP Address' or col == 'Receiver XRP Address':
                    valid_df = valid_df[valid_df[col].str.startswith('r', na=False)] #Ensure correct address format.

            except Exception as e:
                logging.warning(f"Validation error in column {col}: {e}")
                st.warning(f"Validation error in column {col}: {e}")
                # Potentially drop rows with validation errors here.
    return valid_df

def transform_transaction_data(df):
    """Transforms transaction data (example: calculate fiat amount, categorize transactions)."""
    if 'Transaction Amount (XRP)' in df.columns and 'Exchange Rate (XRP to Fiat)' in df.columns:
        df['Transaction Amount (Fiat)'] = df['Transaction Amount (XRP)'] * df['Exchange Rate (XRP to Fiat)']

    if 'Transaction Type' in df.columns:
        df['Transaction Category'] = df['Transaction Type'].apply(lambda x: 'Financial' if x in ['Payment', 'Trade'] else 'Other')

    return df

def enrich_transaction_data(df):
    """Enriches transaction data (example: calculate transaction risk)."""
    if 'AML Risk Score' in df.columns:
        df['Transaction Risk'] = df['AML Risk Score'].apply(lambda x: 'High' if x > 70 else ('Medium' if x > 30 else 'Low'))
    if 'Transaction Amount (XRP)' in df.columns:
        df['Transaction Size Category'] = pd.cut(df['Transaction Amount (XRP)'], bins = [0,10,100,1000, 100000000000], labels = ['Tiny','Small','Medium', 'Large'])
    return df

def generate_schema(df):
    """Generates schema from DataFrame."""
    schema = {}
    for column, dtype in df.dtypes.items():
        schema[column] = str(dtype)
    return schema

def run_transaction_etl(transaction_filepath, transaction_filetype='parquet'):
    """Transaction data ETL pipeline."""
    transactions_df = load_data(transaction_filepath, transaction_filetype)

    if transactions_df is None:
        return None, None

    transaction_schema = {col: str(dtype) for col, dtype in transactions_df.dtypes.items()}

    transactions_df = validate_data(transactions_df, transaction_schema)
    transactions_df = transform_transaction_data(transactions_df)
    transactions_df = enrich_transaction_data(transactions_df)

    return transactions_df, generate_schema(transactions_df)