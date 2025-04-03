import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath, file_type='parquet'):
    """Loads customer data from various file types."""
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
    """Validates customer data against a schema with advanced checks."""
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
                # Add more complex validation rules here (e.g., regex, range checks)
            except Exception as e:
                logging.warning(f"Validation error in column {col}: {e}")
                st.warning(f"Validation error in column {col}: {e}")
                # You could choose to drop rows with validation errors here.
    return valid_df

def transform_customer_data(df):
    """Transforms customer data (example: standardize addresses)."""
    if 'Residential Address (Physical and Postal)' in df.columns:
        df['Standardized Address'] = df['Residential Address (Physical and Postal)'].str.upper()
    return df

def generate_schema(df):
    """Generates schema from DataFrame."""
    schema = {}
    for column, dtype in df.dtypes.items():
        schema[column] = str(dtype)
    return schema

def run_customer_etl(customer_filepath, customer_filetype='parquet'):
    """Customer data ETL pipeline."""
    customers_df = load_data(customer_filepath, customer_filetype)

    if customers_df is None:
        return None, None

    customer_schema = {col: str(dtype) for col, dtype in customers_df.dtypes.items()}

    customers_df = validate_data(customers_df, customer_schema)
    customers_df = transform_customer_data(customers_df)

    return customers_df, generate_schema(customers_df)