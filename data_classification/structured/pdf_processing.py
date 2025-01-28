import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
def convert_pdf_to_csv(pdf_path, output_csv_path):
    transactions = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()  
            for table in tables:
                for row in table:  
                    if row:  
                        transactions.append(row)  
    df = pd.DataFrame(transactions)
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.to_csv(output_csv_path, index=False, header=False)
    return df
def save_csv(df, output_path):
    if isinstance(df, pd.DataFrame):
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("The input is not a valid DataFrame.")
    return output_path
def store_data_in_db(df, db_name, user, password, host, port, table_name):
    try:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data has been successfully stored in the {table_name} table of the {db_name} database.")
    except Exception as e:
        print(f"Error storing data in the database: {e}")
def generate_debit_credit_visualizations(csv_file):
    try:
        df = pd.read_csv(csv_file, header=1) 
    except Exception as e:
        raise FileNotFoundError(f"Error loading the CSV file: {e}")
    df.columns = df.columns.str.strip().str.replace(r'\W+', '', regex=True)
    df = df[~df.apply(lambda row: row.astype(str).isin(df.columns).any(), axis=1)]
    if 'Debit' not in df.columns or 'Credit' not in df.columns:
        raise ValueError("The expected columns 'Debit' and 'Credit' are missing from the CSV.")
    def clean_column_data(df, column):
        df[column] = pd.to_numeric(df[column].replace({',': '', '₹': '', '': None}, regex=True), errors='coerce')
        return df
    df = clean_column_data(df, 'Debit')
    df = clean_column_data(df, 'Credit')
    df = df.dropna(subset=['Debit', 'Credit'])
    total_debit = df['Debit'].sum()
    total_credit = df['Credit'].sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Debit', 'Credit'], [total_debit, total_credit], color=['#1f77b4', '#ff7f0e']) 
    ax.set_title('Total Debit vs Total Credit', fontsize=16)
    ax.set_ylabel('Amount', fontsize=14)
    ax.set_xlabel('Category', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    chart_path = "data/visualizations/debit_credit_comparison.png"
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path
