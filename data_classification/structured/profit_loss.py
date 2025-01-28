import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pytesseract
from PIL import Image
import streamlit as st
import os

# OCR Extraction
def extract_profit_loss_data(image_path, csv_path):
    try:
        image = Image.open(image_path)
        ocr_result = pytesseract.image_to_string(image)
        lines = ocr_result.split("\n")
        data = []
        start_processing = False

        def contains_numeric(value):
            return any(char.isdigit() for char in value)

        for line in lines:
            clean_line = line.strip().replace("$", "").replace(",", "")
            if "2015" in clean_line and "2016" in clean_line and "2017" in clean_line:
                start_processing = True
                continue

            if start_processing and clean_line:
                if contains_numeric(clean_line):
                    data.append(clean_line.split())
                else:
                    data.append([clean_line, "", "", ""])

        header = ["Description", "2015", "2016", "2017"]
        rows = []

        for line in data:
            if not line or len(line) < 4:
                continue
            description = " ".join(line[:-3])  
            row = [description] + [str(value) if value.replace('.', '', 1).isdigit() else value for value in line[-3:]]
            rows.append(row)

        df = pd.DataFrame(rows, columns=header)
        df = df.dropna(how="all")  
        df = df.dropna(subset=['Description', '2015', '2016', '2017'], how='any')

        # Save extracted data to CSV
        df.to_csv(csv_path, index=False)
        return df
    except Exception as e:
        st.error(f"Error occurred in OCR extraction: {e}")
        return pd.DataFrame()

# Store data in PostgreSQL
def store_profit_loss_to_postgresql(df, db_name, user, password, host, port, table_name):
    if df.empty:
        st.error("The data provided is not valid or empty.")
        return
    try:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        st.success(f"Data successfully stored in PostgreSQL table: {table_name}")
    except Exception as e:
        st.error(f"Error while storing data in PostgreSQL: {e}")

# Visualization creation
def create_visualizations(df, bar_chart_path, pie_chart_path):
    os.makedirs(os.path.dirname(bar_chart_path), exist_ok=True)
    os.makedirs(os.path.dirname(pie_chart_path), exist_ok=True)

    required_columns = ['Description', '2015', '2016', '2017']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing columns in data for visualization.")
        return

    df[['2015', '2016', '2017']] = df[['2015', '2016', '2017']].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['2015', '2016', '2017'], inplace=True)

    if df.empty:
        st.error("No valid data available for visualization.")
        return

    df.set_index('Description', inplace=True)

    # Bar Chart Visualization
    plt.figure(figsize=(12, 6))
    df[['2015', '2016', '2017']].plot(kind='line', marker='o', linestyle='-', title='Yearly Financial Data')
    plt.ylabel('Amount')
    plt.xlabel('Description')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(bar_chart_path)
    plt.close()

    # Pie Chart Visualization
    df[['2015', '2016', '2017']].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title='Yearly Financial Data - Pie Chart')
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(pie_chart_path)
    plt.close()
