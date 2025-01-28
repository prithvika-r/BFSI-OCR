import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pytesseract
from PIL import Image
import streamlit as st
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_profit_loss_data(image_path, csv_path):
    try:
        image = Image.open(image_path)
        ocr_result = pytesseract.image_to_string(image.convert("L"))
        lines = ocr_result.split("\n")
        data = []
        start_processing = False

        def contains_numeric(value):
            return any(char.isdigit() for char in value)

        for line in lines:
            clean_line = (
                line.replace("$", "")
                .replace(",", "")
                .replace("~", "")
                .replace("*", "")
                .replace("“", "")
                .replace("”", "")
                .strip()
            )
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
        description = ""

        for line in data:
            if not line or len(line) < 4:
                continue
            if len(line) > 1:
                description = " ".join(line[:-3])
                row = [description] + [str(value) if value.replace('.', '', 1).isdigit() else value for value in line[-3:]]
                rows.append(row)
            else:
                if len(line) == 4:
                    row = [line[0]] + [str(value) if value.replace('.', '', 1).isdigit() else value for value in line[1:]]
                    rows.append(row)

        df = pd.DataFrame(rows, columns=header)
        df = df.dropna(how="all")
        df = df.dropna(subset=['Description', '2015', '2016', '2017'], how='any')
        df.to_csv(csv_path, index=False)
        return df

    except Exception as e:
        st.error(f"Error occurred in OCR extraction: {e}")
        return None


def store_profit_loss_to_postgresql(df, db_name, user, password, host, port, table_name):
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            st.success(f"Data successfully stored in PostgreSQL table: {table_name}")
        except Exception as e:
            st.error(f"Error occurred while storing data in PostgreSQL: {e}")
    else:
        st.error("The data provided is not valid or empty.")


def create_visualizations(df, bar_chart_path, pie_chart_path):
    if df is None or df.empty:
        st.error("No data available for visualization.")
        return

    os.makedirs(os.path.dirname(bar_chart_path), exist_ok=True)
    os.makedirs(os.path.dirname(pie_chart_path), exist_ok=True)

    required_columns = ['Description', '2015', '2016', '2017']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns for visualization: {', '.join(missing_columns)}")
        return

    df = df.dropna(subset=['2015', '2016', '2017'])
    df.set_index('Description', inplace=True)
    df[['2015', '2016', '2017']] = df[['2015', '2016', '2017']].apply(pd.to_numeric, errors='coerce')

    # Bar Chart
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

    # Pie Chart
    df[['2015', '2016', '2017']].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title='Yearly Financial Data - Pie Chart')
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(pie_chart_path)
    plt.close()
