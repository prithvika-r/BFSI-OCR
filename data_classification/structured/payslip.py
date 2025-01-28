import pytesseract
import cv2
import re
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from io import BytesIO
import numpy as np
import streamlit as st
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_comparison_table(text: str):
    table_data = []
    lines = text.split("\n")
    table_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Earnings" in line and "Amount" in line:
            table_started = True
            continue
        if table_started:
            if "Amount in Words" in line:
                break
            match = re.match(r"([A-Za-z\s]+)\s+([\d,]+(?:\.\d{1,2})?)\s+([A-Za-z\s]+)?\s+([\d,]+(?:\.\d{1,2})?)?", line)
            if match:
                earnings, amount_1, deductions, amount_2 = match.groups()
                deductions = deductions if deductions else ""
                amount_2 = amount_2 if amount_2 else ""
                table_data.append([earnings.strip(), amount_1.strip(), deductions.strip(), amount_2.strip()])
    return table_data
def extract_key_value_table(text: str):
    table_data = []
    lines = text.split("\n")
    table_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Earnings" in line or "Earnings (£)" in line:
            table_started = True
            continue
        if table_started:
            if "Net Pay" in line or "Amount in Words" in line:
                break
            match = re.match(r"(.+?)\s+([\d.,]+)$", line)
            if match:
                key, value = match.groups()
                table_data.append([key.strip(), value.strip()])
    return table_data
def process_image_to_csv(image, csv_filename: str):
    img = cv2.imdecode(np.asarray(bytearray(image.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--psm 4'
    text = pytesseract.image_to_string(gray, config=custom_config)
    comparison_table_data = extract_comparison_table(text)
    if not comparison_table_data:
        comparison_table_data = extract_key_value_table(text)
    if comparison_table_data:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if len(comparison_table_data[0]) == 4:
                writer.writerow(["Earnings", "Amount", "Deductions", "Amount 2"])
            else:
                writer.writerow(["Earning", "Amount"])
            writer.writerows(comparison_table_data)
import pandas as pd
from sqlalchemy import create_engine

def store_csv_to_postgresql(csv_file_path, db_name, user, password, host, port, table_name):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if dataframe is empty
        if df.empty:
            print(f"Warning: The CSV file {csv_file_path} is empty.")
            return
        
        # Connect to the PostgreSQL database
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        
        # Store data into the PostgreSQL table
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data from {csv_file_path} successfully stored into {table_name} table in PostgreSQL.")
    
    except Exception as e:
        print(f"Error: {e}")

def visualize_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # For 'Earnings' and 'Amount' columns
    if "Earnings" in df.columns and "Amount" in df.columns:
        # Bar chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Earnings", y="Amount", data=df, palette="viridis", ax=ax1)
        ax1.set_title("Earnings vs Amount")
        ax1.set_xlabel("Earnings")
        ax1.set_ylabel("Amount")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        # Pie chart
        earnings_sum = df.groupby("Earnings")["Amount"].sum()
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(
            earnings_sum.values, 
            labels=earnings_sum.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("Set3", len(earnings_sum)),
        )
        ax2.set_title("Earnings Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    # For 'Earning' and 'Amount' columns
    elif "Earning" in df.columns and "Amount" in df.columns:
        # Bar chart
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Earning", y="Amount", data=df, palette="viridis", ax=ax3)
        ax3.set_title("Earning vs Amount")
        ax3.set_xlabel("Earning")
        ax3.set_ylabel("Amount")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

        # Pie chart
        earnings_sum_alt = df.groupby("Earning")["Amount"].sum()
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        ax4.pie(
            earnings_sum_alt.values, 
            labels=earnings_sum_alt.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("Set3", len(earnings_sum_alt)),
        )
        ax4.set_title("Earning Distribution")
        plt.tight_layout()
        st.pyplot(fig4)
