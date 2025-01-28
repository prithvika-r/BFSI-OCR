import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pytesseract
from PIL import Image
import streamlit as st
import os

def extract_profit_loss_data(image_path, csv_path):
    try:
        image = Image.open(image_path)
        ocr_result = pytesseract.image_to_string(image)
        lines = ocr_result.split("\n")
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
        print(f"Error occurred in OCR extraction: {e}")
        return None

import pandas as pd
from sqlalchemy import create_engine

def store_profit_loss_to_postgresql(df, db_name, user, password, host, port, table_name):
    if isinstance(df, pd.DataFrame):
        if df.empty:
            print(f"Warning: The DataFrame is empty. No data to store in {table_name}.")
            return
        try:
            # Create a connection to PostgreSQL
            engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
            
            # Store data in the table
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Data successfully stored in PostgreSQL table: {table_name} from {db_name}")
        
        except Exception as e:
            print(f"Error occurred while storing data in PostgreSQL: {e}")
    else:
        print("The data provided is not a pandas DataFrame.")

def fetch_data_from_postgresql(db_name, user, password, host, port, table_name):
    try:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        query = f'SELECT * FROM {table_name};'
        df_from_db = pd.read_sql(query, engine)
        return df_from_db
    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {e}")
        return None

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def create_visualizations(df, bar_chart_path, pie_chart_path):
    # Ensure directories for saving charts exist
    os.makedirs(os.path.dirname(bar_chart_path), exist_ok=True)
    os.makedirs(os.path.dirname(pie_chart_path), exist_ok=True)
    
    # Check for required columns
    required_columns = ['Description', '2015', '2016', '2017']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    
    # Drop rows with missing values in the year columns
    df = df.dropna(subset=['2015', '2016', '2017'])
    df.set_index('Description', inplace=True)
    
    # Ensure data is numeric
    df[['2015', '2016', '2017']] = df[['2015', '2016', '2017']].apply(pd.to_numeric, errors='coerce')

    # Create the bar chart (line plot)
    plt.figure(figsize=(12, 6))
    df[['2015', '2016', '2017']].plot(kind='line', marker='o', linestyle='-', title='Yearly Financial Data')
    plt.ylabel('Amount')
    plt.xlabel('Description')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    # Display and save the bar chart
    st.pyplot(plt)
    plt.savefig(bar_chart_path)  
    plt.close()  # Close the plot to prevent further issues

    # Create the pie chart
    df[['2015', '2016', '2017']].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title='Yearly Financial Data - Pie Chart')
    plt.tight_layout()

    # Display and save the pie chart
    st.pyplot(plt)
    plt.savefig(pie_chart_path)  
    plt.close()  # Close the plot to prevent further issues


