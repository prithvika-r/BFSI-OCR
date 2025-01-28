import pytesseract
import cv2
import pandas as pd
import re
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_data_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 6")
        print("Raw Extracted Text:\n", text)
        extracted_data = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\S*[\\/]\S*', '', line)
            line = re.sub(r'[^\x00-\x7F]+', '', line)
            line = re.sub(r'\s+', ' ', line)  
            numeric_values = re.findall(r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%?))", line)
            description = re.sub(r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%?))", "", line).strip()
            if description and numeric_values:
                last_numeric = numeric_values[-1] 
                extracted_data.append([description, last_numeric])
        return extracted_data
    except Exception as e:
        print(f"Error extracting data from image: {e}")
        return []

def save_to_csv(extracted_data, csv_output_path):
    try:
        df = pd.DataFrame(extracted_data, columns=["Description", "Balance"])
        df["Balance"] = df["Balance"].str.replace("$", "").str.replace(",", "").apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=["Balance"], inplace=True)
        df.to_csv(csv_output_path, index=False, header=True)
        print(f"CSV file saved to {csv_output_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def store_csv_to_postgresql(csv_file_path, db_name, user, password, host, port, table_name):
    try:
        df = pd.read_csv(csv_file_path)
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data from {csv_file_path} successfully stored into {table_name} table in PostgreSQL.")
    except Exception as e:
        print(f"Error storing data from {csv_file_path} into PostgreSQL: {e}")
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

def create_visualizationss(df, bar_chart_path, pie_chart_path):
    try:
        # Barplot with custom bar widths and spacing
        plt.figure(figsize=(12, 6))

        # Define positions for the bars
        bar_positions = np.arange(len(df))

        # Define bar width (this controls the individual width of the bars)
        bar_width = 0.7  # Adjust this for wider/narrower bars

        # Create the bars with custom width
        bars = plt.bar(bar_positions, df['Balance'], width=bar_width, color=plt.cm.viridis(np.linspace(0, 1, len(df))))

        # Add labels
        plt.title("Amount per Description", fontsize=16)
        plt.xlabel("Description", fontsize=14)
        plt.ylabel("Balance", fontsize=14)
        plt.xticks(bar_positions, df['Description'], rotation=45, ha="right")

        # Add the legend outside the plot (avoid overlap)
        plt.legend(bars, df['Description'], title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save bar chart and display
        plt.savefig(bar_chart_path)
        plt.show()

        # Pie Chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            df["Balance"],
            labels=df["Description"],
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.Paired.colors
        )
        plt.title("Distribution of Amounts", fontsize=16)
        plt.tight_layout()
        plt.savefig(pie_chart_path)
        plt.show()

        # Close all plots after saving
        plt.close('all')
        print(f"Visualizations saved: Bar chart ({bar_chart_path}), Pie chart ({pie_chart_path})")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
