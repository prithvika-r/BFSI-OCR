import pytesseract
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

def extract_from_description_to_csv(image_path, csv_output_path):
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_string(image)
    lines = ocr_result.split("\n")
    capture_data = False
    descriptions = []
    amounts = []
    for line in lines:
        clean_line = line.strip()
        if "Description of Services" in clean_line:
            capture_data = True
            continue  
        if capture_data:
            if "Subtotal" in clean_line or clean_line == "":
                break
            if "$" in clean_line:
                description, amount = clean_line.rsplit('$', 1)  
                descriptions.append(description.strip())
                amounts.append(float(amount.strip().replace(",", "").replace("$", ""))) 
            elif clean_line: 
                descriptions.append(clean_line.strip())
                amounts.append(0.0) 
    max_len = max(len(descriptions), len(amounts))
    descriptions.extend([''] * (max_len - len(descriptions)))  
    amounts.extend([0.0] * (max_len - len(amounts)))  
    # Save to CSV
    data = {"Description": descriptions, "Amount ($)": amounts}
    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    print(f"Extracted data saved to {csv_output_path}")
    return df



def store_csv_to_postgresql(csv_file_path, db_name, user, password, host, port, table_name):
    # Try to establish a connection to PostgreSQL first
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'
    engine = create_engine(connection_string)

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise ValueError("The CSV file is empty or invalid.")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data from {csv_file_path} successfully stored into {table_name} table in PostgreSQL.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except OperationalError as oe:
        print(f"Error connecting to PostgreSQL: {oe}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_visualizations(df, bar_chart_path, pie_chart_path):
    # Bar Chart Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Description", y="Amount ($)", palette="viridis")
    plt.title("Amount per Description", fontsize=16)
    plt.xlabel("Description", fontsize=14)
    plt.ylabel("Amount ($)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    # Pie Chart Visualization
    plt.figure(figsize=(8, 8))
    plt.pie(df["Amount ($)"], labels=df["Description"], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Distribution of Amounts", fontsize=16)
    plt.tight_layout()
    plt.savefig(pie_chart_path)
    
    plt.show()

