import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def process_json_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    data = data.get('AccountStatementOverAPIResponse', {}).get('Data', {}).get('AccountStatementReportResponseBody', {}).get('data', [])
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("No transaction data found in the provided JSON.")
    
def create_json_visualizations(json_data, bar_chart_path, pie_chart_path):
    df = pd.DataFrame(json_data)
    if df.empty:
        print("Data is empty. No visualizations can be created.")
        return
    print(df.dtypes)
    numeric_cols = df.select_dtypes(include=['number'])
    if numeric_cols.empty:
        print("No numeric data found for visualization.")
        return
    numeric_means = numeric_cols.mean()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['serialNumber'], y=numeric_cols['amount'], palette='viridis')
    plt.title("Transaction Amount by Serial Number")
    plt.xlabel('Serial Number')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    transaction_amounts = df['amount']
    plt.figure(figsize=(8, 8))
    plt.pie(transaction_amounts, labels=[f"Transaction {i+1}" for i in range(len(transaction_amounts))], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(transaction_amounts)))
    plt.title("Proportion of Transaction Amounts")
    plt.tight_layout()
    plt.savefig(pie_chart_path)
    plt.close()


